#!/usr/bin/env python3
"""Train a 225-subgoal Double DQN over a frozen PPO-Attention policy."""

import argparse
import csv
import math
from collections import deque
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from ppo_agent import HighLevelAgent
from sim2d_env import SCENARIOS, Sim2DEnv, export_world, make_layout
from util import FrameStack, SubgoalProgress, SubgoalSpace


def candidates(env, space):
    points, mask, exact = space.candidates(env.state)
    if not mask.any():
        raise RuntimeError('No safe subgoal: scenario={} step={} pose=({:.4f},{:.4f},{:.4f}) '
                           'scan_min={:.6f} scan_max={:.6f} mask_count=0'.format(
                               env.kind, env.steps, env.info['x'], env.info['y'], env.info['yaw'],
                               env.ranges.min(), env.ranges.max()))
    return points, mask, exact


def run_segment(agent, env, space, history, total_steps, budget, deterministic=False):
    state = history.snapshot()
    points, mask, exact = candidates(env, space)
    index = agent.select_action(state, mask, total_steps, deterministic)
    env.set_subgoal(space.world_point(points[index], env.info))
    low_state = env.low_state()
    progress = SubgoalProgress(float(low_state[362]) * 5 * math.sqrt(2), float(low_state[363]) * math.pi,
                               .1 if index == exact else .3, env.v_max, env.w_max)
    before = env.astar_distance()
    elapsed = 0.
    for steps in range(1, budget + 1):
        action = agent.low_action(low_state)
        next_state, terminated, truncated, info = env.step(action)
        history.append(next_state)
        low_state = env.low_state()
        elapsed += info['sim_duration']
        reason = progress.update(float(low_state[362]) * 5 * math.sqrt(2),
                                 float(low_state[363]) * math.pi, elapsed)
        if terminated or truncated:
            reason = info['outcome']
        elif steps == budget:
            reason = 'checkpoint_cut'
        env.info.update(subgoal_reason=reason, segment_steps=steps, segment_elapsed=elapsed,
                        subgoal_distance=float(low_state[362]) * 5 * math.sqrt(2),
                        best_subgoal_distance=progress.best_distance,
                        best_subgoal_heading=progress.best_heading,
                        no_progress_time=elapsed - progress.last_progress, segment_time_limit=progress.limit)
        env.refresh()
        if terminated or truncated or reason != 'running':
            break
    # Safe masks make r_out zero. Stalls and time limits are not unsafe-point penalties.
    parts = dict(r_arrival=100. if info['reached'] else 0., r_collision=-100. if info['collision'] else 0.,
                 r_step=-.1 * steps, r_dist=5 * (before - env.astar_distance()), r_out=0.)
    reward = float(sum(parts.values()))
    env.info.update(parts, segment_reward=reward)
    env.refresh()
    return state, index, reward, history.snapshot(), terminated, truncated or (steps == budget and not terminated), steps


def evaluate(agent, env, space, config, suite='validation', baseline=False):
    count = 100 if suite == 'test' else config['eval_episodes']
    seed_base = config['seed'] + (2000000 if suite == 'test' else 1000000)
    rows = []
    for index in range(count):
        kind = SCENARIOS[index % len(SCENARIOS)] if config['scenario'] == 'all' else config['scenario']
        if suite == 'local':
            state, _ = env.reset_local(kind, seed_base + index)
        else:
            state, _ = env.reset(kind, seed_base + index)
        history = FrameStack()
        history.reset(state)
        reward = 0.
        if baseline or suite == 'local':
            terminated = truncated = False
            while not (terminated or truncated):
                _, terminated, truncated, _ = env.step(agent.low_action(env.low_state()))
                env.refresh()
        else:
            terminated = truncated = False
            while not (terminated or truncated):
                _, _, segment_reward, _, terminated, truncated, _ = run_segment(
                    agent, env, space, history, 0, env.timeout_steps - env.steps, deterministic=True)
                reward += segment_reward
        info = env.info
        spl = min(1., env.initial_distance / max(env.path_length, env.initial_distance)) if info['reached'] and env.initial_distance > 0 else 0.
        rows.append(dict(scenario=kind, seed=seed_base + index, outcome=info['outcome'], steps=env.steps,
                         distance=info['distance'], path_length=env.path_length, spl=spl, reward=reward))
    rates = {name: sum(row['outcome'] == name for row in rows) / count for name in ('success', 'collision', 'timeout')}
    print('eval episodes={} success={:.1%} collision={:.1%} timeout={:.1%}'.format(
        count, rates['success'], rates['collision'], rates['timeout']), flush=True)
    return rates, rows


def write_metrics(directory, records):
    records.sort(key=lambda row: (row['eval_success_rate'] if row['eval_episodes'] else -1., row['total_steps']), reverse=True)
    with (directory / 'checkpoint_metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('train', 'eval'), default='train')
    parser.add_argument('--backend', choices=('sim2d', 'gazebo'), default='sim2d')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--low-checkpoint', type=Path, default=Path(__file__).resolve().parent / 'runs/attention_paper/step_180000.pt')
    parser.add_argument('--scenario', choices=('all',) + SCENARIOS, default='all')
    parser.add_argument('--scenario-seed', type=int, default=3)
    parser.add_argument('--export-world', type=Path)
    parser.add_argument('--render', choices=('on', 'off'), default='off')
    parser.add_argument('--eval-suite', choices=('validation', 'test', 'local'), default='validation')
    parser.add_argument('--low-baseline', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'runs')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--max-train-steps', type=int, default=200000)
    parser.add_argument('--save-interval', type=int, default=10000)
    parser.add_argument('--eval-episodes', type=int, default=20)
    parser.add_argument('--timeout-steps', type=int, default=800)
    parser.add_argument('--net-width', type=int, default=256)
    parser.add_argument('--high-lr', type=float, default=3e-4)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--replay-capacity', type=int, default=50000)
    parser.add_argument('--warmup', type=int, default=256)
    parser.add_argument('--target-update', type=int, default=200)
    parser.add_argument('--epsilon-decay-steps', type=int, default=100000)
    parser.add_argument('--gamma', type=float, default=.99)
    parser.add_argument('--max-grad-norm', type=float, default=.5)
    parser.add_argument('--start', type=float, nargs=2)
    parser.add_argument('--goal', type=float, nargs=2)
    parser.add_argument('--yaw', type=float, default=180.)
    parser.add_argument('--robot-model-name', default='turtlebot3')
    parser.add_argument('--sensor-timeout', type=float, default=10.)
    args = parser.parse_args()
    for key in ('max_train_steps', 'save_interval', 'eval_episodes', 'timeout_steps', 'net_width', 'high_lr',
                'batch_size', 'replay_capacity', 'warmup', 'target_update', 'epsilon_decay_steps', 'max_grad_norm', 'sensor_timeout'):
        value = getattr(args, key)
        if not np.isfinite(value) or value <= 0:
            parser.error('--' + key.replace('_', '-') + ' must be finite and positive')
    if not args.batch_size <= args.warmup <= args.replay_capacity or not 0 <= args.gamma <= 1:
        parser.error('Require batch-size <= warmup <= replay-capacity and gamma in [0,1]')
    if args.seed < 0 or args.scenario_seed < 0:
        parser.error('Seeds must be nonnegative')
    if (args.start is None) != (args.goal is None):
        parser.error('--start and --goal must be supplied together')
    if not np.isfinite([args.yaw] + (args.start or []) + (args.goal or [])).all():
        parser.error('Route values must be finite')
    if args.start is not None and args.start == args.goal:
        parser.error('Start and goal must differ')
    if args.export_world:
        if args.scenario == 'all':
            parser.error('--export-world requires a specific --scenario')
        start, goal, yaw = export_world(args.export_world, args.scenario, args.scenario_seed)
        print('world={} start={} goal={} yaw={:.6f}'.format(args.export_world.resolve(), start.tolist(), goal.tolist(), math.degrees(yaw)))
        return
    if args.backend == 'gazebo' and (args.mode != 'eval' or args.eval_suite != 'validation'):
        parser.error('Gazebo supports --mode eval --eval-suite validation only')
    if args.backend == 'sim2d' and args.start is not None:
        parser.error('--start and --goal are for Gazebo evaluation; 2D uses generated scenario tasks')
    if args.backend == 'gazebo' and args.start is None and args.scenario == 'all':
        parser.error('Gazebo requires a specific --scenario, or explicit --start and --goal')
    if args.mode == 'eval' and args.checkpoint is None and not (args.low_baseline or args.eval_suite == 'local'):
        parser.error('High-policy evaluation requires --checkpoint')
    if args.mode == 'train' and (args.low_baseline or args.eval_suite != 'validation'):
        parser.error('Baseline and test/local suites are evaluation-only')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; use --device cpu')
    path = args.checkpoint or args.low_checkpoint
    if not path.is_file():
        parser.error('Checkpoint not found: ' + str(path))
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if args.checkpoint:
        if (saved.get('algorithm') != 'HPPO_ATTENTION' or saved['version'] != 1 or saved['observation_shape'] != (1456,)):
            parser.error('--checkpoint requires HPPO_ATTENTION v1 with four frames')
        for key in ('net_width', 'high_lr', 'batch_size', 'replay_capacity', 'warmup', 'target_update',
                    'epsilon_decay_steps', 'gamma', 'max_grad_norm'):
            setattr(args, key, saved['config'][key])
        if args.mode == 'train':
            for key in ('seed', 'scenario', 'timeout_steps'):
                if getattr(args, key) != saved['config'][key]:
                    parser.error('Resume requires unchanged --' + key.replace('_', '-'))
        low_checkpoint = saved['low_checkpoint']
    else:
        low_checkpoint = saved
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    agent = HighLevelAgent(config, low_checkpoint)
    counters = dict(total_steps=0, episode=0, tasks=0, recent_outcomes=[])
    if args.checkpoint:
        if args.mode == 'train':
            agent.restore(saved)
        else:
            agent.online.load_state_dict(saved['online'])
        counters = saved['counters']
    if args.mode == 'train' and counters['total_steps'] >= args.max_train_steps:
        parser.error('Training budget must exceed restored total_steps')
    low_config = low_checkpoint['config']
    space = SubgoalSpace(low_config['scan_angle_min'], low_config['scan_angle_increment'])
    with ExitStack() as resources:
        if args.backend == 'sim2d':
            env = Sim2DEnv(low_config, args.timeout_steps, args.render == 'on')
        else:
            import rospy
            from turtlebot3_env import HighGazeboEnv
            rospy.init_node('hppo_attention', anonymous=True)
            if args.start is None:
                _, start, goal, yaw = make_layout(args.scenario, args.scenario_seed)
            else:
                start, goal, yaw = args.start, args.goal, math.radians(args.yaw)
            env = HighGazeboEnv(low_config, start, goal, yaw, args.timeout_steps,
                                args.robot_model_name, args.sensor_timeout)
        resources.callback(env.close)
        if args.mode == 'eval':
            rates, rows = evaluate(agent, env, space, config, args.eval_suite, args.low_baseline)
            args.output_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            with (args.output_dir / ('evaluation_' + stamp + '.csv')).open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            return
        from torch.utils.tensorboard import SummaryWriter
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        run_dir = args.output_dir / ('hppo_attention_' + stamp)
        run_dir.mkdir(parents=True)
        writer = resources.enter_context(SummaryWriter(str(run_dir)))
        records = []
        outcomes = deque(counters['recent_outcomes'], maxlen=100)
        history = FrameStack()
        state, _ = env.reset(SCENARIOS[counters['tasks'] % 5] if args.scenario == 'all' else args.scenario,
                             args.seed + counters['tasks'])
        counters['tasks'] += 1
        history.reset(state)
        episode_reward = 0.
        while counters['total_steps'] < args.max_train_steps:
            total = counters['total_steps']
            budget = min(args.save_interval - total % args.save_interval, args.max_train_steps - total,
                         env.timeout_steps - env.steps)
            start, action, reward, next_state, terminated, truncated, steps = run_segment(
                agent, env, space, history, total, budget)
            counters['total_steps'] += steps
            episode_reward += reward
            next_mask = np.zeros(225, bool) if terminated else candidates(env, space)[1]
            agent.replay.add(start, action, reward, next_state, next_mask, terminated, truncated)
            if agent.replay.size >= args.warmup:
                agent.update()
            complete = env.info['outcome'] != 'running'
            if complete:
                counters['episode'] += 1
                outcomes.append(env.info['outcome'])
                writer.add_scalar('episode/reward', episode_reward, counters['episode'])
                writer.add_scalars('episode/outcome_rate_100',
                                   {name: outcomes.count(name) / len(outcomes) for name in ('success', 'collision', 'timeout')}, counters['episode'])
            final = counters['total_steps'] == args.max_train_steps
            due = counters['total_steps'] % args.save_interval == 0 or final
            if due:
                counters['recent_outcomes'] = list(outcomes)
                path = run_dir / 'step_{:06d}.pt'.format(counters['total_steps'])
                agent.save(path, counters)
                if final:
                    agent.save(run_dir / 'last.pt', counters)
                row = dict(checkpoint=path.name, total_steps=counters['total_steps'], episode=counters['episode'],
                           window_episodes=len(outcomes))
                row.update({name + '_rate_100': outcomes.count(name) / len(outcomes) if outcomes else None
                            for name in ('success', 'collision', 'timeout')})
                row.update(eval_episodes=0, eval_success_rate=None, eval_collision_rate=None, eval_timeout_rate=None,
                           **{kind + '_success_rate': None for kind in SCENARIOS})
                records.append(row)
                write_metrics(run_dir, records)
                rates, rows = evaluate(agent, env, space, config)
                row.update(eval_episodes=len(rows), **{'eval_' + name + '_rate': rate for name, rate in rates.items()})
                for kind in SCENARIOS:
                    subset = [sample for sample in rows if sample['scenario'] == kind]
                    row[kind + '_success_rate'] = sum(sample['outcome'] == 'success' for sample in subset) / len(subset) if subset else None
                write_metrics(run_dir, records)
                writer.add_scalars('eval/outcome_rate', rates, counters['total_steps'])
            if (complete or due) and not final:
                state, _ = env.reset(SCENARIOS[counters['tasks'] % 5] if args.scenario == 'all' else args.scenario,
                                     args.seed + counters['tasks'])
                counters['tasks'] += 1
                history.reset(state)
                episode_reward = 0.


if __name__ == '__main__':
    main()
