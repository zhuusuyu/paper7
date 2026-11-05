#!/usr/bin/env python3
"""Train knowledge-guided high-level DQN with a permanently frozen PPO low policy."""

import argparse
import csv
import hashlib
import json
import math
import time
from collections import deque
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from ppo_agent import HighLevelAgent
from sim2d_env import SCENARIOS, Sim2DEnv, export_world, make_layout
from util import (FrameStack, SubgoalProgress, SubgoalSpace, VisitMemory, ObservedMap,
                  candidate_features)

TRACE_FIELDS = ('scenario', 'seed', 'segment', 'neural_choice', 'knowledge_choice', 'fused_choice', 'chosen',
                'alpha', 'confidence', 'preference', 'floor', 'epsilon', 'novelty',
                'exact_goal_available', 'exact_goal_selected',
                'x_start', 'y_start', 'yaw_start', 'x_end', 'y_end', 'yaw_end', 'subgoal_x', 'subgoal_y', 'steps', 'duration',
                'reason', 'path_length', 'astar_before', 'astar_after', 'r_arrival', 'r_collision',
                'r_step', 'r_dist', 'r_out', 'reward')


def candidate_state(env, space, history, memory, terminal=False):
    points, mask, exact = space.candidates(env.state)
    if not terminal and not mask.any():
        raise RuntimeError('No safe subgoal: scenario={} step={} pose=({:.4f},{:.4f},{:.4f}) '
                           'scan_min={:.6f} scan_max={:.6f}'.format(env.kind, env.steps,
                               env.info['x'], env.info['y'], env.info['yaw'], env.ranges.min(), env.ranges.max()))
    c, s = math.cos(env.info['yaw']), math.sin(env.info['yaw'])
    world = points @ np.array([[c, s], [-s, c]]) + env.position
    novelty = memory.features(world, mask, exact)
    progress, valid = memory.mapping.route_features(env.position, env.goal_xy, world, memory)
    features = candidate_features(env.state, points, exact, novelty, progress, valid, space.beams)
    state = np.concatenate((history.snapshot(), features.ravel())).astype(np.float32)
    return state, mask, exact, world, novelty


def reset_episode(env, history, memory, kind, seed):
    state, _ = env.reset(kind, seed)
    history.reset(state)
    memory.reset(env.position)
    memory.mapping = ObservedMap(env.scan_angle_min, env.scan_angle_increment)
    memory.mapping.observe(env.ranges, env.position, env.yaw)
    if env.show_memory:
        env.info.update(memory_points=memory.samples(), candidate_points=np.empty((0, 2)), candidate_novelty=np.empty(0))
    env.refresh()


def run_segment(agent, env, space, history, memory, decisions, budget, deterministic=False):
    state, mask, exact, world, novelty = candidate_state(env, space, history, memory)
    index, selection = agent.select_action(state, mask, decisions, deterministic)
    selection.update(exact_goal_available=exact >= 0 and bool(mask[exact]), exact_goal_selected=index == exact)
    memory.mapping.goal_seen |= exact >= 0 and bool(mask[exact])
    memory.mapping.goal_near |= env.info['distance'] < .5
    env.set_subgoal(world[index])
    low_state = env.low_state()
    progress = SubgoalProgress(float(low_state[362]) * 5 * math.sqrt(2), float(low_state[363]) * math.pi,
                               .1 if index == exact else .3, env.v_max, env.w_max)
    before, path_before = env.astar_distance(), env.path_length
    start = env.position.copy()
    yaw_start = float(env.info['yaw'])
    elapsed = 0.
    for steps in range(1, budget + 1):
        next_state, terminated, truncated, info = env.step(agent.low_action(low_state))
        history.append(next_state)
        memory.record(env.position)
        memory.mapping.observe(env.ranges, env.position, env.yaw)
        low_state = env.low_state()
        elapsed += info['sim_duration']
        reason = progress.update(float(low_state[362]) * 5 * math.sqrt(2), float(low_state[363]) * math.pi, elapsed)
        _, current_mask, current_exact = space.candidates(next_state)
        visible = current_exact >= 0 and bool(current_mask[current_exact])
        near = info['distance'] < .5
        if index != exact:
            if visible and not memory.mapping.goal_seen:
                reason = 'goal_visible'
            elif near and not memory.mapping.goal_near:
                reason = 'goal_near'
        memory.mapping.goal_seen |= visible
        memory.mapping.goal_near |= near
        if terminated or truncated:
            reason = info['outcome']
        elif steps == budget:
            reason = 'checkpoint_cut'
        env.info.update(subgoal_reason=reason, segment_steps=steps, segment_elapsed=elapsed,
                        subgoal_distance=float(low_state[362]) * 5 * math.sqrt(2),
                        best_subgoal_distance=progress.best_distance, best_subgoal_heading=progress.best_heading,
                        no_progress_time=elapsed - progress.last_progress, segment_time_limit=progress.limit,
                        **selection)
        if env.show_memory:
            env.info.update(memory_points=memory.samples(), candidate_points=world[mask], candidate_novelty=novelty[mask],
                            observed_map=memory.mapping, knowledge_point=world[selection['knowledge_choice']],
                            fused_point=world[selection['fused_choice']])
        env.refresh()
        if terminated or truncated or reason != 'running':
            break
    after = env.astar_distance()
    parts = dict(r_arrival=100. if info['reached'] else 0., r_collision=-100. if info['collision'] else 0.,
                 r_step=-.1 * steps, r_dist=5 * (before - after), r_out=0.)
    reward = float(sum(parts.values()))
    env.info.update(parts, segment_reward=reward)
    env.refresh()
    next_state, next_mask, _, _, _ = candidate_state(env, space, history, memory, terminal=terminated)
    if terminated:
        next_mask = np.zeros(225, bool)
    transition = dict(state=state, action=index, reward=reward, next_state=next_state, next_mask=next_mask,
                      terminated=terminated, truncated=truncated or (steps == budget and not terminated), mask=mask, steps=steps)
    record = dict(selection, x_start=float(start[0]), y_start=float(start[1]), yaw_start=yaw_start,
                  x_end=float(env.position[0]), y_end=float(env.position[1]), yaw_end=float(env.info['yaw']),
                  subgoal_x=float(env.subgoal_xy[0]), subgoal_y=float(env.subgoal_xy[1]),
                  steps=steps, duration=elapsed, reason=reason, path_length=env.path_length - path_before,
                  astar_before=before if isinstance(env, Sim2DEnv) else None,
                  astar_after=after if isinstance(env, Sim2DEnv) else None, reward=reward, **parts)
    return transition, record


def evaluate(agent, env, space, config, suite='validation', decisions=0, trace_writer=None):
    count = 100 if suite == 'test' else config['eval_episodes']
    seed_base = {'test': 7000000, 'validation': 5000000}[suite]
    rows = []
    history, memory = FrameStack(), VisitMemory()
    for index in range(count):
        kind = SCENARIOS[index % len(SCENARIOS)] if config['scenario'] == 'all' else config['scenario']
        seed = seed_base + index
        reset_episode(env, history, memory, kind, seed)
        reward = 0.
        records = []
        terminated = truncated = False
        while not (terminated or truncated):
            transition, record = run_segment(agent, env, space, history, memory, decisions,
                                              env.timeout_steps - env.steps, deterministic=True)
            terminated, truncated = transition['terminated'], transition['truncated']
            reward += transition['reward']
            records.append(record)
            if trace_writer is not None:
                trace_writer.writerow(dict(record, scenario=kind, seed=seed, segment=len(records)))
        info = env.info
        reference = env.initial_distance
        spl = min(1., reference / max(env.path_length, reference)) if info['reached'] and reference > 0 else 0.
        repeat = sum(any(math.hypot(r['x_end'] - earlier['x_end'], r['y_end'] - earlier['y_end']) < .5
                         for earlier in records[:max(i - 2, 0)]) for i, r in enumerate(records))
        rows.append(dict(scenario=kind, seed=seed, outcome=info['outcome'], steps=env.steps,
                         distance=info['distance'], path_length=env.path_length, reference_distance=reference,
                         spl=spl, reward=reward, subgoals=len(records),
                         stalled=sum(r['reason'] == 'stalled' for r in records), repeat_ends=repeat,
                         guidance_changed=sum(r['fused_choice'] != r['neural_choice'] for r in records),
                         exact_goal_missed=sum(r['exact_goal_available'] and not r['exact_goal_selected'] for r in records),
                         remaining_astar=env.astar_distance() if isinstance(env, Sim2DEnv) else None))
    rates = {name: sum(row['outcome'] == name for row in rows) / count for name in ('success', 'collision', 'timeout')}
    print('eval episodes={} success={:.1%} collision={:.1%} timeout={:.1%}'.format(
        count, rates['success'], rates['collision'], rates['timeout']), flush=True)
    return rates, rows


def write_rows(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def write_metrics(directory, records):
    records.sort(key=lambda row: (row['eval_success_rate'] if row['eval_episodes'] else -1.,
                                  row['eval_mean_spl'] if row['eval_episodes'] else -1., row['total_steps']), reverse=True)
    write_rows(directory / 'checkpoint_metrics.csv', records)


def save_evaluate(agent, env, space, config, counters, outcomes, run_dir, records, writer, final):
    """Save first; one completed evaluation fills that model's metrics row."""
    counters['recent_outcomes'] = list(outcomes)
    path = run_dir / 'step_{:06d}.pt'.format(counters['total_steps'])
    agent.save(path, counters)
    if final:
        agent.save(run_dir / 'last.pt', counters)
    row = dict(checkpoint=path.name, total_steps=counters['total_steps'], high_decisions=counters['high_decisions'],
               gradient_updates=agent.updates, target_syncs=agent.target_syncs, episode=counters['episode'],
               window_episodes=len(outcomes), high_guidance_floor=agent.guidance_floor(counters['high_decisions']))
    row.update({name + '_rate_100': outcomes.count(name) / len(outcomes) if outcomes else None
                for name in ('success', 'collision', 'timeout')})
    row.update(eval_episodes=0, eval_success_rate=None, eval_collision_rate=None, eval_timeout_rate=None,
               eval_mean_spl=None, **{kind + '_success_rate': None for kind in SCENARIOS})
    records.append(row)
    write_metrics(run_dir, records)
    started = time.perf_counter()
    rates, rows = evaluate(agent, env, space, config, decisions=counters['high_decisions'])
    counters['eval_wall_seconds'] += time.perf_counter() - started
    counters['eval_steps'] += sum(r['steps'] for r in rows)
    row.update(eval_episodes=len(rows), eval_mean_spl=float(np.mean([r['spl'] for r in rows])),
               **{'eval_' + name + '_rate': rate for name, rate in rates.items()})
    for kind in SCENARIOS:
        subset = [r for r in rows if r['scenario'] == kind]
        row[kind + '_success_rate'] = sum(r['outcome'] == 'success' for r in subset) / len(subset) if subset else None
    write_rows(run_dir / 'eval_tasks_step_{:06d}.csv'.format(counters['total_steps']), rows)
    write_metrics(run_dir, records)
    writer.add_scalars('eval/outcome_rate', rates, counters['total_steps'])
    (run_dir / 'progress.json').write_text(json.dumps(counters, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--guidance', choices=('on', 'off'), default='on')
    parser.add_argument('--mode', choices=('train', 'eval'), default='train')
    parser.add_argument('--backend', choices=('sim2d', 'gazebo'), default='sim2d')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--low-checkpoint', type=Path, default=Path(__file__).resolve().parent / 'runs/attention_paper/step_180000.pt')
    parser.add_argument('--scenario', choices=('all',) + SCENARIOS, default='all')
    parser.add_argument('--scenario-seed', type=int, default=3)
    parser.add_argument('--export-world', type=Path)
    parser.add_argument('--render', choices=('on', 'off'), default='off')
    parser.add_argument('--trace', choices=('on', 'off'), default='off', help='Write per-subgoal diagnostics in evaluation mode')
    parser.add_argument('--eval-suite', choices=('validation', 'test'), default='validation')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'runs')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--max-train-steps', type=int, default=200000)
    parser.add_argument('--save-interval', type=int, default=10000)
    parser.add_argument('--eval-episodes', type=int, default=100)
    parser.add_argument('--timeout-steps', type=int, default=800)
    parser.add_argument('--net-width', type=int, default=256)
    parser.add_argument('--high-lr', type=float, default=3e-4)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--replay-capacity', type=int, default=50000)
    parser.add_argument('--warmup', type=int, default=256)
    parser.add_argument('--updates-per-decision', type=int, default=4)
    parser.add_argument('--target-sync-decisions', type=int, default=200)
    parser.add_argument('--epsilon-initial', type=float, default=.3)
    parser.add_argument('--epsilon-final', type=float, default=.02)
    parser.add_argument('--epsilon-decay-decisions', type=int, default=2000)
    parser.add_argument('--gamma', type=float, default=.99)
    parser.add_argument('--discount-reference-steps', type=float, default=25.)
    parser.add_argument('--max-grad-norm', type=float, default=.5)
    parser.add_argument('--high-guidance-floor-initial', type=float, default=.8)
    parser.add_argument('--high-guidance-floor-decisions', type=int, default=2000)
    parser.add_argument('--high-imitation-coef', type=float, default=1.)
    parser.add_argument('--start', type=float, nargs=2)
    parser.add_argument('--goal', type=float, nargs=2)
    parser.add_argument('--yaw', type=float, default=180.)
    parser.add_argument('--robot-model-name', default='turtlebot3')
    parser.add_argument('--sensor-timeout', type=float, default=10.)
    args = parser.parse_args()
    for key in ('max_train_steps', 'save_interval', 'eval_episodes', 'timeout_steps', 'net_width', 'high_lr',
                'batch_size', 'replay_capacity', 'warmup', 'updates_per_decision', 'target_sync_decisions',
                'epsilon_decay_decisions', 'discount_reference_steps', 'max_grad_norm', 'sensor_timeout',
                'high_guidance_floor_decisions'):
        value = getattr(args, key)
        if not np.isfinite(value) or value <= 0:
            parser.error('--' + key.replace('_', '-') + ' must be finite and positive')
    if not args.batch_size <= args.warmup <= args.replay_capacity or not 0 <= args.gamma <= 1:
        parser.error('Require batch-size <= warmup <= replay-capacity and gamma in [0,1]')
    if not (np.isfinite([args.epsilon_initial, args.epsilon_final]).all() and 0 <= args.epsilon_final <= args.epsilon_initial <= 1):
        parser.error('Require 0 <= epsilon-final <= epsilon-initial <= 1')
    if not np.isfinite(args.high_guidance_floor_initial) or not 0 <= args.high_guidance_floor_initial <= 1:
        parser.error('--high-guidance-floor-initial must be finite and in [0,1]')
    if not np.isfinite(args.high_imitation_coef) or args.high_imitation_coef < 0:
        parser.error('--high-imitation-coef must be finite and nonnegative')
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
    if args.trace == 'on' and args.mode != 'eval':
        parser.error('--trace on is evaluation-only')
    if args.backend == 'gazebo' and (args.mode != 'eval' or args.eval_suite != 'validation'):
        parser.error('Gazebo supports --mode eval --eval-suite validation only')
    if args.backend == 'sim2d' and args.start is not None:
        parser.error('--start and --goal are for Gazebo evaluation; 2D uses generated scenario tasks')
    if args.backend == 'gazebo' and args.start is None and args.scenario == 'all':
        parser.error('Gazebo requires a specific --scenario, or explicit --start and --goal')
    if args.mode == 'eval' and args.checkpoint is None:
        parser.error('High-policy evaluation requires --checkpoint')
    if args.mode == 'train' and args.eval_suite != 'validation':
        parser.error('--eval-suite test is evaluation-only')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; use --device cpu')
    path = args.checkpoint or args.low_checkpoint
    if not path.is_file():
        parser.error('Checkpoint not found: ' + str(path))
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if args.checkpoint:
        if (saved.get('algorithm') != 'KGHPPO_ATTENTION' or saved['version'] != 4
                or saved['observation_shape'] != (4606,) or saved['candidate_shape'] != (225, 14) or saved['frame_shape'] != (364,)
                or saved['history_length'] != 4 or saved['memory_config'] != VisitMemory.config):
            parser.error('--checkpoint requires KGHPPO_ATTENTION v4; old high policies require their archived implementation')
        for key in ('net_width', 'high_lr', 'batch_size', 'replay_capacity', 'warmup', 'updates_per_decision',
                    'target_sync_decisions', 'epsilon_initial', 'epsilon_final', 'epsilon_decay_decisions',
                    'gamma', 'discount_reference_steps', 'max_grad_norm', 'high_guidance_floor_initial',
                    'high_guidance_floor_decisions', 'high_imitation_coef', 'guidance'):
            setattr(args, key, saved['config'][key])
        if args.mode == 'train':
            for key in ('seed', 'scenario', 'timeout_steps'):
                if getattr(args, key) != saved['config'][key]:
                    parser.error('Resume requires unchanged --' + key.replace('_', '-'))
        low_checkpoint = saved['low_checkpoint']
        low_source = saved['low_source']
    else:
        low_checkpoint = saved
        low_source = dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    agent = HighLevelAgent(config, low_checkpoint, low_source)
    counters = dict(total_steps=0, high_decisions=0, episode=0, tasks=0, recent_outcomes=[],
                    train_wall_seconds=0., eval_wall_seconds=0., eval_steps=0)
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
            rospy.init_node('kghppo_attention_v2', anonymous=True)
            if args.start is None:
                _, start, goal, yaw = make_layout(args.scenario, args.scenario_seed)
            else:
                start, goal, yaw = args.start, args.goal, math.radians(args.yaw)
            env = HighGazeboEnv(low_config, start, goal, yaw, args.timeout_steps, args.robot_model_name, args.sensor_timeout)
        env.show_memory = args.backend == 'sim2d' and args.render == 'on'
        resources.callback(env.close)
        if args.mode == 'eval':
            args.output_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            trace_writer = None
            if args.trace == 'on':
                handle = resources.enter_context((args.output_dir / ('segments_' + stamp + '.csv')).open('w', newline=''))
                trace_writer = csv.DictWriter(handle, fieldnames=TRACE_FIELDS)
                trace_writer.writeheader()
            _, rows = evaluate(agent, env, space, config, args.eval_suite,
                               counters['high_decisions'], trace_writer)
            write_rows(args.output_dir / ('evaluation_' + stamp + '.csv'), rows)
            return
        from torch.utils.tensorboard import SummaryWriter
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        run_dir = args.output_dir / ('kghppo_attention_v4_' + stamp)
        run_dir.mkdir(parents=True)
        writer = resources.enter_context(SummaryWriter(str(run_dir)))
        records = []
        outcomes = deque(counters['recent_outcomes'], maxlen=100)
        history, memory = FrameStack(), VisitMemory()
        run_started = time.perf_counter()
        (run_dir / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
        reset_episode(env, history, memory, SCENARIOS[counters['tasks'] % 5] if args.scenario == 'all' else args.scenario,
                      args.seed + counters['tasks'])
        counters['tasks'] += 1
        episode_reward = 0.
        train_started = time.perf_counter()
        while counters['total_steps'] < args.max_train_steps:
            total = counters['total_steps']
            budget = min(args.save_interval - total % args.save_interval, args.max_train_steps - total,
                         env.timeout_steps - env.steps)
            transition, _ = run_segment(agent, env, space, history, memory, counters['high_decisions'], budget)
            counters['total_steps'] += transition['steps']
            counters['high_decisions'] += 1
            episode_reward += transition['reward']
            agent.replay.add(**transition)
            if agent.replay.size >= args.warmup:
                diagnostics = agent.learn(counters['high_decisions'])
                writer.add_scalars('high_guidance/weights', diagnostics['guidance'], counters['total_steps'])
                writer.add_scalars('high_optimization/loss', diagnostics['loss'], counters['total_steps'])
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
                counters['train_wall_seconds'] += time.perf_counter() - train_started
                save_evaluate(agent, env, space, config, counters, outcomes, run_dir, records, writer, final)
                train_started = time.perf_counter()
            if (complete or due) and not final:
                reset_episode(env, history, memory, SCENARIOS[counters['tasks'] % 5] if args.scenario == 'all' else args.scenario,
                              args.seed + counters['tasks'])
                counters['tasks'] += 1
                episode_reward = 0.
        timing = dict(train_wall_seconds=counters['train_wall_seconds'], eval_wall_seconds=counters['eval_wall_seconds'],
                      total_wall_seconds=time.perf_counter() - run_started, total_steps=counters['total_steps'], eval_steps=counters['eval_steps'],
                      high_decisions=counters['high_decisions'], gradient_updates=agent.updates)
        (run_dir / 'timing.json').write_text(json.dumps(timing, indent=2) + '\n')


if __name__ == '__main__':
    main()
