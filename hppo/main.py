#!/usr/bin/env python3
"""Two-stage navigation: four-frame Dueling Double DQN over a frozen PPO actor."""

import argparse
import csv
from collections import deque
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from ppo_agent import AutoAgent, HighLevelAgent, PPOAgent
from util import AStarViewer, FrameStack, GridMap, RolloutBuffer, SubgoalProgress, high_level_reward


def write_checkpoint_metrics(run_dir, records):
    records.sort(key=lambda row: (
        row["eval_success_rate"] if row["eval_episodes"] else -1.0, row["total_steps"]
    ), reverse=True)
    with (run_dir / "checkpoint_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)


def save_checkpoint(agent, run_dir, episode, total_steps, updates, recent_outcomes, records, final=False):
    """Save the model and its training statistics before evaluation starts."""
    path = run_dir / "step_{:06d}.pt".format(total_steps)
    agent.save(path, episode, total_steps, updates, recent_outcomes)
    if final:
        agent.save(run_dir / "last.pt", episode, total_steps, updates, recent_outcomes)
    count = len(recent_outcomes)
    record = {
        "checkpoint": path.name, "total_steps": total_steps, "episode": episode,
        "window_episodes": count,
    }
    if isinstance(agent, AutoAgent):
        record.update(phase=agent.phase, low_steps=agent.low_steps, high_steps=agent.high_steps,
                      pretrain_limit=agent.pretrain_limit, switch_step=agent.switch_step, status=agent.status)
        gate = agent.gate_result
        record.update(low_eval_episodes=gate["episodes"] if gate else 0,
                      low_eval_bank_1_success_rate=gate["bank_success_rates"][0] if gate else None,
                      low_eval_bank_2_success_rate=gate["bank_success_rates"][1] if gate else None,
                      low_eval_success_rate=gate["success"] if gate else None,
                      low_eval_collision_rate=gate["collision"] if gate else None,
                      low_eval_timeout_rate=gate["timeout"] if gate else None,
                      low_qualified=gate["passed"] if gate else None)
    else:
        phase = agent.config["stage"]
        record.update(phase=phase, low_steps=total_steps if phase == "low" else 0,
                      high_steps=total_steps if phase == "high" else 0)
    for outcome in ("success", "collision", "timeout"):
        record[outcome + "_rate_100"] = recent_outcomes.count(outcome) / count if count else None
    record.update(eval_episodes=0, eval_success_rate=None, eval_collision_rate=None, eval_timeout_rate=None)
    records.append(record)
    write_checkpoint_metrics(run_dir, records)
    return record


def run_subgoal(agent, env, history, total_steps, max_steps, deterministic=False):
    """Execute one fixed subgoal; exclude zero-command computation gaps from timers."""
    import rospy

    state = history.snapshot()
    index = agent.select_action(state, env.action_mask(), total_steps, deterministic=deterministic)
    low_state = env.set_subgoal(index)
    previous_distance = env.info["astar_distance"]
    elapsed = 0.0
    progress = SubgoalProgress(env.info["subgoal_distance"], env.info["subgoal_heading"],
                               elapsed, env.v_max, env.w_max)
    for step in range(1, max_steps + 1):
        observation, _, terminated, truncated, info = env.step(agent.low_action(low_state))
        history.append(observation)
        low_state = env.low_observation()
        elapsed += info["sim_duration"]
        reason = progress.update(info["subgoal_distance"], info["subgoal_heading"], elapsed)
        if terminated or truncated:
            reason = "success" if info["reached"] else "collision" if info["collision"] else "timeout"
        elif reason == "running" and (step == max_steps or rospy.is_shutdown()):
            reason = "sampling_cut"
        info.update(subgoal_reason=reason, subgoal_steps=step, subgoal_elapsed=elapsed,
                    subgoal_time_limit=progress.time_limit, subgoal_best_distance=progress.best_distance,
                    subgoal_best_heading=progress.best_heading,
                    subgoal_no_progress=elapsed - progress.last_progress)
        finished = reason != "running"
        if finished:
            reward, parts = high_level_reward(previous_distance, info["astar_distance"],
                                              info["reached"], info["collision"], step)
            info["high_reward"] = dict(parts, reward=reward, previous_astar_distance=previous_distance,
                                       astar_distance=info["astar_distance"], reason=reason, steps=step,
                                       elapsed=info["subgoal_elapsed"])
        if env.grid.visualize:
            env.grid.viewer.update(info)
        if finished:
            break
    return state, index, reward, history.snapshot(), terminated, truncated, step


def evaluate_episodes(policy, env, episodes, seed, policy_steps=0, tasks=None):
    """Collect deterministic evaluation rows without touching any learner state."""
    import rospy

    rows = []
    history = FrameStack()
    for index in range(episodes):
        if rospy.is_shutdown():
            break
        state, _ = env.reset(seed=seed + index, task=None if tasks is None else tasks[index])
        start, goal, yaw = env.start_xy.copy(), env.goal_xy.copy(), env.start_yaw
        if env.stage == "high":
            history.reset(state)
        done, episode_reward, elapsed, decisions = False, 0.0, 0.0, 0
        while not done and not rospy.is_shutdown():
            if env.stage == "low":
                action, _, _ = policy.select_action(state, deterministic=True)
                state, reward, terminated, truncated, info = env.step(action)
                elapsed += info["sim_duration"]
            else:
                _, _, reward, _, terminated, truncated, _ = run_subgoal(
                    policy, env, history, policy_steps, env.timeout_steps, deterministic=True,
                )
                info = env.info
                elapsed += info["subgoal_elapsed"]
                decisions += 1
            episode_reward += reward
            done = terminated or truncated
        if done:
            outcome = "success" if info["reached"] else "collision" if info["collision"] else "timeout"
            rows.append(dict(episode=index + 1, start_x=float(start[0]), start_y=float(start[1]),
                             goal_x=float(goal[0]), goal_y=float(goal[1]), initial_yaw=float(yaw),
                             outcome=outcome, steps=info["step"], sim_duration=elapsed,
                             reward=episode_reward, decisions=decisions))
    return rows


def evaluate(agent, env, episodes, seed, total_steps=0):
    """Automatic checkpoints always evaluate full navigation, even in pretraining."""
    previous_stage, previous_steps = env.stage, env.training_steps
    if isinstance(agent, AutoAgent):
        policy, policy_steps = agent.high, agent.high_steps
        env.stage = "high"
    else:
        policy, policy_steps = agent, total_steps
    env.training_steps = 20000  # Independent low-stage evaluation covers the full local curriculum.
    try:
        rows = evaluate_episodes(policy, env, episodes, seed + 1000000, policy_steps)
        if len(rows) != episodes:
            return None
        rates = {name: sum(row["outcome"] == name for row in rows) / episodes
                 for name in ("success", "collision", "timeout")}
        print("eval episodes={} success={:.1%} collision={:.1%} timeout={:.1%}".format(
            episodes, rates["success"], rates["collision"], rates["timeout"]), flush=True)
        return rates
    finally:
        env.stage, env.training_steps = previous_stage, previous_steps


def evaluate_low_gate(agent, env, seed):
    """Two disjoint fixed local banks, each requiring at least 19/20 successes."""
    previous_stage = env.stage
    env.stage = "low"
    try:
        banks = [evaluate_episodes(agent.low, env, 20, seed + 2000000 + bank * 1000, tasks=tasks)
                 for bank, tasks in enumerate(env.grid.low_gate_tasks)]
        if any(len(rows) != 20 for rows in banks):
            return None
        successes = [sum(row["outcome"] == "success" for row in rows) for rows in banks]
        rows = banks[0] + banks[1]
        return dict(episodes=40, bank_success_rates=[count / 20 for count in successes],
                    passed=all(count >= 19 for count in successes),
                    **{name: sum(row["outcome"] == name for row in rows) / 40
                       for name in ("success", "collision", "timeout")})
    finally:
        env.stage = previous_stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("auto", "low", "high"), default="auto")
    parser.add_argument("--mode", choices=("train", "eval"), default="train")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--low-checkpoint", type=Path, help="Pretrained HPPO low-stage model for new high-stage training")
    parser.add_argument("--world-file", type=Path)
    parser.add_argument("--a_star", choices=("on", "off"), default="off", help="Show A* search, path and per-step rewards")
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--max-train-steps", type=int, default=200000, help="Total environment steps, including restored steps")
    parser.add_argument("--pretrain-steps", type=int, default=30000, help="Automatic mode: earliest low-level qualification check")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-suite", choices=("standard", "coverage"), default="standard",
                        help="Manual evaluation: standard tasks or all 12 corner pairs x 10 headings")
    parser.add_argument("--save-interval", type=int, default=10000, help="Training steps between checkpoint saves and evaluations")
    parser.add_argument("--robot-model-name", default="turtlebot3_burger")
    parser.add_argument("--v-max", type=float, default=0.2)
    parser.add_argument("--w-max", type=float, default=1.5)
    parser.add_argument("--action-duration", type=float, default=0.2)
    parser.add_argument("--timeout-steps", type=int, default=800)
    parser.add_argument("--sensor-timeout", type=float, default=10.0, help="Wall-clock sensor/service startup timeout")
    parser.add_argument("--horizon", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--net-width", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--high-lr", type=float, default=3e-4)
    parser.add_argument("--replay-capacity", type=int, default=50000)
    parser.add_argument("--high-warmup", type=int, default=256, help="High-level transitions before learning starts")
    parser.add_argument("--target-update", type=int, default=200, help="Gradient updates between hard target copies")
    parser.add_argument("--epsilon-decay-steps", type=int, default=50000, help="Low-level interaction steps for epsilon decay")
    parser.add_argument("--epsilon-final", type=float, default=0.05)
    args = parser.parse_args()

    # All user configuration checks belong to this entry point.
    positive = ("max_train_steps", "eval_episodes", "save_interval", "v_max", "w_max", "action_duration",
                "timeout_steps", "sensor_timeout", "horizon", "batch_size", "epochs", "net_width",
                "actor_lr", "critic_lr", "max_grad_norm", "high_lr", "replay_capacity", "high_warmup",
                "target_update", "epsilon_decay_steps", "pretrain_steps")
    for key in positive:
        value = getattr(args, key)
        if not np.isfinite(value) or value <= 0:
            parser.error("--{} must be finite and positive".format(key.replace("_", "-")))
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if not (0.0 <= args.gamma <= 1.0 and 0.0 <= args.gae_lambda <= 1.0):
        parser.error("--gamma and --gae-lambda must be in [0, 1]")
    if not 0.0 < args.clip_ratio < 1.0:
        parser.error("--clip-ratio must be in (0, 1)")
    if not np.isfinite(args.epsilon_final) or not 0.0 <= args.epsilon_final <= 1.0:
        parser.error("--epsilon-final must be finite and in [0, 1]")
    if args.stage in ("auto", "high") and not args.batch_size <= args.high_warmup <= args.replay_capacity:
        parser.error("High-stage training requires batch-size <= high-warmup <= replay-capacity")
    if (args.stage == "auto" and args.mode == "train" and args.checkpoint is None
            and not args.pretrain_steps <= args.max_train_steps // 2 < args.max_train_steps):
        parser.error("Automatic mode requires pretrain-steps <= max-train-steps // 2 < max-train-steps")
    if args.eval_suite == "coverage" and (args.mode != "eval" or args.stage == "low"):
        parser.error("--eval-suite coverage requires --mode eval and --stage auto/high")
    if args.low_checkpoint is not None:
        if args.stage != "high" or args.checkpoint is not None:
            parser.error("--low-checkpoint is only for starting a new high stage without --checkpoint")
        if not args.low_checkpoint.is_file():
            parser.error("Low-stage checkpoint does not exist: {}".format(args.low_checkpoint))
    if args.stage == "high" and args.checkpoint is None and args.low_checkpoint is None:
        parser.error("New high-stage training requires --low-checkpoint")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable; use --device cpu")
    if args.mode == "eval" and args.checkpoint is None:
        parser.error("--mode eval requires --checkpoint")
    if args.checkpoint is not None and not args.checkpoint.is_file():
        parser.error("Checkpoint file does not exist: {}".format(args.checkpoint))
    if args.world_file is None:
        import rospkg
        args.world_file = Path(rospkg.RosPack().get_path("turtlebot3_gazebo")) / "worlds/turtlebot3_stage_4.world"
    if not args.world_file.is_file():
        parser.error("World file does not exist: {}".format(args.world_file))
    args.world_file = args.world_file.resolve()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    grid = GridMap(args.world_file)
    if args.stage != "high":
        grid.prepare_local_tasks(seed=args.seed)
    automatic = args.stage == "auto"
    training = args.mode == "train"
    if training:
        from torch.utils.tensorboard import SummaryWriter

    import rospy
    from turtlebot3_env import TurtleBot3Env
    rospy.init_node("tb3_hppo", anonymous=True)
    env = TurtleBot3Env(
        grid, v_max=args.v_max, w_max=args.w_max, action_duration=args.action_duration,
        timeout_steps=args.timeout_steps, sensor_timeout=args.sensor_timeout,
        robot_model_name=args.robot_model_name, stage="low" if automatic else args.stage,
    )
    try:
        config.update(scan_angle_min=env.scan_angle_min, scan_angle_increment=env.scan_angle_increment)
        agent = AutoAgent(config) if automatic else PPOAgent(config) if args.stage == "low" else HighLevelAgent(config)
        counters = {"episode": 0, "total_steps": 0, "updates": 0, "recent_outcomes": []}
        if args.checkpoint is not None:
            counters = agent.load(args.checkpoint, training=training)
        elif args.stage == "high":
            agent.load_low(args.low_checkpoint)
        if training and counters["total_steps"] >= args.max_train_steps:
            parser.error("--max-train-steps must exceed the number of restored environment steps")
        if automatic:
            env.stage = agent.phase
        with ExitStack() as resources:
            if args.a_star == "on":
                viewer = AStarViewer(grid)
                resources.callback(viewer.close)
            if not training:
                if args.eval_suite == "coverage":
                    env.stage = "high"
                    policy = agent.high if automatic else agent
                    policy_steps = agent.high_steps if automatic else counters["total_steps"]
                    tasks = grid.coverage_tasks()
                    rows = evaluate_episodes(policy, env, len(tasks), args.seed + 3000000, policy_steps, tasks)
                    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                    directory = args.output_dir / ("coverage_" + stamp)
                    directory.mkdir(parents=True)
                    with (directory / "coverage_results.csv").open("w", newline="") as handle:
                        fields = ("episode", "start_x", "start_y", "goal_x", "goal_y", "initial_yaw",
                                  "outcome", "steps", "sim_duration", "reward", "decisions")
                        table = csv.DictWriter(handle, fieldnames=fields)
                        table.writeheader()
                        table.writerows(rows)
                    if len(rows) == len(tasks):
                        rates = {name: sum(row["outcome"] == name for row in rows) / len(tasks)
                                 for name in ("success", "collision", "timeout")}
                        print("eval episodes={} success={:.1%} collision={:.1%} timeout={:.1%}".format(
                            len(tasks), rates["success"], rates["collision"], rates["timeout"]), flush=True)
                else:
                    evaluate(agent, env, args.eval_episodes, args.seed, counters["total_steps"])
                return
            total_steps = counters["total_steps"]
            updates = counters["updates"]
            episode = counters["episode"]
            recent_outcomes = deque(counters["recent_outcomes"], maxlen=100)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            run_dir = args.output_dir / (args.stage + "_" + stamp)
            run_dir.mkdir(parents=True)
            writer = resources.enter_context(SummaryWriter(log_dir=str(run_dir), flush_secs=10))
            if args.stage != "high":
                buffer = RolloutBuffer(args.horizon, env.state_dim)
            history = FrameStack()
            checkpoint_records = []
            env.training_steps = agent.low_steps if automatic else total_steps
            state, _ = env.reset(seed=args.seed + episode)
            if env.stage == "high":
                history.reset(state)
            while not rospy.is_shutdown() and total_steps < args.max_train_steps:
                episode_reward = 0.0
                done = False
                while not done and not rospy.is_shutdown() and total_steps < args.max_train_steps:
                    phase = agent.phase if automatic else args.stage
                    learner = (agent.low if phase == "low" else agent.high) if automatic else agent
                    if phase == "low":
                        action, raw_action, log_prob = learner.select_action(state)
                        next_state, reward, terminated, truncated, info = env.step(action)
                        total_steps += 1
                        if automatic:
                            agent.low_steps += 1
                        env.training_steps = agent.low_steps if automatic else total_steps
                    else:
                        step_limit = min(args.max_train_steps - total_steps,
                                         args.save_interval - total_steps % args.save_interval)
                        start_stack, action, reward, next_stack, terminated, truncated, steps = run_subgoal(
                            learner, env, history, agent.high_steps if automatic else total_steps, step_limit,
                        )
                        total_steps += steps
                        if automatic:
                            agent.high_steps += steps
                        info = env.info
                    episode_reward += reward
                    final_step = total_steps == args.max_train_steps
                    checkpoint_due = total_steps % args.save_interval == 0 or final_step
                    at_pretrain_limit = automatic and phase == "low" and agent.low_steps == agent.pretrain_limit
                    gate_due = (automatic and phase == "low" and agent.low_steps >= args.pretrain_steps
                                and (checkpoint_due or at_pretrain_limit))
                    cut = truncated or ((checkpoint_due or gate_due) and not terminated)
                    if phase == "low":
                        buffer.add(state, raw_action, log_prob, reward, next_state, terminated, cut)
                        state = next_state
                    else:
                        next_mask = np.zeros(225, dtype=np.bool_) if terminated else env.action_mask()
                        learner.replay.add(start_stack, action, reward, next_stack, next_mask, terminated, cut)
                        learner.decisions += 1
                    done = terminated or truncated or checkpoint_due or gate_due
                    if terminated or truncated:
                        episode += 1
                        outcome = "success" if info["reached"] else "collision" if info["collision"] else "timeout"
                        recent_outcomes.append(outcome)
                        rates = {name: recent_outcomes.count(name) / len(recent_outcomes)
                                 for name in ("success", "collision", "timeout")}
                        writer.add_scalar("episode/reward", episode_reward, episode)
                        writer.add_scalars("episode/outcome_rate_100", rates, episode)
                        if not rospy.is_shutdown() and not checkpoint_due and not gate_due:
                            state, _ = env.reset(seed=args.seed + episode)
                            if phase == "high":
                                history.reset(state)
                    if phase == "low":
                        if buffer.size == buffer.capacity or ((final_step or gate_due) and buffer.size):
                            learner.update(buffer)
                            updates += 1
                            if automatic:
                                agent.low_updates += 1
                    elif learner.replay.size >= args.high_warmup:
                        learner.update()
                        updates = learner.updates + (agent.low_updates if automatic else 0)
                    failed_gate = False
                    if gate_due:
                        agent.gate_result = evaluate_low_gate(agent, env, args.seed)
                        if agent.gate_result is None:
                            return
                        if agent.gate_result["passed"]:
                            agent.enter_high()
                            env.stage = "high"
                            recent_outcomes.clear()
                        elif at_pretrain_limit:
                            agent.status = "low_not_qualified"
                            failed_gate = True
                    if final_step and automatic:
                        agent.status = "budget_complete"
                    if checkpoint_due or failed_gate:
                        record = save_checkpoint(agent, run_dir, episode, total_steps, updates,
                                                 recent_outcomes, checkpoint_records,
                                                 final=final_step or failed_gate)
                        if checkpoint_due:
                            rates = evaluate(agent, env, args.eval_episodes, args.seed, total_steps)
                            if rates is None:
                                return
                            record.update(eval_episodes=args.eval_episodes,
                                          **{"eval_" + name + "_rate": rate for name, rate in rates.items()})
                            write_checkpoint_metrics(run_dir, checkpoint_records)
                            writer.add_scalars("eval/outcome_rate", rates, total_steps)
                        if failed_gate:
                            raise RuntimeError("底层能力未达标：两套局部任务须分别成功至少 19/20；"
                                               "已保存模型及评估结果，停止训练。")
                    if (checkpoint_due or gate_due) and not final_step and not rospy.is_shutdown():
                        state, _ = env.reset(seed=args.seed + episode)
                        if env.stage == "high":
                            history.reset(state)

    finally:
        env.close()


if __name__ == "__main__":
    main()
