#!/usr/bin/env python3
"""Train or evaluate fuzzy knowledge-guided PPO in a TurtleBot3 Gazebo world."""

import argparse
import csv
from collections import deque
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from ppo_agent import KGPPOAgent
from util import AStarViewer, GridMap, RolloutBuffer


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
        "guidance_alpha": agent.guidance_weight(total_steps),
    }
    for outcome in ("success", "collision", "timeout"):
        record[outcome + "_rate_100"] = recent_outcomes.count(outcome) / count if count else None
    record.update(eval_episodes=0, eval_success_rate=None, eval_collision_rate=None, eval_timeout_rate=None)
    records.append(record)
    write_checkpoint_metrics(run_dir, records)
    return record


def evaluate(agent, env, episodes, seed, total_steps):
    """Run fixed-seed deterministic episodes; an interrupted evaluation has no score."""
    import rospy

    outcomes = []
    for index in range(episodes):
        if rospy.is_shutdown():
            break
        state, _ = env.reset(seed=seed + 1000000 + index)
        done = False
        while not done and not rospy.is_shutdown():
            action, _, _, _ = agent.select_action(state, total_steps, deterministic=True)
            state, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        if done:
            outcome = "success" if info["reached"] else "collision" if info["collision"] else "timeout"
            outcomes.append(outcome)
    if len(outcomes) != episodes:
        return None
    rates = {name: outcomes.count(name) / episodes for name in ("success", "collision", "timeout")}
    print("eval episodes={} success={:.1%} collision={:.1%} timeout={:.1%}".format(
        episodes, rates["success"], rates["collision"], rates["timeout"]), flush=True)
    return rates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "eval"), default="train")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--world-file", type=Path)
    parser.add_argument("--a_star", choices=("on", "off"), default="off", help="Show A* search, path and per-step rewards")
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--max-train-steps", type=int, default=100000, help="Total environment steps, including restored steps")
    parser.add_argument("--eval-episodes", type=int, default=20)
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
    parser.add_argument("--guidance-initial", type=float, default=0.8)
    parser.add_argument("--guidance-steps", type=int, default=80000)
    parser.add_argument("--guidance-exp-rate", type=float, default=3.0)
    args = parser.parse_args()

    # All user configuration checks belong to this entry point.
    positive = ("max_train_steps", "eval_episodes", "save_interval", "v_max", "w_max", "action_duration",
                "timeout_steps", "sensor_timeout", "horizon", "batch_size", "epochs", "net_width",
                "actor_lr", "critic_lr", "max_grad_norm", "guidance_steps", "guidance_exp_rate")
    for key in positive:
        value = getattr(args, key)
        if not np.isfinite(value) or value <= 0:
            parser.error("--{} must be finite and positive".format(key.replace("_", "-")))
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if not np.isfinite(args.guidance_initial) or not 0.0 <= args.guidance_initial <= 1.0:
        parser.error("--guidance-initial must be finite and in [0, 1]")
    if not (0.0 <= args.gamma <= 1.0 and 0.0 <= args.gae_lambda <= 1.0):
        parser.error("--gamma and --gae-lambda must be in [0, 1]")
    if not 0.0 < args.clip_ratio < 1.0:
        parser.error("--clip-ratio must be in (0, 1)")
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
    training = args.mode == "train"
    if training:
        from torch.utils.tensorboard import SummaryWriter

    import rospy
    from turtlebot3_env import TurtleBot3Env
    rospy.init_node("tb3_kgppo", anonymous=True)
    env = TurtleBot3Env(
        grid, v_max=args.v_max, w_max=args.w_max, action_duration=args.action_duration,
        timeout_steps=args.timeout_steps, sensor_timeout=args.sensor_timeout,
        robot_model_name=args.robot_model_name,
    )
    try:
        config.update(scan_angle_min=env.scan_angle_min, scan_angle_increment=env.scan_angle_increment)
        agent = KGPPOAgent(config)
        counters = {"episode": 0, "total_steps": 0, "updates": 0, "recent_outcomes": []}
        if args.checkpoint is not None:
            counters = agent.load(args.checkpoint, training=training)
        if training and counters["total_steps"] >= args.max_train_steps:
            parser.error("--max-train-steps must exceed the number of restored environment steps")
        with ExitStack() as resources:
            if args.a_star == "on":
                viewer = AStarViewer(grid)
                resources.callback(viewer.close)
            if not training:
                evaluate(agent, env, args.eval_episodes, args.seed, counters["total_steps"])
                return
            total_steps = counters["total_steps"]
            updates = counters["updates"]
            episode = counters["episode"]
            recent_outcomes = deque(counters["recent_outcomes"], maxlen=100)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            run_dir = args.output_dir / ("train_" + stamp)
            run_dir.mkdir(parents=True)
            writer = resources.enter_context(SummaryWriter(log_dir=str(run_dir), flush_secs=10))
            buffer = RolloutBuffer(args.horizon, env.state_dim)
            checkpoint_records = []
            state, _ = env.reset(seed=args.seed + episode)
            while not rospy.is_shutdown() and total_steps < args.max_train_steps:
                episode_reward = 0.0
                done = False
                while not done and not rospy.is_shutdown() and total_steps < args.max_train_steps:
                    action, raw_action, log_prob, alpha = agent.select_action(state, total_steps)
                    next_state, reward, terminated, truncated, info = env.step(action)
                    total_steps += 1
                    episode_reward += reward
                    final_step = total_steps == args.max_train_steps
                    checkpoint_due = total_steps % args.save_interval == 0 or final_step
                    buffer.add(state, raw_action, log_prob, alpha, reward, next_state, terminated,
                               truncated or (checkpoint_due and not terminated))
                    state = next_state
                    done = terminated or truncated or checkpoint_due
                    if terminated or truncated:
                        episode += 1
                        outcome = "success" if info["reached"] else "collision" if info["collision"] else "timeout"
                        recent_outcomes.append(outcome)
                        rates = {name: recent_outcomes.count(name) / len(recent_outcomes)
                                 for name in ("success", "collision", "timeout")}
                        writer.add_scalar("episode/reward", episode_reward, episode)
                        writer.add_scalars("episode/outcome_rate_100", rates, episode)
                        if not rospy.is_shutdown() and not checkpoint_due:
                            state, _ = env.reset(seed=args.seed + episode)
                    if buffer.size == buffer.capacity or (final_step and buffer.size):
                        agent.update(buffer)
                        updates += 1
                    if checkpoint_due:
                        record = save_checkpoint(agent, run_dir, episode, total_steps, updates,
                                                 recent_outcomes, checkpoint_records, final=final_step)
                        rates = evaluate(agent, env, args.eval_episodes, args.seed, total_steps)
                        if rates is None:
                            return
                        record.update(eval_episodes=args.eval_episodes,
                                      **{"eval_" + name + "_rate": rate for name, rate in rates.items()})
                        write_checkpoint_metrics(run_dir, checkpoint_records)
                        writer.add_scalars("eval/outcome_rate", rates, total_steps)
                        if not final_step and not rospy.is_shutdown():
                            state, _ = env.reset(seed=args.seed + episode)
    finally:
        env.close()


if __name__ == "__main__":
    main()
