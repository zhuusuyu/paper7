#!/usr/bin/env python3
"""Train or evaluate a basic PPO policy in a TurtleBot3 Gazebo world."""

import argparse
import csv
from collections import deque
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from ppo_agent import PPOAgent
from util import GridMap, RolloutBuffer


def save_checkpoint(agent, run_dir, episode, total_steps, updates, recent_outcomes, records, final=False):
    """Save weights first, then rank the checkpoints by their training-window score."""
    path = run_dir / "step_{:06d}.pt".format(total_steps)
    agent.save(path, episode, total_steps, updates, recent_outcomes)
    if final:
        agent.save(run_dir / "last.pt", episode, total_steps, updates, recent_outcomes)
    count = len(recent_outcomes)
    record = {
        "checkpoint": path.name, "total_steps": total_steps, "episode": episode,
        "window_episodes": count,
    }
    for outcome in ("success", "collision", "timeout"):
        record[outcome + "_rate_100"] = recent_outcomes.count(outcome) / count if count else None
    records.append(record)
    records.sort(key=lambda row: (
        row["success_rate_100"] if row["window_episodes"] else -1.0, row["total_steps"]
    ), reverse=True)
    with (run_dir / "checkpoint_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=record.keys())
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "eval"), default="train")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--world-file", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--max-train-steps", type=int, default=100000, help="Total environment steps, including restored steps")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--save-interval", type=int, default=5000, help="Environment steps between checkpoints")
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
    args = parser.parse_args()

    # All user configuration checks belong to this entry point.
    positive = ("max_train_steps", "eval_episodes", "save_interval", "v_max", "w_max", "action_duration",
                "timeout_steps", "sensor_timeout", "horizon", "batch_size", "epochs", "net_width",
                "actor_lr", "critic_lr", "max_grad_norm")
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
    print("world_file={}".format(args.world_file), flush=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    grid = GridMap(args.world_file)
    agent = PPOAgent(config)
    training = args.mode == "train"
    counters = {"episode": 0, "total_steps": 0, "updates": 0, "recent_outcomes": []}
    if args.checkpoint is not None:
        counters = agent.load(args.checkpoint, training=training)
    if training and counters["total_steps"] >= args.max_train_steps:
        parser.error("--max-train-steps must exceed the number of restored environment steps")
    if training:
        from torch.utils.tensorboard import SummaryWriter

    import rospy
    from turtlebot3_env import TurtleBot3Env
    rospy.init_node("tb3_ppo", anonymous=True)
    env = TurtleBot3Env(
        grid, v_max=args.v_max, w_max=args.w_max, action_duration=args.action_duration,
        timeout_steps=args.timeout_steps, sensor_timeout=args.sensor_timeout,
        robot_model_name=args.robot_model_name,
    )
    try:
        total_steps = counters["total_steps"] if training else 0
        updates = counters["updates"] if training else 0
        episode = counters["episode"] if training else 0
        recent_outcomes = deque(counters["recent_outcomes"] if training else (), maxlen=100)
        print("state_dim=364 action_dim=2 device={}".format(args.device), flush=True)
        with ExitStack() as resources:
            if training:
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                run_dir = args.output_dir / ("train_" + stamp)
                run_dir.mkdir(parents=True)
                writer = resources.enter_context(SummaryWriter(log_dir=str(run_dir), flush_secs=10))
                buffer = RolloutBuffer(args.horizon, env.state_dim)
                checkpoint_records = []
                print("output={}".format(run_dir), flush=True)
            state, reset_info = env.reset(seed=args.seed + episode)
            print("reset episode={} start=({:.2f}, {:.2f}) goal=({:.2f}, {:.2f})".format(
                episode + 1, reset_info["x"], reset_info["y"],
                reset_info["goal_x"], reset_info["goal_y"]), flush=True)
            while not rospy.is_shutdown() and (
                total_steps < args.max_train_steps if training else episode < args.eval_episodes
            ):
                episode_reward, steps = 0.0, 0
                done = False
                while not done and not rospy.is_shutdown() and (not training or total_steps < args.max_train_steps):
                    action, raw_action, log_prob = agent.select_action(state, deterministic=not training)
                    next_state, reward, terminated, truncated, info = env.step(action)
                    steps += 1
                    total_steps += 1
                    episode_reward += reward
                    if training:
                        buffer.add(state, raw_action, log_prob, reward, next_state, terminated, truncated)
                    state = next_state
                    done = terminated or truncated
                    if done:
                        episode += 1
                        outcome = "success" if info["reached"] else "collision" if info["collision"] else "timeout"
                        recent_outcomes.append(outcome)
                        rates = {name: recent_outcomes.count(name) / len(recent_outcomes)
                                 for name in ("success", "collision", "timeout")}
                        if training:
                            writer.add_scalar("episode/reward", episode_reward, episode)
                            writer.add_scalars("episode/outcome_rate_100", rates, episode)
                        print("episode={} total_steps={} steps={} reward={:.3f} "
                              "success_100={:.1%} collision_100={:.1%} timeout_100={:.1%}".format(
                                  episode, total_steps, steps, episode_reward,
                                  rates["success"], rates["collision"], rates["timeout"]), flush=True)
                        if not rospy.is_shutdown() and (
                            total_steps < args.max_train_steps if training else episode < args.eval_episodes
                        ):
                            state, reset_info = env.reset(seed=args.seed + episode)
                            print("reset episode={} start=({:.2f}, {:.2f}) goal=({:.2f}, {:.2f})".format(
                                episode + 1, reset_info["x"], reset_info["y"],
                                reset_info["goal_x"], reset_info["goal_y"]), flush=True)
                    if training and buffer.size == buffer.capacity:
                        agent.update(buffer)
                        updates += 1
                    # At the budget boundary, save only after the final partial update below.
                    if training and total_steps % args.save_interval == 0 and total_steps < args.max_train_steps:
                        save_checkpoint(agent, run_dir, episode, total_steps, updates,
                                        recent_outcomes, checkpoint_records)
            if training and total_steps == args.max_train_steps:
                if buffer.size:
                    agent.update(buffer)
                    updates += 1
                save_checkpoint(agent, run_dir, episode, total_steps, updates,
                                recent_outcomes, checkpoint_records, final=True)
                print("training complete: total_steps={} episodes={} updates={}".format(
                    total_steps, episode, updates), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
