#!/usr/bin/env python3
"""Evaluate the best saved PPO-Attention policy on a fixed route, without A*."""

import argparse
import csv
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rospy
import torch
from geometry_msgs.msg import Pose, Twist

from ppo_agent import PPOAgent
from turtlebot3_env import TurtleBot3Env


class EvalEnv(TurtleBot3Env):
    """Reuse ROS transport and cleanup; override the task and omit map/reward logic."""

    def __init__(self, config, robot_model_name, start, goal, yaw):
        self.start_xy = np.asarray(start, dtype=np.float64)
        self.goal_xy = np.asarray(goal, dtype=np.float64)
        self.start_yaw = math.radians(yaw)
        super().__init__(None, v_max=config["v_max"], w_max=config["w_max"],
                         action_duration=config["action_duration"], timeout_steps=config["timeout_steps"],
                         sensor_timeout=config["sensor_timeout"], robot_model_name=robot_model_name)

    def _observe(self, scan, odom):
        ranges = np.asarray(scan.ranges, dtype=np.float32)
        ranges = np.clip(np.nan_to_num(ranges, nan=0., posinf=3.5, neginf=0.), 0., 3.5)
        position = odom.pose.pose.position
        q = odom.pose.pose.orientation
        yaw = math.atan2(2. * (q.w * q.z + q.x * q.y), 1. - 2. * (q.y * q.y + q.z * q.z))
        delta = self.goal_xy - np.array([position.x, position.y])
        distance = float(np.hypot(*delta))
        heading = (math.atan2(delta[1], delta[0]) - yaw + math.pi) % (2. * math.pi) - math.pi
        twist = odom.twist.twist
        # Keep the trained stage-4 scale; House coordinates do not change the observation definition.
        obs = np.concatenate((ranges / 3.5, np.array([
            twist.linear.x / self.v_max, twist.angular.z / self.w_max,
            distance / (5. * math.sqrt(2.)), heading / math.pi,
        ], dtype=np.float32))).astype(np.float32)
        return obs, dict(step=self.steps, x=float(position.x), y=float(position.y), yaw=yaw,
                         goal_x=float(self.goal_xy[0]), goal_y=float(self.goal_xy[1]),
                         distance=distance, heading=heading, min_scan=float(ranges.min()))

    def reset(self):
        self.stop()
        self.reset_world()
        self._set_pose(self.robot_model_name, self.start_xy, self.start_yaw)
        if self.goal_model_name in self.world_properties().model_names:
            self._delete_goal(self.goal_model_name)
        time.sleep(0.2)
        pose = Pose()
        pose.position.x, pose.position.y = map(float, self.goal_xy)
        pose.orientation.w = 1.
        response = self.spawn_model(self.goal_model_name, self._goal_sdf, "", pose, "world")
        if not response.success:
            raise RuntimeError("Cannot spawn goal: " + response.status_message)
        with self._condition:
            counts = self._scan_count, self._odom_count
        started = rospy.Time.now().to_sec()
        try:
            self.unpause_sim()
            rospy.sleep(0.15)
            scan, odom = self._wait_messages(counts, started)
        finally:
            self.pause_sim()
        self.steps = 0
        return self._observe(scan, odom)

    def step(self, action):
        with self._condition:
            counts = self._scan_count, self._odom_count
        command = Twist()
        command.linear.x, command.angular.z = map(float, action)
        started = rospy.Time.now().to_sec()
        self.cmd_pub.publish(command)
        try:
            self.unpause_sim()
            rospy.sleep(self.action_duration)
        finally:
            self.pause_sim()
        scan, odom = self._wait_messages(counts, started)
        self.steps += 1
        obs, info = self._observe(scan, odom)
        collision = info["min_scan"] < 0.15
        reached = info["distance"] < 0.1 and not collision
        terminated = reached or collision
        truncated = self.steps >= self.timeout_steps and not terminated
        info["outcome"] = "success" if reached else "collision" if collision else "timeout" if truncated else "running"
        if terminated or truncated:
            self.stop()
        return obs, None, terminated, truncated, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("house", "stage4"), default="house",
                        help="Connect to an already launched scene; this does not start Gazebo")
    parser.add_argument("--start", type=float, nargs=2, required=True, metavar=("X", "Y"))
    parser.add_argument("--goal", type=float, nargs=2, required=True, metavar=("X", "Y"))
    parser.add_argument("--yaw", type=float, default=180., help="Initial world yaw in degrees; no rotation action")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).resolve().parent / "runs/attention_paper")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()

    if not np.isfinite([*args.start, *args.goal, args.yaw]).all():
        parser.error("Route coordinates and yaw must be finite")
    if args.start == args.goal or args.episodes < 1:
        parser.error("Start and goal must differ and --episodes must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu")
    metrics = args.model_dir / "checkpoint_metrics.csv"
    if not metrics.is_file():
        parser.error("Missing model scores: {}".format(metrics))
    with metrics.open(newline="") as handle:
        candidates = [row for row in csv.DictReader(handle) if int(row["eval_episodes"]) > 0]
    if not candidates:
        parser.error("No completed evaluations in {}".format(metrics))
    best = max(candidates, key=lambda row: (float(row["eval_success_rate"]), int(row["total_steps"])))
    checkpoint = args.model_dir / best["checkpoint"]
    if not checkpoint.is_file():
        parser.error("Best checkpoint does not exist: {}".format(checkpoint))
    saved = torch.load(checkpoint, map_location=args.device, weights_only=False)
    if (saved.get("algorithm") != "PPO_ATTENTION" or saved["version"] != 1
            or saved["observation_shape"] != (364,)):
        parser.error("A complete PPO_ATTENTION version-1 checkpoint is required")
    config = dict(saved["config"], device=args.device)
    agent = PPOAgent(config)
    for name in ("actor", "fuzzy", "fusion", "critic"):
        module = getattr(agent, name)
        module.load_state_dict(saved[name])
        module.requires_grad_(False).eval()

    rospy.init_node("tb3_ppo_attention_eval", anonymous=True)
    env = EvalEnv(config, {"house": "turtlebot3", "stage4": "turtlebot3_burger"}[args.scene],
                  args.start, args.goal, args.yaw)
    try:
        for key in ("scan_angle_min", "scan_angle_increment"):
            if getattr(env, key) != config[key]:
                raise ValueError("Live {} differs from the trained policy".format(key))
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        trajectory = Path(__file__).resolve().parent / "runs" / ("eval_" + args.scene + "_" + stamp + ".csv")
        trajectory.parent.mkdir(parents=True, exist_ok=True)
        print("model={} source_eval_success={:.1%}".format(checkpoint, float(best["eval_success_rate"])), flush=True)
        print("trajectory={}".format(trajectory), flush=True)
        with trajectory.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=(
                "episode", "step", "sim_time", "x", "y", "yaw", "goal_x", "goal_y", "distance", "heading",
                "min_scan", "v_cmd", "w_cmd", "alpha_v", "alpha_w", "confidence", "outcome",
            ))
            writer.writeheader()
            for episode in range(1, args.episodes + 1):
                if rospy.is_shutdown():
                    break
                state, info = env.reset()
                writer.writerow(dict(info, episode=episode, sim_time=rospy.Time.now().to_sec(), outcome="start"))
                done = False
                while not done and not rospy.is_shutdown():
                    action, _, _, _, guidance = agent.select_action(state, saved["total_steps"], deterministic=True)
                    state, _, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    writer.writerow(dict(info, episode=episode, sim_time=rospy.Time.now().to_sec(),
                                         v_cmd=float(action[0]), w_cmd=float(action[1]), alpha_v=float(guidance[0]),
                                         alpha_w=float(guidance[1]), confidence=float(guidance[2])))
                    handle.flush()
                if done:
                    print("eval episode={} outcome={} steps={} final_distance={:.6f}m".format(
                        episode, info["outcome"], info["step"], info["distance"]), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
