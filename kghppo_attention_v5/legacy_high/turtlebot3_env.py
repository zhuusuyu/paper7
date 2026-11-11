"""One TurtleBot3 in a ROS1 Gazebo world."""

from contextlib import ExitStack
import math
import threading
import time

import numpy as np
import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import DeleteModel, GetWorldProperties, SetModelState, SpawnModel
from geometry_msgs.msg import Pose, Quaternion, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty

from util import navigation_reward


class TurtleBot3Env:
    state_dim = 364
    action_dim = 2

    def __init__(self, grid, v_max=0.2, w_max=1.5, action_duration=0.2,
                 timeout_steps=800, sensor_timeout=10.0, robot_model_name="turtlebot3_burger"):
        self.grid = grid
        self.v_max, self.w_max = v_max, w_max
        self.action_duration = action_duration
        self.timeout_steps = timeout_steps
        self.sensor_timeout = sensor_timeout
        self.robot_model_name = robot_model_name
        self.goal_model_name = "ppo_goal"
        self._closed = False
        self._close_lock = threading.RLock()
        self._condition = threading.Condition()
        self._scan_count = self._odom_count = 0
        self.rng = np.random.default_rng()
        for name in ("reset_world", "pause_physics", "unpause_physics", "set_model_state",
                     "spawn_sdf_model", "delete_model", "get_world_properties"):
            rospy.wait_for_service("/gazebo/" + name, timeout=sensor_timeout)
        self.reset_world = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        self.pause_sim = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.unpause_sim = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.set_model = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
        self.spawn_model = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
        self.delete_model = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)
        self.world_properties = rospy.ServiceProxy("/gazebo/get_world_properties", GetWorldProperties)
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.scan_sub = rospy.Subscriber("/scan", LaserScan, self._scan_cb, queue_size=1)
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self._odom_cb, queue_size=1)
        rospy.on_shutdown(self.close)

        with ExitStack() as initialization:
            initialization.callback(self.close)
            deadline = time.monotonic() + sensor_timeout
            while self.cmd_pub.get_num_connections() == 0:
                if rospy.is_shutdown():
                    raise rospy.ROSInterruptException("ROS stopped during initialization")
                if time.monotonic() >= deadline:
                    raise TimeoutError("No subscriber connected to /cmd_vel")
                time.sleep(0.01)
            self.stop()
            try:
                self.unpause_sim()
                scan, _ = self._wait_messages((0, 0), not_before=0.0)
                if len(scan.ranges) != 360 or not math.isclose(
                    abs(scan.angle_increment) * 359, 2.0 * math.pi, abs_tol=0.05
                ):
                    raise ValueError("/scan must contain 360 beams covering a full circle")
                self.scan_angle_min = float(scan.angle_min)
                self.scan_angle_increment = float(scan.angle_increment)
            finally:
                self.pause_sim()
            for name in self.world_properties().model_names:
                if name == self.goal_model_name or (name.startswith("tb3_ppo_") and name.endswith("_goal")):
                    self._delete_goal(name)
            initialization.pop_all()

        self._goal_sdf = """<sdf version='1.6'><model name='{name}'>
          <static>true</static><link name='link'><visual name='visual'>
          <pose>0 0 0.01 0 0 0</pose><geometry><cylinder>
          <radius>0.2</radius><length>0.02</length></cylinder></geometry>
          <material><ambient>0 1 0 1</ambient><diffuse>0 1 0 1</diffuse></material>
          </visual></link></model></sdf>""".format(name=self.goal_model_name)

    def _scan_cb(self, message):
        with self._condition:
            self._scan = message
            self._scan_count += 1
            self._condition.notify_all()

    def _odom_cb(self, message):
        with self._condition:
            self._odom = message
            self._odom_count += 1
            self._condition.notify_all()

    def _wait_messages(self, counts, not_before):
        deadline = time.monotonic() + self.sensor_timeout
        with self._condition:
            while not rospy.is_shutdown():
                if self._scan_count > counts[0] and self._odom_count > counts[1]:
                    scan_stamp = self._scan.header.stamp.to_sec()
                    odom_stamp = self._odom.header.stamp.to_sec()
                    # Read messages from this action, without waiting for a future
                    # timestamp that cannot be reached while physics is paused.
                    if scan_stamp > not_before and odom_stamp > not_before:
                        return self._scan, self._odom
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError("Timed out waiting for fresh /scan and /odom messages")
                self._condition.wait(min(remaining, 0.1))
        raise rospy.ROSInterruptException("ROS stopped while waiting for sensors")

    def _set_pose(self, model_name, xy, yaw):
        state = ModelState()
        state.model_name = model_name
        state.reference_frame = "world"
        state.pose.position.x, state.pose.position.y = map(float, xy)
        state.pose.orientation = Quaternion(0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))
        response = self.set_model(state)
        if not response.success:
            raise RuntimeError("Cannot position {}: {}".format(model_name, response.status_message))

    def _delete_goal(self, name):
        response = self.delete_model(name)
        if not response.success:
            raise RuntimeError("Cannot delete {}: {}".format(name, response.status_message))

    def _observe(self, scan, odom):
        ranges = np.asarray(scan.ranges, dtype=np.float32)
        ranges = np.nan_to_num(ranges, nan=0.0, posinf=3.5, neginf=0.0)
        ranges = np.clip(ranges, 0.0, 3.5)
        position = odom.pose.pose.position
        q = odom.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        delta = self.goal_xy - np.array([position.x, position.y])
        distance = float(np.hypot(*delta))
        heading = (math.atan2(delta[1], delta[0]) - yaw + math.pi) % (2.0 * math.pi) - math.pi
        twist = odom.twist.twist
        obs = np.concatenate((ranges / 3.5, np.array([
            twist.linear.x / self.v_max, twist.angular.z / self.w_max,
            distance / (2.0 * self.grid.world_half * math.sqrt(2.0)), heading / math.pi,
        ], dtype=np.float32))).astype(np.float32)
        info = {
            "x": float(position.x), "y": float(position.y), "yaw": yaw,
            "goal_x": float(self.goal_xy[0]), "goal_y": float(self.goal_xy[1]),
            "distance": distance, "heading": heading, "min_scan": float(ranges.min()),
            "astar_distance": self.grid.distance((position.x, position.y), self.goal_xy),
        }
        return obs, info

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.stop()
        self.reset_world()
        start_index, goal_index = self.rng.choice(4, size=2, replace=False)
        self.start_xy = self.grid.task_points[start_index].copy()
        self.goal_xy = self.grid.task_points[goal_index].copy()
        self.start_yaw = math.pi / 2.0 if self.start_xy[1] < 0 else -math.pi / 2.0
        self._set_pose(self.robot_model_name, self.start_xy, self.start_yaw)
        if self.goal_model_name in self.world_properties().model_names:
            self._delete_goal(self.goal_model_name)
        # Let the GUI process deletion, including stale-goal cleanup at startup.
        time.sleep(0.2)
        pose = Pose()
        pose.position.x, pose.position.y = map(float, self.goal_xy)
        pose.orientation.w = 1.0
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
        obs, info = self._observe(scan, odom)
        self.previous_distance = info["astar_distance"]
        info.update(step=0, reached=False, collision=False, timeout=False)
        if self.grid.visualize:
            self.grid.viewer.update(info)
        return obs, info

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
        reward, reached, collision, parts = navigation_reward(
            self.previous_distance, info["astar_distance"], info["distance"], info["min_scan"],
        )
        terminated = reached or collision
        truncated = self.steps >= self.timeout_steps and not terminated
        info.update(parts, step=self.steps, previous_astar_distance=self.previous_distance,
                    reward=reward, reached=reached, collision=collision, timeout=truncated)
        self.previous_distance = info["astar_distance"]
        if terminated or truncated:
            self.stop()
        if self.grid.visualize:
            self.grid.viewer.update(info)
        return obs, reward, terminated, truncated, info

    def stop(self):
        self.cmd_pub.publish(Twist())

    def close(self):
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            self.stop()
            try:
                self.pause_sim()
                if self.goal_model_name in self.world_properties().model_names:
                    self._delete_goal(self.goal_model_name)
            finally:
                self.scan_sub.unregister()
                self.odom_sub.unregister()


class HighGazeboEnv(TurtleBot3Env):
    """Evaluation adapter: final-goal observations, fixed subgoals, no global map."""

    def __init__(self, low_config, start, goal, yaw, timeout_steps, robot_model_name, sensor_timeout):
        self.start_xy = np.asarray(start, dtype=np.float64)
        self.goal_xy = np.asarray(goal, dtype=np.float64)
        self.start_yaw = yaw
        self.subgoal_model_name = 'hppo_subgoal'
        super().__init__(None, low_config['v_max'], low_config['w_max'], low_config['action_duration'],
                         timeout_steps, sensor_timeout, robot_model_name)
        for key in ('scan_angle_min', 'scan_angle_increment'):
            if getattr(self, key) != low_config[key]:
                self.close()
                raise ValueError('Live ' + key + ' differs from the frozen low policy')
        if self.subgoal_model_name in self.world_properties().model_names:
            self._delete_goal(self.subgoal_model_name)

    def _observe(self, scan, odom):
        from util import goal_observation
        self.ranges = np.clip(np.nan_to_num(np.asarray(scan.ranges, np.float32),
                                           nan=0., posinf=3.5, neginf=0.), 0., 3.5)
        position = odom.pose.pose.position
        q = odom.pose.pose.orientation
        self.position = np.array([position.x, position.y])
        self.yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.velocity = np.array([odom.twist.twist.linear.x, odom.twist.twist.angular.z])
        state, distance, heading = goal_observation(self.ranges, self.position, self.yaw, self.velocity,
                                                  self.goal_xy, self.v_max, self.w_max)
        collision = self.ranges.min() < .15
        reached = distance < .1 and not collision
        timeout = self.steps >= self.timeout_steps and not (collision or reached)
        info = dict(x=float(position.x), y=float(position.y), yaw=self.yaw, distance=distance, heading=heading,
                    min_scan=float(self.ranges.min()), step=self.steps, reached=reached, collision=collision,
                    timeout=timeout, sim_time=rospy.Time.now().to_sec(), sim_duration=self.last_duration,
                    outcome='collision' if collision else 'success' if reached else 'timeout' if timeout else 'running')
        return state, info

    def reset(self, kind, seed):
        self.kind = kind
        self.stop()
        self.reset_world()
        self._set_pose(self.robot_model_name, self.start_xy, self.start_yaw)
        names = self.world_properties().model_names
        for name in (self.goal_model_name, self.subgoal_model_name):
            if name in names:
                self._delete_goal(name)
        time.sleep(.2)
        pose = Pose()
        pose.position.x, pose.position.y = map(float, self.goal_xy)
        pose.orientation.w = 1.
        response = self.spawn_model(self.goal_model_name, self._goal_sdf, '', pose, 'world')
        if not response.success:
            raise RuntimeError('Cannot spawn goal: ' + response.status_message)
        with self._condition:
            counts = self._scan_count, self._odom_count
        started = rospy.Time.now().to_sec()
        try:
            self.unpause_sim()
            rospy.sleep(.15)
            scan, odom = self._wait_messages(counts, started)
        finally:
            self.pause_sim()
        self.steps = 0
        self.last_duration = self.path_length = self.initial_distance = 0.
        self.subgoal_xy = self.goal_xy.copy()
        self.state, self.info = self._observe(scan, odom)
        return self.state.copy(), dict(self.info)

    def low_state(self):
        from util import goal_observation
        return goal_observation(self.ranges, self.position, self.yaw, self.velocity,
                                self.subgoal_xy, self.v_max, self.w_max)[0]

    def set_subgoal(self, point):
        self.subgoal_xy = np.asarray(point).copy()
        if self.subgoal_model_name in self.world_properties().model_names:
            self._delete_goal(self.subgoal_model_name)
        pose = Pose()
        pose.position.x, pose.position.y = map(float, point)
        pose.orientation.w = 1.
        sdf = self._goal_sdf.replace(self.goal_model_name, self.subgoal_model_name).replace('0 1 0 1', '.7 .1 .9 1')
        response = self.spawn_model(self.subgoal_model_name, sdf, '', pose, 'world')
        if not response.success:
            raise RuntimeError('Cannot spawn subgoal: ' + response.status_message)

    def step(self, action):
        with self._condition:
            counts = self._scan_count, self._odom_count
        command = Twist()
        command.linear.x, command.angular.z = map(float, action)
        old = self.position.copy()
        self.cmd_pub.publish(command)
        started = rospy.Time.now().to_sec()
        try:
            self.unpause_sim()
            rospy.sleep(self.action_duration)
        finally:
            self.pause_sim()
        self.last_duration = rospy.Time.now().to_sec() - started
        scan, odom = self._wait_messages(counts, started)
        self.steps += 1
        self.state, self.info = self._observe(scan, odom)
        self.path_length += float(np.linalg.norm(self.position - old))
        terminated = self.info['reached'] or self.info['collision']
        truncated = self.info['timeout']
        if terminated or truncated:
            self.stop()
        return self.state.copy(), terminated, truncated, dict(self.info)

    def astar_distance(self):
        return 0.  # Deployment does not construct a map or compute reward distances.

    def refresh(self):
        pass  # Gazebo draws the robot and the two goal models.

    def close(self):
        with self._close_lock:
            if self._closed:
                return
            try:
                if self.subgoal_model_name in self.world_properties().model_names:
                    self._delete_goal(self.subgoal_model_name)
            finally:
                super().close()
