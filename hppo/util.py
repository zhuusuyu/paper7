"""Networks, temporal observations, buffers, static-world A*, and rewards."""

import heapq
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal

TRAINING_VERSION = 3
STEP_BACKEND = "gazebo_ros_continuous_apsr"


def curriculum_stage(steps):
    return 0 if steps < 10000 else 1 if steps < 20000 else 2


class Actor(nn.Module):
    def __init__(self, state_dim, width, v_max, w_max):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 2),
        )
        self.log_std = nn.Parameter(torch.full((2,), -0.5))
        self.register_buffer("action_scale", torch.tensor([v_max / 2.0, w_max]))
        self.register_buffer("action_bias", torch.tensor([v_max / 2.0, 0.0]))

    def forward(self, states):
        return Normal(self.network(states), self.log_std.exp())

    def to_action(self, raw_actions):
        return raw_actions.tanh() * self.action_scale + self.action_bias

    def log_prob(self, distribution, raw_actions):
        # log(1 - tanh(u)^2), stable even when tanh(u) rounds to +/-1.
        log_tanh_jacobian = 2.0 * (
            math.log(2.0) - raw_actions - F.softplus(-2.0 * raw_actions)
        )
        return (
            distribution.log_prob(raw_actions)
            - log_tanh_jacobian - self.action_scale.log()
        ).sum(dim=-1)


class Critic(nn.Module):
    def __init__(self, state_dim, width):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1),
        )

    def forward(self, states):
        return self.network(states).squeeze(-1)


class DuelingQNetwork(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1456, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(),
        )
        self.value = nn.Linear(width, 1)
        self.advantage = nn.Linear(width, 225)

    def forward(self, states):
        features = self.network(states)
        advantage = self.advantage(features)
        return self.value(features) + advantage - advantage.mean(dim=-1, keepdim=True)


class FrameStack:
    """Four consecutive environment observations, oldest first."""

    def reset(self, state):
        self.frames = np.tile(state, (4, 1)).astype(np.float32)
        return self.snapshot()

    def append(self, state):
        self.frames[:-1] = self.frames[1:]
        self.frames[-1] = state

    def snapshot(self):
        return self.frames.reshape(1456).copy()


class SubgoalSpace:
    """Fixed polar candidates; safety uses only the current lidar observation."""

    def __init__(self, scan_angle_min, scan_angle_increment):
        self.radii = np.repeat((np.arange(15) + 0.5) * 3.5 / 15, 15)
        self.angles = np.tile(np.arange(15) * 2 * math.pi / 15, 15)
        beams = scan_angle_min + np.arange(360) * scan_angle_increment
        error = (beams[None, :] - self.angles[:, None] + math.pi) % (2 * math.pi) - math.pi
        # Include adjacent beams at each disk's angular boundary. Requiring the
        # whole covered fan to be clear up to r + 0.15 is conservative.
        half_angle = np.arcsin(np.minimum(0.15 / self.radii, 1.0)) + abs(scan_angle_increment) / 2
        self.covered_beams = np.abs(error) <= half_angle[:, None]

    def mask(self, state):
        ranges = state[:360] * 3.5
        clear = (~self.covered_beams | (ranges[None, :] > self.radii[:, None] + 0.15)).all(axis=1)
        return (self.radii > 0.5) & clear

    def world_xy(self, index, info):
        angle = info["yaw"] + self.angles[index]
        return np.array([info["x"], info["y"]]) + self.radii[index] * np.array([
            math.cos(angle), math.sin(angle),
        ])


class SubgoalProgress:
    """Track meaningful progress toward the current fixed world-coordinate goal."""

    def __init__(self, distance, heading, started, v_max, w_max):
        self.started = self.last_progress = started
        self.best_distance = self.distance_anchor = distance
        self.best_heading = self.heading_anchor = abs(heading)
        self.time_limit = max(6.0, 3.0 + 2 * max(distance - 0.5, 0) / v_max + 2 * abs(heading) / w_max)

    def update(self, distance, heading, now):
        self.best_distance = min(self.best_distance, distance)
        self.best_heading = min(self.best_heading, abs(heading))
        if (self.distance_anchor - self.best_distance >= 0.05
                or self.heading_anchor - self.best_heading >= math.radians(15)):
            self.distance_anchor, self.heading_anchor = self.best_distance, self.best_heading
            self.last_progress = now
        if distance < 0.5:
            return "subgoal_reached"
        if now - self.last_progress >= 3.0 - 1e-9:
            return "stalled"
        if now - self.started >= self.time_limit - 1e-9:
            return "time_limit"
        return "running"


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity, self.size, self.position = capacity, 0, 0
        self.data = {
            "states": np.empty((capacity, 1456), dtype=np.float32),
            "next_states": np.empty((capacity, 1456), dtype=np.float32),
            "actions": np.empty(capacity, dtype=np.int64),
            "rewards": np.empty(capacity, dtype=np.float32),
            "next_masks": np.empty((capacity, 225), dtype=np.bool_),
            "terminated": np.empty(capacity, dtype=np.bool_),
            "truncated": np.empty(capacity, dtype=np.bool_),
        }

    def add(self, state, action, reward, next_state, next_mask, terminated, truncated):
        values = (state, next_state, action, reward, next_mask, terminated, truncated)
        for key, value in zip(self.data, values):
            self.data[key][self.position] = value
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        indices = np.random.choice(self.size, batch_size, replace=False)
        return {key: torch.as_tensor(value[indices], device=device) for key, value in self.data.items()}

    def state_dict(self):
        return {"capacity": self.capacity, "size": self.size, "position": self.position,
                "data": {key: value[:self.size].copy() for key, value in self.data.items()}}

    def load_state_dict(self, state):
        if state["capacity"] != self.capacity:
            raise ValueError("Replay capacity differs from the checkpoint")
        self.size, self.position = state["size"], state["position"]
        for key, value in state["data"].items():
            self.data[key][:self.size] = value


class RolloutBuffer:
    def __init__(self, capacity, state_dim):
        self.capacity = capacity
        self.size = 0
        self.data = {
            "states": np.empty((capacity, state_dim), dtype=np.float32),
            "next_states": np.empty((capacity, state_dim), dtype=np.float32),
            "raw_actions": np.empty((capacity, 2), dtype=np.float32),
            "log_probs": np.empty(capacity, dtype=np.float32),
            "rewards": np.empty(capacity, dtype=np.float32),
            "terminated": np.empty(capacity, dtype=np.bool_),
            "truncated": np.empty(capacity, dtype=np.bool_),
        }

    def add(self, state, raw_action, log_prob, reward, next_state, terminated, truncated):
        values = (state, next_state, raw_action, log_prob, reward, terminated, truncated)
        for key, value in zip(self.data, values):
            self.data[key][self.size] = value
        self.size += 1

    def tensors(self, device):
        return {
            key: torch.as_tensor(value[:self.size], device=device)
            for key, value in self.data.items()
        }

    def clear(self):
        self.size = 0


def compute_gae(rewards, values, next_values, terminated, truncated, gamma, gae_lambda):
    """Timeouts bootstrap their final observation but never the following episode."""
    deltas = rewards + gamma * next_values * (~terminated) - values
    advantages = torch.empty_like(rewards)
    tail = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
    episode_ends = terminated | truncated
    for index in reversed(range(rewards.shape[0])):
        tail = deltas[index] + gamma * gae_lambda * (~episode_ends[index]) * tail
        advantages[index] = tail
    return advantages, advantages + values


def _world_pose(parent, element):
    """Compose planar include/model/link/collision poses."""
    pose_element = element.find("pose")
    if pose_element is not None and pose_element.get("relative_to"):
        raise ValueError("World map requires parent-relative SDF poses")
    pose = np.fromstring(element.findtext("pose", "0 0 0 0 0 0"), sep=" ")
    if pose.shape != (6,) or np.any(np.abs(pose[3:5]) > 1e-8):
        raise ValueError("World map supports six-value poses with yaw rotation only")
    x, y, yaw = parent
    c, s = math.cos(yaw), math.sin(yaw)
    return x + c * pose[0] - s * pose[1], y + s * pose[0] + c * pose[1], yaw + pose[5]


def _model_file(uri, directory):
    if uri.startswith("model://"):
        import rospkg
        roots = [Path(value) for value in os.environ.get("GAZEBO_MODEL_PATH", "").split(os.pathsep) if value]
        roots.extend((directory.parent / "models", Path.home() / ".gazebo/models",
                      Path(rospkg.RosPack().get_path("turtlebot3_gazebo")) / "models"))
        relative = uri[len("model://"):]
        for root in roots:
            path = root / relative
            if path.exists():
                break
        else:
            raise FileNotFoundError("Cannot resolve Gazebo model URI: " + uri)
    else:
        path = Path(uri[len("file://"):] if uri.startswith("file://") else uri)
        if not path.is_absolute():
            path = directory / path
    if path.is_dir():
        config = ET.parse(path / "model.config").getroot()
        path = path / config.findtext("sdf")
    return path.resolve()


def _static_models(container, parent_pose, directory, parent_static=False):
    """Expand local model includes and yield static models in world coordinates."""
    for child in container:
        if child.tag not in ("model", "include"):
            continue
        override = None
        model_directory = directory
        if child.tag == "include":
            uri = child.findtext("uri").strip()
            if uri in ("model://sun", "model://ground_plane"):
                continue
            path = _model_file(uri, directory)
            model = ET.parse(path).getroot().find("model")
            if model is None:
                raise ValueError("Included SDF must contain a model: {}".format(path))
            include_pose = _world_pose(parent_pose, child)
            override = child.findtext("static")
            model_directory = path.parent
        else:
            model = child
            include_pose = parent_pose
        static_text = override if override is not None else model.findtext("static", str(parent_static))
        if static_text.lower() not in ("true", "1"):
            continue
        pose = _world_pose(include_pose, model)
        yield model, pose
        yield from _static_models(model, pose, model_directory, parent_static=True)


class GridMap:
    resolution = 0.2
    inflation_radius = 0.15
    world_half = 2.5

    def __init__(self, world_file):
        count = int(math.floor(self.world_half / self.resolution))
        self.centers = np.arange(-count, count + 1) * self.resolution
        self.visualize = False
        x, y = np.meshgrid(self.centers, self.centers)
        self.occupied = np.zeros(x.shape, dtype=np.bool_)
        half_cell = self.resolution / 2.0
        corners = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
        world_file = Path(world_file).resolve()
        world = ET.parse(world_file).getroot().find("world")
        if world is None:
            raise ValueError("Expected an SDF world file")
        self.world_name = world.attrib["name"]
        if not self.world_name:
            raise ValueError("Gazebo world must have a name")
        for model, model_pose in _static_models(world, (0.0, 0.0, 0.0), world_file.parent):
            for link in model.findall("link"):
                link_pose = _world_pose(model_pose, link)
                for collision in link.findall("collision"):
                    px, py, yaw = _world_pose(link_pose, collision)
                    c, s = math.cos(yaw), math.sin(yaw)
                    dx, dy = x - px, y - py
                    geometry = collision.find("geometry")
                    box, cylinder = geometry.find("box"), geometry.find("cylinder")
                    if box is not None:
                        half_x, half_y, _ = np.fromstring(box.findtext("size"), sep=" ") / 2.0
                        # Separating-axis test: cell axes and rotated rectangle axes.
                        overlap = (
                            (np.abs(dx) <= half_x * abs(c) + half_y * abs(s) + half_cell)
                            & (np.abs(dy) <= half_x * abs(s) + half_y * abs(c) + half_cell)
                            & (np.abs(c * dx + s * dy) <= half_x + half_cell * (abs(c) + abs(s)))
                            & (np.abs(-s * dx + c * dy) <= half_y + half_cell * (abs(c) + abs(s)))
                        )
                        # Disjoint rectangles attain their minimum distance at
                        # a vertex and the other rectangle. Inflation is round.
                        qx = dx[..., None] + half_cell * corners[:, 0]
                        qy = dy[..., None] + half_cell * corners[:, 1]
                        cell_distance_sq = (
                            np.maximum(np.abs(c * qx + s * qy) - half_x, 0.0) ** 2
                            + np.maximum(np.abs(-s * qx + c * qy) - half_y, 0.0) ** 2
                        ).min(axis=-1)
                        vx = c * half_x * corners[:, 0] - s * half_y * corners[:, 1]
                        vy = s * half_x * corners[:, 0] + c * half_y * corners[:, 1]
                        box_distance_sq = (
                            np.maximum(np.abs(dx[..., None] - vx) - half_cell, 0.0) ** 2
                            + np.maximum(np.abs(dy[..., None] - vy) - half_cell, 0.0) ** 2
                        ).min(axis=-1)
                        # Include touching boundaries despite floating-point roundoff.
                        overlap |= np.minimum(cell_distance_sq, box_distance_sq) <= self.inflation_radius ** 2 + 1e-12
                    elif cylinder is not None:
                        radius = float(cylinder.findtext("radius")) + self.inflation_radius
                        overlap = (
                            np.maximum(np.abs(dx) - half_cell, 0.0) ** 2
                            + np.maximum(np.abs(dy) - half_cell, 0.0) ** 2
                        ) <= radius ** 2 + 1e-12
                    else:
                        raise ValueError("World map supports box and cylinder collision geometry only")
                    self.occupied |= overlap

        self.free_cells = np.argwhere(~self.occupied)  # Row-major order resolves ties.
        if len(self.free_cells) == 0:
            raise ValueError("The rasterized world has no free cells")
        self.free_xy = self.centers[self.free_cells[:, ::-1]]
        first = tuple(self.free_cells[0])
        visited, pending = {first}, [first]
        while pending:
            for neighbor, _ in self.neighbors(pending.pop()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    pending.append(neighbor)
        if len(visited) != len(self.free_cells):
            raise ValueError("The rasterized grid contains disconnected free regions")
        self.task_points = np.array([[2, 2], [2, -2], [-2, 2], [-2, -2]], dtype=np.float64)
        for start in self.task_points:
            cell = self.nearest_free_cell(start)
            if not np.array_equal(self.cell_xy(cell), start):
                raise ValueError("A task corner is occupied in the rasterized map")
            for goal in self.task_points:
                self.distance(start, goal)

    def cell_xy(self, cell):
        row, col = cell
        return np.array([self.centers[col], self.centers[row]])

    def prepare_local_tasks(self, seed=3):
        """Precompute local pairs whose segment touches no occupied grid cell."""
        starts, goals = [], []
        cell_low = self.centers[np.argwhere(self.occupied)[:, ::-1]] - self.resolution / 2
        cell_high = cell_low + self.resolution
        for index, start in enumerate(self.free_xy):
            delta = self.free_xy - start
            distances = np.linalg.norm(delta, axis=1)
            candidates = np.flatnonzero((distances >= 0.75) & (distances <= 3.5))
            directions = delta[candidates, None, :]
            # Slab intersection, including segments parallel to a cell edge.
            parallel = directions == 0
            first = np.full((len(candidates), len(cell_low), 2), -np.inf)
            second = np.full_like(first, np.inf)
            np.divide(cell_low[None, :, :] - start, directions, out=first, where=~parallel)
            np.divide(cell_high[None, :, :] - start, directions, out=second, where=~parallel)
            enter = np.minimum(first, second)
            leave = np.maximum(first, second)
            outside = parallel & ((start < cell_low) | (start > cell_high))[None, :, :]
            enter[outside] = np.inf
            intersects = np.maximum(enter.max(axis=-1), 0) <= np.minimum(leave.min(axis=-1), 1)
            visible = candidates[~intersects.any(axis=1)]
            starts.extend([index] * len(visible))
            goals.extend(visible.tolist())
        if not starts:
            raise ValueError("No free local tasks with distance 0.75 to 3.5 m")
        self.local_pairs = np.column_stack((starts, goals))
        distances = np.linalg.norm(self.free_xy[self.local_pairs[:, 1]] - self.free_xy[self.local_pairs[:, 0]], axis=1)
        self.curriculum_pairs = (np.flatnonzero(distances <= 1.25), np.flatnonzero(distances <= 2.0),
                                 np.arange(len(self.local_pairs)))
        self.low_gate_tasks = [[], []]
        rng = np.random.default_rng(seed + 2000000)
        for low, high in ((0.75, 1.25), (1.25, 2.0), (2.0, 2.75), (2.75, 3.5)):
            indices = np.flatnonzero((distances >= low if low == 0.75 else distances > low) & (distances <= high))
            if len(indices) < 10:
                raise ValueError("Local evaluation needs ten distinct pairs in distance band ({}, {}]".format(low, high))
            selected = rng.choice(indices, size=10, replace=False)
            for bank in range(2):
                for index, heading in zip(selected[bank * 5:(bank + 1) * 5], (-160, -90, 0, 90, 160)):
                    start, goal = self.free_xy[self.local_pairs[index]]
                    delta = goal - start
                    yaw = (math.atan2(delta[1], delta[0]) - math.radians(heading) + math.pi) % (2 * math.pi) - math.pi
                    self.low_gate_tasks[bank].append((start.copy(), goal.copy(), yaw))

    def coverage_tasks(self):
        tasks = []
        for start_index, start in enumerate(self.task_points):
            yaw = math.pi / 2 if start[1] < 0 else -math.pi / 2
            for goal_index, goal in enumerate(self.task_points):
                if start_index != goal_index:
                    for perturbation in (-15, -10, -5, 0, 5, 10, 15, -7.5, 7.5, 0):
                        tasks.append((start.copy(), goal.copy(), yaw + math.radians(perturbation)))
        return tasks

    def nearest_free_cell(self, xy):
        # Equivalent to nearest-grid mapping followed by nearest-free projection.
        index = np.argmin(np.sum((self.free_xy - np.asarray(xy)) ** 2, axis=1))
        return tuple(self.free_cells[index])

    def neighbors(self, cell):
        row, col = cell
        height, width = self.occupied.shape
        for dr, dc in ((-1, 0), (0, -1), (0, 1), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            nr, nc = row + dr, col + dc
            if not (0 <= nr < height and 0 <= nc < width) or self.occupied[nr, nc]:
                continue
            if dr and dc and (self.occupied[row + dr, col] or self.occupied[row, col + dc]):
                continue
            yield (nr, nc), self.resolution * (math.sqrt(2.0) if dr and dc else 1.0)

    def distance(self, position, goal):
        start, target = self.nearest_free_cell(position), self.nearest_free_cell(goal)
        best = {start: 0.0}
        queue = [(0.0, 0.0, start)]
        if self.visualize:
            parents, expanded = {}, []
        while queue:
            _, cost, cell = heapq.heappop(queue)
            if cost > best[cell]:
                continue
            if self.visualize:
                expanded.append(cell)
            if cell == target:
                if self.visualize:
                    path = [target]
                    while path[-1] != start:
                        path.append(parents[path[-1]])
                    self.last_path = self.centers[np.asarray(path[::-1])[:, ::-1]]
                    self.last_expanded = np.asarray(expanded)
                return cost
            for neighbor, edge_cost in self.neighbors(cell):
                candidate = cost + edge_cost
                if candidate < best.get(neighbor, math.inf):
                    best[neighbor] = candidate
                    if self.visualize:
                        parents[neighbor] = cell
                    dr, dc = abs(neighbor[0] - target[0]), abs(neighbor[1] - target[1])
                    heuristic = self.resolution * (max(dr, dc) + (math.sqrt(2.0) - 1.0) * min(dr, dc))
                    heapq.heappush(queue, (candidate + heuristic, candidate, neighbor))
        raise RuntimeError("No A* path on the validated static grid")


class AStarViewer:
    """Optional GUI for the same A* search and rewards used by the environment."""

    def __init__(self, grid):
        import matplotlib
        matplotlib.use("TkAgg")
        from matplotlib import pyplot as plt
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch

        self.grid, self.plt, self.closed = grid, plt, False
        plt.ion()
        self.figure, axis = plt.subplots(figsize=(9.4, 7.6))
        self.figure.canvas.manager.set_window_title("TurtleBot3 A*")
        self.figure.subplots_adjust(left=0.08, right=0.67, bottom=0.16, top=0.88)
        edges = np.r_[grid.centers - grid.resolution / 2.0, grid.centers[-1] + grid.resolution / 2.0]
        extent = (edges[0], edges[-1], edges[0], edges[-1])
        # Display world +x upward and +y leftward.
        axis.imshow(grid.occupied.T, origin="lower", extent=extent,
                    cmap=ListedColormap(["white", "#64748b"]), vmin=0, vmax=1)
        self.expanded_image = axis.imshow(
            np.ma.masked_all(grid.occupied.shape), origin="lower", extent=extent,
            cmap=ListedColormap(["#93c5fd"]), vmin=0, vmax=1, alpha=0.7,
        )
        self.path_line, = axis.plot([], [], color="#e11d48", linewidth=2, label="A* path")
        self.robot, = axis.plot([], [], "o", color="#16a34a", markersize=7, label="Robot")
        self.goal, = axis.plot([], [], "*", color="#eab308", markeredgecolor="#854d0e",
                               markersize=13, label="Goal")
        self.subgoal, = axis.plot([], [], "^", color="#7c3aed", markersize=9, label="Subgoal")
        axis.set(xlim=(grid.world_half, -grid.world_half), ylim=(-grid.world_half, grid.world_half),
                 xlabel="y / m", ylabel="x / m", aspect="equal",
                 title="A* | grid {:.2f} m | inflation {:.2f} m".format(grid.resolution, grid.inflation_radius))
        axis.set_xticks(np.arange(-2, 3))
        axis.set_yticks(np.arange(-2, 3))
        axis.set_xticks(edges, minor=True)
        axis.set_yticks(edges, minor=True)
        axis.tick_params(which="minor", bottom=False, left=False)
        axis.grid(which="minor", color="#cbd5e1", linewidth=0.5)
        self.status_text = self.figure.text(0.69, 0.88, "Waiting for reset", va="top", family="monospace", fontsize=9)
        self.figure.legend(handles=[Patch(facecolor="#64748b", label="Occupied (inflated)"),
                                    Patch(facecolor="#93c5fd", label="Expanded"),
                                    self.path_line, self.robot, self.goal, self.subgoal],
                           loc="lower center", bbox_to_anchor=(0.37, 0.01), ncol=3, fontsize=8)
        self.figure.canvas.mpl_connect("close_event", self.close)
        grid.viewer, grid.visualize = self, True
        plt.show(block=False)
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def update(self, info):
        expanded = np.zeros(self.grid.occupied.shape, dtype=bool)
        expanded[tuple(self.grid.last_expanded.T)] = True
        self.expanded_image.set_data(np.ma.masked_where(~expanded, np.ones(expanded.shape)).T)
        path = self.grid.last_path
        self.path_line.set_data(path[:, 1], path[:, 0])
        self.robot.set_data([info["y"]], [info["x"]])
        self.goal.set_data([info["goal_y"]], [info["goal_x"]])
        lines = ["step: {}".format(info["step"])]
        if info["stage"] == "high":
            lines.append("global d_A*: {:.6f} m".format(info["astar_distance"]))
            if "subgoal_x" in info:
                self.subgoal.set_data([info["subgoal_y"]], [info["subgoal_x"]])
                lines.append("subgoal d: {:.6f} m".format(info["subgoal_distance"]))
                lines.append("heading: {:+.6f} rad".format(info["subgoal_heading"]))
                lines.append("reason: {}".format(info["subgoal_reason"]))
                lines.append("segment steps: {}".format(info["subgoal_steps"]))
                lines.append("active elapsed: {:.6f} s".format(info["subgoal_elapsed"]))
                lines.append("limit: {:.6f} s".format(info["subgoal_time_limit"]))
                lines.append("no progress: {:.6f} s".format(info["subgoal_no_progress"]))
                lines.append("best d: {:.6f} m".format(info["subgoal_best_distance"]))
                lines.append("best |h|: {:.6f} rad".format(info["subgoal_best_heading"]))
            else:
                self.subgoal.set_data([], [])
            lines.extend(["", "last high decision:"])
            if "high_reward" in info:
                high = info["high_reward"]
                lines.append("d_A* prev: {:.6f} m".format(high["previous_astar_distance"]))
                lines.append("d_A* end : {:.6f} m".format(high["astar_distance"]))
                for key in ("r_arrival", "r_step", "r_dist", "r_collision", "r_out", "reward"):
                    lines.append("{}: {:+.6f}".format(key, high[key]))
            else:
                lines.append("reward: —")
        else:
            self.subgoal.set_data([], [])
            if info["step"] == 0:
                lines.append("d_A* prev: —")
            else:
                lines.append("d_A* prev: {:.6f} m".format(info["previous_astar_distance"]))
            lines.extend(["d_A* now : {:.6f} m".format(info["astar_distance"]), ""])
            for key in ("r_arrival", "r_step", "r_dist", "r_collision", "reward"):
                lines.append("{}: —".format(key) if info["step"] == 0 else "{}: {:+.6f}".format(key, info[key]))
        lines.extend(["", "sim step: {:.6f} s".format(info["sim_duration"]),
                      "scan age: {:.6f} s".format(info["scan_age"]),
                      "odom age: {:.6f} s".format(info["odom_age"])])
        self.status_text.set_text("\n".join(lines))
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()

    def close(self, event=None):
        if self.closed:
            return
        self.closed = True
        self.grid.visualize = False
        self.plt.close(self.figure)


def navigation_reward(previous_distance, distance, goal_distance, min_scan,
                      goal_radius=0.5, collision_distance=0.15):
    collision = min_scan < collision_distance
    reached = goal_distance < goal_radius and not collision
    parts = {
        "r_arrival": 100.0 if reached else 0.0,
        "r_step": -0.1,
        "r_dist": 5 * (previous_distance - distance),
        "r_collision": -100.0 if collision else 0.0,
    }
    return float(sum(parts.values())), reached, collision, parts


def high_level_reward(previous_distance, distance, reached, collision, steps, undesirable=False):
    parts = {
        "r_arrival": 100.0 if reached and not collision else 0.0,
        "r_step": -0.1 * steps,
        "r_dist": 5 * (previous_distance - distance),
        "r_collision": -100.0 if collision else 0.0,
        "r_out": -20.0 if undesirable else 0.0,
    }
    return float(sum(parts.values())), parts
