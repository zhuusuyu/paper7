"""Networks, on-policy storage, static-world A*, and navigation reward."""

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


class FuzzyActor(nn.Module):
    """Eleven fixed rules with 28 antecedent and 11 rule multipliers."""

    def __init__(self, scan_angle_min, scan_angle_increment, v_max, w_max):
        super().__init__()
        angles = scan_angle_min + torch.arange(360, dtype=torch.float64) * scan_angle_increment
        angles = (angles + math.pi) % (2.0 * math.pi) - math.pi
        self.register_buffer("sector_mask", torch.stack((
            (angles >= -math.pi / 4) & (angles < math.pi / 4),
            (angles >= math.pi / 4) & (angles < 3 * math.pi / 4),
            (angles >= -3 * math.pi / 4) & (angles < -math.pi / 4),
        )))
        # Columns: goal heading, front risk, left risk, right risk; -1 is don't-care.
        indices = torch.tensor([
            [1, 0, -1, -1], [2, 0, -1, -1], [0, 0, -1, -1],
            [-1, 1, 0, 1], [-1, 1, 1, 0], [-1, 1, 1, 1],
            [-1, 0, 1, -1], [-1, 0, -1, 1],
            [2, 1, 0, -1], [0, 1, -1, 0], [1, 1, 0, -1],
        ])
        care = indices >= 0
        centers, sigmas = torch.zeros(11, 4), torch.ones(11, 4)
        for dim in range(4):
            center_table = torch.tensor([-0.58, 0.0, 0.58]) if dim == 0 else torch.tensor([0.0, 1.0])
            sigma_table = torch.tensor([0.20, 0.16, 0.20]) if dim == 0 else torch.tensor([0.35, 0.35])
            centers[care[:, dim], dim] = center_table[indices[care[:, dim], dim]]
            sigmas[care[:, dim], dim] = sigma_table[indices[care[:, dim], dim]]
        self.register_buffer("rule_indices", indices)
        self.register_buffer("care_mask", care)
        self.register_buffer("membership_centers", centers)
        self.register_buffer("membership_sigmas", sigmas)
        self.register_buffer("rule_priorities", torch.tensor([1.0] * 3 + [2.0] * 8))
        self.register_buffer("action_limits", torch.tensor([v_max, w_max]))
        self.register_buffer("anchor_actions", torch.tensor([
            [0.18, 0.0], [0.11, 0.65], [0.11, -0.65],
            [0.025, 1.0], [0.025, -1.0], [0.0, 0.0],
            [0.07, -0.75], [0.07, 0.75],
            [0.025, 0.90], [0.025, -0.90], [0.015, 0.80],
        ]) * torch.tensor([v_max / 0.2, w_max / 1.5]))
        self.antecedent_logits = nn.Parameter(torch.zeros(int(care.sum())))
        self.rule_logits = nn.Parameter(torch.zeros(11))

    def semantic_features(self, states):
        ranges = states[:, None, :360] * 3.5
        nearest = ranges.masked_fill(~self.sector_mask, torch.inf).amin(dim=-1)
        risks = ((0.70 - nearest) / (0.70 - 0.15)).clamp(0.0, 1.0)
        return torch.cat((states[:, 363:364], risks), dim=-1)

    def forward(self, states):
        features = self.semantic_features(states)
        log_scales = torch.zeros_like(self.membership_centers).masked_scatter(
            self.care_mask, F.logsigmoid(self.antecedent_logits),
        )
        log_membership = -0.5 * (
            (features[:, None, :] - self.membership_centers) / self.membership_sigmas
        ).square() + log_scales
        log_activation = log_membership.masked_fill(~self.care_mask, torch.inf).amin(dim=-1)
        log_evidence = log_activation + self.rule_priorities.log() + F.logsigmoid(self.rule_logits)
        return log_evidence.softmax(dim=-1) @ self.anchor_actions


class RolloutBuffer:
    def __init__(self, capacity, state_dim):
        self.capacity = capacity
        self.size = 0
        self.data = {
            "states": np.empty((capacity, state_dim), dtype=np.float32),
            "next_states": np.empty((capacity, state_dim), dtype=np.float32),
            "raw_actions": np.empty((capacity, 2), dtype=np.float32),
            "log_probs": np.empty(capacity, dtype=np.float32),
            "alphas": np.empty(capacity, dtype=np.float32),
            "rewards": np.empty(capacity, dtype=np.float32),
            "terminated": np.empty(capacity, dtype=np.bool_),
            "truncated": np.empty(capacity, dtype=np.bool_),
        }

    def add(self, state, raw_action, log_prob, alpha, reward, next_state, terminated, truncated):
        values = (state, next_state, raw_action, log_prob, alpha, reward, terminated, truncated)
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
        self.figure, axis = plt.subplots(figsize=(8.4, 6.2))
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
        axis.set(xlim=(grid.world_half, -grid.world_half), ylim=(-grid.world_half, grid.world_half),
                 xlabel="y / m", ylabel="x / m", aspect="equal",
                 title="A* | grid {:.2f} m | inflation {:.2f} m".format(grid.resolution, grid.inflation_radius))
        axis.set_xticks(np.arange(-2, 3))
        axis.set_yticks(np.arange(-2, 3))
        axis.set_xticks(edges, minor=True)
        axis.set_yticks(edges, minor=True)
        axis.tick_params(which="minor", bottom=False, left=False)
        axis.grid(which="minor", color="#cbd5e1", linewidth=0.5)
        self.status_text = self.figure.text(0.71, 0.79, "Waiting for reset", va="top", family="monospace", fontsize=10)
        self.figure.legend(handles=[Patch(facecolor="#64748b", label="Occupied (inflated)"),
                                    Patch(facecolor="#93c5fd", label="Expanded"),
                                    self.path_line, self.robot, self.goal],
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
        if info["step"] == 0:
            lines.append("d_A* prev: —")
        else:
            lines.append("d_A* prev: {:.6f} m".format(info["previous_astar_distance"]))
        lines.extend(["d_A* now : {:.6f} m".format(info["astar_distance"]), ""])
        for key in ("r_arrival", "r_step", "r_dist", "r_collision", "reward"):
            lines.append("{}: —".format(key) if info["step"] == 0 else "{}: {:+.6f}".format(key, info[key]))
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
        "r_step": -1,
        "r_dist": 5 * (previous_distance - distance),
        "r_collision": -100.0 if collision else 0.0,
    }
    return float(sum(parts.values())), reached, collision, parts
