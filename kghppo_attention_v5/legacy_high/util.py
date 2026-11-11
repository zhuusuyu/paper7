"""Spatial attention, fuzzy knowledge, hypernetwork fusion, storage and A*."""

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
    """Single-frame 60-sector attention and a squashed Gaussian neural policy."""

    def __init__(self, width, v_max, w_max, scan_angle_min, scan_angle_increment, prior_scale):
        super().__init__()
        angles = scan_angle_min + torch.arange(360, dtype=torch.float64) * scan_angle_increment
        directions = torch.stack((angles.cos(), angles.sin()), dim=-1).float()
        self.register_buffer("beam_directions", directions)
        self.register_buffer("sector_directions", F.normalize(directions.reshape(60, 6, 2).mean(1), dim=-1))
        self.prior_scale = prior_scale
        self.embedding = nn.Sequential(nn.Linear(18, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.score = nn.Linear(64, 1)
        self.feature = nn.Sequential(nn.Linear(64, 32), nn.Tanh())
        self.network = nn.Sequential(
            nn.Linear(36, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 2),
        )
        self.log_std = nn.Parameter(torch.full((2,), -0.5))
        self.register_buffer("action_scale", torch.tensor([v_max / 2.0, w_max]))
        self.register_buffer("action_bias", torch.tensor([v_max / 2.0, 0.0]))

    def forward(self, states, attention_prior):
        points = (states[:, :360, None] * self.beam_directions).reshape(-1, 60, 12)
        tokens = torch.cat((points, self.sector_directions[None].expand(len(states), -1, -1),
                            states[:, None, 360:364].expand(-1, 60, -1)), dim=-1)
        embedded = self.embedding(tokens)
        logits = self.score(embedded).squeeze(-1)
        attention = (logits + self.prior_scale * attention_prior.detach()).softmax(dim=-1)
        pooled = (attention[:, :, None] * self.feature(embedded)).sum(dim=1)
        mean = self.network(torch.cat((pooled, states[:, 360:364]), dim=-1))
        return Normal(mean, self.log_std.exp()), attention

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


class FuzzyKnowledge(nn.Module):
    """STA-KGRL's 16 spatial rules with 42 antecedent and 16 rule multipliers."""

    def __init__(self, scan_angle_min, scan_angle_increment, v_max, w_max):
        super().__init__()
        angles = scan_angle_min + torch.arange(360, dtype=torch.float64) * scan_angle_increment
        angles = (angles + math.pi) % (2 * math.pi) - math.pi
        directions = torch.stack((angles.cos(), angles.sin()), dim=-1).reshape(60, 6, 2).mean(1)
        sector_angles = torch.atan2(directions[:, 1], directions[:, 0])
        masks = []
        for values in (angles, sector_angles):
            masks.append(torch.stack((
                (values >= -math.pi / 4) & (values < math.pi / 4),
                (values >= math.pi / 4) & (values < 3 * math.pi / 4),
                (values >= 3 * math.pi / 4) | (values < -3 * math.pi / 4),
                (values >= -3 * math.pi / 4) & (values < -math.pi / 4),
            )))
        self.register_buffer("direction_mask", masks[0])
        # Columns: normalized heading, front/left/rear/right risk; -1 is don't-care.
        indices = torch.tensor([
            [1, 0, -1, -1, -1], [2, 0, -1, -1, -1], [0, 0, -1, -1, -1],
            [-1, 1, 0, -1, 1], [-1, 1, 1, -1, 0], [-1, 1, 1, -1, 1],
            [-1, 0, 1, -1, -1], [-1, 0, -1, -1, 1],
            [2, 1, 0, -1, -1], [0, 1, -1, -1, 0], [1, 1, 0, -1, -1],
            [-1, 0, -1, 1, -1], [-1, 0, 1, 1, -1], [-1, 0, -1, 1, 1],
            [-1, 1, 0, 1, -1], [-1, 1, -1, 1, 0],
        ])
        care = indices >= 0
        centers, sigmas = torch.zeros(16, 5), torch.ones(16, 5)
        for dim in range(5):
            center_table = torch.tensor([-0.58, 0., 0.58]) if dim == 0 else torch.tensor([0., 1.])
            sigma_table = torch.tensor([0.20, 0.16, 0.20]) if dim == 0 else torch.tensor([0.35, 0.35])
            centers[care[:, dim], dim] = center_table[indices[care[:, dim], dim]]
            sigmas[care[:, dim], dim] = sigma_table[indices[care[:, dim], dim]]
        self.register_buffer("rule_indices", indices)
        self.register_buffer("care_mask", care)
        self.register_buffer("membership_centers", centers)
        self.register_buffer("membership_sigmas", sigmas)
        self.register_buffer("rule_priorities", torch.tensor([1.] * 3 + [2.] * 13))
        self.register_buffer("rule_sector_mask", (indices[:, 1:] == 1).float() @ masks[1].float())
        self.register_buffer("action_limits", torch.tensor([v_max, w_max]))
        self.register_buffer("anchor_actions", torch.tensor([
            [0.18, 0.], [0.11, 0.65], [0.11, -0.65],
            [0.025, 1.], [0.025, -1.], [0., 0.], [0.07, -0.75], [0.07, 0.75],
            [0.025, 0.90], [0.025, -0.90], [0.015, 0.80],
            [0.13, 0.], [0.09, -0.45], [0.09, 0.45], [0.01, 0.90], [0.01, -0.90],
        ]) * torch.tensor([v_max / 0.22, w_max / 1.5]))
        self.antecedent_logits = nn.Parameter(torch.zeros(int(care.sum())))
        self.rule_logits = nn.Parameter(torch.zeros(16))

    def forward(self, states):
        ranges = states[:, :360] * 3.5
        nearest = ranges[:, None, :].masked_fill(~self.direction_mask, torch.inf).amin(dim=-1)
        risks = ((0.70 - nearest) / 0.55).clamp(0., 1.)
        semantic = torch.cat((states[:, 363:364], risks), dim=-1)
        log_scales = torch.zeros_like(self.membership_centers).masked_scatter(
            self.care_mask, math.log(2.) + F.logsigmoid(self.antecedent_logits),
        )
        log_membership = -0.5 * ((semantic[:, None, :] - self.membership_centers)
                                 / self.membership_sigmas).square() + log_scales
        log_activation = log_membership.masked_fill(~self.care_mask, torch.inf).amin(dim=-1)
        log_evidence = (log_activation + self.rule_priorities.log()
                        + math.log(2.) + F.logsigmoid(self.rule_logits))
        weights = log_evidence.softmax(dim=-1)
        goal_scale = 0.6 + 0.4 * (states[:, 362] / 0.2).clamp(0., 1.)
        effective_actions = self.anchor_actions[None] * torch.stack(
            (goal_scale, torch.ones_like(goal_scale)), dim=-1,
        )[:, None, :]
        action = (weights[:, :, None] * effective_actions).sum(dim=1)
        variance = (weights * ((effective_actions - action[:, None, :])
                               / self.action_limits).square().mean(dim=-1)).sum(dim=-1)
        confidence = -torch.expm1(-log_evidence.exp().sum(dim=-1)) * torch.exp(-4. * variance)
        sector_risk = ((0.70 - ranges.reshape(-1, 60, 6).amin(dim=-1)) / 0.55).clamp(0., 1.)
        directional_attention = weights @ self.rule_sector_mask
        return {"action": action, "risks": risks, "confidence": confidence,
                "attention_prior": (1. + directional_attention) * sector_risk,
                "rule_weights": weights, "directional_attention": directional_attention}


class HyperFusion(nn.Module):
    """Generate two convex expert-weight rows, one for each physical action."""

    def __init__(self, v_max, w_max):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(13, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 4))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        self.register_buffer("action_limits", torch.tensor([v_max, w_max]))

    def forward(self, states, neural_action, knowledge, floors):
        context = torch.cat((states[:, 360:364], knowledge["risks"], knowledge["confidence"][:, None],
                             neural_action / self.action_limits, knowledge["action"] / self.action_limits), dim=-1)
        preferences = self.network(context.detach()).reshape(-1, 2, 2).softmax(dim=-1)[:, :, 1]
        alpha = knowledge["confidence"].detach()[:, None] * (
            floors[:, None] + (1. - floors[:, None]) * preferences
        )
        mean_action = (1. - alpha) * neural_action + alpha * knowledge["action"]
        return mean_action, alpha


class RolloutBuffer:
    def __init__(self, capacity, state_dim):
        self.capacity = capacity
        self.size = 0
        self.data = {
            "states": np.empty((capacity, state_dim), dtype=np.float32),
            "next_states": np.empty((capacity, state_dim), dtype=np.float32),
            "raw_actions": np.empty((capacity, 2), dtype=np.float32),
            "log_probs": np.empty(capacity, dtype=np.float32),
            "floors": np.empty(capacity, dtype=np.float32),
            "guidance": np.empty((capacity, 3), dtype=np.float32),
            "rewards": np.empty(capacity, dtype=np.float32),
            "terminated": np.empty(capacity, dtype=np.bool_),
            "truncated": np.empty(capacity, dtype=np.bool_),
        }

    def add(self, state, raw_action, log_prob, floor, guidance, reward, next_state, terminated, truncated):
        values = (state, next_state, raw_action, log_prob, floor, guidance, reward, terminated, truncated)
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
                      goal_radius=0.1, collision_distance=0.15):
    collision = min_scan < collision_distance
    reached = goal_distance < goal_radius and not collision
    parts = {
        "r_arrival": 100.0 if reached else 0.0,
        "r_step": -1,
        "r_dist": 5 * (previous_distance - distance),
        "r_collision": -100.0 if collision else 0.0,
    }
    return float(sum(parts.values())), reached, collision, parts


class HighQNetwork(nn.Module):
    """Permutation-equivariant candidate attention and shared dueling heads."""
    def __init__(self, width=256):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(1456, width), nn.Tanh(), nn.Linear(width, 128), nn.Tanh())
        self.embedding = nn.Sequential(nn.Linear(14, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.score = nn.Sequential(nn.Linear(192, 64), nn.Tanh(), nn.Linear(64, 1))
        self.value = nn.Sequential(nn.Linear(192, width), nn.Tanh(), nn.Linear(width, 1))
        self.advantage = nn.Sequential(nn.Linear(256, width), nn.Tanh(), nn.Linear(width, 1))

    def forward(self, states, masks, prior):
        context = self.encoder(states[:, :1456])
        tokens = self.embedding(states[:, 1456:].reshape(-1, 225, 14))
        joint = torch.cat((context[:, None].expand(-1, 225, -1), tokens), -1)
        attention = (self.score(joint).squeeze(-1) + prior.detach()).masked_fill(~masks, -torch.inf).softmax(-1)
        pooled = (attention[:, :, None] * tokens).sum(1)
        value = self.value(torch.cat((context, pooled), -1))
        advantage = self.advantage(torch.cat((joint, pooled[:, None].expand(-1, 225, -1)), -1)).squeeze(-1)
        mean = (advantage * masks).sum(-1, keepdim=True) / masks.sum(-1, keepdim=True)
        return value + advantage - mean, attention


class FrameStack:
    def reset(self, state):
        self.frames = np.tile(state, (4, 1)).astype(np.float32)
        return self.snapshot()

    def append(self, state):
        self.frames[:-1] = self.frames[1:]
        self.frames[-1] = state

    def snapshot(self):
        return self.frames.reshape(1456).copy()


class VisitMemory:
    """Odometry-only spatial samples; candidate features contain no scene map."""
    config = dict(capacity=64, spacing=.5, sigma=.5, exclude_recent=2)

    def reset(self, position):
        self.points = np.zeros((64, 2), dtype=np.float64)
        self.points[0] = position
        self.count, self.cursor = 1, 1
        self.last = np.asarray(position).copy()

    def record(self, position):
        if np.linalg.norm(position - self.last) >= .5:
            self.points[self.cursor] = position
            self.last = np.asarray(position).copy()
            self.cursor = (self.cursor + 1) % 64
            self.count = min(self.count + 1, 64)

    def features(self, world_points, mask, exact):
        valid = np.arange(64) < self.count
        valid[(self.cursor - np.arange(1, 3)) % 64] = False
        similarity = np.exp(-np.square(world_points[:, None] - self.points).sum(-1) / (2 * .5 ** 2))
        novelty = (1. - (similarity * valid).max(-1)).astype(np.float32)
        if exact >= 0 and mask[exact]:
            novelty[exact] = 1.
        return novelty

    def samples(self):
        return self.points[:self.count].copy()


def goal_observation(ranges, position, yaw, velocity, goal, v_max, w_max):
    """One observation definition shared by the 2D and ROS adapters."""
    delta = np.asarray(goal) - position
    distance = float(np.linalg.norm(delta))
    heading = (math.atan2(delta[1], delta[0]) - yaw + math.pi) % (2 * math.pi) - math.pi
    obs = np.concatenate((np.clip(ranges, 0, 3.5) / 3.5, [velocity[0] / v_max, velocity[1] / w_max,
                           distance / (5 * math.sqrt(2)), heading / math.pi])).astype(np.float32)
    return obs, distance, heading


class SubgoalSpace:
    def __init__(self, angle_min, angle_increment):
        self.radii = np.repeat((np.arange(15) + .5) * 3.5 / 15, 15)
        self.angles = np.tile(np.arange(15) * 2 * math.pi / 15, 15)
        self.points = self.radii[:, None] * np.stack((np.cos(self.angles), np.sin(self.angles)), -1)
        self.beams = angle_min + np.arange(360) * angle_increment
        self.directions = np.stack((np.cos(self.beams), np.sin(self.beams)), -1)
        self.angle_increment = abs(angle_increment)

    def candidates(self, state):
        """Require lidar-confirmed clearance over the entire swept safety disk.

        Each straight corridor is a capsule: two circular caps and a rectangle.
        Intersect all beams with that capsule, including the sensor/base offset.
        The final-goal point is tested in the same batch before replacing a bin.
        """
        distance, heading = float(state[362]) * 5 * math.sqrt(2), float(state[363]) * math.pi
        goal_in_range = 0 < distance < 3.5
        goal = distance * np.array([math.cos(heading), math.sin(heading)])
        points = np.vstack((self.points, goal)) if goal_in_range else self.points
        lengths = np.linalg.norm(points, axis=1)
        axes = points / lengths[:, None]
        # Coordinates are relative to the laser, 0.032 m behind base_link.
        start = np.array([.032, 0.])
        ends = points + start
        radius = .15 + np.maximum(np.linalg.norm(ends, axis=1), .032) * math.sin(self.angle_increment / 2)
        centers = np.stack((np.broadcast_to(start, ends.shape), ends), 1)
        projection = centers @ self.directions.T
        cross = centers[:, :, 0, None] * self.directions[:, 1] - centers[:, :, 1, None] * self.directions[:, 0]
        discriminant = radius[:, None, None] ** 2 - cross ** 2
        cap_far = projection + np.sqrt(np.maximum(discriminant, 0.))
        far = np.where(discriminant >= 0., np.maximum(cap_far, 0.), 0.).max(1)

        # Slab intersection with the rectangle along the segment.
        normals = np.stack((-axes[:, 1], axes[:, 0]), -1)
        basis = np.stack((axes, normals), 1)
        ray = basis @ self.directions.T
        origin = -(basis @ start)
        lower = np.column_stack((np.zeros(len(points)), -radius))
        upper = np.column_stack((lengths, radius))
        nonzero = abs(ray) > 1e-12
        inverse = np.divide(1., ray, out=np.zeros_like(ray), where=nonzero)
        first = (lower - origin)[:, :, None] * inverse
        second = (upper - origin)[:, :, None] * inverse
        inside = ((origin >= lower) & (origin <= upper))[:, :, None]
        entry = np.where(nonzero, np.minimum(first, second), np.where(inside, -np.inf, np.inf)).max(1)
        exit_ = np.where(nonzero, np.maximum(first, second), np.where(inside, np.inf, -np.inf)).min(1)
        far = np.maximum(far, np.where(exit_ >= np.maximum(entry, 0.), exit_, 0.))
        clear = (state[None, :360] * 3.5 > far).all(1)

        result = self.points.copy()
        mask = clear[:225] & (self.radii > .5)
        exact = -1
        if goal_in_range and clear[-1]:
            exact = int(np.square(self.points - goal).sum(1).argmin())
            result[exact] = goal
            mask[exact] = True
        return result, mask, exact

    @staticmethod
    def world_point(point, info):
        c, s = math.cos(info['yaw']), math.sin(info['yaw'])
        return np.array([info['x'], info['y']]) + np.array([[c, -s], [s, c]]) @ point


class SubgoalProgress:
    def __init__(self, distance, heading, reach_radius, v_max, w_max):
        self.distance_improvement = .01 if reach_radius == .1 else .05
        self.heading_improvement = math.radians(5 if reach_radius == .1 else 15)
        self.stall_duration = 6. if reach_radius == .1 else 3.
        self.reach_radius = reach_radius
        self.best_distance = self.distance_anchor = distance
        self.best_heading = self.heading_anchor = abs(heading)
        self.last_progress = 0.
        self.limit = max(6., 3 + 2 * max(distance - reach_radius, 0) / v_max + 2 * abs(heading) / w_max)

    def update(self, distance, heading, elapsed):
        self.best_distance = min(self.best_distance, distance)
        self.best_heading = min(self.best_heading, abs(heading))
        if self.distance_anchor - self.best_distance >= self.distance_improvement or self.heading_anchor - self.best_heading >= self.heading_improvement:
            self.distance_anchor, self.heading_anchor = self.best_distance, self.best_heading
            self.last_progress = elapsed
        if distance < self.reach_radius:
            return 'subgoal_reached'
        if elapsed - self.last_progress >= self.stall_duration - 1e-9:
            return 'stalled'
        if elapsed >= self.limit - 1e-9:
            return 'time_limit'
        return 'running'


class HighReplay:
    def __init__(self, capacity):
        self.capacity = capacity
        self.size = self.position = 0
        self.data = dict(states=np.empty((capacity, 4606), np.float32),
                         next_states=np.empty((capacity, 4606), np.float32),
                         actions=np.empty(capacity, np.int64), rewards=np.empty(capacity, np.float32),
                         next_masks=np.empty((capacity, 225), bool),
                         terminated=np.empty(capacity, bool), truncated=np.empty(capacity, bool),
                         masks=np.empty((capacity, 225), bool), steps=np.empty(capacity, np.int64))

    def add(self, state, action, reward, next_state, next_mask, terminated, truncated, mask, steps):
        for key, value in zip(self.data, (state, next_state, action, reward, next_mask, terminated, truncated,
                                         mask, steps)):
            self.data[key][self.position] = value
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, count, device):
        indices = np.random.choice(self.size, count, replace=False)
        return {k: torch.as_tensor(v[indices], device=device) for k, v in self.data.items()}

    def state_dict(self):
        return dict(size=self.size, position=self.position,
                    data={key: value[:self.size].copy() for key, value in self.data.items()})

    def load_state_dict(self, saved):
        self.size, self.position = saved['size'], saved['position']
        for key, value in saved['data'].items():
            self.data[key][:self.size] = value


class HighFuzzyKnowledge(nn.Module):
    """Eight rules, 15 actual antecedents and 8 rule multipliers.

    Route availability gates rules, rather than being a trainable antecedent.
    Candidate columns: x,y,length,sin,cos,E,A,P,C,O,N,R,V,T.
    """
    def __init__(self, low_config):
        super().__init__()
        rules = ({5: 1}, {11: 1, 8: 1}, {11: 1, 13: 1}, {11: 1, 10: 1},
                 {11: 0, 5: 0}, {8: 0, 5: 0}, {9: 1, 10: 1}, {6: 1, 8: 1})
        care = torch.zeros(8, 14, dtype=torch.bool)
        centers = torch.zeros(8, 14)
        for row, rule in enumerate(rules):
            for col, val in rule.items():
                care[row, col], centers[row, col] = True, val
        self.register_buffer('care_mask', care)
        self.register_buffer('membership_centers', centers)
        self.register_buffer('consequents', torch.tensor([4., 2., 1., .5, -2., -1., .5, .5]))
        self.antecedent_logits = nn.Parameter(torch.zeros(15))
        self.rule_logits = nn.Parameter(torch.zeros(8))
        angles = low_config['scan_angle_min'] + torch.arange(360) * low_config['scan_angle_increment']
        angles = (angles + math.pi) % (2 * math.pi) - math.pi
        self.register_buffer('direction_masks', torch.stack((
            (angles >= -math.pi/4) & (angles < math.pi/4),
            (angles >= math.pi/4) & (angles < 3*math.pi/4),
            (angles >= 3*math.pi/4) | (angles < -3*math.pi/4),
            (angles >= -3*math.pi/4) & (angles < -math.pi/4))))

    def forward(self, states, masks):
        features = states[:, 1456:].reshape(-1, 225, 14)
        scales = torch.zeros_like(self.membership_centers).masked_scatter(
            self.care_mask, math.log(2.) + F.logsigmoid(self.antecedent_logits))
        membership = -.5 * ((features[:, :, None] - self.membership_centers) / .35).square() + scales
        log_evidence = membership.masked_fill(~self.care_mask, torch.inf).amin(-1)
        log_evidence = log_evidence + math.log(2.) + F.logsigmoid(self.rule_logits)
        route = features[:, :, 12] > .5
        enabled = torch.ones_like(log_evidence, dtype=torch.bool)
        enabled[:, :, 1:5] = route[:, :, None]
        enabled[:, :, 6:] = ~route[:, :, None]
        log_evidence = log_evidence.masked_fill(~enabled, -torch.inf)
        weights = log_evidence.softmax(-1)
        evidence = log_evidence.exp()
        scores = (evidence * self.consequents).sum(-1) / 4.
        count = masks.sum(-1)
        mean = (scores * masks).sum(-1) / count
        variance = ((scores - mean[:, None]).square() * masks).sum(-1) / count
        consequent_mean = (weights * self.consequents).sum(-1)
        conflict = (weights * ((self.consequents - consequent_mean[:, :, None]) / 4.).square()).sum(-1)
        best = scores.masked_fill(~masks, -torch.inf).argmax(-1, keepdim=True)
        coverage = -torch.expm1(-evidence.sum(-1).gather(1, best).squeeze(1))
        agreement = torch.exp(-4 * conflict.gather(1, best).squeeze(1))
        maximum = scores.masked_fill(~masks, -torch.inf).amax(-1)
        minimum = scores.masked_fill(~masks, torch.inf).amin(-1)
        amplitude = scores.abs().masked_fill(~masks, 0.).amax(-1)
        contrast = ((maximum - minimum) / (amplitude + 1e-6)).clamp(0., 1.)
        confidence = coverage * agreement * contrast
        ranges = states[:, 1092:1452] * 3.5
        risks = ((.70 - ranges[:, None].masked_fill(~self.direction_masks, torch.inf).amin(-1)) / .55).clamp(0., 1.)
        return dict(scores=scores, mean=mean, variance=variance, confidence=confidence, risks=risks,
                    route=(features[:, :, 12] * masks).sum(-1) / count,
                    revisit=((1. - features[:, :, 10]) * masks).sum(-1) / count)


class HighHyperFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(13, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 2))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, states, masks, floors, neural_q, knowledge):
        count = masks.sum(-1, keepdim=True)
        mean = (neural_q * masks).sum(-1, keepdim=True) / count
        sigma = (((neural_q - mean).square() * masks).sum(-1, keepdim=True) / count + 1e-6).sqrt()
        knowledge_z = (knowledge['scores'] - knowledge['mean'][:, None]) / (knowledge['variance'][:, None] + 1e-6).sqrt()
        context = torch.cat((states[:, 1452:1456], knowledge['risks'], knowledge['route'][:, None],
                             knowledge['revisit'][:, None], knowledge['confidence'][:, None],
                             sigma.log1p().tanh(), knowledge['variance'][:, None].sqrt().tanh()), -1)
        preference = self.network(context.detach()).softmax(-1)[:, 1]
        alpha = knowledge['confidence'].detach() * (floors + (1. - floors) * preference)
        mixed = mean + (1. - alpha[:, None]) * (neural_q - mean) + alpha[:, None] * sigma.detach() * knowledge_z
        q = torch.where(alpha[:, None] > 0, mixed, neural_q)
        knowledge_log = knowledge_z.masked_fill(~masks, -torch.inf).log_softmax(-1).masked_fill(~masks, 0.)
        neural_log = ((neural_q - mean) / sigma.detach()).masked_fill(~masks, -torch.inf).log_softmax(-1).masked_fill(~masks, 0.)
        probability = knowledge_log.exp() * masks
        imitation = alpha.detach() * (probability.detach() * (knowledge_log.detach() - neural_log)).sum(-1)
        exploration = (1. - alpha[:, None]) * masks / count + alpha[:, None] * probability
        return dict(q=q, imitation=imitation, exploration=exploration, alpha=alpha,
                    confidence=knowledge['confidence'], preference=preference)


class GuidedHighQNetwork(nn.Module):
    def __init__(self, width, low_config, guidance=True):
        super().__init__()
        self.guidance = guidance
        self.neural = HighQNetwork(width)
        self.knowledge = HighFuzzyKnowledge(low_config)
        self.fusion = HighHyperFusion()

    def forward(self, states, masks, floors):
        knowledge = self.knowledge(states, masks)
        prior = (knowledge['scores'] - knowledge['mean'][:, None]) / (knowledge['variance'][:, None] + 1e-6).sqrt()
        neural, attention = self.neural(states, masks, prior if self.guidance else torch.zeros_like(prior))
        if self.guidance:
            output = self.fusion(states, masks, floors, neural, knowledge)
        else:
            zero = states.new_zeros(len(states))
            output = dict(q=neural, imitation=zero, exploration=masks / masks.sum(-1, keepdim=True),
                          alpha=zero, confidence=knowledge['confidence'], preference=zero)
        output.update(neural_q=neural, knowledge_q=knowledge['scores'], revisit=knowledge['revisit'], attention=attention)
        return output


class ObservedMap:
    """Episode-local occupancy evidence. Inputs contain no simulator geometry."""
    resolution = .1
    config = dict(resolution=.1, inflation=.15, free_evidence=-.4, occupied_evidence=.85,
                  log_odds_limit=4., max_range_hit_margin=.05, frontier_radius=1.)

    def __init__(self, angle_min, angle_increment):
        self.angles = angle_min + np.arange(360) * angle_increment
        self.origin = np.zeros(2, dtype=int)
        self.log_odds = np.zeros((0, 0), np.float32)
        self.anchor = None
        self.frontiers = np.empty((0, 2))
        self.path = np.empty((0, 2))
        self.goal_seen = self.goal_near = False

    def observe(self, ranges, position, yaw):
        directions = np.stack((np.cos(self.angles + yaw), np.sin(self.angles + yaw)), -1)
        sensor = np.asarray(position) - .032 * np.array([math.cos(yaw), math.sin(yaw)])
        # Five noise standard deviations below max range avoids treating max-range
        # measurement noise as a ring of obstacles in the 2D sensor model.
        hits = ranges < 3.45
        distances = np.arange(0., 3.501, .05)
        samples = sensor + directions[:, None] * distances[None, :, None]
        valid = distances[None] <= (ranges - .05 * hits)[:, None]
        free = np.rint(samples[valid] / self.resolution).astype(int)
        occupied = np.rint((sensor + directions[hits] * ranges[hits, None]) / self.resolution).astype(int)
        all_cells = np.concatenate((free, occupied))
        lo, hi = all_cells.min(0) - 2, all_cells.max(0) + 3
        if self.log_odds.size:
            lo = np.minimum(lo, self.origin)
            hi = np.maximum(hi, self.origin + self.log_odds.shape)
        shape = tuple(hi - lo)
        if shape != self.log_odds.shape:
            expanded = np.zeros(shape, np.float32)
            offset = self.origin - lo
            expanded[offset[0]:offset[0] + self.log_odds.shape[0],
                     offset[1]:offset[1] + self.log_odds.shape[1]] = self.log_odds
            self.log_odds, self.origin = expanded, lo
        free -= self.origin
        occupied -= self.origin
        free_ids = np.unique(free[:, 0] * self.log_odds.shape[1] + free[:, 1])
        occupied_ids = np.unique(occupied[:, 0] * self.log_odds.shape[1] + occupied[:, 1])
        before = self.log_odds.flat[occupied_ids].copy()
        self.log_odds.flat[free_ids] -= .4
        self.log_odds.flat[occupied_ids] = before + .85
        np.clip(self.log_odds, -4., 4., out=self.log_odds)

    @staticmethod
    def search(free, start):
        """Eight-connected Dijkstra, with no diagonal corner cutting."""
        distance = np.full(free.shape, np.inf)
        parent = np.full(free.shape, -1, np.int32)
        start = tuple(start)
        distance[start] = 0.
        queue = [(0., start[0], start[1])]
        nx, ny = free.shape
        while queue:
            cost, x, y = heapq.heappop(queue)
            if cost > distance[x, y]:
                continue
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                xx, yy = x + dx, y + dy
                if not (0 <= xx < nx and 0 <= yy < ny and free[xx, yy]):
                    continue
                if dx and dy and not (free[x + dx, y] and free[x, y + dy]):
                    continue
                candidate = cost + (.1 * math.sqrt(2) if dx and dy else .1)
                if candidate < distance[xx, yy]:
                    distance[xx, yy] = candidate
                    parent[xx, yy] = x * ny + y
                    heapq.heappush(queue, (candidate, xx, yy))
        return distance, parent

    def route_features(self, position, goal, world_points, memory):
        occupied = self.log_odds > 0
        inflated = np.zeros_like(occupied)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                source = (slice(max(0, -dx), min(occupied.shape[0], occupied.shape[0] - dx)),
                          slice(max(0, -dy), min(occupied.shape[1], occupied.shape[1] - dy)))
                target = (slice(max(0, dx), min(occupied.shape[0], occupied.shape[0] + dx)),
                          slice(max(0, dy), min(occupied.shape[1], occupied.shape[1] + dy)))
                inflated[target] |= occupied[source]
        free = (self.log_odds < 0) & ~inflated
        root = np.rint(np.asarray(position) / .1).astype(int) - self.origin
        result = np.full(len(world_points), .5, np.float32)
        available = np.zeros(len(world_points), np.float32)
        self.path = np.empty((0, 2))
        self.frontiers = np.empty((0, 2))
        if not free[tuple(root)]:
            return result, available
        start_distance, _ = self.search(free, root)
        reachable = np.isfinite(start_distance)
        goal_cell = np.rint(np.asarray(goal) / .1).astype(int) - self.origin
        goal_known = ((goal_cell >= 0).all() and (goal_cell < free.shape).all()
                      and reachable[tuple(goal_cell)])
        unknown = self.log_odds == 0
        frontier = np.zeros_like(free)
        frontier[1:] |= unknown[:-1]
        frontier[:-1] |= unknown[1:]
        frontier[:, 1:] |= unknown[:, :-1]
        frontier[:, :-1] |= unknown[:, 1:]
        frontier &= reachable
        frontier_cells = np.argwhere(frontier)
        self.frontiers = (frontier_cells + self.origin) * .1
        if goal_known:
            target = goal_cell
            self.anchor = None
        else:
            retained = False
            if self.anchor is not None:
                cell = np.rint(self.anchor / .1).astype(int) - self.origin
                retained = ((cell >= 0).all() and (cell < free.shape).all() and frontier[tuple(cell)]
                            and np.linalg.norm(self.anchor - position) >= .5)
            if retained:
                target = cell
            elif len(frontier_cells):
                offsets = np.array([(x, y) for x in range(-10, 11) for y in range(-10, 11) if x*x + y*y <= 100])
                neighbors = frontier_cells[:, None] + offsets
                inside = ((neighbors >= 0) & (neighbors < np.array(free.shape))).all(-1)
                clipped = np.clip(neighbors, 0, np.array(free.shape) - 1)
                information = ((~inside) | unknown[clipped[:, :, 0], clipped[:, :, 1]]).mean(1)
                novelty = memory.features(self.frontiers, np.ones(len(frontier_cells), bool), -1)
                costs = (start_distance[tuple(frontier_cells.T)] + np.linalg.norm(self.frontiers - goal, axis=1)
                         + 2 * (1 - novelty) - information)
                target = frontier_cells[int(costs.argmin())]
                self.anchor = (target + self.origin) * .1
            else:
                self.anchor = None
                return result, available
        target_distance, parent = self.search(free, target)
        cells = np.rint(world_points / .1).astype(int) - self.origin
        inside = ((cells >= 0) & (cells < free.shape)).all(1)
        indices = np.flatnonzero(inside)
        connected = np.isfinite(target_distance[tuple(cells[indices].T)])
        indices = indices[connected]
        lengths = start_distance[tuple(cells[indices].T)]
        delta = target_distance[tuple(root)] - target_distance[tuple(cells[indices].T)]
        result[indices] = np.clip((1 + delta / np.maximum(lengths, .1)) / 2, 0, 1)
        available[indices] = 1.
        cell = tuple(root)
        route = [cell]
        while cell != tuple(target):
            cell = np.unravel_index(parent[cell], free.shape)
            route.append(cell)
        self.path = (np.array(route) + self.origin) * .1
        return result, available


def candidate_features(state, points, exact, novelty, route_progress, route_valid, beams):
    """14 observable features shared by guided and matched unguided policies."""
    ranges = state[:360] * 3.5
    distance, heading = float(state[362]) * 5 * math.sqrt(2), float(state[363]) * math.pi
    goal = distance * np.array([math.cos(heading), math.sin(heading)])
    lengths = np.linalg.norm(points, axis=1)
    angles = np.arctan2(points[:, 1], points[:, 0])
    endpoints = ranges[:, None] * np.stack((np.cos(beams), np.sin(beams)), -1) + [-.032, 0.]
    clearance = np.clip((np.linalg.norm(points[:, None] - endpoints, axis=-1).min(1) - .15) / .15, 0, 1)
    angular_error = (beams[None] - angles[:, None] + math.pi) % (2 * math.pi) - math.pi
    fan = abs(angular_error) <= math.pi / 12
    openness = np.clip((fan * ranges).sum(1) / fan.sum(1), 0, 1)
    is_goal = np.zeros(225)
    if exact >= 0:
        is_goal[exact] = 1.
    alignment = (1 + np.cos(angles - heading)) / 2
    progress = np.clip((1 + (distance - np.linalg.norm(goal - points, axis=1)) / np.maximum(lengths, 1e-6)) / 2, 0, 1)
    return np.column_stack((points / 3.5, lengths / 3.5, np.sin(angles), np.cos(angles),
                            is_goal, alignment, progress, clearance, openness, novelty,
                            route_progress, route_valid, (1 + np.cos(angles)) / 2)).astype(np.float32)
