"""Small ROS-free differential-drive simulator and reproducible wall layouts."""

import heapq
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from util import goal_observation

SCENARIOS = ('u_wall', 'dead_end', 'rooms', 'door', 'narrow')


def make_layout(kind, seed):
    """Construct valid geometry directly; there is no rejection-sampling loop."""
    rng = np.random.default_rng(seed)
    width = float(rng.uniform(.6, 1.2))
    walls = [[-6.1, -6.1, -5.9, 6.1], [5.9, -6.1, 6.1, 6.1],
             [-5.9, -6.1, 5.9, -5.9], [-5.9, 5.9, 5.9, 6.1]]
    if kind in ('u_wall', 'dead_end'):
        half = float(rng.uniform(1.2, 2.)) if kind == 'u_wall' else width / 2 + .1
        bottom = -2. if kind == 'u_wall' else -3.
        top = float(rng.uniform(1.6, 2.4))
        walls += [[-half - .1, bottom, -half + .1, top], [half - .1, bottom, half + .1, top],
                  [-half - .1, bottom - .2, half + .1, bottom]]
        start = [0., bottom + .8]
        goal = [0., bottom - 1.5] if kind == 'u_wall' else [3., -2.2]
        yaw = -math.pi / 2
    elif kind == 'rooms':
        door_y = float(rng.choice([1.6, 2., 2.4]))
        door_x = float(rng.choice([2., 2.4, 2.8]))
        # Four rooms connected by three doors, so the full free area is connected.
        walls += [[-.1, -5.9, .1, door_y - width / 2], [-.1, door_y + width / 2, .1, 5.9],
                  [-5.9, -.1, -door_x - width / 2, .1], [-door_x + width / 2, -.1, -.1, .1],
                  [.1, -.1, door_x - width / 2, .1], [door_x + width / 2, -.1, 5.9, .1]]
        start, goal, yaw = [-4., -4.], [4., 4.], math.pi / 4
    elif kind == 'door':
        door_y = float(rng.choice([-2., -1., 0., 1., 2.]))
        walls += [[-.1, -5.9, .1, door_y - width / 2], [-.1, door_y + width / 2, .1, 5.9]]
        start, goal, yaw = [-4., -2.], [4., 2.], 0.
    elif kind == 'narrow':
        # Complement of an S-shaped union of three corridor rectangles.
        free = np.array([[-5.4, -2 - width / 2, 1 + width / 2, -2 + width / 2],
                         [1 - width / 2, -2 - width / 2, 1 + width / 2, 2 + width / 2],
                         [1 - width / 2, 2 - width / 2, 5.4, 2 + width / 2]])
        xs = np.unique(np.r_[-5.9, 5.9, free[:, [0, 2]].ravel()])
        ys = np.unique(np.r_[-5.9, 5.9, free[:, [1, 3]].ravel()])
        for y0, y1 in zip(ys[:-1], ys[1:]):
            for x0, x1 in zip(xs[:-1], xs[1:]):
                mid = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
                if not ((mid >= free[:, :2]).all(1) & (mid <= free[:, 2:]).all(1)).any():
                    walls.append([x0, y0, x1, y1])
        start, goal, yaw = [-4.6, -2.], [4.6, 2.], 0.
    else:
        raise ValueError('Unknown scenario: ' + kind)
    turns = int(rng.integers(4))
    angle = turns * math.pi / 2
    rotation = np.array([[round(math.cos(angle)), -round(math.sin(angle))],
                         [round(math.sin(angle)), round(math.cos(angle))]])
    walls = np.array(walls, dtype=np.float64)
    corners = np.stack((walls[:, :2], walls[:, 2:], walls[:, [0, 3]], walls[:, [2, 1]]), 1) @ rotation.T
    walls = np.concatenate((corners.min(1), corners.max(1)), 1)
    return walls, rotation @ start, rotation @ goal, (yaw + angle + math.pi) % (2 * math.pi) - math.pi


def wall_distance(points, walls):
    points = np.asarray(points)
    gap = np.maximum(np.maximum(walls[:, :2] - points[..., None, :], points[..., None, :] - walls[:, 2:]), 0)
    return np.linalg.norm(gap, axis=-1).min(-1)


def raycast(origin, directions, walls):
    # Parallel rays are handled by slab intervals, without dividing by zero.
    nonzero = abs(directions) > 1e-12
    inverse = np.divide(1., directions, out=np.zeros_like(directions), where=nonzero)
    first = (walls[None, :, :2] - origin) * inverse[:, None]
    second = (walls[None, :, 2:] - origin) * inverse[:, None]
    inside = (origin >= walls[:, :2]) & (origin <= walls[:, 2:])
    low = np.where(nonzero[:, None], np.minimum(first, second), np.where(inside[None], -np.inf, np.inf))
    high = np.where(nonzero[:, None], np.maximum(first, second), np.where(inside[None], np.inf, -np.inf))
    near, far = low.max(-1), high.min(-1)
    hits = np.where(far >= np.maximum(near, 0), np.maximum(near, 0), np.inf)
    return np.minimum(hits.min(1), 3.5)


class DistanceField:
    """One reverse Dijkstra per task; rewards query the fixed goal distance field."""
    resolution = .2

    def __init__(self, walls, start, goal):
        self.centers = np.arange(-6., 6.01, self.resolution)
        yy, xx = np.meshgrid(self.centers, self.centers, indexing='ij')
        points = np.stack((xx, yy), -1)
        gap = np.maximum(np.maximum(walls[:, :2] - points[..., None, :] - .1,
                                    points[..., None, :] - walls[:, 2:] - .1), 0)
        self.occupied = (np.linalg.norm(gap, axis=-1) <= .15 + 1e-9).any(-1)
        self.free_cells = np.argwhere(~self.occupied)
        self.free_xy = points[~self.occupied]
        if wall_distance(np.array([start, goal]), walls).min() <= .25:
            raise ValueError('Task anchors lack collision clearance')
        self.distances = np.full(self.occupied.shape, np.inf)
        target = self.cell(goal)
        self.distances[target] = 0.
        queue = [(0., target)]
        while queue:
            cost, cell = heapq.heappop(queue)
            if cost > self.distances[cell]:
                continue
            for neighbor, step in self.neighbors(cell):
                proposed = cost + step
                if proposed < self.distances[neighbor]:
                    self.distances[neighbor] = proposed
                    heapq.heappush(queue, (proposed, neighbor))
        if not np.isfinite(self.distances[~self.occupied]).all():
            raise ValueError('Generated layout has disconnected inflated free space')
        self.initial_distance = self.distance(start)

    def neighbors(self, cell):
        row, col = cell
        for dr, dc in ((-1, 0), (0, -1), (0, 1), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            nr, nc = row + dr, col + dc
            if not (0 <= nr < len(self.centers) and 0 <= nc < len(self.centers)) or self.occupied[nr, nc]:
                continue
            if dr and dc and (self.occupied[row + dr, col] or self.occupied[row, col + dc]):
                continue
            yield (nr, nc), self.resolution * (math.sqrt(2) if dr and dc else 1)

    def cell(self, position):
        return tuple(self.free_cells[np.square(self.free_xy - position).sum(1).argmin()])

    def distance(self, position):
        return float(self.distances[self.cell(position)])


def export_world(path, kind, seed):
    walls, start, goal, yaw = make_layout(kind, seed)
    DistanceField(walls, start, goal)
    root = ET.Element('sdf', version='1.6')
    world = ET.SubElement(root, 'world', name='default')
    for model in ('sun', 'ground_plane'):
        ET.SubElement(ET.SubElement(world, 'include'), 'uri').text = 'model://' + model
    physics = ET.SubElement(world, 'physics', name='default_physics', type='ode')
    ET.SubElement(physics, 'max_step_size').text = '.001'
    ET.SubElement(physics, 'real_time_update_rate').text = '1000'
    for index, (x0, y0, x1, y1) in enumerate(walls):
        model = ET.SubElement(world, 'model', name='wall_' + str(index))
        ET.SubElement(model, 'static').text = 'true'
        ET.SubElement(model, 'pose').text = '{} {} .5 0 0 0'.format((x0 + x1) / 2, (y0 + y1) / 2)
        link = ET.SubElement(model, 'link', name='link')
        for tag in ('collision', 'visual'):
            item = ET.SubElement(link, tag, name=tag)
            ET.SubElement(ET.SubElement(ET.SubElement(item, 'geometry'), 'box'), 'size').text = '{} {} 1'.format(x1 - x0, y1 - y0)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)
    return start, goal, yaw


class Sim2DEnv:
    def __init__(self, low_config, timeout_steps=800, render=False):
        self.v_max, self.w_max = low_config['v_max'], low_config['w_max']
        self.action_duration = low_config['action_duration']
        self.timeout_steps = timeout_steps
        self.scan_angle_min, self.scan_angle_increment = low_config['scan_angle_min'], low_config['scan_angle_increment']
        self.beams = self.scan_angle_min + np.arange(360) * self.scan_angle_increment
        self.viewer = NavigationViewer() if render else None
        self.scene = None

    def reset(self, kind, seed):
        self.kind, self.seed = kind, seed
        self.rng = np.random.default_rng(seed)
        self.walls, self.position, self.goal_xy, self.yaw = make_layout(kind, seed)
        self.field = DistanceField(self.walls, self.position, self.goal_xy)
        self.initial_distance = self.field.initial_distance
        self.velocity = np.zeros(2)
        self.subgoal_xy = self.goal_xy.copy()
        self.steps = 0
        self.sim_time = self.path_length = 0.
        self.trajectory = [self.position.copy()]
        self.state, self.info = self._observe(False)
        if self.viewer:
            self.viewer.reset(self)
        return self.state.copy(), dict(self.info)

    def _observe(self, collision):
        origin = self.position - .032 * np.array([math.cos(self.yaw), math.sin(self.yaw)])
        directions = np.stack((np.cos(self.beams + self.yaw), np.sin(self.beams + self.yaw)), -1)
        distances = raycast(origin, directions, self.walls)
        ranges = np.clip(distances + self.rng.normal(0, .01, 360), 0, 3.5)
        ranges[distances < .12] = 0.
        self.ranges = ranges
        state, distance, heading = goal_observation(ranges, self.position, self.yaw, self.velocity,
                                                  self.goal_xy, self.v_max, self.w_max)
        collision = collision or ranges.min() < .15
        reached = distance < .1 and not collision
        timeout = self.steps >= self.timeout_steps and not (collision or reached)
        info = dict(x=float(self.position[0]), y=float(self.position[1]), yaw=self.yaw,
                    distance=distance, heading=heading, min_scan=float(ranges.min()), step=self.steps,
                    goal_x=float(self.goal_xy[0]), goal_y=float(self.goal_xy[1]),
                    reached=reached, collision=collision, timeout=timeout, sim_time=self.sim_time,
                    sim_duration=self.action_duration if self.steps else 0.,
                    outcome='collision' if collision else 'success' if reached else 'timeout' if timeout else 'running')
        return state, info

    def low_state(self):
        return goal_observation(self.ranges, self.position, self.yaw, self.velocity,
                                self.subgoal_xy, self.v_max, self.w_max)[0]

    def set_subgoal(self, point):
        self.subgoal_xy = np.asarray(point).copy()

    def astar_distance(self):
        return self.field.distance(self.position)

    def refresh(self):
        if self.viewer:
            self.viewer.update(self)

    def step(self, action):
        self.velocity = np.asarray(action, dtype=np.float64)
        intervals = int(math.ceil(self.action_duration / .01))
        dt = self.action_duration / intervals
        collision = False
        travelled = 0.
        for _ in range(intervals):
            v, w = self.velocity
            turn = w * dt
            self.position += v * dt * np.sinc(turn / (2 * math.pi)) * np.array([
                math.cos(self.yaw + turn / 2), math.sin(self.yaw + turn / 2)])
            self.yaw = (self.yaw + turn + math.pi) % (2 * math.pi) - math.pi
            travelled += abs(v) * dt
            if wall_distance(self.position, self.walls) < .105:
                collision = True
                break
        self.path_length += travelled
        self.sim_time += self.action_duration
        self.steps += 1
        self.trajectory.append(self.position.copy())
        self.state, self.info = self._observe(collision)
        return self.state.copy(), self.info['reached'] or self.info['collision'], self.info['timeout'], dict(self.info)

    def close(self):
        if self.viewer:
            self.viewer.close()


class NavigationViewer:
    def __init__(self):
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection, PatchCollection
        self.plt, self.LineCollection, self.PatchCollection = plt, LineCollection, PatchCollection
        self.closed = False
        plt.ion()
        self.figure, self.axis = plt.subplots(figsize=(9, 8))
        self.figure.canvas.mpl_connect('close_event', lambda event: setattr(self, 'closed', True))
        self.axis.set(aspect='equal', xlim=(6.2, -6.2), ylim=(-6.2, 6.2), xlabel='y (m, positive left)', ylabel='x (m, positive up)')
        self.walls = PatchCollection([], facecolor='#444444')
        self.axis.add_collection(self.walls)
        self.laser = LineCollection([], linewidths=.3, colors='#87c2e5', alpha=.5)
        self.axis.add_collection(self.laser)
        self.path, = self.axis.plot([], [], color='#e67e22', linewidth=1)
        self.robot, = self.axis.plot([], [], 'o', color='#2068bc')
        self.heading, = self.axis.plot([], [], color='#2068bc')
        self.goal, = self.axis.plot([], [], '*', color='green', markersize=14, label='final goal (fixed per episode)')
        self.subgoal, = self.axis.plot([], [], '^', color='#963ec2', markersize=9, label='current subgoal')
        self.visits = self.axis.scatter([], [], s=16, color='#2266aa', marker='x', label='odometry samples')
        self.candidates = self.axis.scatter([], [], s=16, c=[], cmap='RdYlGn', vmin=0, vmax=1, alpha=.65)
        self.known = self.axis.scatter([], [], s=3, color='#aaaaaa', alpha=.25, label='observed free')
        self.occupied = self.axis.scatter([], [], s=5, color='#111111', label='observed occupied')
        self.frontiers = self.axis.scatter([], [], s=6, color='#00aaaa', label='frontiers')
        self.route, = self.axis.plot([], [], '--', color='#00aaaa', label='observed-map reference')
        self.teacher, = self.axis.plot([], [], 's', color='#cc3399', label='knowledge choice')
        self.fused, = self.axis.plot([], [], '+', color='#111111', markersize=10, label='fused choice')
        self.axis.legend(loc='upper right', fontsize=7)
        self.figure.colorbar(self.candidates, ax=self.axis, label='candidate novelty', fraction=.04, pad=.03)
        self.text = self.figure.text(.12, .01, '', fontsize=9, family='monospace')
        self.figure.subplots_adjust(bottom=.18)

    def reset(self, env):
        from matplotlib.patches import Rectangle
        if self.closed:
            return
        self.walls.set_paths([Rectangle((y0, x0), y1 - y0, x1 - x0) for x0, y0, x1, y1 in env.walls])
        self.axis.set_title('{} / seed {}'.format(env.kind, env.seed))
        self.update(env)

    def update(self, env):
        if self.closed:
            return
        trajectory = np.asarray(env.trajectory)
        self.path.set_data(trajectory[:, 1], trajectory[:, 0])
        self.robot.set_data([env.position[1]], [env.position[0]])
        tip = env.position + .3 * np.array([math.cos(env.yaw), math.sin(env.yaw)])
        self.heading.set_data([env.position[1], tip[1]], [env.position[0], tip[0]])
        self.goal.set_data([env.goal_xy[1]], [env.goal_xy[0]])
        self.subgoal.set_data([env.subgoal_xy[1]], [env.subgoal_xy[0]])
        origin = env.position - .032 * np.array([math.cos(env.yaw), math.sin(env.yaw)])
        ends = origin + env.ranges[:, None] * np.stack((np.cos(env.beams + env.yaw), np.sin(env.beams + env.yaw)), -1)
        self.laser.set_segments(np.stack((np.broadcast_to(origin, ends.shape), ends), 1)[..., ::-1])
        info = env.info
        visits = np.asarray(info.get('memory_points', np.empty((0, 2))))
        candidates = np.asarray(info.get('candidate_points', np.empty((0, 2))))
        self.visits.set_offsets(visits[:, ::-1])
        self.candidates.set_offsets(candidates[:, ::-1])
        self.candidates.set_array(np.asarray(info.get('candidate_novelty', np.empty(0))))
        if 'observed_map' in info:
            mapping = info['observed_map']
            free = (np.argwhere(mapping.log_odds < 0) + mapping.origin) * .1
            occupied = (np.argwhere(mapping.log_odds > 0) + mapping.origin) * .1
            self.known.set_offsets(free[:, ::-1])
            self.occupied.set_offsets(occupied[:, ::-1])
            self.frontiers.set_offsets(mapping.frontiers[:, ::-1])
            self.route.set_data(mapping.path[:, 1], mapping.path[:, 0])
            self.teacher.set_data([info['knowledge_point'][1]], [info['knowledge_point'][0]])
            self.fused.set_data([info['fused_point'][1]], [info['fused_point'][0]])
        else:
            for artist in (self.known, self.occupied, self.frontiers):
                artist.set_offsets(np.empty((0, 2)))
            for artist in (self.route, self.teacher, self.fused):
                artist.set_data([], [])
        reward = '{:+.4f}'.format(info['segment_reward']) if 'segment_reward' in info else 'pending'
        self.text.set_text('step={}  final={:.3f}m  subgoal={:.3f}m\nsegment={}  elapsed={:.2f}s  no progress={:.2f}s / limit={:.2f}s\nreason={}  segment reward={}'.format(
            env.steps, info['distance'], info.get('subgoal_distance', float(np.linalg.norm(env.subgoal_xy - env.position))),
            info.get('segment_steps', 0), info.get('segment_elapsed', 0.), info.get('no_progress_time', 0.),
            info.get('segment_time_limit', 0.),
            info.get('subgoal_reason', 'reset'), reward))
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()

    def close(self):
        self.closed = True
        self.plt.close(self.figure)
