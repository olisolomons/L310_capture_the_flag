from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import matplotlib.pylab as plt
import matplotlib.patches as patches
import numpy as np
import os
import re
import scipy.signal
import yaml

# Constants used for indexing.

X = 0
Y = 1
XY = slice(None, 2)
YAW = 2

# Constants for occupancy grid.
FREE = 0
UNKNOWN = 1
OCCUPIED = 2

ROBOT_RADIUS = 0.105 / 2.
GOAL_POSITION = np.array([1.5, 1.5], dtype=np.float32)  # Any orientation is good.
START_POSE = np.array([-1.5, -1.5, 0.], dtype=np.float32)
MAX_ITERATIONS = 400


def sample_random_position(occupancy_grid):
    while True:
        position = (np.random.random(2) - 0.5) * 8
        if not occupancy_grid.is_occupied(position):
            return position


def angle_difference(a, b):
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def angle(vector):
    x, y = vector
    return np.arctan2(y, x)


def find_arc(node_a, node_b):
    center, radius = find_circle(node_a, node_b)
    start_angle = angle(node_a.pose[XY] - center)
    end_angle = angle(node_b.pose[XY] - center)
    # clockwise or anticlockwise
    angle_direction = 1 if 0 < np.cross(node_a.direction, center - node_a.position) else -1
    # correct end_angle to ensure that the correct arc is used
    if angle_direction * end_angle < angle_direction * start_angle:
        end_angle += angle_direction * np.pi * 2

    return start_angle, end_angle, angle_direction


def adjust_pose(node, final_position, occupancy_grid):
    final_pose = node.pose.copy()
    final_pose[XY] = final_position
    angle_to_final = angle(final_position - node.pose[XY])
    final_pose[YAW] -= angle_difference(node.pose[YAW], angle_to_final) * 2

    final_node = Node(final_pose)

    # check for intersection
    center, radius = find_circle(node, final_node)
    grid_cell_size = 4 / occupancy_grid.values.shape[0]
    angle_step = grid_cell_size / radius

    start_angle, end_angle, angle_direction = find_arc(node, final_node)

    # find all angles along the the circle that are part of the arc
    intermediary_angles = np.arange(start_angle, end_angle, angle_direction * angle_step)
    # find all the points in the arc
    intermediary_poses = radius * np.stack((np.cos(intermediary_angles), np.sin(intermediary_angles)), axis=1) + center
    # if the arc intersects wth an obstacle, discard the node
    if not np.all(np.logical_not(occupancy_grid.is_occupied(intermediary_poses))):
        return None
    else:
        return final_node


# Defines an occupancy grid.
class OccupancyGrid(object):
    def __init__(self, values, origin, resolution):
        self._original_values = values.copy()
        self._values = values.copy()
        # Inflate obstacles (using a convolution).
        inflated_grid = np.zeros_like(values)
        inflated_grid[values == OCCUPIED] = 1.
        w = 2 * int(ROBOT_RADIUS / resolution) + 1
        inflated_grid = scipy.signal.convolve2d(inflated_grid, np.ones((w, w)), mode='same')
        self._values[inflated_grid > 0.] = OCCUPIED
        self._origin = np.array(origin[:2], dtype=np.float32)
        self._origin -= resolution / 2.
        assert origin[YAW] == 0.
        self._resolution = resolution

    @property
    def values(self):
        return self._values

    @property
    def resolution(self):
        return self._resolution

    @property
    def origin(self):
        return self._origin

    def draw(self):
        plt.imshow(self._original_values.T, interpolation='none', origin='lower',
                   extent=[self._origin[X],
                           self._origin[X] + self._values.shape[0] * self._resolution,
                           self._origin[Y],
                           self._origin[Y] + self._values.shape[1] * self._resolution])
        plt.set_cmap('gray_r')

    def get_index(self, position):
        idx = ((position - self._origin) / self._resolution).astype(np.int32)
        if len(idx.shape) == 2:
            idx[:, 0] = np.clip(idx[:, 0], 0, self._values.shape[0] - 1)
            idx[:, 1] = np.clip(idx[:, 1], 0, self._values.shape[1] - 1)
            return (idx[:, 0], idx[:, 1])
        idx[0] = np.clip(idx[0], 0, self._values.shape[0] - 1)
        idx[1] = np.clip(idx[1], 0, self._values.shape[1] - 1)
        return tuple(idx)

    def get_position(self, i, j):
        return np.array([i, j], dtype=np.float32) * self._resolution + self._origin

    def is_occupied(self, position):
        return self._values[self.get_index(position)] == OCCUPIED

    def is_free(self, position):
        return self._values[self.get_index(position)] == FREE


# Defines a node of the graph.
class Node(object):
    def __init__(self, pose):
        self._pose = pose.copy()
        self._neighbors = []
        self._parent = None
        self._cost = 0.

    @property
    def pose(self):
        return self._pose

    def add_neighbor(self, node):
        self._neighbors.append(node)

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, node):
        self._parent = node

    @property
    def neighbors(self):
        return self._neighbors

    @property
    def position(self):
        return self._pose[:2]

    @property
    def yaw(self):
        return self._pose[YAW]

    @property
    def direction(self):
        return np.array([np.cos(self._pose[YAW]), np.sin(self._pose[YAW])], dtype=np.float32)

    @property
    def cost(self):
        return self._cost

    @cost.setter
    def cost(self, c):
        self._cost = c


def rrt(start_pose, goal_position, occupancy_grid):
    # RRT builds a graph one node at a time.
    graph = []
    start_node = Node(start_pose)
    final_node = None
    if occupancy_grid.is_occupied(goal_position):
        print('Goal position is not in the free space.')
        return start_node, final_node
    graph.append(start_node)
    for _ in range(MAX_ITERATIONS):
        position = sample_random_position(occupancy_grid)
        # With a random chance, draw the goal position.
        if np.random.rand() < .05:
            position = goal_position
        # Find closest node in graph.
        # In practice, one uses an efficient spatial structure (e.g., quadtree).
        potential_parent = sorted(((n, np.linalg.norm(position - n.position)) for n in graph), key=lambda x: x[1])
        # Pick a node at least some distance away but not too far.
        # We also verify that the angles are aligned (within pi / 4).
        u = None
        for n, d in potential_parent:
            if .2 < d < 3 and n.direction.dot(position - n.position) / d > 0.70710678118:
                u = n
                break
        else:
            continue
        v = adjust_pose(u, position, occupancy_grid)
        if v is None:
            continue
        u.add_neighbor(v)
        v.parent = u
        graph.append(v)
        if np.linalg.norm(v.position - goal_position) < .2:
            final_node = v
            break
    return start_node, final_node


def find_circle(node_a, node_b):
    def perpendicular(v):
        w = np.empty_like(v)
        w[X] = -v[Y]
        w[Y] = v[X]
        return w

    db = perpendicular(node_b.direction)
    dp = node_a.position - node_b.position
    t = np.dot(node_a.direction, db)
    if np.abs(t) < 1e-3:
        # By construction node_a and node_b should be far enough apart,
        # so they must be on opposite end of the circle.
        center = (node_b.position + node_a.position) / 2.
        radius = np.linalg.norm(center - node_b.position)
    else:
        radius = np.dot(node_a.direction, dp) / t
        center = radius * db + node_b.position
    return center, np.abs(radius)


def read_pgm(filename, byteorder='>'):
    """Read PGM file."""
    with open(filename, 'rb') as fp:
        buf = fp.read()
    try:
        header, width, height, maxval = re.search(
            b'(^P5\s(?:\s*#.*[\r\n])*'
            b'(\d+)\s(?:\s*#.*[\r\n])*'
            b'(\d+)\s(?:\s*#.*[\r\n])*'
            b'(\d+)\s(?:\s*#.*[\r\n]\s)*)', buf).groups()
    except AttributeError:
        raise ValueError('Invalid PGM file: "{}"'.format(filename))
    maxval = int(maxval)
    height = int(height)
    width = int(width)
    img = np.frombuffer(buf,
                        dtype='u1' if maxval < 256 else byteorder + 'u2',
                        count=width * height,
                        offset=len(header)).reshape((height, width))
    return img.astype(np.float32) / 255.


def draw_solution(start_node, final_node=None):
    ax = plt.gca()

    def draw_path(u, v, arrow_length=.1, color=(.8, .8, .8), lw=1):
        du = u.direction
        plt.arrow(u.pose[X], u.pose[Y], du[0] * arrow_length, du[1] * arrow_length,
                  head_width=.05, head_length=.1, fc=color, ec=color)
        dv = v.direction
        plt.arrow(v.pose[X], v.pose[Y], dv[0] * arrow_length, dv[1] * arrow_length,
                  head_width=.05, head_length=.1, fc=color, ec=color)
        center, radius = find_circle(u, v)
        du = u.position - center
        theta1 = np.arctan2(du[1], du[0])
        dv = v.position - center
        theta2 = np.arctan2(dv[1], dv[0])
        # Check if the arc goes clockwise.
        if np.cross(u.direction, du).item() > 0.:
            theta1, theta2 = theta2, theta1
        ax.add_patch(patches.Arc(center, radius * 2., radius * 2.,
                                 theta1=theta1 / np.pi * 180., theta2=theta2 / np.pi * 180.,
                                 color=color, lw=lw))

    points = []
    s = [(start_node, None)]  # (node, parent).
    while s:
        v, u = s.pop()
        if hasattr(v, 'visited'):
            continue
        v.visited = True
        # Draw path from u to v.
        if u is not None:
            draw_path(u, v)
        points.append(v.pose[:2])
        for w in v.neighbors:
            s.append((w, v))

    points = np.array(points)
    plt.scatter(points[:, 0], points[:, 1], s=10, marker='o', color=(.8, .8, .8))
    if final_node is not None:
        plt.scatter(final_node.position[0], final_node.position[1], s=10, marker='o', color='k')
        # Draw final path.
        v = final_node
        while v.parent is not None:
            draw_path(v.parent, v, color='k', lw=2)
            v = v.parent


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Uses RRT to reach the goal.')
    parser.add_argument('--map', action='store', default='map', help='Which map to use.')
    args, unknown = parser.parse_known_args()

    # Load map.
    with open(args.map + '.yaml') as fp:
        data = yaml.load(fp)
    img = read_pgm(os.path.join(os.path.dirname(args.map), data['image']))
    occupancy_grid = np.empty_like(img, dtype=np.int8)
    occupancy_grid[:] = UNKNOWN
    occupancy_grid[img < .1] = OCCUPIED
    occupancy_grid[img > .9] = FREE
    # Transpose (undo ROS processing).
    occupancy_grid = occupancy_grid.T
    # Invert Y-axis.
    occupancy_grid = occupancy_grid[:, ::-1]
    occupancy_grid = OccupancyGrid(occupancy_grid, data['origin'], data['resolution'])

    # Run RRT.
    start_node, final_node = rrt(START_POSE, GOAL_POSITION, occupancy_grid)

    # Plot environment.
    fig, ax = plt.subplots()
    occupancy_grid.draw()
    plt.scatter(.3, .2, s=10, marker='o', color='green', zorder=1000)
    draw_solution(start_node, final_node)
    plt.scatter(START_POSE[0], START_POSE[1], s=10, marker='o', color='green', zorder=1000)
    plt.scatter(GOAL_POSITION[0], GOAL_POSITION[1], s=10, marker='o', color='red', zorder=1000)

    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([-.5 - 2., 2. + .5])
    plt.ylim([-.5 - 2., 2. + .5])

    plt.show()


def get_path(final_node):
    # Construct path from RRT solution.
    if final_node is None:
        return []
    path_reversed = []
    path_reversed.append(final_node)
    while path_reversed[-1].parent is not None:
        path_reversed.append(path_reversed[-1].parent)
    path = list(reversed(path_reversed))
    # Put a point every 5 cm.
    distance = 0.05
    offset = 0.
    points_x = []
    points_y = []
    for u, v in zip(path, path[1:]):
        center, radius = find_circle(u, v)
        du = u.position - center
        theta1 = np.arctan2(du[1], du[0])
        dv = v.position - center
        theta2 = np.arctan2(dv[1], dv[0])
        # Check if the arc goes clockwise.
        clockwise = np.cross(u.direction, du).item() > 0.
        # Generate a point every 5cm apart.
        da = distance / radius
        offset_a = offset / radius
        if clockwise:
            da = -da
            offset_a = -offset_a
            if theta2 > theta1:
                theta2 -= 2. * np.pi
        else:
            if theta2 < theta1:
                theta2 += 2. * np.pi
        angles = np.arange(theta1 + offset_a, theta2, da)
        offset = distance - (theta2 - angles[-1]) * radius
        points_x.extend(center[X] + np.cos(angles) * radius)
        points_y.extend(center[Y] + np.sin(angles) * radius)
    return zip(points_x, points_y)
