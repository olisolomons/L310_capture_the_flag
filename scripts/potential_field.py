from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import matplotlib.pylab as plt
import numpy as np

WALL_OFFSET = 2.
CYLINDER_POSITIONS = [np.array([0.5, 0], dtype=np.float32), np.array([0, 0.5], dtype=np.float32)]
# CYLINDER_POSITIONS = [np.array([0, 0], dtype=np.float32)]
# CYLINDER_POSITIONS = [np.array([.3, .2], dtype=np.float32)]
CYLINDER_RADII = [.3] * len(CYLINDER_POSITIONS)
GOAL_POSITION = np.array([1.5, 1.5], dtype=np.float32)
START_POSITION = np.array([-1.5, -1.5], dtype=np.float32)
MAX_SPEED = .25


def get_velocity_to_reach_goal(position, goal_position):
    return normalize(goal_position - position)


def get_velocity_to_avoid_obstacles(position, obstacle_positions, obstacle_radii):
    v = np.zeros(2, dtype=np.float32)
    # MISSING: Compute the velocity field needed to avoid the obstacles
    # In the worst case there might a large force pushing towards the
    # obstacles (consider what is the largest force resulting from the
    # get_velocity_to_reach_goal function). Make sure to not create
    # speeds that are larger than max_speed for each obstacle. Both obstacle_positions
    # and obstacle_radii are lists.

    for obstacle_position, obstacle_radius in zip(obstacle_positions, obstacle_radii):
        away_from = position - obstacle_position
        v += normalize(away_from) * (np.linalg.norm(away_from) / (np.linalg.norm(away_from) - obstacle_radius - 0.05) - 1)

    return 0.25 * v


def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-2:
        return np.zeros_like(v)
    return v / n


def cap(v, max_speed):
    n = np.linalg.norm(v)
    if n > max_speed:
        return v / n * max_speed
    return v


def get_velocity(position, goal, obstacle_positions, obstacle_radii, mode='all', speed_cap=MAX_SPEED):
    if mode in ('goal', 'all'):
        v_goal = get_velocity_to_reach_goal(position, goal)
    else:
        v_goal = np.zeros(2, dtype=np.float32)
    if mode in ('obstacle', 'all'):
        v_avoid = get_velocity_to_avoid_obstacles(
            position,
            obstacle_positions,
            obstacle_radii)
    else:
        v_avoid = np.zeros(2, dtype=np.float32)
    v = v_goal + v_avoid
    return cap(v, max_speed=speed_cap)


def plot_obstacle_function():
    from matplotlib import pyplot as plt
    x = np.linspace(1.001, 8, 100)
    y = x / (np.abs(x) - 1) - 1
    r = np.empty(x.shape)
    r.fill(1)
    plt.ylim([0, 12])
    plt.xlim([0, 8])
    plt.plot(x, y, label='speed=$\\frac{distance}{distance-1}-1$')
    plt.plot(r, np.linspace(0, 12, 100), label='distance=radius')
    ticks = np.arange(1, 8)
    plt.xticks(ticks, ["$%d\\times r$" % tick for tick in ticks])
    plt.xlabel("distance")
    plt.ylabel("speed contribution from a single obstacle")
    plt.legend()


def angle_difference(a, b):
    return (a - b + np.pi) % (2 * np.pi) - np.pi


if __name__ == '__main__':
    # plot_obstacle_function()
    parser = argparse.ArgumentParser(description='Runs obstacle avoidance with a potential field')
    parser.add_argument('--mode', action='store', default='all', help='Which velocity field to plot.',
                        choices=['obstacle', 'goal', 'all'])
    parser.add_argument('--bug', action='store_true', help='Use BUG algorithm to escape local minima')
    args, unknown = parser.parse_known_args()

    fig, ax = plt.subplots()
    # Plot field.
    X, Y = np.meshgrid(np.linspace(-WALL_OFFSET, WALL_OFFSET, 30),
                       np.linspace(-WALL_OFFSET, WALL_OFFSET, 30))
    U = np.zeros_like(X)
    V = np.zeros_like(X)
    for i in range(len(X)):
        for j in range(len(X[0])):
            velocity = get_velocity(np.array([X[i, j], Y[i, j]]), args.mode)
            U[i, j] = velocity[0]
            V[i, j] = velocity[1]
    plt.quiver(X, Y, U, V, units='width')

    # Plot environment.
    for CYLINDER_POSITION, CYLINDER_RADIUS in zip(CYLINDER_POSITIONS, CYLINDER_RADII):
        ax.add_artist(plt.Circle(CYLINDER_POSITION, CYLINDER_RADIUS, color='gray'))
    plt.plot([-WALL_OFFSET, WALL_OFFSET], [-WALL_OFFSET, -WALL_OFFSET], 'k')
    plt.plot([-WALL_OFFSET, WALL_OFFSET], [WALL_OFFSET, WALL_OFFSET], 'k')
    plt.plot([-WALL_OFFSET, -WALL_OFFSET], [-WALL_OFFSET, WALL_OFFSET], 'k')
    plt.plot([WALL_OFFSET, WALL_OFFSET], [-WALL_OFFSET, WALL_OFFSET], 'k')

    # Plot a simple trajectory from the start position.
    # Uses Euler integration.
    dt = 0.01
    x = START_POSITION
    positions = [x]

    if args.bug:
        do_bug = False
        bug_angle = None
        bug_threshold = 1e-1
        rotate_angle = np.pi * 0.475  # almost a quarter turn
        rotate_quarter = np.array(
            [[np.cos(rotate_angle), -np.sin(rotate_angle)], [np.sin(rotate_angle), np.cos(rotate_angle)]]
        )

        for t in np.arange(0., 20., dt):
            v = get_velocity(x, args.mode)

            obstacle = cap(get_velocity_to_avoid_obstacles(x, CYLINDER_POSITIONS, CYLINDER_RADII),
                           max_speed=MAX_SPEED)
            bug_v = cap(np.matmul(rotate_quarter, obstacle),
                    max_speed=MAX_SPEED)

            # check bug entry conditions
            if not do_bug and np.linalg.norm(v) < bug_threshold and np.linalg.norm(GOAL_POSITION-x)>0.2:
                do_bug = True
                bug_angle = np.arctan2(v[1], v[0])
            # check bug exit conditions
            if do_bug \
                    and np.linalg.norm(v) > bug_threshold \
                    and abs(angle_difference(np.arctan2(bug_v[1], bug_v[0]), bug_angle)) < np.pi / 24:
                do_bug = False

            # if in BUG mode, circumnavigate obstacles
            if do_bug:
                v=bug_v

            x = x + v * dt

            positions.append(x)
    else:
        for t in np.arange(0., 20., dt):
            v = get_velocity(x, args.mode)
            x = x + v * dt
            positions.append(x)
    positions = np.array(positions)
    plt.plot(positions[:, 0], positions[:, 1], lw=2, c='r')

    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([-.5 - WALL_OFFSET, WALL_OFFSET + .5])
    plt.ylim([-.5 - WALL_OFFSET, WALL_OFFSET + .5])
    plt.show()
