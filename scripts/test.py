#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
from collections import deque

import rospy
import sys
import numpy as np

import robot_control
import rrt
from slam import SLAM
from constants import *

directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../python')
sys.path.insert(0, directory)


class Exploring(object):
    def __init__(self, ground_truth_pose, planner, robot_number, rate_limiter):
        # type: (robot_control.GroundTruthPose, robot_control.MovementPlanner, int) -> None
        self.planner = ground_truth_pose
        self.rrt_controller = planner
        self.robot_number = robot_number
        self.rate_limiter = rate_limiter

        grid_cell_size = 2 * SENSOR_RANGE / np.sqrt(2)
        number_of_cells = int(np.ceil(WALL_OFFSET * 2 / grid_cell_size))
        grid_cell_size = WALL_OFFSET * 2 / number_of_cells
        self.visited_grid = np.zeros((number_of_cells, number_of_cells), dtype=np.bool)
        self.visited_grid_center_coords = np.stack(np.meshgrid(
            *(
                 np.linspace(
                     -WALL_OFFSET + grid_cell_size / 2,
                     WALL_OFFSET - grid_cell_size / 2, number_of_cells
                 ),
             ) * 2
        ), axis=2)

        waypoints_array = np.array(list(np.ndindex(*self.visited_grid.shape))).reshape(self.visited_grid.shape + (-1,))
        self.my_indices = self.get_range(self.visited_grid.shape)
        my_waypoints = waypoints_array[self.my_indices]
        self.choose_order(my_waypoints)
        my_waypoints[1::2, :] = my_waypoints[1::2, ::-1]

        self.waypoints = (tuple(index) for index in my_waypoints.reshape((-1, 2)))
        self.current_waypoint = next(self.waypoints)

        self.reachable_cache = {'queue': deque(maxlen=4), 'map': {}}

    def get_range(self, grid_shape):
        lower_number = 0
        upper_number = TEAM_SIZE

        dim_index = 0

        ranges = [[0, grid_shape[0]], [0, grid_shape[1]]]

        while upper_number - lower_number > 1:
            mid = (lower_number + upper_number) // 2
            if self.robot_number < mid:
                upper_number = mid
                ranges[dim_index][1] = sum(ranges[dim_index]) // 2
            else:
                lower_number = mid
                ranges[dim_index][0] = sum(ranges[dim_index]) // 2

            dim_index = (dim_index + 1) % len(grid_shape)

        return tuple(slice(*r) for r in ranges)

    def choose_order(self, waypoints):
        while not self.planner.ready:
            self.rate_limiter.sleep()

        pos = self.planner.pose[:2]
        grid_dists = self.visited_grid_center_coords - pos
        grid_dists = np.linalg.norm(grid_dists, axis=2)

        def grid_dist_at(x, y):
            return grid_dists[tuple(waypoints[x, y])]

        bottom = min(grid_dist_at(0, 0), grid_dist_at(0, -1))
        top = min(grid_dist_at(-1, 0), grid_dist_at(-1, -1))
        if top < bottom:
            waypoints[:] = waypoints[::-1]

        if grid_dist_at(0, -1) < grid_dist_at(0, 0):
            waypoints[:, :] = waypoints[:, ::-1]

    def update(self):
        done_waypoint = False
        waypoint = self.visited_grid_center_coords[self.current_waypoint]
        try:
            if not np.any(np.linalg.norm(waypoint - OBSTACLE_POSITIONS, axis=1) < OBSTACLE_RADIUS):
                done_waypoint = self.rrt_controller.navigate_towards(waypoint)
        except robot_control.NavigationError:
            if self.slam.occupancy_grid.is_occupied(waypoint):
                done_waypoint = True

        if done_waypoint:
            try:
                self.visited_grid[self.current_waypoint] = True
                self.current_waypoint = next(self.waypoints)
                print("next waypoint:", self.current_waypoint)
            except StopIteration:
                return True

        else:
            return False

    def flood_occupied(self, start_point, radius=6):
        start_point_t = tuple(start_point)
        cache_queue = self.reachable_cache['queue']
        cache_map = self.reachable_cache['map']

        if start_point_t in cache_map:
            result, attempts = cache_map[start_point_t]
            if attempts < 10:
                cache_map[start_point_t] = (result, attempts + 1)
                return result
            else:
                del cache_map[start_point_t]
                cache_queue.remove(start_point_t)

        result = self.do_flood_occupied(start_point, radius)
        if len(cache_queue) == cache_queue.maxlen:
            old_point = cache_queue.popleft()
            del cache_map[old_point]

        cache_queue.append(start_point_t)
        cache_map[start_point_t] = (result, 0)

        return result

    def do_flood_occupied(self, start_point, radius=6):
        # type: (np.ndarray, int) -> bool
        print('do_flood_occupied')
        start_point = np.array(self.slam.occupancy_grid.get_index(start_point))

        to_flood = deque()
        to_flood.append(tuple(start_point))

        flooded = set()

        while to_flood:
            p = np.array(to_flood.popleft())
            if np.linalg.norm(p - start_point) > radius:
                return False

            for angle in np.arange(4) * np.pi / 2:
                direction_vector = np.array([np.cos(angle), np.sin(angle)])
                new_point = tuple(np.int32(np.round(p + direction_vector)))
                free = self.slam.occupancy_grid.values[new_point] != rrt.OCCUPIED
                if new_point not in flooded and free:
                    to_flood.append(new_point)
                    flooded.add(new_point)

        return True


def run(args):
    rospy.init_node('rrt_navigation')

    ns_prefix = 'robot%d_%d' % (args.team, args.number)

    # Update control every 100 ms.
    rate_limiter = rospy.Rate(100)
    command_velocity = robot_control.CommandVelocity('/%s/cmd_vel' % ns_prefix)
    # slam = SLAM(args.team, args.number)
    ground_truth_pose = robot_control.GroundTruthPose((args.team, args.number))

    planner = robot_control.PotentialField(command_velocity, ground_truth_pose, args.team, args.number)
    exploring_behaviour = Exploring(ground_truth_pose, planner, args.number, rate_limiter)

    while not rospy.is_shutdown():
        # slam.update()

        # if not (slam.ready and ground_truth_pose.ready):
        if not (ground_truth_pose.ready and planner.ready):
            rate_limiter.sleep()
            continue

        if exploring_behaviour is not None:
            done = exploring_behaviour.update()
            if done:
                exploring_behaviour = None
                command_velocity.set_velocity(0, 0)
                print('%s done' % ns_prefix)

        rate_limiter.sleep()


if __name__ == '__main__':
    np.seterr(all='raise')
    parser = argparse.ArgumentParser(description='Runs RRT navigation')
    parser.add_argument('--team', help='Robot team', type=int)
    parser.add_argument('--number', help='Robot number', type=int)

    args, unknown = parser.parse_known_args()
    try:
        run(args)
    except rospy.ROSInterruptException:
        pass
