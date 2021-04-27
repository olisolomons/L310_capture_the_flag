#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
from collections import deque
import json

import rospy
import sys
import numpy as np
import std_msgs.msg

import robot_control
import rrt
from slam import SLAM
from constants import *
from potential_field import angle_difference

directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../python')
sys.path.insert(0, directory)


class Exploring(object):
    def __init__(self, ground_truth_pose, planner, robot_number, rate_limiter):
        # type: (robot_control.GroundTruthPose, robot_control.PotentialField, int, rospy.Rate) -> None
        self.ground_truth_pose = ground_truth_pose
        self.planner = planner
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

        self.waypoints = [tuple(index) for index in my_waypoints.reshape((-1, 2))]
        self.current_waypoint = 0

        self.area_requests = rospy.Publisher('/area_requests', std_msgs.msg.String, queue_size=3)
        rospy.Subscriber('/area_requests', std_msgs.msg.String, self.on_area_request)

        self.collected_adverts = []
        self.awaiting_adverts = None

    def on_area_request(self, msg):
        msg = json.loads(msg.data)
        if msg['type'] == 'area_request':
            if self.current_waypoint < len(self.waypoints) - 1:
                closest = min(
                    np.linalg.norm(np.array(msg['position']) - point)
                    for wp in self.waypoints[self.current_waypoint + 1:]
                    for point in (self.visited_grid_center_coords[wp],)
                )
                advert = {'type': 'area_advert', 'sender': self.robot_number, 'closest': closest}
                self.area_requests.publish(json.dumps(advert))
        elif msg['type'] == 'area_advert':
            if not (self.current_waypoint < len(self.waypoints)):
                self.collected_adverts.append(msg)
        elif msg['type'] == 'specific_request':
            if msg['recipient'] == self.robot_number:
                remaining = len(self.waypoints) - self.current_waypoint
                if remaining > 0:
                    mid = (len(self.waypoints) + self.current_waypoint + 1) // 2
                    mine, yours = self.waypoints[:mid], self.waypoints[mid:]
                    self.waypoints = mine

                    reply = {'type': 'grant', 'recipient': msg['sender'], 'waypoints': yours}
                    self.area_requests.publish(json.dumps(reply))
        elif msg['type'] == 'grant':
            if msg['recipient'] == self.robot_number:
                self.waypoints = [tuple(waypoint) for waypoint in msg['waypoints']]
                self.current_waypoint = 0

                self.awaiting_adverts = None
                self.collected_adverts = []

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

        pos = self.ground_truth_pose.pose[:2]
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

        waypoint_index = 0
        if self.current_waypoint < len(self.waypoints):
            waypoint_index = self.waypoints[self.current_waypoint]
            waypoint = self.visited_grid_center_coords[waypoint_index]

            if not np.any(np.linalg.norm(waypoint - OBSTACLE_POSITIONS, axis=1) < OBSTACLE_RADIUS):
                done_waypoint = self.planner.navigate_towards(waypoint)
        else:
            done_waypoint = True

        if done_waypoint:
            if self.current_waypoint < len(self.waypoints):
                self.visited_grid[waypoint_index] = True
                self.current_waypoint += 1
                print("next waypoint:", waypoint_index)
            else:
                if self.awaiting_adverts is None:
                    self.planner.command_velocity.set_velocity(0, 0)

                    self.send_area_request()
                elif rospy.Time.now().to_sec() - self.awaiting_adverts > 0.5:
                    if self.collected_adverts:
                        best = min(self.collected_adverts, key=lambda advert: advert['closest'])
                        msg = {'type': 'specific_request', 'recipient': best['sender'], 'sender': self.robot_number}
                        self.area_requests.publish(json.dumps(msg))

                        self.send_area_request()
                    else:
                        return True
        else:
            return False

    def send_area_request(self):
        self.awaiting_adverts = rospy.Time.now().to_sec()
        request = {'type': 'area_request', 'position': self.ground_truth_pose.pose[:2].tolist()}
        self.area_requests.publish(json.dumps(request))
        self.collected_adverts = []


class CirclingBehaviour(object):
    def __init__(self, center, radius, pose, planner, command_velocity, team, number):
        # type: (np.ndarray, float, robot_control.GroundTruthPose, robot_control.PotentialField,robot_control.CommandVelocity, int, int) -> None
        self.center = center
        self.radius = radius
        self.pose_gt = pose
        self.planner = planner
        self.command_velocity = command_velocity
        self.team = team
        self.robot_number = number

        self.all_poses = [pose] if number == 0 else [
            pose if i == number else robot_control.GroundTruthPose((team, i))
            for i in range(TEAM_SIZE)
        ]

    @property
    def ready(self):
        return all(p.ready for p in self.all_poses)

    def update(self):
        average_phase = sum(
            np.array([np.cos(relative_angle), np.sin(relative_angle)])
            for i, pose_gt in enumerate(self.all_poses)
            for to_center in (pose_gt.pose[:2] - self.center,)
            for target_angle in (np.pi * 2 / TEAM_SIZE * i,)
            for actual_angle in (np.arctan2(to_center[1], to_center[0]),)
            for relative_angle in (actual_angle - target_angle,)
        )
        average_angle = np.arctan2(average_phase[1], average_phase[0])
        angular_speed = SPEED / self.radius * 0.8
        target_offset = np.pi * 2 / TEAM_SIZE * self.robot_number
        my_next_angle = average_angle + target_offset + angular_speed
        desired_position = np.array([np.cos(my_next_angle), np.sin(my_next_angle)])
        desired_position = desired_position * self.radius + self.center

        to_goal = desired_position - self.pose_gt.pose[:2]
        if np.linalg.norm(to_goal) > 0.35:
            self.planner.navigate_towards(desired_position)
        else:
            u, w = robot_control.feedback_linearized(self.pose_gt.pose, to_goal, EPSILON)
            self.command_velocity.set_velocity(max(u, 0), w)


RELATIVE_SEPARATION = 0
RELATIVE_BEARING = 1
desired_formation = np.array([
    [0.3, 2 * np.pi / 3], [0.35, np.pi / 3], [0.3, -2 * np.pi / 3], [0.35, -np.pi / 3]
])
# control_gains = np.array([0.75, 0.75])
control_gains = np.array([2, 8])


class Formation(object):
    def __init__(self, team, robot_number, pose_gt, command_velocity):
        # type: (int, int, robot_control.GroundTruthPose, robot_control.CommandVelocity) -> None
        self.robot_number = robot_number
        self.desired_relative_location = desired_formation[robot_number - 1]
        self.pose_gt = pose_gt
        self.command_velocity = command_velocity

        self.leader_gt = robot_control.GroundTruthPose((team, 0))

    @property
    def ready(self):
        return self.leader_gt.ready

    def update(self):
        pose = self.pose_gt.pose
        holonomic_point = pose[:2] + EPSILON * np.array([np.cos(pose[YAW]), np.sin(pose[YAW])])
        leader_pose = self.leader_gt.pose
        relative_location = self.relative_location(leader_pose, holonomic_point)

        relative_orientation = angle_difference(leader_pose[YAW], pose[YAW])
        gamma = angle_difference(relative_orientation + relative_location[RELATIVE_BEARING], 0)

        separation = relative_location[RELATIVE_SEPARATION]
        g_matrix = np.array([
            [np.cos(gamma), EPSILON * np.sin(gamma)],
            [-np.sin(gamma) / separation, EPSILON * np.cos(gamma) / separation]
        ])

        f_matrix = np.array([
            [-np.cos(relative_location[RELATIVE_BEARING]), 0],
            [np.sin(relative_location[RELATIVE_BEARING]) / separation, -1]
        ])

        leader_twist = self.leader_gt.twist

        u, w = np.matmul(
            np.linalg.inv(g_matrix),
            control_gains * (self.desired_relative_location - relative_location) - np.matmul(
                f_matrix, leader_twist
            )
        )

        self.command_velocity.set_velocity(
            np.clip(u, -SPEED, SPEED * 1.5),
            np.clip(w, -np.pi / 2, np.pi / 2)
        )

    @staticmethod
    def relative_location(leader_pose, pose):
        vector_leader_to_this = pose[:2] - leader_pose[:2]
        separation = np.linalg.norm(vector_leader_to_this)

        vector_leader_to_this_angle = np.arctan2(vector_leader_to_this[1], vector_leader_to_this[0])

        bearing = angle_difference(vector_leader_to_this_angle, leader_pose[YAW])

        return np.array([separation, bearing])


def formation_robot1_0_path(planner, pose_gt):
    # type: (robot_control.PotentialField, robot_control.GroundTruthPose) -> None
    if not hasattr(formation_robot1_0_path, 'at_goal'):
        formation_robot1_0_path.at_goal = False

    if not formation_robot1_0_path.at_goal and pose_gt.pose[0] < 3:
        formation_robot1_0_path.at_goal = planner.navigate_towards(np.array([3, -1]))
    else:
        planner.navigate_towards(np.array([3, 1]))


def run(args):
    rospy.init_node('rrt_navigation')

    ns_prefix = 'robot%d_%d' % (args.team, args.number)

    # Update control every 100 ms.
    rate_limiter = rospy.Rate(100)
    command_velocity = robot_control.CommandVelocity('/%s/cmd_vel' % ns_prefix)
    # slam = SLAM(args.team, args.number)
    ground_truth_pose = robot_control.GroundTruthPose((args.team, args.number))

    planner = robot_control.PotentialField(
        command_velocity, ground_truth_pose, args.team, args.number,
        avoid_team=args.task != 'formation',
        speed=0.125 if args.task == 'formation' else SPEED
    )

    exploring_behaviour = [Exploring(ground_truth_pose, planner, args.number, rate_limiter)]
    circling_behaviour = CirclingBehaviour(
        np.array([3, 1]), 0.225,
        ground_truth_pose, planner, command_velocity,
        args.team, args.number
    )
    formation_behaviour = Formation(args.team, args.number, ground_truth_pose, command_velocity)

    def explore():
        if exploring_behaviour[0] is not None:
            done = exploring_behaviour[0].update()
            if done:
                exploring_behaviour[0] = None
                command_velocity.set_velocity(0, 0)
                print('%s done' % ns_prefix)

    behaviour = {
        'explore': explore,
        'circle': circling_behaviour.update,
        'formation': (lambda: formation_robot1_0_path(planner,
                                                      ground_truth_pose)) if args.number == 0 else formation_behaviour.update
        # 'formation': explore if args.number == 0 else formation_behaviour.update
    }[args.task]

    while not rospy.is_shutdown():
        # slam.update()

        # if not (slam.ready and ground_truth_pose.ready):
        if not (ground_truth_pose.ready and planner.ready and formation_behaviour.ready):
            rate_limiter.sleep()
            continue

        behaviour()

        rate_limiter.sleep()


if __name__ == '__main__':
    np.seterr(all='raise')
    parser = argparse.ArgumentParser(description='Runs RRT navigation')
    parser.add_argument('--team', help='Robot team', type=int)
    parser.add_argument('--number', help='Robot number', type=int)
    parser.add_argument('--task', help='The task to perform', type=str)

    args, unknown = parser.parse_known_args()
    try:
        run(args)
    except rospy.ROSInterruptException:
        pass
