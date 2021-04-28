#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os

import rospy
import sys
import numpy as np

import robot_control
from exploring import Exploring
from constants import *
from potential_field import angle_difference

directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../python')
sys.path.insert(0, directory)


class CirclingBehaviour(object):
    def __init__(self, center, radius, pose, planner, command_velocity, team, number, team_size):
        # type: (np.ndarray, float, robot_control.GroundTruthPose, robot_control.PotentialField,robot_control.CommandVelocity, int, int, int) -> None
        self.center = center
        self.radius = radius
        self.pose_gt = pose
        self.planner = planner
        self.command_velocity = command_velocity
        self.team = team
        self.robot_number = number
        self.team_size = team_size

        self.all_poses = [pose] if number == 0 else [
            pose if i == number else robot_control.GroundTruthPose((team, i))
            for i in range(team_size)
        ]

    @property
    def ready(self):
        return all(p.ready for p in self.all_poses)

    def update(self):
        average_phase = sum(
            np.array([np.cos(relative_angle), np.sin(relative_angle)])
            for i, pose_gt in enumerate(self.all_poses)
            for to_center in (pose_gt.pose[:2] - self.center,)
            for target_angle in (np.pi * 2 / self.team_size * i,)
            for actual_angle in (np.arctan2(to_center[1], to_center[0]),)
            for relative_angle in (actual_angle - target_angle,)
        )
        average_angle = np.arctan2(average_phase[1], average_phase[0])
        angular_speed = SPEED / self.radius * 0.8
        target_offset = np.pi * 2 / self.team_size * self.robot_number
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
    [[0.3, 2 * np.pi / 3], [0.35, np.pi / 4], [0.3, -2 * np.pi / 3], [0.35, -np.pi / 4]],
    [[0.35, -np.pi / 4], [0.35, np.pi / 4], [0.425, 0], [0.7, np.pi * 0.6]]
])
control_gains = np.array([0.75, 0.75])


# control_gains = np.array([2, 8])


class Formation(object):
    def __init__(self, team, robot_number, pose_gt, command_velocity, formation_number):
        # type: (int, int, robot_control.GroundTruthPose, robot_control.CommandVelocity, int) -> None
        self.robot_number = robot_number
        if robot_number <= desired_formation.shape[1]:
            self.desired_relative_location = desired_formation[formation_number][robot_number - 1]
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

        relative_location_diff = self.desired_relative_location - relative_location
        relative_location_diff[RELATIVE_BEARING] = angle_difference(
            self.desired_relative_location[RELATIVE_BEARING],
            relative_location[RELATIVE_BEARING]
        )

        u, w = np.matmul(
            np.linalg.inv(g_matrix),
            control_gains * relative_location_diff - np.matmul(
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
        command_velocity, ground_truth_pose, args.team, args.number, args.team_size,
        avoid_team=not args.task.startswith('formation'),
        speed=0.125 if args.task.startswith('formation') else SPEED
    )

    exploring_behaviour = Exploring(ground_truth_pose, planner, args.number, args.team_size, rate_limiter)
    circling_behaviour = CirclingBehaviour(
        np.array([3, 1]), 0.225,
        ground_truth_pose, planner, command_velocity,
        args.team, args.number, args.team_size
    )
    formation_number = 0
    if args.task.startswith('formation-'):
        formation_name = args.task.split('-')[1]
        formation_number = ['x', 'decoy'].index(formation_name)
    formation_behaviour = Formation(args.team, args.number, ground_truth_pose, command_velocity, formation_number)

    explore_path_log = []

    def explore():
        if not hasattr(explore, 'done'):
            done = exploring_behaviour.update()
            explore_path_log.append(ground_truth_pose.pose[:2].tolist())
            if done:
                explore.done = True
                command_velocity.set_velocity(0, 0)
                print('%s done' % ns_prefix)

                path_file_path = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    'path_log/%d_%d.json' % (args.team_size, args.number)
                )
                with open(path_file_path, 'w') as path_file:
                    json.dump(explore_path_log, path_file)

    formation_function = formation_behaviour.update
    if args.number == 0:
        formation_function = lambda: formation_robot1_0_path(planner, ground_truth_pose)

    behaviour = {
        'explore': explore,
        'circle': circling_behaviour.update,
        'formation-x': formation_function,
        'formation-decoy': formation_function
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
    parser.add_argument('--team_size', help='Robot team size', type=int)
    parser.add_argument('--number', help='Robot number', type=int)
    parser.add_argument('--task', help='The task to perform', type=str)

    args, unknown = parser.parse_known_args()
    try:
        run(args)
    except rospy.ROSInterruptException:
        pass
