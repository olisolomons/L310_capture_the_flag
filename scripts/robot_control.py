import heapq

import numpy as np
import rospy
from gazebo_msgs.msg import ModelStates
from tf.transformations import euler_from_quaternion

import rrt
import potential_field
import slam
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path
from constants import *


def feedback_linearized(pose, velocity, epsilon):
    yaw = pose[YAW]
    u = velocity[0] * np.cos(yaw) + velocity[1] * np.sin(yaw)  # [m/s]
    w = (-velocity[0] * np.sin(yaw) + velocity[1] * np.cos(yaw)) / epsilon  # [rad/s] going counter-clockwise.
    return u, w


def get_velocity(position, path_points):
    print 'get_velocity', position
    v = np.zeros_like(position)
    if len(path_points) == 0:
        return v
    # Stop moving if the goal is reached.
    if np.linalg.norm(position - path_points[-1]) < .2:
        return v

    (i1, _), (i2, _) = heapq.nsmallest(2, enumerate(path_points), key=lambda (i, p): np.linalg.norm(p - position))

    p1, p2 = path_points[list(sorted((i1, i2)))]

    v_line_dir = p2 - p1
    v_correction = p2 - position
    v = 3 * v_line_dir + v_correction

    return v / np.linalg.norm(v) * SPEED


class CommandVelocity(object):
    def __init__(self, path):
        self.publisher = rospy.Publisher(path, Twist, queue_size=5)

    def set_velocity(self, u, w):
        vel_msg = Twist()
        vel_msg.linear.x = u
        vel_msg.angular.z = w
        self.publisher.publish(vel_msg)


class NavigationError(Exception):
    pass


class MovementPlanner(object):
    def navigate_towards(self, goal):
        # type: (Rrt, np.ndarray) -> bool
        return False


class Rrt(MovementPlanner):
    REPLAN_INTERVAL = 3

    def __init__(self, slam, command_velocity, ns_prefix):
        # type: (Rrt, slam.SLAM, CommandVelocity, str) -> None
        self.slam = slam
        self.command_velocity = command_velocity
        self.ns_prefix = ns_prefix

        self.previous_time = rospy.Time.now().to_sec()
        self.current_path = []
        self.current_goal = np.zeros(2)

        self.previous_pose = None
        self.previous_speed = 0
        self.reversing = 0

        self.path_publisher = rospy.Publisher('/path', Path, queue_size=1)
        self.frame_id = 0

    def navigate_towards(self, goal):
        # type: (Rrt, np.ndarray) -> bool

        current_time = rospy.Time.now().to_sec()
        if self.reversing > 1:
            self.command_velocity.set_velocity(-SPEED, 0)
            self.reversing -= 1

            return False
        elif self.reversing == 1:
            self.reversing -= 1

            self.calculate_path(goal)
            self.previous_time = current_time
            self.previous_pose = None
            self.previous_speed = -SPEED

        goal_reached = np.linalg.norm(self.slam.pose[:2] - goal) < .2
        if goal_reached:
            return True

        time_since = current_time - self.previous_time

        is_no_path = self.current_path is None or len(self.current_path) == 0
        is_path_old = not np.all(self.current_goal == goal)
        if is_no_path or is_path_old or time_since > self.REPLAN_INTERVAL:
            if self.previous_pose is not None and time_since > self.REPLAN_INTERVAL / 2:
                actual_speed = np.linalg.norm(self.slam.pose[:2] - self.previous_pose) / time_since
                if self.previous_speed > 0 and actual_speed < self.previous_speed / 2:
                    self.reversing -= 1
                else:
                    self.reversing = 0
                if self.reversing < -5:
                    self.reversing = 5

            self.previous_pose = self.slam.pose[:2].copy()

            self.calculate_path(goal)
            self.previous_time = current_time

        v = get_velocity(self.slam.pose[:2], np.array(self.current_path, dtype=np.float32))
        u, w = feedback_linearized(self.slam.pose, v, epsilon=EPSILON)

        self.previous_speed = u

        self.command_velocity.set_velocity(u, w)

        return False

    def calculate_path(self, goal):
        path_candidates = (
            (
                new_path,
                (
                    np.linalg.norm(new_path[1:] - new_path[:-1], axis=1).sum()
                    if len(new_path) > 0
                    else np.inf
                )
            ) for i in range(3)
            for start_node, final_node in (rrt.rrt(self.slam.pose.copy(), goal, self.slam.occupancy_grid),)
            for new_path in (np.array(rrt.get_path(final_node)),)
        )
        self.current_goal = goal
        new_path = min(path_candidates, key=lambda path: path[1])[0]
        if len(new_path) == 0:
            raise NavigationError('Unable to navigate to goal')
        self.current_path = new_path

        path_msg = Path()
        path_msg.header.seq = self.frame_id
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = '/%s/map' % self.ns_prefix
        for u in self.current_path:
            pose_msg = PoseStamped()
            pose_msg.header.seq = self.frame_id
            pose_msg.header.stamp = path_msg.header.stamp
            pose_msg.header.frame_id = path_msg.header.frame_id
            pose_msg.pose.position.x = u[X]
            pose_msg.pose.position.y = u[Y]
            path_msg.poses.append(pose_msg)
        self.path_publisher.publish(path_msg)

        self.frame_id += 1


class GroundTruthPose(object):
    def __init__(self, robot_id, name_fmt='turtlebot3_%d_%dburger'):
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.callback)
        self._pose = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
        self._name = name_fmt % robot_id

    def callback(self, msg):
        idx = [i for i, n in enumerate(msg.name) if n == self._name]
        if not idx:
            raise ValueError('Specified name "{}" does not exist.'.format(self._name))
        idx = idx[0]
        self._pose[X] = msg.pose[idx].position.x
        self._pose[Y] = msg.pose[idx].position.y
        _, _, yaw = euler_from_quaternion([
            msg.pose[idx].orientation.x,
            msg.pose[idx].orientation.y,
            msg.pose[idx].orientation.z,
            msg.pose[idx].orientation.w])
        self._pose[YAW] = yaw

    @property
    def ready(self):
        return not np.isnan(self._pose[0])

    @property
    def pose(self):
        return self._pose


class PotentialField(MovementPlanner):
    def __init__(self, command_velocity, ground_truth_pose, robot_team, robot_number):
        # type: (CommandVelocity, GroundTruthPose, int, int) -> None
        self.command_velocity = command_velocity
        self.ground_truth_pose = ground_truth_pose

        self.team_poses = [GroundTruthPose((robot_team, i)) for i in range(TEAM_SIZE) if i != robot_number]

    @property
    def ready(self):
        return all(pose_ground_truth.ready for pose_ground_truth in self.team_poses)

    def navigate_towards(self, goal):
        # type: (np.ndarray) -> bool
        pose = self.ground_truth_pose.pose

        goal_reached = np.linalg.norm(pose[:2] - goal) < .2
        if goal_reached:
            return True

        obstacles = list(OBSTACLE_POSITIONS) + [pose_gt.pose[:2] for pose_gt in self.team_poses]
        radii = [OBSTACLE_RADIUS] * len(OBSTACLE_POSITIONS) + [0.175] * (TEAM_SIZE - 1)

        v = potential_field.get_velocity(pose[:2], goal, obstacles, radii)

        u, w = feedback_linearized(pose, v, epsilon=EPSILON)

        self.command_velocity.set_velocity(u, w)

        return False
