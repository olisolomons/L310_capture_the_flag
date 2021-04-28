import json

import numpy as np
import rospy
import std_msgs.msg

import robot_control

from constants import *


class Exploring(object):
    def __init__(self, ground_truth_pose, planner, robot_number, team_size, rate_limiter):
        # type: (robot_control.GroundTruthPose, robot_control.PotentialField, int, int, rospy.Rate) -> None
        self.ground_truth_pose = ground_truth_pose
        self.planner = planner
        self.robot_number = robot_number
        self.team_size = team_size
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

    def get_range(self, grid_shape):
        lower_number = 0
        upper_number = self.team_size

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

    def on_area_request(self, msg):
        msg = json.loads(msg.data)
        if msg['type'] == 'area_request':
            if self.current_waypoint < len(self.waypoints) - 1:
                work_achievable = self.calculate_work_achievable(msg)
                if work_achievable > 0:
                    advert = {'type': 'area_advert', 'sender': self.robot_number, 'work_achievable': work_achievable}
                    self.area_requests.publish(json.dumps(advert))
        elif msg['type'] == 'area_advert':
            if not (self.current_waypoint < len(self.waypoints)):
                self.collected_adverts.append(msg)
        elif msg['type'] == 'specific_request':
            if msg['recipient'] == self.robot_number:
                remaining = len(self.waypoints) - self.current_waypoint

                if remaining > 1:
                    work_achievable = self.calculate_work_achievable(msg)
                    waypoints_achievable = 0
                    work_for_waypoints = 0
                    while work_for_waypoints < work_achievable / 2 and waypoints_achievable < len(self.waypoints) - 1:
                        end = len(self.waypoints) - waypoints_achievable
                        ind1, ind2 = self.waypoints[-waypoints_achievable - 2:end]
                        p1 = self.visited_grid_center_coords[ind1]
                        p2 = self.visited_grid_center_coords[ind2]
                        work_for_waypoints += np.linalg.norm(p1 - p2)

                        waypoints_achievable += 1

                    mid = len(self.waypoints) - waypoints_achievable - 1
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

    def calculate_work_achievable(self, msg):
        closest = min(
            np.linalg.norm(np.array(msg['position']) - point)
            for wp in self.waypoints[self.current_waypoint + 1:]
            for point in (self.visited_grid_center_coords[wp],)
        )
        remaining_distance = sum(
            np.linalg.norm(p2 - p1)
            for i in range(self.current_waypoint, len(self.waypoints) - 1)
            for ind1, ind2 in ([tuple(self.waypoints[i]), tuple(self.waypoints[i + 1])],)
            for p1 in (self.visited_grid_center_coords[ind1],)
            for p2 in (self.visited_grid_center_coords[ind2],)
        )
        work_achievable = remaining_distance - closest
        return work_achievable

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
                        best = max(self.collected_adverts, key=lambda advert: advert['work_achievable'])
                        request = {
                            'type': 'specific_request', 'recipient': best['sender'],
                            'sender': self.robot_number,
                            'position': self.ground_truth_pose.pose[:2].tolist()
                        }
                        self.area_requests.publish(json.dumps(request))

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
