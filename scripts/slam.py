import threading
import traceback

import numpy as np
import cv2
import rospy
from nav_msgs.msg import OccupancyGrid
from tf import TransformListener

from tf.transformations import euler_from_quaternion

import rrt

X = 0
Y = 1
YAW = 2


class SLAM(object):
    def __init__(self, team, number):
        self.robot_id = (team, number)

        print('/robot%d_%d/map' % self.robot_id)
        rospy.Subscriber('/robot%d_%d/map' % self.robot_id, OccupancyGrid, self.callback)
        self._tf = TransformListener()
        self._occupancy_grid = None
        self._pose = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
        self.frames_exist = (False, False)

    def callback(self, msg):
        values = np.array(msg.data, dtype=np.int8).reshape((msg.info.width, msg.info.height))
        processed = np.empty_like(values)
        processed[:] = rrt.FREE
        processed[values < 0] = rrt.UNKNOWN
        dilated_occupied = cv2.dilate(np.uint8(values > 50), np.ones((5, 5), dtype=np.uint8)) == 1
        processed[dilated_occupied] = rrt.OCCUPIED
        processed = processed.T
        origin = [msg.info.origin.position.x, msg.info.origin.position.y, 0.]
        resolution = msg.info.resolution
        self._occupancy_grid = rrt.OccupancyGrid(processed, origin, resolution)

    def update(self):
        # type: () -> None
        # Get pose w.r.t. map.
        a = 'robot%d_%d/occupancy_grid' % self.robot_id
        b = 'robot%d_tf_%d/base_link' % self.robot_id
        a_exists, b_exists = self.frames_exist
        if a_exists and b_exists:
            try:
                t = rospy.Time(0)
                position, orientation = self._tf.lookupTransform(a, b, t)
                self._pose[X] = position[X]
                self._pose[Y] = position[Y]
                _, _, self._pose[YAW] = euler_from_quaternion(orientation)
            except Exception as e:
                traceback.print_exc()
                print(e)
                self.frames_exist = self._tf.frameExists(a), self._tf.frameExists(b)
        else:
            print('Unable to find: %s=%s, %s=%s' % (a, a_exists, b, b_exists))
            self.frames_exist = self._tf.frameExists(a), self._tf.frameExists(b)

    @property
    def ready(self):
        return self._occupancy_grid is not None and not np.isnan(self._pose[0])

    @property
    def pose(self):
        return self._pose.copy()

    @property
    def occupancy_grid(self):
        # type: () -> rrt.OccupancyGrid
        return self._occupancy_grid
