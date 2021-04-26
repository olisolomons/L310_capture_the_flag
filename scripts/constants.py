import numpy as _np
SPEED = 0.2
EPSILON = 0.1
SENSOR_RANGE = 1.

X = 0
Y = 1
YAW = 2

WALL_OFFSET = 4.

OBSTACLE_POSITIONS = _np.array([
    (0, 0),
    (-2, 0), (2, 0),
    (-1, 2), (1, 2), (-1, -2), (1, -2),
    (3, 2), (-3, 2), (3, -2), (-3, -2)
])
OBSTACLE_RADIUS = 0.3

TEAM_SIZE = 5

