#############################################################################################################
##
##  Parameters
##
#############################################################################################################
import numpy as np
import cv2

from enum import Enum

class Parameters():
    
    # for line & lane
    x_size = 512
    y_size = 256
    resize_ratio = 8
    grid_x = x_size//resize_ratio  #64
    grid_y = y_size//resize_ratio  #32

    threshold_point = 0.65 #0.88 #0.93 #0.95 #0.93
    threshold_instance = 0.15

    # test parameter
    color = [(0,0,0), (255,0,0), (0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(255,255,255),(100,255,0),(100,0,255),(255,100,0),(0,100,255),(255,0,100),(0,255,100)]
    grid_location = np.zeros((grid_y, grid_x, 2))
    for y in range(grid_y):
        for x in range(grid_x):
            grid_location[y][x][0] = x
            grid_location[y][x][1] = y
            
    num_iter = 45
    threshold_RANSAC = 0.1
    ratio_inliers = 0.1

    # expand

    point_in_lane = 0
    source_points = np.float32([
    [0, y_size],
    [0, (5/9)*y_size],
    [x_size, (5/9)*y_size],
    [x_size, y_size]
    ])
    
    destination_points = np.float32([
    [0 * x_size, y_size],
    [0 * x_size, 0],
    [x_size - (0 * x), 0],
    [x_size - (0 * x), y_size]
    ])
    
    perspective_transform = cv2.getPerspectiveTransform(source_points, destination_points)
    inverse_perspective_transform = cv2.getPerspectiveTransform( destination_points, source_points)


class VehicleLabels(Enum):
    car=0
    truck=1
    bus=2
    motorbike=3
    other=4
    bicycle=5

class Objects(Enum):
    car=0
    w65=1
    i10=2
    i12=3
    i13=4
    pne=5
    p19=6
    p23=7
    i5=8
    i6=9
    stop=10