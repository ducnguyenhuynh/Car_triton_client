import cv2
import time
import numpy as np
from numpy.testing._private.utils import print_assert_equal
from lane_client import LaneClient
from obj_client import ObjClient


if __name__ == '__main__':
    laneClient = LaneClient()
    objClient = ObjClient()


    image = cv2.imread("/home/duc/Desktop/triton_client/data-tfs-00000883.jpg")
    
    start_point = time.time()
    xs, ys= laneClient.getPrediction(image)
    fits = laneClient.getFits()
    boxes, scores, classes = objClient.getPrediction(image)
    image_output = laneClient.getImageResult()
    image_output = objClient.getImageResult(image_output)
    end_point = time.time()

    fps = 1/(end_point - start_point)
    print("fps: ", fps)
    cv2.imshow("image_output", image_output)
    cv2.waitKey(0)