import cv2
import time
import numpy as np
from threading import Thread
from lane_client import LaneClient
from obj_client import ObjClient
from queue import Queue

laneClient = LaneClient()
objClient = ObjClient()

lane_queue = Queue()
obj_queue = Queue()

lane_queue_result = Queue()
obj_queue_result = Queue()



def sendrequest(model_name, q, q_result):
    if q.empty():
        return
    else:
        image = q.get()
        if model_name == "pinet_1block":
            laneClient.getPrediction(image)
            fits = laneClient.getFits()
            laneClient.getMask()
            return  fits
        
        
        return objClient.getPrediction(image)


class record():
    
    def __init__(self, width, height):

        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter('result.avi', self.fourcc, 30, (width,height))
    
    def write(self, image):
        self.out.write(image)
    
    def release(self):
        self.out.release()


if __name__ == '__main__':

    cam = cv2.VideoCapture("demo.avi")
    recor = record(512,256)

    while cam.isOpened():
        ret, frame = cam.read()
        if not ret:
            break

        # lane_queue.put(frame)
        # obj_queue.put(frame)

        start_point = time.time()


        # thread_1 = Thread(target=sendrequest, args=("pinet_1block",lane_queue,lane_queue_result))
        # thread_2 = Thread(target=sendrequest, args=("yolov4-tiny-3l",obj_queue, obj_queue_result))
        
        # thread_1.start()
        # thread_2.start()
        # thread_1.join()
        # thread_2.join()

        # forward
        xs, ys= laneClient.getPrediction(frame)
        fits = laneClient.getFits()
        laneClient.getMask()
        boxes, scores, classes = objClient.getPrediction(frame)
        
        ##################################
        image_output = laneClient.getImageResult()
        image_output = objClient.getImageResult(image_output)

        image_output = cv2.resize(image_output,(512,256))
        end_point = time.time()
        fps = 1/(end_point - start_point)
        cv2.putText(image_output, "FPS : "+ str(fps), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
        cv2.imshow("image_output", image_output)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        cv2.resize
        recor.write(image_output)