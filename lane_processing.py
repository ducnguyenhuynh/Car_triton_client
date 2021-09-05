import numpy as np
import cv2
import math
from parameters import Parameters
p = Parameters()

def warp_image(img):
    image_size = (img.shape[1], img.shape[0])
    warped_img = cv2.warpPerspective(img, p.perspective_transform, image_size, flags=cv2.INTER_LINEAR)
    return warped_img

def preprocess(image):
    image = cv2.resize(image,(512,256))
    warped = warp_image(image)
    image_processed = np.rollaxis(warped, axis=2, start=0)/255.0
    return image_processed

