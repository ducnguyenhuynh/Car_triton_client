import numpy as np
import cv2
import math
from parameters import Parameters
p = Parameters()



def generate_result(confidence, offsets, instance, thresh):

    mask = confidence[0] > thresh

    grid = p.grid_location[mask]
    offset = offsets[mask]
    feature = instance[mask]

    lane_feature = []
    x = []
    y = []
    
    for i in range(len(grid)):
        if (np.sum(feature[i]**2))>=0:
            point_x = int((offset[i][0]+grid[i][0])*p.resize_ratio)
            point_y = int((offset[i][1]+grid[i][1])*p.resize_ratio)
            if point_x > p.x_size or point_x < 0 or point_y > p.y_size or point_y < 0 : 
                continue
            if len(lane_feature) == 0:
                lane_feature.append(feature[i])
                x.append([point_x])
                y.append([point_y])
            else:
                # flag = 0
                # index = 0
                min_feature_index = -1
                min_feature_dis = 10000
                for feature_idx, j in enumerate(lane_feature):
                    dis = np.linalg.norm((feature[i] - j)**2)
                    if min_feature_dis > dis:
                        min_feature_dis = dis
                        min_feature_index = feature_idx
                if min_feature_dis <= p.threshold_instance:
                    lane_feature[min_feature_index] = (lane_feature[min_feature_index]*len(x[min_feature_index]) + feature[i])/(len(x[min_feature_index])+1)
                    x[min_feature_index].append(point_x)
                    y[min_feature_index].append(point_y)
                elif len(lane_feature) < 20:
                    lane_feature.append(feature[i])
                    x.append([point_x])
                    y.append([point_y])
    
    return x, y


def eliminate_fewer_points(x, y):
    # eliminate fewer points
    out_x = []
    out_y = []
    for i, j in zip(x, y):
        if len(i) > 10:
            out_x.append(i)
            out_y.append(j)     
    return out_x, out_y  

def postprocess(trt_outputs, img_w, img_h, conf_th):
    confidences, offsets, instances, features = trt_outputs
    confidence = confidences[0]

    offset = np.rollaxis(offsets[0], axis=2, start=0)
    offset = np.rollaxis(offset, axis=2, start=0)

    instance = np.rollaxis(instances[0], axis=2, start=0)
    instance = np.rollaxis(instance, axis=2, start=0)
    x, y = generate_result(confidence, offset, instance, p.threshold_point)
        # print("-----------------------------------------")
    x, y = eliminate_fewer_points(x, y)
    
    
    return x, y