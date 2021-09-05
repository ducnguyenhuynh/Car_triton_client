from __future__ import print_function

import ctypes

import numpy as np
import cv2



def _preprocess_yolo(img, input_shape):
    """Preprocess an image before TRT YOLO inferencing.

    # Args
        img: int8 numpy array of shape (img_h, img_w, 3)
        input_shape: a tuple of (H, W)

    # Returns
        preprocessed img: float32 numpy array of shape (3, H, W)
    """
    img = cv2.resize(img, (input_shape[1], input_shape[0]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1)).astype(np.float32)
    img /= 255.0
    return img


def _nms_boxes(detections, nms_threshold):
    """Apply the Non-Maximum Suppression (NMS) algorithm on the bounding
    boxes with their confidence scores and return an array with the
    indexes of the bounding boxes we want to keep.

    # Args
        detections: Nx7 numpy arrays of
                    [[x, y, w, h, box_confidence, class_id, class_prob],
                     ......]
    """
    x_coord = detections[:, 0]
    y_coord = detections[:, 1]
    width = detections[:, 2]
    height = detections[:, 3]
    box_confidences = detections[:, 4] * detections[:, 6]
    
    areas = width * height
    ordered = box_confidences.argsort()[::-1]

    keep = list()
    while ordered.size > 0:
            i = ordered[0]
            keep.append(i)
            xx1 = np.maximum(x_coord[i], x_coord[ordered[1:]])
            yy1 = np.maximum(y_coord[i], y_coord[ordered[1:]])
            xx2 = np.minimum(x_coord[i] + width[i], x_coord[ordered[1:]] + width[ordered[1:]])
            yy2 = np.minimum(y_coord[i] + height[i], y_coord[ordered[1:]] + height[ordered[1:]])

            width1 = np.maximum(0.0, xx2 - xx1 + 1)
            height1 = np.maximum(0.0, yy2 - yy1 + 1)
            intersection = width1 * height1
            union = (areas[i] + areas[ordered[1:]] - intersection)

            iou = intersection / union

            indexes = np.where(iou <= nms_threshold)[0]
            ordered = ordered[indexes + 1]
    keep = np.array(keep).astype(int)
    return keep
    


def _postprocess_yolo(trt_outputs, img_w, img_h, conf_th, nms_threshold=0.3):
    """Postprocess TensorRT outputs.

    # Args
        trt_outputs: a list of 2 or 3 tensors, where each tensor
                    contains a multiple of 7 float32 numbers in
                    the order of [x, y, w, h, box_confidence, class_id, class_prob]
        conf_th: confidence threshold

    # Returns
        boxes, scores, classes (after NMS)
    """
    
    detections = np.concatenate(
        [o.reshape(-1, 7) for o in trt_outputs], axis=0)

    # drop detections with score lower than conf_th
    box_scores = detections[:, 4] * detections[:, 6]
    pos = np.where(box_scores >= conf_th)
    detections = detections[pos]

    # scale x, y, w, h from [0, 1] to pixel values
    index = detections[:, :4] > 1
    index += detections[:, :4] < 0
    index = [i for i in range(len(index)) if any(index[i])]
    
    detections = np.delete(detections, index, 0)
    detections[:, 0] *= img_w
    detections[:, 1] *= img_h
    detections[:, 2] *= img_w
    detections[:, 3] *= img_h

    # NMS
    nms_detections = np.zeros((0, 7), dtype=detections.dtype)

    for class_id in set(detections[:, 5]):
        idxs = np.where(detections[:, 5] == class_id)
        cls_detections = detections[idxs]

        keep = _nms_boxes(cls_detections, nms_threshold)
        nms_detections = np.concatenate(
            [nms_detections, cls_detections[keep]], axis=0)
    if len(nms_detections) == 0:
        boxes = np.zeros((0, 4), dtype=np.int)
        scores = np.zeros((0, 1), dtype=np.float32)
        classes = np.zeros((0, 1), dtype=np.float32)
        return boxes, scores, classes
    else:
        xx = nms_detections[:, 0].reshape(-1, 1)
        yy = nms_detections[:, 1].reshape(-1, 1)
        ww = nms_detections[:, 2].reshape(-1, 1)
        hh = nms_detections[:, 3].reshape(-1, 1)
        
        order = np.logical_not(ww[:,0] <= 1)

        boxes = np.concatenate([xx, yy, xx+ww, yy+hh], axis=1) + 0.5
        boxes = boxes.astype(np.int)
        
        scores = nms_detections[:, 4] * nms_detections[:, 6]
        classes = nms_detections[:, 5]
        return boxes[order], scores[order], classes[order]

