#!/usr/bin/env python

import argparse
import numpy as np
import sys
import cv2
import time

import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

from obj_with_plugins import _preprocess_yolo, _postprocess_yolo
from render import render_box, render_filled_box, get_text_size, render_text, RAND_COLORS
from parameters import Objects

class ObjClient(object):
    def __init__(self, model_name = "yolov4-tiny-3l", url="0.0.0.0:8001", confidence=0.5, nms = 0.5):
        self.model_name = model_name
        self.url = url
        self.confidence = confidence
        self.nms = nms

        # Create server context
        try:
            self.triton_client = grpcclient.InferenceServerClient(
                url=url,
                verbose=False,
                ssl=False,
                root_certificates=None,
                private_key=None,
                certificate_chain=None)
        except Exception as e:
            print("context creation failed: " + str(e))
            sys.exit()
        
        # Health check
        if not self.triton_client.is_server_live():
            print("FAILED : is_server_live")
            sys.exit(1)

        if not self.triton_client.is_server_ready():
            print("FAILED : is_server_ready")
            sys.exit(1)
        
        if not self.triton_client.is_model_ready(self.model_name):
            print("FAILED : is_model_ready")
            sys.exit(1)


        self.inputs = []
        self.outputs = []
        self.inputs.append(grpcclient.InferInput('input', [1, 3, 224, 224], "FP32"))
        self.outputs.append(grpcclient.InferRequestedOutput('detections'))

    def getPrediction(self, input_image):
        
        self.image = input_image
        input_image_buffer = _preprocess_yolo(input_image, (224,224,3))
        input_image_buffer = np.expand_dims(input_image_buffer, axis=0)
        self.inputs[0].set_data_from_numpy(input_image_buffer)


        results = self.triton_client.infer(model_name=self.model_name,
                                    inputs=self.inputs,
                                    outputs=self.outputs,
                                    client_timeout=None)

        result_boxes = results.as_numpy('detections')

        self.boxes, self.scores, self.classes = _postprocess_yolo([result_boxes], input_image.shape[1], input_image.shape[0] , self.confidence, self.nms)

        self.classes = [Objects(int(classID)).name for classID in self.classes]
        return self.boxes, self.scores, self.classes
    
    def getImageResult(self, image = None):
        '''
        :param box: (x1, y1, x2, y2) - box coordinates
        '''
        if image is None:
            image = self.image
        for i, det in enumerate(self.boxes):
            ratio_x = image.shape[0] / self.image.shape[0]
            ratio_y = image.shape[1] / self.image.shape[1]      

            box = det[0]*ratio_y, det[1]*ratio_x, det[2]*ratio_y, det[3]*ratio_x
            
            confidence = self.scores[i]
            
            # print(f"{self.classes[i]}: {confidence}")
            
            input_image = render_box(image, box, color=tuple(RAND_COLORS[i % 64].tolist()))
            size = get_text_size(input_image, f"{self.classes[i]}: {confidence:.2f}", normalised_scaling=0.6)
            result_image = render_filled_box(input_image, (box[0] - 3, box[1] - 3, box[0] + size[0], box[1] + size[1]), color=(220, 220, 220))
            image = render_text(result_image, f"{self.classes[i]}: {confidence:.2f}", (box[0], box[1]), color=(30, 30, 30), normalised_scaling=0.5)
            
            
        return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',
                        choices=['dummy', 'image', 'video'],
                        default='dummy',
                        help='Run mode. \'dummy\' will send an emtpy buffer to the server to test if inference works. \'image\' will process an image. \'video\' will process a video.')
    parser.add_argument('input',
                        type=str,
                        nargs='?',
                        help='Input file to load from in image or video mode')
    FLAGS = parser.parse_args()
    client = ObjClient()
    image = cv2.imread(str(FLAGS.input))
    boxes, scores, classes = client.getPrediction(image)
    image_output = client.getImageResult()
    cv2.imshow("image result", image_output)
    cv2.waitKey(0)

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('mode',
#                         choices=['dummy', 'image', 'video'],
#                         default='dummy',
#                         help='Run mode. \'dummy\' will send an emtpy buffer to the server to test if inference works. \'image\' will process an image. \'video\' will process a video.')
#     parser.add_argument('input',
#                         type=str,
#                         nargs='?',
#                         help='Input file to load from in image or video mode')
#     parser.add_argument('-m',
#                         '--model',
#                         type=str,
#                         required=False,
#                         default='yolov4-tiny-3l',
#                         help='Inference model name, default yolov4')
#     parser.add_argument('-u',
#                         '--url',
#                         type=str,
#                         required=False,
#                         default='0.0.0.0:8001',
#                         help='Inference server URL, default localhost:8001')
#     parser.add_argument('-o',
#                         '--out',
#                         type=str,
#                         required=False,
#                         default='',
#                         help='Write output into file instead of displaying it')
#     parser.add_argument('-c',
#                         '--confidence',
#                         type=float,
#                         required=False,
#                         default=0.4,
#                         help='Confidence threshold for detected objects, default 0.8')
#     parser.add_argument('-n',
#                         '--nms',
#                         type=float,
#                         required=False,
#                         default=0.2,
#                         help='Non-maximum suppression threshold for filtering raw boxes, default 0.5')
#     parser.add_argument('-f',
#                         '--fps',
#                         type=float,
#                         required=False,
#                         default=24.0,
#                         help='Video output fps, default 24.0 FPS')
#     parser.add_argument('-i',
#                         '--model-info',
#                         action="store_true",
#                         required=False,
#                         default=False,
#                         help='Print model status, configuration and statistics')
#     parser.add_argument('-v',
#                         '--verbose',
#                         action="store_true",
#                         required=False,
#                         default=False,
#                         help='Enable verbose client output')
#     parser.add_argument('-t',
#                         '--client-timeout',
#                         type=float,
#                         required=False,
#                         default=None,
#                         help='Client timeout in seconds, default no timeout')
#     parser.add_argument('-s',
#                         '--ssl',
#                         action="store_true",
#                         required=False,
#                         default=False,
#                         help='Enable SSL encrypted channel to the server')
#     parser.add_argument('-r',
#                         '--root-certificates',
#                         type=str,
#                         required=False,
#                         default=None,
#                         help='File holding PEM-encoded root certificates, default none')
#     parser.add_argument('-p',
#                         '--private-key',
#                         type=str,
#                         required=False,
#                         default=None,
#                         help='File holding PEM-encoded private key, default is none')
#     parser.add_argument('-x',
#                         '--certificate-chain',
#                         type=str,
#                         required=False,
#                         default=None,
#                         help='File holding PEM-encoded certicate chain default is none')
#     parser.add_argument('-b',
#                         '--batch',
#                         type=int,
#                         required=False,
#                         default=None,
#                         help='batch size')
#     FLAGS = parser.parse_args()

#     # Create server context
#     try:
#         triton_client = grpcclient.InferenceServerClient(
#             url=FLAGS.url,
#             verbose=FLAGS.verbose,
#             ssl=FLAGS.ssl,
#             root_certificates=FLAGS.root_certificates,
#             private_key=FLAGS.private_key,
#             certificate_chain=FLAGS.certificate_chain)
#     except Exception as e:
#         print("context creation failed: " + str(e))
#         sys.exit()

#     # Health check
#     if not triton_client.is_server_live():
#         print("FAILED : is_server_live")
#         sys.exit(1)

#     if not triton_client.is_server_ready():
#         print("FAILED : is_server_ready")
#         sys.exit(1)
    
#     if not triton_client.is_model_ready(FLAGS.model):
#         print("FAILED : is_model_ready")
#         sys.exit(1)

#     if FLAGS.model_info:
#         # Model metadata
#         try:
#             metadata = triton_client.get_model_metadata(FLAGS.model)
#             print(metadata)
#         except InferenceServerException as ex:
#             if "Request for unknown model" not in ex.message():
#                 print("FAILED : get_model_metadata")
#                 print("Got: {}".format(ex.message()))
#                 sys.exit(1)
#             else:
#                 print("FAILED : get_model_metadata")
#                 sys.exit(1)

#         # Model configuration
#         try:
#             config = triton_client.get_model_config(FLAGS.model)
#             if not (config.config.name == FLAGS.model):
#                 print("FAILED: get_model_config")
#                 sys.exit(1)
#             print(config)
#         except InferenceServerException as ex:
#             print("FAILED : get_model_config")
#             print("Got: {}".format(ex.message()))
#             sys.exit(1)

   
#     if FLAGS.mode == 'image':
#         print("Running in 'image' mode")
#         if not FLAGS.input:
#             print("FAILED: no input image")
#             sys.exit(1)
        
#         inputs = []
#         outputs = []
#         inputs.append(grpcclient.InferInput('input', [1, 3, 224, 224], "FP32"))
#         outputs.append(grpcclient.InferRequestedOutput('detections'))

#         print("Creating buffer from image file...")
#         input_image = cv2.imread(str(FLAGS.input))
#         if input_image is None:
#             print(f"FAILED: could not load input image {str(FLAGS.input)}")
#             sys.exit(1)

#         input_image_buffer = _preprocess_yolo(input_image, (224,224,3))
#         input_image_buffer = np.expand_dims(input_image_buffer, axis=0)
#         inputs[0].set_data_from_numpy(input_image_buffer)

#         print("Invoking inference...")
#         results = triton_client.infer(model_name=FLAGS.model,
#                                     inputs=inputs,
#                                     outputs=outputs,
#                                     client_timeout=FLAGS.client_timeout)
#         if FLAGS.model_info:
#             statistics = triton_client.get_inference_statistics(model_name=FLAGS.model)
#             if len(statistics.model_stats) != 1:
#                 print("FAILED: get_inference_statistics")
#                 sys.exit(1)
#             print(statistics)
#         print("Done")

#         result_detections = results.as_numpy('detections')
        

#         print(f"Received result buffer of size {result_detections.shape}")

#         boxes, scores, classes = _postprocess_yolo([result_detections], input_image.shape[1], input_image.shape[0] , FLAGS.confidence, FLAGS.nms)
       
#         print(boxes.shape)
#         # print(detected_objects)
#         for i, det in enumerate(boxes):
#             box = det
#             confidence = scores[i]
#             classID = int(classes[i])
            
#             print(f"{Objects(classID).name}: {confidence}")
            
#             input_image = render_box(input_image, box, color=tuple(RAND_COLORS[classID % 64].tolist()))
#             size = get_text_size(input_image, f"{Objects(classID).name}: {confidence:.2f}", normalised_scaling=0.6)
#             input_image = render_filled_box(input_image, (box[0] - 3, box[1] - 3, box[0] + size[0], box[1] + size[1]), color=(220, 220, 220))
#             input_image = render_text(input_image, f"{Objects(classID).name}: {confidence:.2f}", (box[0], box[1]), color=(30, 30, 30), normalised_scaling=0.5)
            
#         if FLAGS.out:
#             cv2.imwrite(FLAGS.out, input_image)
#             print(f"Saved result to {FLAGS.out}")
#         else:
#             cv2.imshow('image', input_image)
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()
