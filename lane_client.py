#!/usr/bin/env python

import argparse
import numpy as np
import sys
import cv2
import time

from numpy.testing._private.utils import print_assert_equal
import util 

import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

from parameters import Parameters
from lane_processing import preprocess, warp_image
from lane_postprocessing import postprocess

p = Parameters()

class LaneClient(object):
    def  __init__(self, model_name = "pinet_1block", url="0.0.0.0:8001", confidence = 0.6):
        self.model_name = model_name
        self.url = url
        self.confidence = confidence
        self.colours = np.array([[78, 142, 255], [204, 237, 221], [92, 252, 255], [92, 255, 195], [159, 150, 255], [53, 150, 100]])
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
        
        # create message
        
        if self.model_name.split("_")[1].split("b")[0] == '1':
            self.inputs = []
            self.outputs = []
            self.inputs.append(grpcclient.InferInput('input.1', [1, 3, 256, 512], "FP32"))
            self.outputs.append(grpcclient.InferRequestedOutput('1790'))
            self.outputs.append(grpcclient.InferRequestedOutput('1924'))
            self.outputs.append(grpcclient.InferRequestedOutput('1933'))
            self.outputs.append(grpcclient.InferRequestedOutput('1942'))
            self.mod = 1
        elif self.model_name.split("_")[1].split("b")[0] == '2':
            self.inputs.append(grpcclient.InferInput('input.1', [1, 3, 256, 512], "FP32"))
            self.outputs.append(grpcclient.InferRequestedOutput('1790'))
            self.outputs.append(grpcclient.InferRequestedOutput('1925'))
            self.outputs.append(grpcclient.InferRequestedOutput('1934'))
            self.outputs.append(grpcclient.InferRequestedOutput('1943'))
            self.outputs.append(grpcclient.InferRequestedOutput('2016'))
            self.outputs.append(grpcclient.InferRequestedOutput('2150'))
            self.outputs.append(grpcclient.InferRequestedOutput('2159'))
            self.outputs.append(grpcclient.InferRequestedOutput('2168'))
            self.mod = 2
        else:
            print("System currently dont support model > 2 block")
            sys.exit(1)
        
    def getPrediction(self, image):
        
        input_image = cv2.resize(image,(512,256))
        self.image = input_image
        self.warped = warp_image(input_image)
        input_image_buffer = np.rollaxis(self.warped, axis=2, start=0)/255.0
        input_image_buffer = np.expand_dims(input_image_buffer, axis=0).astype("float32")

        self.inputs[0].set_data_from_numpy(input_image_buffer)

        results = self.triton_client.infer(model_name=self.model_name,
                                    inputs=self.inputs,
                                    outputs=self.outputs,
                                    client_timeout=None)
        if self.mod == 1:
            result_features = results.as_numpy('1790')
            result_confs = results.as_numpy('1924')
            result_offsets = results.as_numpy('1933')
            result_instances = results.as_numpy('1942')
        if self.mod == 2:
            result_features = results.as_numpy('2016')
            result_confs = results.as_numpy('2150')
            result_offsets = results.as_numpy('2159')
            result_instances = results.as_numpy('2168')

        self.xs, self.ys = postprocess([result_confs, result_offsets, result_instances, result_features], input_image.shape[1], input_image.shape[0] , self.confidence)
        return self.xs, self.ys

    def getFits(self):
        fits = np.array([np.polyfit(_y, _x, 1) for _x, _y in zip(self.xs, self.ys)]) 
        self.fits = util.adjust_fits(fits)
        return fits


    def getMask(self):
        warp = np.zeros_like(self.image)
        y = np.linspace(20, 256, 4)

        for i, fit in enumerate(self.fits[:-1]):
            x_0 = np.array([np.poly1d(fit)(_y) for _y in y ])
            x_1 = np.array([np.poly1d(self.fits[i+1])(_y) for _y in y])

            pts_left = np.array([np.transpose(np.vstack([x_0, y]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([x_1, y])))])

            color = self.colours[i]

            pts = np.hstack((pts_left, pts_right))
            cv2.fillPoly(warp, np.int_([pts]), (int(color[0]),int(color[1]),int(color[2])))
        
        self.mask = cv2.warpPerspective(warp, p.inverse_perspective_transform, (warp.shape[1], warp.shape[0]))
        return self.mask
    

    def getImageResult(self, image=None):
        if image is None:
            try:
                result = cv2.addWeighted(self.image, 1, self.mask, 0.7, 0.3)
            except:
                self.getMask()
                result = cv2.addWeighted(self.image, 1, self.mask, 0.7, 0.3)

        else:
            try:
                result = cv2.addWeighted(image, 1, self.mask, 0.7, 0.3)
            except:
                self.getMask()
                result = cv2.addWeighted(image, 1, self.mask, 0.7, 0.3)
        return result



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
    client = LaneClient()
    image = cv2.imread(str(FLAGS.input))
    xs, ys = client.getPrediction(image)
    fits = client.getFits()
    image_output = client.getImageResult()
    cv2.imshow("image result", image_output)
    cv2.waitKey(0)
    
# original

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
#                         default='pinet_2block',
#                         help='Inference model name, default yolov4')
#     parser.add_argument('-u',
#                         '--url',
#                         type=str,
#                         required=False,
#                         default='192.168.31.46:8001',
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
#                         default=0.8,
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
#         # inputs.append(grpcclient.InferInput('input.1', [1, 3, 256, 512], "FP32"))
#         # outputs.append(grpcclient.InferRequestedOutput('1790'))
#         # outputs.append(grpcclient.InferRequestedOutput('1924'))
#         # outputs.append(grpcclient.InferRequestedOutput('1933'))
#         # outputs.append(grpcclient.InferRequestedOutput('1942'))

#         inputs.append(grpcclient.InferInput('input.1', [1, 3, 256, 512], "FP32"))
#         outputs.append(grpcclient.InferRequestedOutput('1790'))
#         outputs.append(grpcclient.InferRequestedOutput('1925'))
#         outputs.append(grpcclient.InferRequestedOutput('1934'))
#         outputs.append(grpcclient.InferRequestedOutput('1943'))
#         outputs.append(grpcclient.InferRequestedOutput('2016'))
#         outputs.append(grpcclient.InferRequestedOutput('2150'))
#         outputs.append(grpcclient.InferRequestedOutput('2159'))
#         outputs.append(grpcclient.InferRequestedOutput('2168'))


#         print("Creating buffer from image file...")
#         input_image = cv2.imread(str(FLAGS.input))
#         if input_image is None:
#             print(f"FAILED: could not load input image {str(FLAGS.input)}")
#             sys.exit(1)

#         input_image_buffer = preprocess(input_image)
#         input_image_buffer = np.expand_dims(input_image_buffer, axis=0).astype("float32")
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

#         # result_features = results.as_numpy('1790')
#         # result_confs = results.as_numpy('1924')
#         # result_offsets = results.as_numpy('1933')
#         # result_instances = results.as_numpy('1942')

#         result_features = results.as_numpy('2016')
#         result_confs = results.as_numpy('2150')
#         result_offsets = results.as_numpy('2159')
#         result_instances = results.as_numpy('2168')

#         xs, ys = postprocess([result_confs, result_offsets, result_instances, result_features], input_image.shape[1], input_image.shape[0] , FLAGS.confidence, FLAGS.nms)


#         image = cv2.resize(input_image,(512,256))
#         warped = warp_image(image)
#         result_image = util.draw_points(xs, ys, warped)
        
#         cv2.waitKey(0)

#         if FLAGS.out:
#             cv2.imwrite(FLAGS.out, input_image)
#             print(f"Saved result to {FLAGS.out}")
#         else:
#             cv2.imshow("image",result_image)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()