# Applications

MIVisionX has several applications built on top of OpenVX and its modules, it uses AMD optimized libraries to build applications that can be used as prototypes or used as models to develop products.

## Prerequisites
* [MIVisionX](https://github.com/ROCm/MIVisionX/blob/master/README.md#prerequisites) installed

## Bubble Pop

This sample [application](./bubble_pop) creates bubbles and donuts to pop using OpenVX & OpenCV functionality.

<p align="center"> <img width="90%" src="https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/vx-pop-app.gif"> </p>

## Cloud Application

This sample [application](./cloud_inference/README.md#cloud-inference-application) does inference using a client-server system.

<p align="center"> <a href="http://www.youtube.com/watch?v=0GLmnrpMSYs"> <img width="90%" src="https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/inferenceVideo.png"> </a></p>

## DG Test

This sample [application](./dg_test/README.md#amd-dgtest) is used to recognize hand written digits.

<p align="center"> <img src="https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/DGtest.gif"> </p>

# MIVisionX Inference Analyzer

[MIVisionX Inference Analyzer Application](./mivisionx_inference_analyzer/README.md#mivisionx-python-inference-analyzer) using pre-trained `ONNX` / `NNEF` / `Caffe` models to analyze and summarize images.

<p align="center"><img width="90%" src="https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/inference_analyzer.gif" /></p>

## MIVisionX OpenVX Classsification

This sample [application](./mivisionx_openvx_classifier/README.md) shows how to run supported pre-trained caffe models with MIVisionX RunTime.

<p align="center"> <img src="https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/mivisionx_openvx_classifier_imageClassification.png"></p>

# MIVisionX Validation Tool

[MIVisionX ML Model Validation Tool](./mivisionx_validation_tool/README.md#mivisionx-python-ml-model-validation-tool) using pre-trained `ONNX` / `NNEF` / `Caffe` models to analyze, summarize, & validate.

<p align="center"><img width="90%" src="https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/validation-2.png" /></p>

## MIVisionX WinML Classification

This sample [application](./mivisionx_winml_classifier/README.md) shows how to run supported ONNX models with MIVisionX RunTime on Windows.

<p align="center"> <img width="60%" src="https://raw.githubusercontent.com/ROCm/MIVisionX/master/apps/mivisionx_winml_classifier/images/MIVisionX-ImageClassification-WinML.png"> </p>

## MIVisionX WinML YoloV2

This sample [application](./mivisionx_winml_yolov2/README.md#yolov2-using-amd-winml-extension) shows how to run tiny yolov2(20 classes) with MIVisionX RunTime on Windows.

<p align="center"> <img width="60%" src="https://raw.githubusercontent.com/ROCm/MIVisionX/master/apps/mivisionx_winml_yolov2/image/cat-yolo.jpg"> </p>

## Optical Flow

This sample [application](./optical_flow/README.md#openvx-samples) we will create an OpenVX graph to run Optical Flow on a video/live. This sample application uses <a href="https://en.wikipedia.org/wiki/OpenCV" target="_blank">OpenCV</a> to decode input video and display the output.

<p align="center"> <img width="60%" src="https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/optical_flow_video.gif"> </p>

## External Application

* [MIVisionX-Classifier](https://github.com/kiritigowda/MIVisionX-Classifier) - This application runs know CNN image classifiers on live/pre-recorded video stream.

* [YOLOv2](https://github.com/kiritigowda/YoloV2NCS) - Run tiny yolov2 (20 classes) with AMD's MIVisionX

* [Traffic Vision](https://github.com/srohit0/trafficVision#traffic-vision) - This app detects cars/buses in live traffic at a phenomenal 50 frames/sec with HD resolution (1920x1080) using deep learning network Yolo-V2. The model used in the app is optimized for inferencing performance on AMD-GPUs using the MIVisionX toolkit.

<p align="center"> <img width="70%" src="https://raw.githubusercontent.com/srohit0/trafficVision/master/media/traffic_viosion.gif" /> </p>

* [RGBDSLAMv2-MIVisionX](https://github.com/ICURO-AI-LAB/RGBDSLAMv2-MIVisionX) - This is an implementation of RGBDSLAM_V2 that utilizes AMD MIVisionX for feature detection and ROCm OpenCL for offloading computations to Radeon GPUs. This application is used to create 3D maps using RGB-D Cameras.
