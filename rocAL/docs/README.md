# rocAL User Guide

Today’s deep learning applications require loading and pre-processing data efficiently to achieve high processing throughput.  This requires creating efficient processing pipelines fully utilizing the underlying hardware capabilities. Some examples are load and decode data, do a variety of augmentations, color-format conversions, etc.
Deep learning frameworks require supporting multiple data formats and augmentations to adapt to a variety of data-sets and models.

AMD ROCm Augmentation Library (rocAL) is designed to efficiently do such processing pipelines from both images and video as well as from a variety of storage formats.
These pipelines are programmable by the user using both C++ and Python APIs.

## User Guide Chapters

* [Chapter 1 - Overview](user_guide/ch1.md)
* [Chapter 2 - Architecture Components](user_guide/ch2.md)
* [Chapter 3 - Installation](user_guide/ch3.md)
* [Chapter 4 - Using with Python API](user_guide/ch4.md)
* [Chapter 5 - Framework Integration](user_guide/ch5.md)
* [Chapter 6 - Using with C++ API](user_guide/ch6.md)

## Key Components of rocAL

* Full processing pipeline support for data_loading, meta-data loading, augmentations, and data-format conversions for training and inference.
* Being able to do processing on CPU or Radeon GPU (with OpenCL or HIP backend)
* Ease of integration with framework plugins in Python
* Support variety of augmentation operations through AMD’s Radeon Performance Primitives (RPP).
* All available public and open-sourced under ROCm.

## Prerequisites

Refer [rocAL Prerequisites](https://github.com/ROCm/rocAL#prerequisites)

## Build instructions

Refer [rocAL build instructions](https://github.com/ROCm/rocAL#build-instructions)

## rocAL Python

* rocAL Python package has been created using Pybind11 which enables data transfer between rocAL C++ API and Python API.
* rocal Python Bindings has both PyTorch and TensorFlow framework support.
* Various reader format support including FileReader, COCOReader, and TFRecordReader.
* [examples folder](https://github.com/ROCm/rocAL/docs/exmaples) has sample implementations for PyTorch and Tensorflow training and inference pipeline.

## rocAL Python API

### amd.rocal.fn

* Contains the image augmentations & file read and decode operations which are linked to rocAL C++ API
* All ops (listed below) are supported for the single input image and batched inputs.

| Image Augmentation |   Reader and Decoder      |        Geometric Ops         |     Audio Augmentation     |
| :----------------: | :-----------------------: | :--------------------------: | :------------------------: |
|    Color Twist     |   Image File Reader       |     Crop Mirror Normalize    |    PreEmphasis Filter      |
| Color Temperature  |      Caffe Reader         |          Crop Resize         |      Non-Silent Region     |
|     Brightness     |      Caffe2 Reader        |            Resize            |          Resample          |
|  Gamma Correction  |      CIFAR10 Reader       |          Random Crop         |        Spectrogram         |
|        Snow        |       COCO Reader         |         Warp Affine          |        Mel-Filter Bank     |
|        Rain        |     TF Record Reader      |           Fish Eye           |          ToDecibels        |
|        Blur        |   MXNet Record Reader     |        Lens Correction       |          Normalize         |
|       Jitter       |    Video File Reader      |           Rotate             |                            |  
|        Hue         |     Image Decoder         |            Crop              |                            |
|     Saturation     | Image Decoder Random Crop |            Flip              |                            |
|        Fog         |      Video Decoder        |    Resize Crop Mirror        |                            |
|      Contrast      |      Audio Decoder        | Resize Crop Mirror Normalize |                            |
|      Vignette      |                           |            Slice             |                            |
|     SNP Noise      |                           |                              |                            |
|      Pixelate      |                           |                              |                            |
|       Blend        |                           |                              |                            |
|      Exposure      |                           |                              |                            |

### amd.rocal.pipeline

* Contains Pipeline class which has all the data needed to build and run the rocAL graph.
* Contains support for context/graph creation, verify and run the graph.
* Has data transfer functions to exchange data between frameworks and rocAL
* define_graph functionality has been implemented to add nodes to build a pipeline graph.

### amd.rocal.types

amd.rocal.types are enums exported from C++ API to python. Some examples include CPU, GPU, FLOAT, FLOAT16, RGB, GRAY, etc.

### amd.rocal.plugin.pytorch

* Contains ROCALGenericIterator for Pytorch.
* ROCALClassificationIterator class implements iterator for image classification and return images with corresponding labels.
* ROCALAudioIterator class for audio tasks and returns audio data, corresponding labels and its roi.
* From the above classes, any hybrid iterator pipeline can be created by adding augmentations.
* See example [PyTorch Simple Example](./examples/pytorch/). Requires PyTorch.

### amd.rocal.plugin.tf

* Contains ROCALIterator for TensorFlow.
* Any hybrid iterator pipeline can be created by adding augmentations.
* See example [Tensorflow Simple Example](./examples/tf/). Requires TensorFlow.

### installing rocAL python plugin (Python 3.9+)

* Build and install RPP
* Build and install MIVisionX
* Build and install [rocAL](https://github.com/ROCm/rocAL/)

