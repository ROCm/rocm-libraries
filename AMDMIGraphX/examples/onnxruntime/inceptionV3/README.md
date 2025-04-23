# invepctionv3 inference with MIGraphX and Onnxruntime

## Description

This example demonstrates how to perform an MIGraphX Python API inceptionv3 inference through onnxruntime. The model used here is from Torchvision's pretrained inceptionv3 model

## Content
- [Basic Setup](#Basic-Setup)
- [**Running this Example**](#Running-this-Example)
- [Example output](#example-Output)

## Basic Setup
Before running inference we must first install MIGraphX using any method then download Onnxruntime into the root system directory  

Starting from project root:
```
$ cd AMDMIGraphX
$ docker build -t migraphx .
$ docker run --device='/dev/kfd' --device='/dev/dri' <your docker settings and mounts> --group-add video -it migraphx
```

The dockerfile will install the latest supported version of ROCm with all the dependencies needed for MIGraphX

Once the docker file has been installed and you're inside the folder run

```
$ rbuild develop -d deps -B build
$ cd build 
$ make -j$(nproc) package && dpkg -i *.deb
```

to verify migraphx has been installed correclty in the docker run dpkg --list or dpkg -l

```
$ dpkg -l | grep migraphx 
$ ii  migraphx                      2.7.0                             amd64        AMD's graph optimizer
$ ii  migraphx-dev                  2.7.0                             amd64        AMD's graph optimizer
```

## Running this Example

This directory contains everything needed to perform an inference once MIGraphX has been installed in the docker container
Once you've build and installed MIGraphX as a deb package, go to the examples folder, run the pre-req script to build and install
onnxruntime and then install the appropriate version of pytorch from project root.

An example command build Onnxruntime is found in ./prereq_steps.sh in MIGraphX Root to build onnxruntime. Pre-build Onnxruntime Wheel file builds are also valid.
Ensure the wheel used is using the same python version you're using on the host system.

```
../../../tools/build_and_test_onnxrt.sh
pip3 install /onnxruntime/build/Release/Linux/dist/*.whl
```

To run this example then do the following

```
$ 
$ cd examples/onnxruntime/inveptionv3
$ ./prereq_steps.sh
$ pip list | grep onnxruntime
$ python invepctionv3.py
```

## Example Output:

For each run the target image was changed. Stock images in the example folder which contains three in-class images types and one out of class image.

For guitars we show three different variants of the same item in a class with different backgrounds as well as background shapes

For the tools (scope.jpg and screwdrivers.jpg) these are both in-class images

For the bird.jpg image, the imagenet_classes.txt generated doesn't contain that of a cockatiel and thus the model attempts to find the closest 
match to the animal found in the image

using bird.jpg. Image of an bird which is in the imagenet class labels defaults to african grey

African grey 0.3860151
inception_v3, Total time = 801.04 ms