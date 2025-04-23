# Resnet50 inference with MIGraphX and Onnxruntime

## Description

This example demonstrates how to perform an MIGraphX Python API resnet50 inference through onnxruntime. The model used here is from Torchvision's pretrained resnet50 model

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
onnxruntime and then install the approrpaite version of pytorch from project root.

```
$ 
$ cd examples/onnxruntime/resnet50
$ ./prereq_steps.sh
$ pip list | grep onnxruntime
$ python resnet50.py
```

## Example Output:

For each run the target image was changed. Stock images in the example folder which contains three in-class images types and one out of class image.

For guitars we show three different variants of the same item in a class with different backgrounds as well as background shapes

For the tools (scope.jpg and screwdrivers.jpg) these are both in-class images

For the bird.jpg image, the imagenet_classes.txt generated doesn't contain that of a cockatiel and thus the model attempts to find the closet 
match to the animal found in the image

using scope.jpg. Image of an oscilliscope which is in the imagenet class labels

oscilloscope 0.99869776
radio 0.00051740184
screen 0.00044122382
monitor 8.556517e-05
tape player 5.1727424e-05
resnet50, time = 61.98 ms


using screwdrivers.jpg. Image of screwdrivers which are in the imagenet class labels

screwdriver 0.96512324
carpenter's kit 0.03258186
hammer 0.0007571124
ballpoint 0.00058690814
can opener 0.00026641905
resnet50, time = 109.67 ms


using guitar.jpg. Image of an non traditional electric guitar shape (Telecaster) which is in the imagenet class labels

electric guitar 0.4413027
acoustic guitar 0.14725313
Band Aid 0.14059556
pick 0.076821454
rule 0.020968033
resnet50, time = 77.01 ms


using guitar2.jpg. Image of a les paul stype electric guitar which is in the imagenet classes

electric guitar 0.7114628
acoustic guitar 0.23226906
banjo 0.044191252
pick 0.0056983875
stage 0.0013321621
resnet50, time = 43.09 ms


using guitar3.jpg. Image of a super strat 7 string style electric guitar which is in the imagenet classes

electric guitar 0.71952045
prayer rug 0.08914763
banjo 0.078091994
acoustic guitar 0.056061186
violin 0.021640636
resnet50, time = 71.71 ms


using bird.jpg. Image of a cockatiel not contained in the imagenet class labels

African grey 0.5883207
kite 0.06284781
goldfinch 0.01847724
macaw 0.014789124
fire screen 0.013297303
resnet50, time = 88.89 ms
