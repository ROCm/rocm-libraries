# rocAL Performance Tests

This application is used to run performance tests on the rocAL API for graphs of depth size 1.

## Pre-requisites

* Ubuntu Linux, [version `16.04` or later](https://www.microsoft.com/software-download/windows10)
* rocAL library
* [OpenCV 3.1](https://github.com/opencv/opencv/releases) or higher
* ROCm Performance Primitives (RPP)

## Build Instructions

  ````absh
  mkdir build
  cd build
  cmake ../
  make
  ````
### running the application

  ````bash
  ./performance_tests [test image folder - required] [image width - required] [image height - required] [test case] [batch size] [0 for CPU, 1 for GPU] [0 for grayscale, 1 for RGB]
  ````
