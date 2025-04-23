# rocAL Performance Tests with Depth

This application can be used to run performance tests on rocAL graphs with depth greater than 1.
This is very similar to the rocAL Performance Tests app except it takes an extra parameter to specify depth.

## Pre-requisites

* Ubuntu Linux, [version `16.04` or later](https://www.microsoft.com/software-download/windows10)
* rocAL library
* [OpenCV 3.1](https://github.com/opencv/opencv/releases) or higher
* ROCm Performance Primitives (RPP)

## Build Instructions

  ````bash
  mkdir build
  cd build
  cmake ../
  make
  ````

### running the application

  ````bash
  ./performance_tests_with_depth [test image folder] [image width] [image height] [test case] [batch size] [graph depth] [0 for CPU, 1 for GPU] [0 for grayscale, 1 for RGB]
  ````
