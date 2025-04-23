# basic test
This application demonstrates a basic usage of rocAL's C API to load images from the disk and tests the functionality for getting image labels and displays the output images in a loop.
<p align="center"><img width="90%" src="https://www.github.com/ROCm/rocAL/docs/data/image_augmentation.png" /></p>

## Pre-requisites

* Ubuntu Linux, [version `16.04` or later](https://www.microsoft.com/software-download/windows10)
* rocAL library
* [OpenCV 3.1](https://github.com/opencv/opencv/releases) or higher
* ROCm Performance Primitives (RPP)

## Build Instructions

  ````shell
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib
  mkdir build
  cd build
  cmake ../
  make
  ````

## Running the application

  ```shell
  ./basic_test <image_dataset_folder - required> <label_text_file_path - required> <test_case:0/1> <processing_device=1/cpu=0>  decode_width decode_height <gray_scale:0/rgb:1> decode_shard_counts <decoder_type:0/1/2>
  ```

NOTE:

* default decoder type : TurboJpeg decoder
* decoder type : 1 - OpenCV decoder
* decoder type : 2 - rocJpeg decoder
