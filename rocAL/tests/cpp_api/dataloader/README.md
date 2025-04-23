# dataloader application

This application demonstrates a basic usage of rocAL's C API to load RAW images from the disk and modify them in different possible ways and displays the output images.
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

### running the application

  ````shell
  ./dataloader <image_dataset_folder> <processing_device=1/cpu=0>  decode_width decode_height batch_size display_on_off
  ````
