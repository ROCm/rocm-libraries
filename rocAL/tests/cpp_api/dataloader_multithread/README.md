# dataloader multithreaded application

This application demonstrates a basic usage of rocAL's C API to use sharded data_loader  in a multithreaded application.
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
  ./dataloader_multithread <image_dataset_folder - required> <num_gpus(gpu:>=1)/(cpu:0)>  <num_shards> <decode_width> <decode_height> <batch_size> <shuffle> <display>
  ````
