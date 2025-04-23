# Image Augmentation Application

This application demonstrates the basic usage of rocAL's C API to load JPEG images from the disk and modify them in different possible ways and displays the output images.

<p align="center"><img width="90%" src="../../docs/data/image_augmentation.png" /></p>

## Pre-requisites

* Linux distribution
  + Ubuntu - `20.04` / `22.04`
* rocAL and it's dependencies
* Optional: OpenCV for display - [4.6.0](https://github.com/opencv/opencv/releases/tag/4.6.0)

## Build Instructions

```bash
 export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib
 mkdir build
 cd build
 cmake ../
 make 
```

### Running the application

```bash
./image_augmentation <path-to-image-dataset>
```
