[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

<p align="center"><img width="50%" src="https://upload.wikimedia.org/wikipedia/commons/d/dd/OpenVX_logo.svg" /></p>

# OpenVX Samples

<a href="https://www.khronos.org/openvx/" target="_blank">Khronos OpenVX&trade;</a> is an open, royalty-free standard for cross-platform acceleration of computer vision applications. OpenVX enables performance and power-optimized computer vision processing, especially important in embedded and real-time use cases such as face, body, and gesture tracking, smart video surveillance, advanced driver assistance systems (ADAS), object and scene reconstruction, augmented reality, visual inspection, robotics and more.

In this sample, we provide OpenVX sample applications to use with any conformant implementation of OpenVX.

* Optical Flow

 ## Optical Flow Sample

In this sample we will create an OpenVX graph to run Optical Flow on a video/live. This sample application uses <a href="https://en.wikipedia.org/wiki/OpenCV" target="_blank">OpenCV</a> to decode input video and display the output.

<p align="center"> <img width="60%" src="https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/optical_flow_video.gif"> </p>

### Prerequisites

* [Conformant OpenVX Implementation](https://github.com/ROCm/MIVisionX)
  
* [OpenCV](https://github.com/opencv/opencv/releases/tag/3.4.0)


### Steps to run the Optical Flow sample

* **Step - 1:** Build and install [Conformant OpenVX Implementation](https://github.com/ROCm/MIVisionX). 

```
Build OpenVX on Linux

* Git Clone project 

      git clone https://github.com/ROCm/MIVisionX.git

* Use CMake to build

      mkdir build && cd build
      cmake ../MIVisionX
      make -j8
      sudo make install
```

* **Step - 2:** Export OpenVX Directory Path

```
export OPENVX_DIR=/opt/rocm/
```


* **Step - 3:** CMake and Build the optical flow application

```
mkdir opticalFlow-build && cd opticalFlow-build
cmake -DOPENVX_INCLUDES=$OPENVX_DIR/include/mivisionx -DOPENVX_LIBRARIES=$OPENVX_DIR/lib/libopenvx.so ../optical_flow
make
```

* **Step - 4:** Run VX Optical Flow application
    
    ```
      Usage:
	      ./opticalFlow --video <Video File>
	      ./opticalFlow --live  <Capture Device ID>
    ```

  + Use Video Option
    ```
    ./opticalFlow --video ../../../data/videos/AMD_driving_virtual_20.mp4
    ```
  + Use Live Device Camera 
    ```
    ./opticalFlow --live 0
    ```
