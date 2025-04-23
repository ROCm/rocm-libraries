<p align="center"><img width="30%" src="https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/OpenVX_logo.png" /></p>

# Khronos OpenVX&trade;

OpenVX&trade; is an open, royalty-free standard for cross platform acceleration of computer vision applications. OpenVX enables performance and power-optimized computer vision processing, especially important in embedded and real-time use cases such as face, body and gesture tracking, smart video surveillance, advanced driver assistance systems (ADAS), object and scene reconstruction, augmented reality, visual inspection, robotics and more.

* Khronos [OpenVX](https://www.khronos.org/openvx/) API offers a set of optimized primitives for low-level image processing, computer vision, and neural net operators. The API provides a simple method to write optimized code that is portable across multiple hardware vendors and platforms.

* OpenVX allows for resource and execution abstractions, which enable hardware vendors to optimize their implementation for their platform. Performance portability across CPUs, GPUs, and special-function hardware are one of the design goals of the OpenVX specification.

* OpenVX is used to build, verify, and coordinate computer vision and neural network graph executions. The graph abstraction enables OpenVX implementation to optimize execution for the underlying hardware.  Using optimized OpenVX conformant implementation, software developers can spend more time on algorithmic innovations without worrying about the performance and portability of their applications.

<p align="center"><img width="80%" src="https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/openvx_software_layers.png" style="background-color:black;" /></p>

* The standard defines graph conventions and execution semantics to address the needs of the developers. The advantage of the graphical interface is the ability of the underlying conformant implementation to optimize the whole graph pipeline instead of specific functions.

* OpenVX specification also defines the VXU or the immediate function library. VXU operators allow developers to use all the OpenVX operators as a directly callable C function without creating a graph first.

* Applications built using the VXU library do not benefit from the optimizations enabled by graph execution. The VXU library can be the simplest way to use OpenVX and is the first step in porting existing vision applications.

## AMD OpenVX&trade;

* AMD OpenVX&trade; is a highly optimized conformant open-source implementation of the [Khronos OpenVX Version 1.3](https://www.khronos.org/registry/OpenVX/specs/1.3/html/OpenVX_Specification_1_3.html) computer vision specification. It allows for rapid prototyping as well as fast execution on a wide range of computer hardware, including small embedded `AMD64` CPUs and large workstation discrete GPUs.

### AMD OpenVX&trade; - Vision Feature Set Conformant Implementation

The Vision Conformance Feature Set includes all the functions and objects in the `Base Feature Set`, plus the vision data objects and vision functions.

#### The Base Feature Set

The purpose is to define a minimal subset of OpenVX features that enable the construction and execution of OpenVX graphs, but it does not contain any specific vision-processing operations.

**Basic Framework Objects**
<table border="1">
  <tr>
    <th style="text-align: center">vx_reference</th>
    <th style="text-align: center">vx_context</th> 
    <th style="text-align: center">vx_graph</th>
    <th style="text-align: center">vx_kernel</th>
  </tr>
  <tr>
    <th style="text-align: center">vx_node</th>
    <th style="text-align: center">vx_parameter</th> 
    <th style="text-align: center">vx_meta_format</th>
    <th style="text-align: center">vx_delay</th>
  </tr>
</table>

**Note:** Details about [Framework Objects](https://www.khronos.org/registry/OpenVX/specs/1.3/html/OpenVX_Specification_1_3.html#sec_framework_objects)

#### The Vision Conformance Feature Set

To provide a basic set of vision processing functions. This set of functions is roughly equivalent to the set of functions available in version 1.1 of the OpenVX specification. In addition to the framework objects included in the Base Feature Set, the Vision Conformance Feature Set includes a set of data objects that the Vision functions operate upon and produce.

**Vision Conformance Data Objects**
<table border="1">
  <tr>
    <th style="text-align: center">vx_array</th>
    <th style="text-align: center">vx_convolution</th> 
    <th style="text-align: center">vx_distribution</th>
    <th style="text-align: center">vx_image</th>
  </tr>
  <tr>
    <th style="text-align: center">vx_lut</th>
    <th style="text-align: center">vx_matrix</th> 
    <th style="text-align: center">vx_pyramid</th>
    <th style="text-align: center">vx_remap</th>
  </tr>
  <tr>
    <th style="text-align: center">vx_scalar</th>
    <th style="text-align: center">vx_threshold</th> 
    <th style="text-align: center">vx_object_array</th>
    <th style="text-align: center">&nbsp; </th>
  </tr>
</table>

**NOTE:** Details about [Data Objects](https://www.khronos.org/registry/OpenVX/specs/1.3/html/OpenVX_Specification_1_3.html#sec_data_objects)

**Vision Conformance Functions**
<table border="1">
  <tr>
    <th style="text-align: center">AbsDiff</th>
    <th style="text-align: center">Add</th> 
    <th style="text-align: center">And</th>
    <th style="text-align: center">Box3x3</th>
    <th style="text-align: center">CannyEdgeDetector</th>
    <th style="text-align: center">ChannelCombine</th>
  </tr>
  <tr>
    <th style="text-align: center">ChannelExtract</th>
    <th style="text-align: center">ColorConvert</th> 
    <th style="text-align: center">ConvertDepth</th>
    <th style="text-align: center">Convolve</th>
    <th style="text-align: center">Dilate3x3</th>
    <th style="text-align: center">EqualizeHist</th>
  </tr>
  <tr>
    <th style="text-align: center">Erode3x3</th>
    <th style="text-align: center">FastCorners</th> 
    <th style="text-align: center">Gaussian3x3</th>
    <th style="text-align: center">GaussianPyramid</th>
    <th style="text-align: center">HarrisCorners</th>
    <th style="text-align: center">HalfScaleGaussian</th>
  </tr>
  <tr>
    <th style="text-align: center">Histogram</th>
    <th style="text-align: center">IntegralImage</th> 
    <th style="text-align: center">LaplacianPyramid</th>
    <th style="text-align: center">LaplacianReconstruct</th>
    <th style="text-align: center">Magnitude</th>
    <th style="text-align: center">MeanStdDev</th>
  </tr>
  <tr>
    <th style="text-align: center">Median3x3</th>
    <th style="text-align: center">MinMaxLoc</th> 
    <th style="text-align: center">Multiply</th>
    <th style="text-align: center">NonLinearFilter</th>
    <th style="text-align: center">Not</th>
    <th style="text-align: center">OpticalFlowPyrLK</th>
  </tr>
  <tr>
    <th style="text-align: center">Or</th>
    <th style="text-align: center">Phase</th> 
    <th style="text-align: center">Remap</th>
    <th style="text-align: center">ScaleImage</th>
    <th style="text-align: center">Sobel3x3</th>
    <th style="text-align: center">Subtract</th>
  </tr>
  <tr>
    <th style="text-align: center">TableLookup</th>
    <th style="text-align: center">Threshold</th> 
    <th style="text-align: center">WarpAffine</th>
    <th style="text-align: center">WarpPerspective</th>
    <th style="text-align: center">WeightedAverage</th>
    <th style="text-align: center">Xor</th>
  </tr>
</table>

**Note:** Details about [Functions](https://www.khronos.org/registry/OpenVX/specs/1.3/html/OpenVX_Specification_1_3.html#group_vision_functions)

### AMD OpenVX&trade; - Features

* The code is highly optimized for both x86 CPU and OpenCL/HIP for GPU
* Supported hardware spans the range from low power embedded APUs, laptops, desktops, and workstation graphics
* Supports `Windows` , `Linux` , and `macOS`
* Includes a “graph optimizer” that looks at the entire processing pipeline and removes/replaces/merges functions to improve performance and minimize bandwidth at runtime 
* Scripting support with [RunVX](../utilities/runvx/README.md) allows for rapid prototyping, without re-compiling at production performance levels

**NOTE:** The amd_openvx project consists of the following components: [AMD OpenVX&trade; Library](openvx/README.md)

**NOTE:** The OpenVX framework provides a mechanism to add new vision functions to OpenVX by 3rd party vendors. Look into amd_openvx_extensions for additional OpenVX modules and utilities.

#### AMD OpenVX&trade; - Extensions
  * **vx_loomsl**: Radeon LOOM stitching library for live 360-degree video applications
  * **vx_nn**: OpenVX neural network module that was built on top of [MIOpen](https://github.com/ROCmSoftwarePlatform/MIOpen)
  * **vx_opencv**: OpenVX module that implemented a mechanism to access OpenCV functionality as OpenVX kernels
  * **vx_rpp**: OpenVX extension providing an interface to some of the ROCm Performance Primitives ([RPP](https://github.com/ROCm/rpp)) functions. This extension is used to enable [rocAL]([../rocAL/README.md](https://github.com/ROCm/rocAL)) to perform image augmentation.
  * **vx_winml**: OpenVX module that implemented a mechanism to access Windows Machine Learning(WinML) functionality as OpenVX kernels

## Pre-requisites

* **CPU**: [AMD64](https://docs.amd.com/bundle/Hardware_and_Software_Reference_Guide/page/Hardware_and_Software_Support.html)
* **GPU**: [AMD Radeon&trade; Graphics](https://docs.amd.com/bundle/Hardware_and_Software_Reference_Guide/page/Hardware_and_Software_Support.html) [optional]
  + Windows: install the latest drivers and OpenCL SDK [download](https://github.com/GPUOpen-LibrariesAndSDKs/OCL-SDK/releases)
  + Linux: install [ROCm](https://rocm.github.io/ROCmInstall.html)
* **APU**: [AMD Radeon&trade; `Mobile`/`Embedded`](https://docs.amd.com/bundle/Hardware_and_Software_Reference_Guide/page/Hardware_and_Software_Support.html) [optional]

## Build Instructions

Build this project to generate AMD OpenVX&trade; library 

* Refer to [openvx/include/VX](openvx/include/VX) for Khronos OpenVX standard header files.
* Refer to [openvx/include/vx_ext_amd.h](openvx/include/vx_ext_amd.h) for vendor extensions in AMD OpenVX&trade; library

### Build using `Visual Studio`

* Install OpenCV with/without contrib [download](https://github.com/opencv/opencv/releases) for RunVX tool to support camera capture and image display (optional)
  + OpenCV_DIR environment variable should point to OpenCV/build folder
* Use amd_openvx/amd_openvx.sln to build for x64 platform
* If AMD GPU (or OpenCL) is not available, set build flag `ENABLE_OPENCL=0`in openvx/openvx.vcxproj and runvx/runvx.vcxproj

**Note:** AMD GPU `HIP` backend is not supported on Windows 

### Build using CMake

* Install CMake 3.10 or later
* Use CMake to configure and generate Makefile

**NOTE:** OpenVX and the OpenVX logo are trademarks of the Khronos Group Inc.
