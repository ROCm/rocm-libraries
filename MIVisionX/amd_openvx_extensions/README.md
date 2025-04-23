# AMD OpenVX&trade; Extensions

The OpenVX framework provides a mechanism to add new vision functions to OpenVX by 3rd party vendors. This project has below OpenVX modules and utilities to extend [AMD OpenVX&trade;](../amd_openvx/README.md#amd-openvx-amd_openvx) (amd_openvx) project, which contains the AMD OpenVX&trade; Core Engine.

* [amd_loomsl](amd_loomsl/README.md): AMD Radeon LOOM stitching library for live 360-degree video applications

<p align="center"><img width="80%" src="https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/loom-2.jpg" /></p>

* [amd_media](amd_media/README.md): `vx_amd_media` is an OpenVX AMD media extension module. This module has mainly two OpenVX extension nodes. `com.amd.amd_media.decode` node for video/jpeg decoding and `com.amd.amd_media.encode` node for video encoding

* [amd_migraphx](amd_migraphx/README.md): amd_migraphx extension integrates the <a href="https://github.com/ROCmSoftwarePlatform/AMDMIGraphX#amd-migraphx" target="_blank"> AMD's MIGraphx </a> into an OpenVX graph for inference.

* [amd_nn](amd_nn/README.md#openvx-neural-network-extension-library-vx_nn): OpenVX neural network module. Learn more about neural net workflow in [Neural Net Model Compiler & Optimizer](../model_compiler/README.md#neural-net-model-compiler--optimizer)

<p align="center"><img width="80%" src="https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/modelCompilerWorkflow.png" /></p>

* [amd_opencv](amd_opencv/README.md#amd-opencv-extension): OpenVX module that implements a mechanism to access OpenCV functionality as OpenVX kernels

* [amd_rpp](amd_openvx_extensions/amd_rpp/README.md): OpenVX extension providing an interface to some of the ROCm Performance Primitives ([RPP](https://github.com/ROCm/rpp)) functions. This extension is used to enable [rocAL](https://github.com/ROCm/rocAL) to perform image augmentation.

* [amd_winml](amd_winml/README.md#amd-winml-extension): WinML extension will allow developers to import a pre-trained ONNX model into an OpenVX graph and add hundreds of different pre & post processing `vision` / `generic` / `user-defined` functions, available in OpenVX and OpenCV interop, to the input and output of the neural net model. This will allow developers to build an end to end application for inference.

<p align="center"><img width="80%" src="https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/winmlFrameWorks.png" /></p>

**NOTE:** OpenVX and the OpenVX logo are trademarks of the Khronos Group Inc.
