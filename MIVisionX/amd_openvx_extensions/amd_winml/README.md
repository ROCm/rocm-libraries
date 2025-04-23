# AMD WinML Extension

The AMD WinML (vx_winml) is an OpenVX module that implements a mechanism to access WinML functionality as OpenVX kernels. These kernels can be accessed from within OpenVX framework using OpenVX API call [vxLoadKernels](https://www.khronos.org/registry/vx/specs/1.0.1/html/da/d83/group__group__user__kernels.html#gae00b6343fbb0126e3bf0f587b09393a3)(context, "vx_winml").

<p align="center"><img width="80%" src="https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/winmlFrameWorks.png" /></p>

The WinML extension will allow developers to import a pre-trained ONNX model into an OpenVX graph and add hundreds of different pre & post-processing `vision` / `generic` / `user-defined` functions, available in OpenVX and OpenCV interop, to the input and output of the neural net model. This will allow developers to build an end to end application for inference.

<p align="center"><img width="100%" src="https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/winmlRuntime.png" /></p>

## List of WinML-interop kernels

The following is a list of WinML functions that have been included in the vx_winml module.

```
 onnxToMivisionX com.winml.onnx_to_mivisionx
 convertImageToTensor com.winml.convert_image_to_tensor
 getTopKLabels com.winml.get_top_k_labels
```

**NOTE:** For the list of OpenVX API calls for WinML-interop refer include/[vx_ext_winml.h](include/vx_ext_winml.h)

## Build Instructions

### Pre-requisites

* Windows `10` / `11`, [version `1809` or later](https://www.microsoft.com/software-download/windows10)
* Windows SDK, [build `17763` or later](https://developer.microsoft.com/en-us/windows/downloads/windows-10-sdk)
* Visual Studio 2019, [version `16.11.5` or later](https://developer.microsoft.com/en-us/windows/downloads)
  + Visual Studio extension for C++/WinRT
* Install the latest AMD [drivers](https://www.amd.com/en/support)
* Install [OpenCL SDK](https://github.com/GPUOpen-LibrariesAndSDKs/OCL-SDK/releases/tag/1.0)
* Install [OpenCV 4.6.0](https://github.com/opencv/opencv/releases/tag/4.6.0)
  + Set `OpenCV_DIR` environment variable to `OpenCV/build` folder
  + Add `%OpenCV_DIR%\x64\vc14\bin` or `%OpenCV_DIR%\x64\vc15\bin` to your `PATH`

### Build using `Visual Studio 2019` on 64-bit Windows `10` / `11`

* Use `amd_openvx_extensions/amd_winml.sln` to build for x64 platform

## Utilities

### MIVisionX WinML Validate

This [utility](utilities/MIVisionX-WinML-Validate/README.md#mivisionx-onnx-model-validation) can be used to test and verify the ONNX model on the Windows platform. If the ONNX model is supported by this utility, the amd_winml extension can import the ONNX model and add other OpenVX nodes for pre & post-processing in a single OpenVX graph to run efficient inference.

**NOTE:** [Samples](utilities/MIVisionX-WinML-Validate/sample/README.md#sample) are available

## Samples

[Samples](samples/README.md#samples) to run inference on a single image and a live camera is provided in the samples folder.

* [SqueezeNet](samples/README.md#sample---squeezenet)
  + [Single Image Inference](samples/README.md#winml-imagegdf---single-image-inference)
  + [Live Inference](samples/README.md#winml-livegdf---live-inference-using-a-camera)
* [FER+ Emotion Recognition](samples/README.md#sample---fer-emotion-recognition)
* [VGG 19](samples/README.md#sample---vgg19)
* [Multiple Models](samples/README.md#winml-live-multiplemodelsgdf---live-inference-using-a-camera)
