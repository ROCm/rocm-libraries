# AMD RPP Extension

The AMD VX RPP extension (vx_rpp) is an OpenVX module that implements an interface to access RPP functionality as OpenVX kernels. These kernels can be accessed from within OpenVX framework using OpenVX API call [vxLoadKernels](https://www.khronos.org/registry/vx/specs/1.0.1/html/da/d83/group__group__user__kernels.html#gae00b6343fbb0126e3bf0f587b09393a3)(context, "vx_rpp").

## List of RPP functions 

The following is a list of RPP functions that have been included in the vx_rpp module.

    Blend                       org.rpp.Blend
    Blur                        org.rpp.Blur
    Brightness                  org.rpp.Brightness
    Color temperature           org.rpp.ColorTemperature
    Contrast                    org.rpp.Contrast
    Exposure                    org.rpp.Exposure
    Fisheye lens effect         org.rpp.Fisheye
    Flip                        org.rpp.Flip
    Fog effect                  org.rpp.Fog
    Gamma Correction            org.rpp.GammaCorrection
    Image Resize                org.rpp.Resize
    Jitter                      org.rpp.Jitter
    Lens Correction             org.rpp.LensCorrection
    Pixelate                    org.rpp.Pixelate
    Rain overlay                org.rpp.Rain
    Resize Crop                 org.rpp.ResizeCrop
    Rotate                      org.rpp.Rotate
    Salt and Pepper noise       org.rpp.NoiseSnp
    Snow                        org.rpp.Snow
    Vignette                    org.rpp.Vignette
    Warp affine                 org.rpp.WarpAffine

**NOTE** - For list of OpenVX API calls for RPP-interop refer include/[vx_ext_rpp.h](include/vx_ext_rpp.h)

## Build Instructions

### Pre-requisites

* AMD OpenVX&trade; library
* [AMD RPP library](https://github.com/ROCm/rpp)
* CMake 3.10 or later
* OpenCL (optional)

### Build using CMake on Linux

* Use CMake to configure and generate Makefile

**NOTE:** OpenVX and the OpenVX logo are trademarks of the Khronos Group Inc.
