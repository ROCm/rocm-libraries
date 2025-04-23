## OpenVX node tests

### Help

The runvxTestAllScript.sh bash script runs runvx for AMD OpenVX&trade; functionalities in HOST/OCL/HIP backends.
- It can optionally run tests for all kernels or a single kernel separately.
- It can optionally generate .bin dumps for the image inputs/outputs.
- It can optionally run HOST/OCL/HIP backends and also run a OCLvsHIP test to compare OCLvsHIP outputs and report inconsistencies (for all kernels or a single kernel separately).
- It requires a user given width, height to run tests.
- It requires a primary RunVX path while running HOST/OCL/HIP backends and optionally a secondary RunVX path while running OCLvsHIP comparison.

### Syntax

Syntax: `./runvxTestAllScript.sh <W> <H> <D> <N> <B> <P> <S> <O>` where:
```
- W     WIDTH of image in pixels
- H     HEIGHT of image pixels
- D     Output bin DUMP toggle (1 = True, 0 = False)
- N     NAME of kernel to run ('ALL' = run all available kernels, '<kernel name>' = run specific kernel name from the list of kernels in this script)
- B     BACKEND (HOST = On CPU / OCL = On GPU with OpenCL backend / HIP = On GPU with HIP backend / OCLvsHIP = On GPU with OCLvsHIP output comparison)
- P     PRIMARY RunVX path (for MIVisionX built with HOST/OCL/HIP backend)
- S     SECONDARY RunVX path (Required only for OCLvsHIP comparison. Primary path used for OCL backend, Secondary path used for HIP backend)
- O     RunVX path OVERRIDE - Optional parameter (0 = Use primary/secondary RunVX paths (DEFAULT), 1 = make-install required backend and then run tests, 2 = Clean build-make-install required backend and then run tests)
```

### Examples

Examples to run a single kernel:
```
./runvxTestAllScript.sh 16 16 1 Add_U8_U8U8_Wrap HOST ../../build_host/install/bin
./runvxTestAllScript.sh 16 16 0 Add_U8_U8U8_Wrap OCL ../../build_ocl/install/bin
./runvxTestAllScript.sh 16 16 1 Add_U8_U8U8_Wrap HIP ../../build_hip/install/bin
./runvxTestAllScript.sh 16 16 1 ChannelExtract_U8U8U8U8_U32 OCLvsHIP ../../build_ocl/install/bin ../../build_hip/install/bin
```

Examples to run all kernels:
```
./runvxTestAllScript.sh 16 16 1 ALL HOST ../../build_host/install/bin
./runvxTestAllScript.sh 16 16 1 ALL OCL ../../build_ocl/install/bin
./runvxTestAllScript.sh 16 16 0 ALL HIP ../../build_hip/install/bin
./runvxTestAllScript.sh 16 16 1 ALL OCLvsHIP ../../build_ocl/install/bin ../../build_hip/install/bin
./runvxTestAllScript.sh 16 16 1 ALL OCLvsHIP ../../build_ocl/install/bin ../../build_hip/install/bin 2
```

**NOTE:** OpenVX and the OpenVX logo are trademarks of the Khronos Group Inc.
