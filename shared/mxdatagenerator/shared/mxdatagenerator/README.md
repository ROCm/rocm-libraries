# mxDataGenerator

> [!CAUTION]
> The mxDataGenerator repository is retired, please use the [ROCm/rocm-libraries](https://github.com/ROCm/rocm-libraries) repository
>

## AMD's Floating Point Data Generator
This library supports data generation for different floating point formats, as well as conversion instructions between lower precision floating points and single precision floating point.

## Formats Supported
- F32 (E8M23)
- FP16 (E5M2)
- BF16 (E8M7)
- OCP MX-FP8 (E4M3)
- OCP MX-BF8 (E5M2)
- OCP MX-FP6 (E2M3)
- OCP MX-BF6 (E3M2)
- OCP MX-FP4 (E2M1)

## Building the library

### Dependencies
- C++ 20
- Compatible Compilers
    - amdclang
    - g++
    - hipcc
### Quick start instruction

To build mxDataGenerator natively
```
git clone git@github.com:ROCm/mxDataGenerator.git
cd mxDataGenerator
mkdir -p build; cd build;
CXX=[clang++|g++|hipcc] cmake ..
make -j
```
