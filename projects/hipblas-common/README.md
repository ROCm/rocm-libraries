# hipBLAS-common

## Quick Start Guide

hipBLAS-common is a header-only library that provides common files for hipBLAS and hipBLASLt.

This section describes how to configure and build the hipBLAS-common project.

### Configure and build

hipBLAS-common provides modern CMake support and relies on native CMake functionality, with the exception of
some project specific options. As such, users are advised to consult the CMake documentation for
general usage questions. Below are usage examples to get started. For details on all configuration
options, see the options section.

```
cd hipBLAS-common
CXX=/opt/rocm/bin/amdclang++             \
cmake -B build                           \
      -S .                               \
      -D CMAKE_INSTALL_PREFIX=<preferred installation path>
cmake --build build --target install
```

### CMake targets

* `roc::hipblas-common`

### Using rmake script

```
 python3 ./rmake.py --install
```
