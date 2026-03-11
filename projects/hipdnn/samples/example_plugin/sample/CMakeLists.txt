# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# =============================================================================
# Sample application demonstrating plugin loading and engine selection.
#
# This executable links against the hipDNN frontend library and exercises
# all plugin loading modes (environment variable, ADDITIVE, ABSOLUTE) as
# well as engine selection by name.  GPU operations (ReLU and ConvFwd)
# use HIP device memory via hipMalloc/hipMemcpy.
# =============================================================================
list(APPEND CMAKE_PREFIX_PATH /opt/rocm /opt/rocm/hip)

find_package(hipdnn_frontend CONFIG REQUIRED)
find_package(hip REQUIRED)

add_executable(example_plugin_sample ExamplePluginSample.cpp)

target_link_libraries(
    example_plugin_sample
    PRIVATE
        hipdnn_frontend
        hip::host
)
