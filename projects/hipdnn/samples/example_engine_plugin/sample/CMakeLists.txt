# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# =============================================================================
# Sample application demonstrating plugin loading, engine selection, and
# correctness verification. Registered as a ctest for automated acceptance
# testing.
#
# This executable links against the hipDNN frontend library and exercises
# all plugin loading modes (ADDITIVE, ABSOLUTE) as well as engine selection
# by name. GPU operations (ReLU and ConvFwd) use HIP device memory via
# hipMalloc/hipMemcpy. Correctness is verified using both hardcoded expected
# values and a CPU reference convolution (via hipdnn_test_sdk).
#
# No-GPU behavior: returns 0 (success) when no GPU is detected, running
# non-GPU portions only.
# =============================================================================
list(APPEND CMAKE_PREFIX_PATH /opt/rocm /opt/rocm/hip)

find_package(hipdnn_frontend CONFIG REQUIRED)
find_package(hipdnn_test_sdk CONFIG REQUIRED)
find_package(hip REQUIRED)

add_executable(example_provider_sample ExampleProviderSample.cpp)

target_link_libraries(
    example_provider_sample
    PRIVATE
        hipdnn_frontend
        hipdnn_test_sdk
        hip::host
        example_provider_compile_options
)

add_test(NAME example_provider_sample COMMAND example_provider_sample)
