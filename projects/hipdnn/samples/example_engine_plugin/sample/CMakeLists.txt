# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# =============================================================================
# Sample application demonstrating plugin loading, engine selection, and
# correctness verification. Registered as a ctest for automated acceptance
# testing.
#
# This executable links against the hipDNN frontend library and exercises
# all plugin loading modes (ADDITIVE, ABSOLUTE) as well as engine selection
# by name. GPU operations (ReLU and ConvFwd) use HIP device memory via
# hipMalloc/hipMemcpy. Correctness is verified using hardcoded expected values.
#
# No-GPU behavior: returns 0 (success) when no GPU is detected, running
# non-GPU portions only.
# =============================================================================
find_package(hipdnn_frontend CONFIG REQUIRED)
find_package(hip REQUIRED)
find_package(Threads REQUIRED)

add_executable(example_provider_sample ExampleProviderSample.cpp)

target_link_libraries(
    example_provider_sample
    PRIVATE
        hipdnn_frontend
        hip::host
        Threads::Threads
        example_provider_compile_options
)

add_test(NAME example_provider_sample COMMAND example_provider_sample)
