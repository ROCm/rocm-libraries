# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

file(GLOB_RECURSE HIP_KERNEL_ENGINE_TESTS ${CMAKE_CURRENT_SOURCE_DIR}/hip_kernel_provider/*.cpp)
target_sources(hip_kernel_provider_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HIP_KERNEL_PROVIDER_GENERATED_SOURCES}
)
