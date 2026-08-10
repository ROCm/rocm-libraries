# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

target_sources(hip_kernel_provider_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/TestPointwiseAddPack.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TestNativeMatchers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TestPointwiseAddDispatchHandler.cpp
)
