# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Standalone developer tool: runs the GPU SDPA reference kernel on .npy inputs
# and writes the fp32 output to .npy, for offline comparison against AOTriton.
# Off by default; gated by BUILD_SDPA_AOTRITON_HARNESS in the parent CMakeLists.

find_package(hip REQUIRED)
find_package(hiprtc REQUIRED)

add_executable(sdpa_aotriton_ref_driver SdpaRefDriver.cpp)

target_compile_features(sdpa_aotriton_ref_driver PRIVATE cxx_std_17)

target_include_directories(sdpa_aotriton_ref_driver PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(sdpa_aotriton_ref_driver
    PRIVATE
        hipdnn_gpu_ref
        hipdnn_test_sdk
        hiprtc::hiprtc
        hip::host
)

set_target_properties(sdpa_aotriton_ref_driver
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
