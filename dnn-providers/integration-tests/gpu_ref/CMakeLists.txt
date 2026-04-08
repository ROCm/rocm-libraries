# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_compile_definitions(__HIP_PLATFORM_AMD__)
find_package(hiprtc REQUIRED)
find_package(hip REQUIRED)

add_subdirectory(kernels)

add_library(
    hipdnn_gpu_ref STATIC
    src/GpuFpReferenceConvolution.cpp
    src/GpuFpReferenceValidation.cpp
    src/GpuIntReferenceValidation.cpp
    src/GpuReferenceValidationFactory.cpp
    src/GpuRefKernelCompiler.cpp
    src/GpuRefValidatorHelpers.cpp
    ${GPU_REF_KERNEL_GENERATED_SOURCES}
)

target_include_directories(
    hipdnn_gpu_ref
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/kernels/types>
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/kernels
)

target_link_libraries(
    hipdnn_gpu_ref
    PUBLIC hipdnn_test_sdk
    PRIVATE hiprtc::hiprtc hip::host
)

target_compile_options(hipdnn_gpu_ref PRIVATE ${INTEGRATION_TESTS_WARNING_COMPILE_OPTIONS})
