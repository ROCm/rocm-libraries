# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_compile_definitions(__HIP_PLATFORM_AMD__)
find_package(hiprtc REQUIRED)
find_package(hip REQUIRED)

add_subdirectory(kernels)

# Join with pipe separator so VERBATIM can pass the value through the shell
# without semicolons being misinterpreted.
list(JOIN GPU_REF_KERNEL_FILES "|" GPU_REF_KERNEL_FILES_PIPE)

# Generate embedded kernel sources at build time
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/kernels/GpuRefKernelSources.cpp
           ${CMAKE_CURRENT_BINARY_DIR}/kernels/GpuRefKernelSources.hpp
    COMMAND ${CMAKE_COMMAND}
        "-DKERNEL_FILES=${GPU_REF_KERNEL_FILES_PIPE}"
        "-DTEMPLATE_DIR=${GPU_REF_KERNEL_TEMPLATE_DIR}"
        "-DOUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/kernels"
        -P "${GPU_REF_KERNEL_EMBED_SCRIPT}"
    DEPENDS ${GPU_REF_KERNEL_FILES}
            ${GPU_REF_KERNEL_TEMPLATE_DIR}/GpuRefKernelSources.hpp.in
            ${GPU_REF_KERNEL_TEMPLATE_DIR}/GpuRefKernelSources.cpp.in
            ${GPU_REF_KERNEL_EMBED_SCRIPT}
    COMMENT "Embedding GPU reference kernel sources"
    VERBATIM
)

add_library(
    hipdnn_gpu_ref STATIC
    src/GpuFpReferenceConvolution.cpp
    src/GpuFpReferenceValidation.cpp
    src/GpuIntReferenceValidation.cpp
    src/GpuReferenceValidationFactory.cpp
    src/GpuRefKernelCompiler.cpp
    src/GpuRefValidatorHelpers.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/kernels/GpuRefKernelSources.cpp
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
