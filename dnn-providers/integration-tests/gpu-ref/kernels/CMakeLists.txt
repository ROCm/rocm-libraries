# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Kernel files to embed as C++ string literals at build time.
set(GPU_REF_KERNEL_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/convolution/GpuRefConvFwd.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/convolution/GpuRefConvBwd.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/convolution/GpuRefConvWrw.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/layernorm/GpuRefLayernormBwd.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/layernorm/GpuRefLayernormFwd.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sdpa/GpuRefSdpaFwd.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rmsnorm/GpuRefRMSNormFwd.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rmsnorm/GpuRefRMSNormBwd.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/batchnorm/GpuRefBatchnormFwdInf.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pointwise/GpuRefPointwiseUnary.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pointwise/GpuRefPointwiseBinary.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/validator/GpuRefValidator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/common/GpuRefScaleUniform.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/types/GpuRefConvArgs.h
    ${CMAKE_CURRENT_SOURCE_DIR}/types/GpuRefLayernormArgs.h
    ${CMAKE_CURRENT_SOURCE_DIR}/types/GpuRefSdpaArgs.h
    ${CMAKE_CURRENT_SOURCE_DIR}/types/GpuRefRMSNormArgs.h
    ${CMAKE_CURRENT_SOURCE_DIR}/types/GpuRefBatchnormArgs.h
    ${CMAKE_CURRENT_SOURCE_DIR}/types/GpuRefPointwiseArgs.h
    ${CMAKE_CURRENT_SOURCE_DIR}/types/GpuRefCommonArgs.h
    ${CMAKE_CURRENT_SOURCE_DIR}/types/GpuRefValidatorArgs.h
    ${CMAKE_CURRENT_SOURCE_DIR}/types/GpuRefTypes.h
    PARENT_SCOPE
)

set(GPU_REF_KERNEL_TEMPLATE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/templates PARENT_SCOPE)
set(GPU_REF_KERNEL_EMBED_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedKernelSources.cmake PARENT_SCOPE)
