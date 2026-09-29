# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/KernelEmbedding.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/KernelTidy.cmake")

# Kernel files to embed. Declared once: add_kernel_tidy_target() below analyses exactly
# what add_kernels_for_embedding() inlines, so a kernel cannot be embedded without also
# being checked.
set(HIP_MLOPS_KERNEL_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/batchnorm/BatchNormFwdInferSpatial.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/batchnorm/BatchNormFwdTrainSpatial.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/batchnorm/BatchNormBwdSpatial.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/batchnorm/BatchnormFunctions.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/batchnorm/Configuration.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/batchnorm/ReductionFunctions.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/batchnorm/StaticUnroll.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rmsnorm/RMSNormFwd.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rmsnorm/RMSNormBwd.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/resample/ResampleFwd.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/resample/ResampleBwd.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/layernorm/LayernormBwd.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/layernorm/LayernormCommon.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/layernorm/LayernormFwd.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/types/VectorTypes.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/common/HipKernelActivation.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/common/HipKernelMath.hpp)

# List kernel files to embed
add_kernels_for_embedding(
    TARGET hip_kernel_provider_impl
    FILES ${HIP_MLOPS_KERNEL_FILES})

# Embedded kernels are never compiled by the build, so nothing would otherwise put them in
# front of clang-tidy. This target does: it runs clang-tidy over them directly, with the
# macros the host plan supplies at hipRTC time force-included from the prelude.
add_kernel_tidy_target(
    NAME hip_mlops_kernel_tidy
    PRELUDE ${CMAKE_CURRENT_SOURCE_DIR}/tidy/KernelTidyPrelude.hpp
    FILES ${HIP_MLOPS_KERNEL_FILES})
