# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/KernelEmbedding.cmake")

# List kernel files to embed
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

add_kernels_for_embedding(
    TARGET hip_kernel_provider_impl
    FILES ${HIP_MLOPS_KERNEL_FILES})

# Compile embedded kernels in an unlinked object library with clang-tidy enabled.
add_kernel_tidy_target(
    NAME hip_mlops_kernel_tidy
    PRELUDE ${CMAKE_CURRENT_SOURCE_DIR}/tidy/KernelTidyDefaults.hpp
    FILES ${HIP_MLOPS_KERNEL_FILES})

# The batchnorm backward kernel has two independent compile-time branches. The default
# target above checks variant 1 with saved statistics; compile the other combinations too
# so clang-tidy and the compiler see every supported path. Headers only contribute include
# directories, so pass them all to match hipRTC's flat view of the embedded files.
set(_hip_mlops_kernel_headers ${HIP_MLOPS_KERNEL_FILES})
list(FILTER _hip_mlops_kernel_headers INCLUDE REGEX "\\.hpp$")
foreach(_bn_variant IN ITEMS 1 2)
    foreach(_bn_usesaved IN ITEMS 0 1)
        if(_bn_variant STREQUAL "1" AND _bn_usesaved STREQUAL "1")
            continue()
        endif()
        add_kernel_tidy_target(
            NAME hip_mlops_batchnorm_bwd_tidy_v${_bn_variant}_saved${_bn_usesaved}
            PRELUDE ${CMAKE_CURRENT_SOURCE_DIR}/tidy/KernelTidyDefaults.hpp
            FILES ${CMAKE_CURRENT_SOURCE_DIR}/batchnorm/BatchNormBwdSpatial.cpp
                  ${_hip_mlops_kernel_headers}
            COMPILE_DEFINITIONS
                HIP_PLUGIN_BN_VARIANT=${_bn_variant}
                HIP_PLUGIN_BN_USESAVED=${_bn_usesaved})
    endforeach()
endforeach()
