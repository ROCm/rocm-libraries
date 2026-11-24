# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/KernelEmbedding.cmake")


# List kernel files to embed
add_kernels_for_embedding(
    TARGET hip_kernel_provider_impl
    FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/batchnorm/BatchNormFwdInferSpatial.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/batchnorm/BatchNormFwdTrainSpatial.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/batchnorm/BatchNormBwdSpatial.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/batchnorm/BatchnormFunctions.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/batchnorm/Configuration.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/batchnorm/ReductionFunctions.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/batchnorm/StaticUnroll.hpp
    # Families migrated to the kernel ingestor keep their sources beside the HipMlops
    # descriptors; this engine embeds them under their bare names until it is retired.
    ${HKP_HIP_MLOPS_INGESTOR_KERNEL_DIR}/resample/ResampleFwd.cpp
    ${HKP_HIP_MLOPS_INGESTOR_KERNEL_DIR}/resample/ResampleBwd.cpp
    ${HKP_HIP_MLOPS_INGESTOR_KERNEL_DIR}/layernorm/LayernormBwd.cpp
    ${HKP_HIP_MLOPS_INGESTOR_KERNEL_DIR}/layernorm/LayernormCommon.hpp
    ${HKP_HIP_MLOPS_INGESTOR_KERNEL_DIR}/layernorm/LayernormFwd.cpp
    ${HKP_HIP_MLOPS_INGESTOR_KERNEL_DIR}/rmsnorm/RMSNormFwd.cpp
    ${HKP_HIP_MLOPS_INGESTOR_KERNEL_DIR}/rmsnorm/RMSNormBwd.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/types/VectorTypes.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/common/HipKernelActivation.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/common/HipKernelMath.hpp)
