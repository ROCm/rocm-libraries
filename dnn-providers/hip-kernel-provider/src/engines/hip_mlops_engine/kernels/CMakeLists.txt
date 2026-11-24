# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/KernelEmbedding.cmake")


# Every family's kernels live beside the HipMlops descriptors, where the kernel ingestor
# compiles them; this engine embeds them under their bare names until it is retired.
add_kernels_for_embedding(
    TARGET hip_kernel_provider_impl
    FILES
    ${HKP_HIP_MLOPS_INGESTOR_KERNEL_DIR}/batchnorm/BatchNormFwdTrainSpatial.cpp
    ${HKP_HIP_MLOPS_INGESTOR_KERNEL_DIR}/batchnorm/BatchNormBwdSpatial.cpp
    ${HKP_HIP_MLOPS_INGESTOR_KERNEL_DIR}/batchnorm/BatchnormFunctions.hpp
    ${HKP_HIP_MLOPS_INGESTOR_KERNEL_DIR}/batchnorm/Configuration.hpp
    ${HKP_HIP_MLOPS_INGESTOR_KERNEL_DIR}/batchnorm/ReductionFunctions.hpp
    ${HKP_HIP_MLOPS_INGESTOR_KERNEL_DIR}/batchnorm/StaticUnroll.hpp
    ${HKP_HIP_MLOPS_INGESTOR_KERNEL_DIR}/batchnorm/BatchNormFwdInferSpatial.cpp
    ${HKP_HIP_MLOPS_INGESTOR_KERNEL_DIR}/resample/ResampleFwd.cpp
    ${HKP_HIP_MLOPS_INGESTOR_KERNEL_DIR}/resample/ResampleBwd.cpp
    ${HKP_HIP_MLOPS_INGESTOR_KERNEL_DIR}/layernorm/LayernormBwd.cpp
    ${HKP_HIP_MLOPS_INGESTOR_KERNEL_DIR}/layernorm/LayernormCommon.hpp
    ${HKP_HIP_MLOPS_INGESTOR_KERNEL_DIR}/layernorm/LayernormFwd.cpp
    ${HKP_HIP_MLOPS_INGESTOR_KERNEL_DIR}/rmsnorm/RMSNormFwd.cpp
    ${HKP_HIP_MLOPS_INGESTOR_KERNEL_DIR}/rmsnorm/RMSNormBwd.cpp
    ${HKP_HIP_MLOPS_INGESTOR_KERNEL_DIR}/types/VectorTypes.hpp
    ${HKP_HIP_MLOPS_INGESTOR_KERNEL_DIR}/common/HipKernelActivation.hpp
    ${HKP_HIP_MLOPS_INGESTOR_KERNEL_DIR}/common/HipKernelMath.hpp)
