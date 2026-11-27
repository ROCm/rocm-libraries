# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/KernelEmbedding.cmake")


# Every operation's kernels live beside the descriptors of the ingestor engine that serves it,
# where the kernel ingestor compiles them; this engine embeds them under their bare names until
# it is retired.
add_kernels_for_embedding(
    TARGET hip_kernel_provider_impl
    FILES
    ${HKP_HIP_MLOPS_INGESTOR_DIR}/batchnorm_training/kernels/BatchNormFwdTrainSpatial.cpp
    ${HKP_HIP_MLOPS_INGESTOR_DIR}/batchnorm_backward/kernels/BatchNormBwdSpatial.cpp
    ${HKP_HIP_MLOPS_INGESTOR_DIR}/include/batchnorm/BatchnormFunctions.hpp
    ${HKP_HIP_MLOPS_INGESTOR_DIR}/include/batchnorm/Configuration.hpp
    ${HKP_HIP_MLOPS_INGESTOR_DIR}/include/batchnorm/ReductionFunctions.hpp
    ${HKP_HIP_MLOPS_INGESTOR_DIR}/include/batchnorm/StaticUnroll.hpp
    ${HKP_HIP_MLOPS_INGESTOR_DIR}/batchnorm_inference/kernels/BatchNormFwdInferSpatial.cpp
    ${HKP_HIP_MLOPS_INGESTOR_DIR}/resample_fwd/kernels/ResampleFwd.cpp
    ${HKP_HIP_MLOPS_INGESTOR_DIR}/resample_bwd/kernels/ResampleBwd.cpp
    ${HKP_HIP_MLOPS_INGESTOR_DIR}/layernorm_bwd/kernels/LayernormBwd.cpp
    ${HKP_HIP_MLOPS_INGESTOR_DIR}/include/layernorm/LayernormCommon.hpp
    ${HKP_HIP_MLOPS_INGESTOR_DIR}/layernorm_fwd/kernels/LayernormFwd.cpp
    ${HKP_HIP_MLOPS_INGESTOR_DIR}/rms_norm_fwd/kernels/RMSNormFwd.cpp
    ${HKP_HIP_MLOPS_INGESTOR_DIR}/rms_norm_bwd/kernels/RMSNormBwd.cpp
    ${HKP_HIP_MLOPS_INGESTOR_DIR}/include/types/VectorTypes.hpp
    ${HKP_HIP_MLOPS_INGESTOR_DIR}/include/common/HipKernelActivation.hpp
    ${HKP_HIP_MLOPS_INGESTOR_DIR}/include/common/HipKernelMath.hpp)
