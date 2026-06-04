# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/KernelEmbedding.cmake")


# List kernel files to embed
add_kernels_for_embedding(${CMAKE_CURRENT_SOURCE_DIR}/batchnorm/BatchNormFwdInferSpatial.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/batchnorm/BatchNormFwdTrainSpatial.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/batchnorm/BatchNormBwdSpatial.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/batchnorm/BatchnormFunctions.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/batchnorm/Configuration.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/batchnorm/ReductionFunctions.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/batchnorm/StaticUnroll.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rmsnorm/RMSNormCommon.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rmsnorm/RMSNormFwd.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rmsnorm/RMSNormBwd.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/layernorm/LayernormFwd.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/types/FloatTypes.h
    ${CMAKE_CURRENT_SOURCE_DIR}/types/VectorTypes.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/types/Bfloat16Dev.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/common/HipKernelActivation.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/common/HipKernelMath.hpp)
