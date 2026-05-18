# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Embed kernels for this engine
add_subdirectory(kernels)

target_sources(hip_kernel_provider_impl PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/HipKernelEngine.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/batchnorm/BatchnormPlanBuilder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/batchnorm/BatchnormFwdInferencePlan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/batchnorm/BatchnormFwdInferenceWithVariancePlan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/batchnorm/BatchnormFwdTrainingPlan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/batchnorm/BatchnormBwdPlan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/batchnorm/BatchnormApplicabilityChecks.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/batchnorm/BatchnormFwdTrainingPlanBuilder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/RMSnorm/RMSnormPlanBuilder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/RMSnorm/RMSnormFwdPlan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/RMSnorm/RMSnormApplicabilityChecks.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/RMSnorm/RMSnormBwdPlan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/RMSnorm/RMSnormBwdPlanBuilder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/layernorm/LayernormApplicabilityChecks.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/layernorm/LayernormFwdPlan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/layernorm/LayernormPlanBuilder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/layernorm/LayernormUtilities.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/ApplicabilityChecks.cpp
)

target_include_directories(hip_kernel_provider_impl PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}/kernels
)

target_compile_definitions(hip_kernel_provider_impl PUBLIC HIPDNN_ENGINE_HIP_KERNEL=1)
