# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# The applicability checks and compile options of the families migrated to the kernel
# ingestor. The ingestor's HipMlops packs call them as this engine's plans do, so both
# engines accept the same graphs and compile the same programs.
target_sources(hip_kernel_provider_impl PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/ApplicabilityChecks.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/resample/ResampleApplicabilityChecks.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/resample/ResampleFwdPlan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/resample/ResampleBwdPlan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/layernorm/LayernormApplicabilityChecks.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/layernorm/LayernormFwdPlan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/layernorm/LayernormBwdPlan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/layernorm/LayernormUtilities.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/RMSnorm/RMSnormApplicabilityChecks.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/RMSnorm/RMSnormFwdPlan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/RMSnorm/RMSnormBwdPlan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/batchnorm/BatchnormApplicabilityChecks.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/batchnorm/BatchnormFwdInferencePlan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/batchnorm/BatchnormFwdInferenceWithVariancePlan.cpp
)

if(NOT ENABLE_HIP_MLOPS_ENGINE)
    return()
endif()

# Embed kernels for this engine
add_subdirectory(kernels)

target_sources(hip_kernel_provider_impl PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/HipMlopsEngine.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/batchnorm/BatchnormPlanBuilder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/batchnorm/BatchnormFwdTrainingPlan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/batchnorm/BatchnormBwdPlan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/batchnorm/BatchnormFwdTrainingPlanBuilder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/RMSnorm/RMSnormPlanBuilder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/RMSnorm/RMSnormBwdPlanBuilder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/layernorm/LayernormPlanBuilder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/resample/ResamplePlanBuilder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/resample/ResampleBwdPlanBuilder.cpp
)

target_include_directories(hip_kernel_provider_impl PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}/kernels
)

target_compile_definitions(hip_kernel_provider_impl PUBLIC HIPDNN_ENGINE_HIP_MLOPS=1)
