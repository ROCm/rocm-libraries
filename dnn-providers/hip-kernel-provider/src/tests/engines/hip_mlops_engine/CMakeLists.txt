# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

target_sources(hip_kernel_provider_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/TestHipMlopsEngine.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/Batchnorm/TestBatchnormPlanBuilder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/Batchnorm/TestBatchnormFwdTrainingActivPlanBuilder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/Batchnorm/TestBatchnormFwdInferencePlan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/Batchnorm/TestBatchnormFwdInferenceWithVariancePlan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/Batchnorm/TestBatchnormFwdTrainingPlan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/Batchnorm/TestBatchnormFwdTrainingActivPlan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/Batchnorm/TestBatchnormBwdPlan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/Batchnorm/TestBatchnormCommon.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/Batchnorm/TestBatchnormApplicabilityChecks.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/Layernorm/TestLayernormPlanBuilder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/Layernorm/TestLayernormFwdPlan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/Layernorm/TestLayernormApplicabilityChecks.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/TestPlanUtils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/RMSnorm/TestRMSnormPlanBuilder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/RMSnorm/TestRMSnormFwdPlan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/RMSnorm/TestRMSnormBwdPlanBuilder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/RMSnorm/TestRMSnormBwdPlan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/RMSnorm/TestRMSnormApplicabilityChecks.cpp

)
