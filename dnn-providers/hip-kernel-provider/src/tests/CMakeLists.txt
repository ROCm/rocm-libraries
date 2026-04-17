# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_compile_definitions(__HIP_PLATFORM_AMD__)
find_package(hip REQUIRED)
find_package(Threads REQUIRED)

find_package(hipdnn_test_sdk CONFIG REQUIRED)
find_package(hipdnn_plugin_sdk CONFIG REQUIRED)

add_executable(
    hip_kernel_provider_tests
    main.cpp
    TestHipKernelContainer.cpp
    TestHipKernelHandle.cpp
    TestHipKernelContext.cpp
    TestHipKernelUtils.cpp
    TestHipKernelPluginPublic.cpp
    engines/TestHipKernelEngine.cpp
    engines/plans/Batchnorm/TestBatchnormPlanBuilder.cpp
    engines/plans/Batchnorm/TestBatchnormFwdInferencePlan.cpp
    engines/plans/Batchnorm/TestBatchnormFwdInferenceWithVariancePlan.cpp
    engines/plans/Batchnorm/TestBatchnormFwdTrainingPlan.cpp
    engines/plans/Batchnorm/TestBatchnormCommon.cpp
    engines/plans/Batchnorm/TestBatchnormApplicabilityChecks.cpp
    engines/plans/Layernorm/TestLayernormPlanBuilder.cpp
    engines/plans/Layernorm/TestLayernormFwdPlan.cpp
    engines/plans/Layernorm/TestLayernormApplicabilityChecks.cpp
    engines/plans/TestPlanUtils.cpp
    engines/plans/RMSnorm/TestRMSnormPlanBuilder.cpp
    engines/plans/RMSnorm/TestRMSnormFwdPlan.cpp
    engines/plans/RMSnorm/TestRMSnormBwdPlanBuilder.cpp
    engines/plans/RMSnorm/TestRMSnormBwdPlan.cpp
    engines/plans/RMSnorm/TestRMSnormApplicabilityChecks.cpp
    hip/TestHipProgramAndKernel.cpp
)

target_compile_options(hip_kernel_provider_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})

target_link_libraries(
    hip_kernel_provider_tests PUBLIC GTest::gtest GTest::gmock hip::host
                                   hip_kernel_provider_private hipdnn_test_sdk hipdnn_plugin_sdk Threads::Threads
)

# make it so tests inside folders can find files without having to use a relative path.
target_include_directories(hip_kernel_provider_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_LIST_DIR}/../../include)

if(ENABLE_ASM_SDPA_ENGINE)
    add_subdirectory(engines/asm_sdpa_engine)
endif()

clang_tidy_check(hip_kernel_provider_tests)

add_unit_test_target(hip_kernel_provider_tests ${CMAKE_CURRENT_BINARY_DIR})
