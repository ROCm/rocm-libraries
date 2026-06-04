# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_compile_definitions(__HIP_PLATFORM_AMD__)
find_package(hip REQUIRED)
find_package(Threads REQUIRED)

# hipdnn SDK dependencies - use targets directly if available (superbuild),
# otherwise find installed packages (true standalone)
if(NOT TARGET hipdnn_test_sdk)
    find_package(hipdnn_test_sdk CONFIG REQUIRED)
endif()
if(NOT TARGET hipdnn_plugin_sdk)
    find_package(hipdnn_plugin_sdk CONFIG REQUIRED)
endif()

add_executable(
    miopen_plugin_tests
    engines/plans/TestMiopenBatchnormApplicabilityChecks.cpp
    engines/plans/TestMiopenBatchnormBwdPlan.cpp
    engines/plans/TestMiopenBatchnormFwdInferencePlan.cpp
    engines/plans/TestMiopenBatchnormFwdInferenceWithVariancePlan.cpp
    engines/plans/TestMiopenBatchnormFwdTrainingActivPlan.cpp
    engines/plans/TestMiopenBatchnormFwdTrainingActivPlanBuilder.cpp
    engines/plans/TestMiopenBatchnormFwdTrainingPlan.cpp
    engines/plans/TestMiopenBatchnormFwdTrainingPlanBuilder.cpp
    engines/plans/TestMiopenBatchnormPlanBuilder.cpp
    engines/plans/TestMiopenConvBwdPlan.cpp
    engines/plans/TestMiopenConvFwdBiasActivPlan.cpp
    engines/plans/TestMiopenConvFwdBiasActivPlanBuilder.cpp
    engines/plans/TestMiopenConvFwdPlan.cpp
    engines/plans/TestMiopenConvPlanBuilder.cpp
    engines/plans/TestMiopenConvWrwPlan.cpp
    engines/TestMiopenEngine.cpp
    common/TestWorkarounds.cpp
    main.cpp
    TestMiopenActivationDescriptor.cpp
    TestMiopenConvDescriptor.cpp
    TestMiopenEngineManager.cpp
    TestMiopenHipdnnEnginePluginHandle.cpp
    TestMiopenHipdnnPluginExecutionContext.cpp
    TestMiopenTensor.cpp
    TestMiopenPluginPublic.cpp
    TestMiopenUtils.cpp
)

target_compile_options(miopen_plugin_tests PRIVATE ${MIOPENPROVIDER_WARNING_COMPILE_OPTIONS})

target_link_libraries(
    miopen_plugin_tests PUBLIC GTest::gtest GTest::gmock hip::host
                               miopen_plugin_private hipdnn_test_sdk hipdnn_plugin_sdk Threads::Threads
)

# make it so tests inside folders can find the mocks folder without having to use a relative path.
target_include_directories(miopen_plugin_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

clang_tidy_check(miopen_plugin_tests)

add_unit_test_target(miopen_plugin_tests ${CMAKE_CURRENT_BINARY_DIR})
