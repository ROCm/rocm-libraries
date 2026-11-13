# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_compile_definitions(__HIP_PLATFORM_AMD__)
find_package(hip REQUIRED)
find_package(Threads REQUIRED)

# SDK dependencies - use targets directly in superbuild, find_package otherwise
if(NOT TARGET hipdnn_test_sdk)
    find_package(hipdnn_test_sdk CONFIG REQUIRED)
endif()

if(NOT TARGET hipdnn_plugin_sdk)
    find_package(hipdnn_plugin_sdk CONFIG REQUIRED)
endif()

add_executable(
    hipblaslt_plugin_tests
    engines/TestPlanBuilderEngine.cpp
    main.cpp
    TestHipblasltEngineManager.cpp
    TestHipblasltHandleFactory.cpp
    TestHipblasltHipdnnEnginePluginHandle.cpp
    TestHipblasltHipdnnPluginExecutionContext.cpp
    TestHipblasltPluginApi.cpp
    engines/rocblas/TestMatmulProblem.cpp
    $<$<BOOL:${HIPBLASLTPROVIDER_ENABLE_ROCBLAS}>:engines/rocblas/TestGpuRocblasMatmulPlan.cpp>
    $<$<BOOL:${HIPBLASLTPROVIDER_ENABLE_CONV}>:engines/conv/TestConvGemmFormulation.cpp>
    $<$<BOOL:${HIPBLASLTPROVIDER_ENABLE_CONV}>:engines/conv/TestConvRepackFormulation.cpp>
    $<$<BOOL:${HIPBLASLTPROVIDER_ENABLE_CONV}>:engines/conv/TestConvReducedPrecision.cpp>
    $<$<BOOL:${HIPBLASLTPROVIDER_ENABLE_CONV}>:engines/conv/TestConvRepackLaunch.cpp>
    $<$<BOOL:${HIPBLASLTPROVIDER_ENABLE_CONV}>:engines/conv/TestGpuRepackKernels.cpp>
    $<$<BOOL:${HIPBLASLTPROVIDER_ENABLE_CONV}>:engines/conv/TestGpuConvGemm.cpp>
)

# The hipBLASLt engine's own tests, and the plugin API cases that drive it.
if(HIPBLASLTPROVIDER_ENABLE_HIPBLASLT)
    target_sources(hipblaslt_plugin_tests PRIVATE
        engines/plans/TestHipblasltMatmulPlan.cpp
        engines/plans/TestHipblasltMatmulPlanBuilder.cpp
        $<$<BOOL:${HIPBLASLTPROVIDER_ENABLE_MX_GEMM}>:engines/plans/TestHipblasltMxMatmulPlan.cpp>
        $<$<BOOL:${HIPBLASLTPROVIDER_ENABLE_MX_GEMM}>:engines/plans/TestHipblasltMxMatmulPlanBuilder.cpp>
        TestHipblasltEnginePluginApi.cpp
        TestHipblasltMatmulDesc.cpp
        TestHipblasltMatrixLayout.cpp
        TestHipblasltMatrixTransformDesc.cpp
        TestHipblasltUtils.cpp
    )
endif()

target_compile_options(hipblaslt_plugin_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})

target_link_libraries(
    hipblaslt_plugin_tests PUBLIC GTest::gtest GTest::gmock hip::host
                                  hipblaslt_plugin_private hipdnn_test_sdk hipdnn_plugin_sdk Threads::Threads
)

# make it so tests inside folders can find the mocks folder without having to use a relative path.
target_include_directories(hipblaslt_plugin_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

clang_tidy_check(hipblaslt_plugin_tests)

add_unit_test_target(hipblaslt_plugin_tests ${CMAKE_CURRENT_BINARY_DIR})
