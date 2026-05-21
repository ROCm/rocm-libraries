# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

find_package(Threads REQUIRED)

# hipdnn_test_sdk supplies the shared log-recording test-harness setup
# that other providers' unit suites use. Fall back to the installed
# CONFIG package when the superbuild target is not in scope.
if(NOT TARGET hipdnn_test_sdk)
    find_package(hipdnn_test_sdk CONFIG REQUIRED)
endif()

add_executable(
    ck_dsl_provider_unit_tests
    main.cpp
    CompileServiceBridgeTest.cpp
    EmbeddedInterpreterTest.cpp
    ConvImplicitGemmAdapterTest.cpp
    ConvImplicitGemmPlanBuilderTest.cpp
    GraphSignatureTest.cpp
    HipModuleSmokeTest.cpp
    JitCacheTest.cpp
    PerfMeasurementTest.cpp
)

target_link_libraries(
    ck_dsl_provider_unit_tests
    PRIVATE GTest::gtest
            GTest::gmock
            hip::host
            ck_dsl_provider_private
            hipdnn_test_sdk
            hipdnn_plugin_sdk
            Python3::Python
            pybind11::embed
            Threads::Threads
)

target_include_directories(ck_dsl_provider_unit_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/Tests.cmake)
add_unit_test_target(ck_dsl_provider_unit_tests ${CMAKE_CURRENT_BINARY_DIR})

# Register the test directly with ctest under a known name so the
# orchestrator's `ctest -R ck_dsl_provider` filter finds it even when
# the provider is built outside the superbuild's finalize_test_targets()
# pipeline. Harmless when the standalone alias also fires.
include(CTest)
if(NOT TEST ck_dsl_provider_unit_tests_direct)
    add_test(NAME ck_dsl_provider_unit_tests_direct
             COMMAND ck_dsl_provider_unit_tests)
endif()
