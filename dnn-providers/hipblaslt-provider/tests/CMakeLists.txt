# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_compile_definitions(__HIP_PLATFORM_AMD__)
find_package(hip REQUIRED)
find_package(Threads REQUIRED)

if(NOT BUILD_PLUGIN_AS_DEPENDENCY)
    find_package(hipdnn_test_sdk CONFIG REQUIRED)
    find_package(hipdnn_plugin_sdk CONFIG REQUIRED)
endif()

add_executable(
    hipblaslt_plugin_tests
    engines/TestHipblasltEngine.cpp
    main.cpp
    TestHipblasltEngineManager.cpp
    TestHipblasltEnginePluginApi.cpp
    TestHipblasltHandleFactory.cpp
    TestHipblasltHipdnnEnginePluginHandle.cpp
    TestHipblasltHipdnnPluginExecutionContext.cpp
    TestHipblasltPluginApi.cpp
    TestHipblasltUtils.cpp
)

target_compile_options(hipblaslt_plugin_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})

target_link_libraries(
    hipblaslt_plugin_tests PUBLIC GTest::gtest GTest::gmock hip::host
                                  hipblaslt_plugin_private hipdnn_test_sdk hipdnn_plugin_sdk Threads::Threads
)

# make it so tests inside folders can find the mocks folder without having to use a relative path.
target_include_directories(hipblaslt_plugin_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_unit_test_target(hipblaslt_plugin_tests ${CMAKE_CURRENT_BINARY_DIR})
