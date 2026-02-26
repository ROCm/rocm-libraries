# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

find_package(Threads REQUIRED)

find_package(hipdnn_test_sdk CONFIG REQUIRED)
find_package(hipdnn_plugin_sdk CONFIG REQUIRED)

add_executable(
    hip_kernel_plugin_integration_tests
    main.cpp
    IntegrationNoEngines.cpp
)

target_compile_options(hip_kernel_plugin_integration_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})

target_link_libraries(
    hip_kernel_plugin_integration_tests PUBLIC GTest::gtest hip::host
        hip_kernel_plugin_private hipdnn_test_sdk hipdnn_plugin_sdk Threads::Threads
)

# Ensure that the hip_kernel_plugin is built before the integration tests
add_dependencies(hip_kernel_plugin_integration_tests hip_kernel_plugin)

add_integration_test_target(hip_kernel_plugin_integration_tests ${CMAKE_CURRENT_BINARY_DIR})
