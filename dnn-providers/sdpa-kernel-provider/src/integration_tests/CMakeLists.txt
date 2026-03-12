# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

find_package(Threads REQUIRED)

find_package(hipdnn_test_sdk CONFIG REQUIRED)
find_package(hipdnn_plugin_sdk CONFIG REQUIRED)

add_executable(
    sdpa_kernel_plugin_integration_tests
    main.cpp
    IntegrationNoEngines.cpp
)

target_compile_options(sdpa_kernel_plugin_integration_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})

# To use the build tree plugin, we use the path relative to the test binary instead of the installed
# libhipdnn_backend.so HIPDNN_PLUGIN_ENGINE_SUBDIR is provided by the sdk
target_compile_definitions(
    sdpa_kernel_plugin_integration_tests
    PRIVATE
        PLUGIN_PATH="../$<IF:$<PLATFORM_ID:Windows>,bin,lib>/${HIPDNN_PLUGIN_ENGINE_SUBDIR}/sdpa_kernel_plugin"
)

target_link_libraries(
    sdpa_kernel_plugin_integration_tests PUBLIC GTest::gtest hip::host hipdnn_test_sdk hipdnn_plugin_sdk Threads::Threads
)

# Ensure that the sdpa_kernel_plugin is built before the integration tests
add_dependencies(sdpa_kernel_plugin_integration_tests sdpa_kernel_plugin)

clang_tidy_check(sdpa_kernel_plugin_integration_tests)

add_integration_test_target(sdpa_kernel_plugin_integration_tests ${CMAKE_CURRENT_BINARY_DIR})
