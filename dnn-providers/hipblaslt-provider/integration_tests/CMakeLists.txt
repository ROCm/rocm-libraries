# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

find_package(Threads REQUIRED)

# SDK dependencies - use targets directly in superbuild, find_package otherwise
if(NOT TARGET hipdnn_test_sdk)
    find_package(hipdnn_test_sdk CONFIG REQUIRED)
endif()

if(NOT TARGET hipdnn_plugin_sdk)
    find_package(hipdnn_plugin_sdk CONFIG REQUIRED)
endif()

add_executable(
    hipblaslt_plugin_integration_tests
    main.cpp
    IntegrationGpuMatmul.cpp
    IntegrationGpuMatmulActiv.cpp
    IntegrationGpuMatmulBias.cpp
    IntegrationGpuMatmulBiasActiv.cpp
)

target_compile_options(hipblaslt_plugin_integration_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})

# To use the build tree plugin, we use the path relative to the test binary instead of the installed
# libhipdnn_backend.so HIPDNN_PLUGIN_ENGINE_SUBDIR is provided by the sdk
target_compile_definitions(
    hipblaslt_plugin_integration_tests
    PRIVATE
        PLUGIN_PATH="../$<IF:$<PLATFORM_ID:Windows>,bin,lib>/${HIPDNN_PLUGIN_ENGINE_SUBDIR}/hipblaslt_plugin"
)

target_link_libraries(
    hipblaslt_plugin_integration_tests PUBLIC GTest::gtest hip::host hipdnn_test_sdk hipdnn_plugin_sdk Threads::Threads
)

# Ensure that the hipblaslt_plugin is built before the hipblaslt_plugin_integration_tests, as the
# integration tests depend on the hipBLASLt Provider Plugin
add_dependencies(hipblaslt_plugin_integration_tests hipblaslt_plugin)

add_integration_test_target(hipblaslt_plugin_integration_tests ${CMAKE_CURRENT_BINARY_DIR} LABELS slow)
