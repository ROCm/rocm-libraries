# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

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
    miopen_plugin_integration_tests
    main.cpp
    IntegrationGpuBatchnormUnsupportedDataTypes.cpp
    IntegrationGpuBatchnormTrainingUnhappySpatial.cpp
    IntegrationGpuBatchnormUnhappyActivation.cpp
    IntegrationGpuBatchnormUnhappyLayouts.cpp
    IntegrationGpuBenchmarkingKnob.cpp
    IntegrationGpuDeterministic.cpp
)

target_compile_options(miopen_plugin_integration_tests PRIVATE ${MIOPENPROVIDER_WARNING_COMPILE_OPTIONS})

# To use the build tree plugin, we use the path relative to the test binary instead of the installed
# libhipdnn_backend.so HIPDNN_PLUGIN_ENGINE_SUBDIR is provided by the sdk
target_compile_definitions(
    miopen_plugin_integration_tests
    PRIVATE
        PLUGIN_PATH="../$<IF:$<PLATFORM_ID:Windows>,bin,lib>/${HIPDNN_PLUGIN_ENGINE_SUBDIR}/miopen_plugin"
)

target_link_libraries(
    miopen_plugin_integration_tests PUBLIC GTest::gtest hip::host hipdnn_test_sdk hipdnn_plugin_sdk Threads::Threads
)

# Ensure that the miopen_plugin is built before the integration tests
add_dependencies(miopen_plugin_integration_tests miopen_plugin)

clang_tidy_check(miopen_plugin_integration_tests)

add_integration_test_target(miopen_plugin_integration_tests ${CMAKE_CURRENT_BINARY_DIR} LABELS slow)
