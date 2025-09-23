# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

if(NOT BUILD_PLUGIN_AS_DEPENDENCY)
    find_package(GTest CONFIG REQUIRED)
endif()

include(GoogleTest)
find_package(Threads REQUIRED)


if(HIP_DNN_SKIP_TESTS)
    message(STATUS "Skipping MIOpen plugin integration tests.")
    return()
endif()

add_executable(miopen_plugin_integration_test
    main.cpp
    IntegrationGpuBatchnormBackward.cpp
    IntegrationGpuBatchnormForwardInference.cpp
    IntegrationGpuConvForward.cpp
)

target_compile_options(miopen_plugin_integration_test PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
target_compile_definitions(miopen_plugin_integration_test PRIVATE
    PLUGIN_PATH="${MIOPEN_LEGACY_PLUGIN_PATH}"
)

target_link_libraries(miopen_plugin_integration_test PUBLIC 
    GTest::gtest
    hip::host
    miopen_legacy_plugin
    hipdnn_frontend
    hipdnn_sdk
    Threads::Threads
)


if(BUILD_PLUGIN_AS_DEPENDENCY)
    clang_tidy_check(miopen_plugin_integration_test)
    
    add_integration_test_target(miopen_plugin_integration_test ${CMAKE_CURRENT_BINARY_DIR})
endif()