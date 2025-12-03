# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include(GoogleTest)

if(HIP_DNN_SKIP_TESTS)
    message(STATUS "Skipping plugin_sdk tests")
    return()
endif()

find_package(Threads REQUIRED)

add_executable(
    hipdnn_plugin_sdk_tests
    main.cpp
    TestPluginException.cpp
    TestPluginHelpers.cpp
    TestPluginLastErrorManager.cpp
    TestPluginDataTypeHelpers.cpp
    TestNodeWrapper.cpp
    TestGraphWrapper.cpp
    TestEngineConfigWrapper.cpp
    TestEngineDetailsWrapper.cpp
)

target_compile_options(hipdnn_plugin_sdk_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
target_link_libraries(
    hipdnn_plugin_sdk_tests GTest::gtest GTest::gmock Threads::Threads hipdnn_plugin_sdk
)

clang_tidy_check(hipdnn_plugin_sdk_tests)
add_unit_test_target(hipdnn_plugin_sdk_tests ${CMAKE_CURRENT_BINARY_DIR})
