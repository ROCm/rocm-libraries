# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include(GoogleTest)

if(HIPDNN_SKIP_TESTS)
    message(STATUS "Skipping Plugin SDK tests")
    return()
endif()

add_compile_definitions(__HIP_PLATFORM_AMD__)
find_package(hip REQUIRED)
find_package(Threads REQUIRED)

add_executable(hipdnn_plugin_sdk_tests
    main.cpp
    TestPluginDataTypeHelpers.cpp
    TestPluginException.cpp
    TestPluginHelpers.cpp
    TestPluginLastErrorManager.cpp
    TestKnobFactory.cpp
    TestEngineManager.cpp
    TestEnginePluginContainer.cpp
    TestKnobSettingFactory.cpp
    TestSdkLogging.cpp
    TestVersion.cpp
)

target_compile_options(hipdnn_plugin_sdk_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS} ${HIPDNN_NO_RTTI_OPTIONS})

target_link_libraries(
    hipdnn_plugin_sdk_tests
    GTest::gtest
    GTest::gmock
    Threads::Threads
    hipdnn_plugin_sdk
    hipdnn_test_sdk
    hip::host
)

clang_tidy_check(hipdnn_plugin_sdk_tests)
add_unit_test_target(hipdnn_plugin_sdk_tests ${CMAKE_CURRENT_BINARY_DIR})
