# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

find_package(Threads REQUIRED)

# hipdnn_test_sdk for the LogRecorder + CPU reference convolution.
# Fall back to the installed CONFIG package when the superbuild
# target is not in scope.
if(NOT TARGET hipdnn_test_sdk)
    find_package(hipdnn_test_sdk CONFIG REQUIRED)
endif()

add_executable(
    ck_dsl_provider_integration_tests
    main.cpp
    EndToEndConvDemo.cpp
    EndToEndGemmDemo.cpp
    EndToEndSdpaDemo.cpp
    GemmBLayoutTest.cpp
)

target_link_libraries(
    ck_dsl_provider_integration_tests
    PRIVATE GTest::gtest
            hip::host
            ck_dsl_provider_private
            hipdnn_test_sdk
            hipdnn_plugin_sdk
            Threads::Threads
)

target_include_directories(ck_dsl_provider_integration_tests
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/..
)

include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/Tests.cmake)
add_integration_test_target(ck_dsl_provider_integration_tests ${CMAKE_CURRENT_BINARY_DIR})

# Standalone ctest alias so a `ctest -R ck_dsl_provider_integration`
# filter finds the test even when the provider is built outside the
# superbuild's finalize_test_targets() pipeline.
include(CTest)
if(NOT TEST ck_dsl_provider_integration_tests_direct)
    add_test(NAME ck_dsl_provider_integration_tests_direct
             COMMAND ck_dsl_provider_integration_tests)
endif()
