# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include(GoogleTest)

if(HIPDNN_SKIP_TESTS)
    message(STATUS "Skipping Flatbuffers SDK tests")
    return()
endif()

find_package(Threads REQUIRED)

add_executable(hipdnn_flatbuffers_sdk_tests
    main.cpp
    TestVersion.cpp
)

target_compile_options(hipdnn_flatbuffers_sdk_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})

target_link_libraries(
    hipdnn_flatbuffers_sdk_tests
    GTest::gtest
    GTest::gmock
    Threads::Threads
    hipdnn_flatbuffers_sdk
    hipdnn_test_sdk
)

clang_tidy_check(hipdnn_flatbuffers_sdk_tests)
add_unit_test_target(hipdnn_flatbuffers_sdk_tests ${CMAKE_CURRENT_BINARY_DIR})
