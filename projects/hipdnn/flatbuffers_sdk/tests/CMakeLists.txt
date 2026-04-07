# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include(GoogleTest)

if(HIPDNN_SKIP_TESTS)
    message(STATUS "Skipping Flatbuffers SDK tests")
    return()
endif()

find_package(Threads REQUIRED)

add_compile_definitions(__HIP_PLATFORM_AMD__)
find_package(hip REQUIRED)

add_executable(
    hipdnn_flatbuffers_sdk_tests
    main.cpp
    TestVersion.cpp
    flatbuffer_utilities/TestEngineConfigWrapper.cpp
    flatbuffer_utilities/TestEngineDetailsWrapper.cpp
    flatbuffer_utilities/TestGraphWrapper.cpp
    flatbuffer_utilities/TestNodeWrapper.cpp
    flatbuffer_utilities/TestKnobWrapper.cpp
    flatbuffer_utilities/TestKnobSettingWrapper.cpp
    flatbuffer_utilities/TestTensorAttributesWrapper.cpp
    utilities/TestJson.cpp
)

target_compile_options(hipdnn_flatbuffers_sdk_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS} ${HIPDNN_NO_RTTI_OPTIONS})

target_include_directories(
    hipdnn_flatbuffers_sdk_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../test_sdk/include
)

target_link_libraries(
    hipdnn_flatbuffers_sdk_tests
    GTest::gtest
    GTest::gmock
    Threads::Threads
    hipdnn_flatbuffers_sdk
    flatbuffers::flatbuffers
    hip::host
    ${CMAKE_DL_LIBS}
)

clang_tidy_check(hipdnn_flatbuffers_sdk_tests)
add_unit_test_target(hipdnn_flatbuffers_sdk_tests ${CMAKE_CURRENT_BINARY_DIR})
