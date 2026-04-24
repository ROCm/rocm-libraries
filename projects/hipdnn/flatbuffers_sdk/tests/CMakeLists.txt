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
    utilities/TestPointwiseValidation.cpp
)

target_compile_options(hipdnn_flatbuffers_sdk_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS} ${HIPDNN_NO_RTTI_OPTIONS})

# Fetch flatbuffers include dirs explicitly. Although hipdnn_flatbuffers_sdk propagates these
# transitively for compilation, per-file clang-tidy (CXX_CLANG_TIDY) does not pick them up
# automatically. We add them here and again as -extra-arg below so clang-tidy can resolve
# flatbuffers headers.
get_target_property(_fb_include_dirs flatbuffers::flatbuffers INTERFACE_INCLUDE_DIRECTORIES)

target_include_directories(
    hipdnn_flatbuffers_sdk_tests
    PRIVATE
        ${_fb_include_dirs}
)

target_link_libraries(
    hipdnn_flatbuffers_sdk_tests
    PRIVATE
    GTest::gtest
    GTest::gmock
    Threads::Threads
    hipdnn_flatbuffers_sdk
    hipdnn_test_sdk
    flatbuffers::flatbuffers
    hip::host
    ${CMAKE_DL_LIBS}
)

clang_tidy_check(hipdnn_flatbuffers_sdk_tests)

# clang-tidy's CXX_CLANG_TIDY does not receive the HIP extra-args that batch
# run-clang-tidy gets (e.g. -isystem /opt/rocm/include). Flatbuffers headers
# live under that path, so we add them here so clang-tidy can resolve them.
if(ENABLE_CLANG_TIDY AND _fb_include_dirs)
    get_target_property(_tidy_cmd hipdnn_flatbuffers_sdk_tests CXX_CLANG_TIDY)
    if(_tidy_cmd)
        foreach(_include IN LISTS _fb_include_dirs)
            list(APPEND _tidy_cmd "-extra-arg=-isystem" "-extra-arg=${_include}")
        endforeach()
        set_target_properties(hipdnn_flatbuffers_sdk_tests PROPERTIES CXX_CLANG_TIDY "${_tidy_cmd}")
    endif()
endif()

add_unit_test_target(hipdnn_flatbuffers_sdk_tests ${CMAKE_CURRENT_BINARY_DIR})
