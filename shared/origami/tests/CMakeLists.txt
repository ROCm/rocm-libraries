# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

find_package(Catch2 3 QUIET CONFIG)
if(NOT Catch2_FOUND)
    if(ORIGAMI_ENABLE_FETCH)
        include(FetchContent)
        FetchContent_Declare(
            Catch2
            GIT_REPOSITORY https://github.com/catchorg/Catch2.git
            GIT_TAG devel
        )
        FetchContent_MakeAvailable(Catch2)
    else()
        message(FATAL_ERROR "Failed to find Catch2")
    endif()
endif()

add_executable(origami-tests)

target_sources(origami-tests
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/test_gemm.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/test_origami.cpp"
        # "${CMAKE_CURRENT_SOURCE_DIR}/test_variance_issue.cpp"
)

target_include_directories(origami-tests
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_link_libraries(origami-tests
    PRIVATE
        roc::origami
        Catch2::Catch2WithMain
)

include(CTest)
include(Catch)
catch_discover_tests(origami-tests)
