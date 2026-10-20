#[[
MIT License

Copyright (c) 2026 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
]]

# Find before fetching: TheRock and the rocm-libraries superbuild each supply one shared
# GoogleTest, and fetching a second copy there redefines the `gtest` target and fails the
# configure. The pin below must stay in step with TheRock's third-party/googletest and
# cmake/modules/shared_third_party.cmake.
if(TARGET GTest::gtest)
    message(STATUS "GoogleTest: using the GTest::gtest target already defined by the enclosing build")
    return()
endif()

find_package(GTest QUIET)

if(GTest_FOUND)
    if(GTest_DIR)
        set(_rpp_gtest_origin "${GTest_DIR}")
    elseif(GTEST_INCLUDE_DIRS)
        set(_rpp_gtest_origin "${GTEST_INCLUDE_DIRS}")
    else()
        set(_rpp_gtest_origin "location not reported by find_package")
    endif()
    if(GTest_VERSION)
        set(_rpp_gtest_origin "${GTest_VERSION}, ${_rpp_gtest_origin}")
    endif()
    message(STATUS "GoogleTest: provided by find_package (${_rpp_gtest_origin})")
    unset(_rpp_gtest_origin)
else()
    message(STATUS "GoogleTest: not found by find_package; fetching the pinned v1.17.0")

    include(FetchContent)

    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

    # These postdate this suite's 3.14 floor, so they are opt-in by version.
    set(_rpp_gtest_declare_args "")
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
        list(APPEND _rpp_gtest_declare_args OVERRIDE_FIND_PACKAGE DOWNLOAD_EXTRACT_TIMESTAMP FALSE)
    endif()
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.25)
        list(APPEND _rpp_gtest_declare_args SYSTEM)
    endif()

    FetchContent_Declare(GTest
        URL      https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz
        URL_HASH SHA256=65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c
        ${_rpp_gtest_declare_args})

    # Must be static on Windows: gmock's class-static members lack __declspec(dllexport),
    # so a shared gmock_main.dll fails to link.
    set(_rpp_gtest_old_build_shared_libs ${BUILD_SHARED_LIBS})
    set(BUILD_SHARED_LIBS OFF)
    FetchContent_MakeAvailable(GTest)
    set(BUILD_SHARED_LIBS ${_rpp_gtest_old_build_shared_libs})
    unset(_rpp_gtest_old_build_shared_libs)
    unset(_rpp_gtest_declare_args)

    # googletest exports plain target names; consumers expect the FindGTest-style aliases.
    foreach(_lib gtest gtest_main gmock gmock_main)
        if(TARGET ${_lib} AND NOT TARGET GTest::${_lib})
            add_library(GTest::${_lib} ALIAS ${_lib})
        endif()
    endforeach()

    message(STATUS "GoogleTest: fetched into ${gtest_SOURCE_DIR}")
endif()
