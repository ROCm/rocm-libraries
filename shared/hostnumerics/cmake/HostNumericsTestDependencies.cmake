# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

option(HOSTNUMERICS_ENABLE_FETCH "Allow test builds to fetch missing dependencies."
       ${PROJECT_IS_TOP_LEVEL})

if(TARGET GTest::gtest_main)
    return()
endif()
# Basic assertions work with older installed packages too.
find_package(GTest 1.12.1 QUIET)
if(NOT GTest_FOUND)
    if(NOT HOSTNUMERICS_ENABLE_FETCH)
        message(FATAL_ERROR "HostNumerics tests require GTest 1.12.1 or newer.")
    endif()
    include(FetchContent)
    # GoogleTest belongs to the test build, never the installed component.
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    # Match the parent runtime on MSVC, independently of static/shared libraries.
    if(MSVC AND (NOT CMAKE_MSVC_RUNTIME_LIBRARY OR CMAKE_MSVC_RUNTIME_LIBRARY MATCHES "DLL"))
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    endif()
    # Match the pin in cmake/modules/shared_third_party.cmake in the superbuild.
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz
        URL_HASH SHA256=65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c
        SYSTEM
    )
    FetchContent_MakeAvailable(googletest)
endif()
