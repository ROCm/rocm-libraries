# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(TARGET GTest::gtest_main)
    return()
endif()
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
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip
        URL_HASH SHA256=24564e3b712d3eb30ac9a85d92f7d720f60cc0173730ac166f27dda7fed76cb2
    )
    FetchContent_MakeAvailable(googletest)
endif()
