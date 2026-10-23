# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Let embedding checks detect this helper even when imported targets are local
# to the component directory. A directory property resets on each configure.
set_property(DIRECTORY "${PROJECT_SOURCE_DIR}"
    PROPERTY HOSTNUMERICS_TEST_DEPENDENCIES_LOADED TRUE)

option(HOSTNUMERICS_ENABLE_FETCH "Allow test builds to fetch missing dependencies."
       ${PROJECT_IS_TOP_LEVEL})

if(TARGET GTest::gtest_main)
    # Source builds record VERSION on the target; installed packages expose
    # GTest_VERSION instead. Do not silently accept an old or unversioned target.
    get_target_property(_hostnumerics_gtest_version GTest::gtest_main VERSION)
    if(NOT _hostnumerics_gtest_version)
        set(_hostnumerics_gtest_version "${GTest_VERSION}")
    endif()
    if(NOT _hostnumerics_gtest_version OR _hostnumerics_gtest_version VERSION_LESS 1.12.1)
        message(FATAL_ERROR
            "HostNumerics tests require GTest 1.12.1 or newer; the existing "
            "GTest::gtest_main target has version '${_hostnumerics_gtest_version}'. "
            "Provide a compatible target with VERSION or GTest_VERSION set.")
    endif()
    unset(_hostnumerics_gtest_version)
    return()
endif()
# Retain the repository's older 1.12.1 compatibility baseline: these assertions
# do not require newer APIs. Fetch the current superbuild pin separately below.
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
