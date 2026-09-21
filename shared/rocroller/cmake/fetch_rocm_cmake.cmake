# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Attempt to find ROCmCMakeBuildTools in the system, if not found, fetch from source. To install
# ROCmCMakeBuildTools from source, see: https://github.com/ROCm/rocm-cmake

find_package(ROCmCMakeBuildTools QUIET CONFIG)

if(NOT ROCmCMakeBuildTools_FOUND)
    message(STATUS "ROCmCMakeBuildTools not found. Fetching from source...")
    include(FetchContent)
    fetchcontent_declare(
        rocm-cmake
        GIT_REPOSITORY https://github.com/ROCm/rocm-cmake.git
        GIT_TAG c01b4f1fd36a94d26c76e7f617b57577b3b84275
        GIT_SHALLOW TRUE
    )

    fetchcontent_getproperties(rocm-cmake)
    if(NOT rocm-cmake_POPULATED)
        fetchcontent_populate(rocm-cmake)
        message(
            STATUS
                "Added ROCm CMake modules path: ${rocm-cmake_SOURCE_DIR}/share/rocmcmakebuildtools/cmake"
        )
        list(APPEND CMAKE_MODULE_PATH ${rocm-cmake_SOURCE_DIR}/share/rocmcmakebuildtools/cmake)
    endif()
endif()

include(ROCMInstallTargets)
include(ROCMSetupVersion)
