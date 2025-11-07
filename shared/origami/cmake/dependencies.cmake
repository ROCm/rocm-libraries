# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Dependencies

# Git
find_package(Git REQUIRED)

# Workaround until hcc & hip cmake modules fixes symlink logic in their config files.
# (Thanks to rocBLAS devs for finding workaround for this problem!)
list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)

find_package(ROCmCMakeBuildTools QUIET CONFIG)

if(NOT ROCmCMakeBuildTools_FOUND)
    message(STATUS "ROCmCMakeBuildTools not found. Fetching from source...")
    include(FetchContent)
    FetchContent_Declare(
        rocm-cmake GIT_REPOSITORY https://github.com/ROCm/rocm-cmake.git GIT_TAG rocm-6.4.3
        GIT_SHALLOW TRUE
    )

    FetchContent_GetProperties(rocm-cmake)
    if(NOT rocm-cmake_POPULATED)
        FetchContent_Populate(rocm-cmake)
        message(
            STATUS
                "Added ROCm CMake modules path: ${rocm-cmake_SOURCE_DIR}/share/rocmcmakebuildtools/cmake"
        )
        list(APPEND CMAKE_MODULE_PATH ${rocm-cmake_SOURCE_DIR}/share/rocmcmakebuildtools/cmake)
    endif()
endif()

include(ROCMSetupVersion)
include(ROCMCreatePackage)
include(ROCMInstallTargets)
include(ROCMPackageConfigHelpers)
include(ROCMInstallSymlinks)
include(ROCMCheckTargetIds OPTIONAL)
include(ROCMClients)
include(ROCMHeaderWrapper)
