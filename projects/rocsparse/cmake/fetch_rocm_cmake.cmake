# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Fetch and configure ROCm CMake build tools

include(FetchContent)

# Workaround until hip cmake modules fixes symlink logic in config files
list(APPEND CMAKE_PREFIX_PATH /opt/rocm /opt/rocm/hip)

# ROCm CMake package - prefer ROCmCMakeBuildTools (modern), fallback to ROCM (legacy)
find_package(ROCmCMakeBuildTools 0.11.0 CONFIG QUIET PATHS "${ROCM_PATH}")
if(NOT ROCmCMakeBuildTools_FOUND)
    find_package(ROCM 0.7.3 CONFIG QUIET PATHS "${ROCM_PATH}")
    if(NOT ROCM_FOUND)
        message(STATUS "ROCmCMakeBuildTools not found. Fetching from GitHub...")
        set(rocm_cmake_tag "rocm-6.4.0" CACHE STRING "rocm-cmake tag to download")
        FetchContent_Declare(
            rocm-cmake
            GIT_REPOSITORY https://github.com/ROCm/rocm-cmake.git
            GIT_TAG ${rocm_cmake_tag}
            SOURCE_SUBDIR "DISABLE ADDING TO BUILD"
        )
        FetchContent_MakeAvailable(rocm-cmake)
        find_package(ROCmCMakeBuildTools CONFIG REQUIRED NO_DEFAULT_PATH
            PATHS "${rocm-cmake_SOURCE_DIR}"
        )
    endif()
endif()

# Include ROCm CMake modules
include(ROCMSetupVersion)
include(ROCMCreatePackage)
include(ROCMInstallTargets)
include(ROCMPackageConfigHelpers)
include(ROCMInstallSymlinks)
include(ROCMCheckTargetIds)
include(ROCMClients)
include(ROCMHeaderWrapper)

