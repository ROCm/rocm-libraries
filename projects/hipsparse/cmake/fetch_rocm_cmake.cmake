# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

include(FetchContent)

find_package(ROCmCMakeBuildTools 0.11.0 QUIET CONFIG PATHS ${CMAKE_PREFIX_PATH})
if(NOT ROCmCMakeBuildTools_FOUND)
    find_package(ROCM 0.7.3 QUIET CONFIG PATHS ${CMAKE_PREFIX_PATH}) # deprecated fallback
    if(NOT ROCM_FOUND)
        message(STATUS "ROCmCMakeBuildTools not found. Fetching...")
        set(rocm_cmake_tag "rocm-6.4.0" CACHE STRING "rocm-cmake tag to download")
        FetchContent_Declare(
            rocm-cmake
            GIT_REPOSITORY https://github.com/ROCm/rocm-cmake.git
            GIT_TAG ${rocm_cmake_tag}
            SOURCE_SUBDIR "DISABLE ADDING TO BUILD"
        )
        FetchContent_MakeAvailable(rocm-cmake)
        find_package(ROCmCMakeBuildTools CONFIG REQUIRED NO_DEFAULT_PATH PATHS "${rocm-cmake_SOURCE_DIR}")
    endif()
endif()

include(ROCMSetupVersion)
include(ROCMCreatePackage)
include(ROCMInstallTargets)
include(ROCMPackageConfigHelpers)
include(ROCMInstallSymlinks)
include(ROCMClients)
include(ROCMHeaderWrapper)
