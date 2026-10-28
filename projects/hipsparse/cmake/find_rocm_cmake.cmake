# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

option(ALLOW_FETCH_DEPS "Download rocm-cmake and GoogleTest if they are not found in the build environment" OFF)
include(FetchContent)
if(NOT ALLOW_FETCH_DEPS)
    set(FETCHCONTENT_FULLY_DISCONNECTED ON)
endif()

find_package(ROCmCMakeBuildTools 0.11.0 CONFIG QUIET PATHS ${CMAKE_PREFIX_PATH} ${ROCM_PATH} /opt/rocm)
if(NOT ROCmCMakeBuildTools_FOUND)
    if(NOT ALLOW_FETCH_DEPS)
        message(FATAL_ERROR "ROCmCMakeBuildTools >= 0.11.0 not found. Install the rocm-cmake package and pass its "
                            "install prefix with -DROCM_PATH or CMAKE_PREFIX_PATH, set ROCmCMakeBuildTools_DIR to "
                            "its share/rocmcmakebuildtools/cmake directory, or configure with -DALLOW_FETCH_DEPS=ON "
                            "to download it.")
    endif()
    message(STATUS "ROCmCMakeBuildTools not found. Fetching rocm-cmake because ALLOW_FETCH_DEPS is ON")
    FetchContent_Declare(
        rocm-cmake
        GIT_REPOSITORY https://github.com/ROCm/rocm-cmake.git
        GIT_TAG b3959201846b9d9250a6f2a386ad43be04d1d051 # rocm-7.2.4
        SOURCE_SUBDIR "DISABLE ADDING TO BUILD"
    )
    FetchContent_MakeAvailable(rocm-cmake)
    find_package(ROCmCMakeBuildTools CONFIG REQUIRED NO_DEFAULT_PATH PATHS "${rocm-cmake_SOURCE_DIR}")
endif()

include(ROCMSetupVersion)
include(ROCMCreatePackage)
include(ROCMInstallTargets)
include(ROCMPackageConfigHelpers)
include(ROCMInstallSymlinks)
include(ROCMClients)
include(ROCMHeaderWrapper)
