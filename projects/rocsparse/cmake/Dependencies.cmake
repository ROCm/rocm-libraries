# ########################################################################
# Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ########################################################################

# Dependencies

option(ALLOW_FETCH_DEPS "Download rocm-cmake, GoogleTest and the test matrices if they are not found in the build environment" OFF)
include(FetchContent)
if(NOT ALLOW_FETCH_DEPS)
  set(FETCHCONTENT_FULLY_DISCONNECTED ON)
endif()

# Workaround until hcc & hip cmake modules fixes symlink logic in their config files.
# (Thanks to rocBLAS devs for finding workaround for this problem!)
list(APPEND CMAKE_PREFIX_PATH /opt/rocm /opt/rocm/hip)

# ROCm cmake package
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
include(ROCMCheckTargetIds)
include(ROCMClients)
include(ROCMHeaderWrapper)
