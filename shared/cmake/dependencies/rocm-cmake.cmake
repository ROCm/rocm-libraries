# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

include_guard(GLOBAL)

include(FetchContent)

if(ROCmCMakeBuildTools_FOUND)
  message(STATUS "Found existing ROCmCMakeBuildTools")
else()
  find_package(ROCmCMakeBuildTools 0.11.0 CONFIG QUIET PATHS "${ROCM_ROOT}")
endif()
if(NOT ROCmCMakeBuildTools_FOUND)
  message(STATUS "ROCm CMake not found. Fetching...")
  set(ROCM_CMAKE_GIT_TAG "rocm-7.2.4" CACHE STRING "rocm-cmake tag to download")
  FetchContent_Declare(
    rocm-cmake
    GIT_REPOSITORY https://github.com/ROCm/rocm-cmake.git
    GIT_TAG        ${ROCM_CMAKE_GIT_TAG}
    ${SOURCE_SUBDIR_ARG}
  )
  # No need to expose rocm-cmake build targets. Instead populate is good enough.
  # FetchContent_MakeAvailable(rocm-cmake)
  FetchContent_Populate(rocm-cmake)
  find_package(
    ROCmCMakeBuildTools
    CONFIG REQUIRED NO_DEFAULT_PATH
    PATHS "${rocm-cmake_SOURCE_DIR}"
  )
  message(STATUS "ROCm CMake populated at ${ROCmCMakeBuildTools_DIR}")
endif()

set(ROCM_DISABLE_CHECKS OFF CACHE BOOL "")
# Override 'rocm_check_toolchain_var' in ROCmCMakeBuildTools so that we can
# easily disable it for depedencies.
macro(rocm_check_toolchain_var var access value list_file)
  if(NOT ROCM_DISABLE_CHECKS)
    _rocm_check_toolchain_var("${var}" "${access}" "${value}" "${list_file}")
  endif()
endmacro()

include(ROCMSetupVersion)
include(ROCMCreatePackage)
include(ROCMInstallTargets)
include(ROCMPackageConfigHelpers)
include(ROCMInstallSymlinks)
include(ROCMCheckTargetIds)
include(ROCMClients)
if(BUILD_DOCS)
  include(ROCMSphinxDoc)
endif()
