# MIT License
#
# Copyright (c) 2018-2025 Advanced Micro Devices, Inc. All rights reserved.
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

if(NOT EXTERNAL_DEPS_FORCE_DOWNLOAD)
  find_package(ROCmCMakeBuildTools 0.7.3 CONFIG QUIET PATHS "${ROCM_ROOT}")
endif()
if(NOT ROCmCMakeBuildTools_FOUND)
  message(STATUS "ROCm CMake not found. Fetching...")
  # We don't really want to consume the build and test targets of ROCm CMake.
  # CMake 3.18 allows omitting them, even though there's a CMakeLists.txt in source root.
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
    set(SOURCE_SUBDIR_ARG SOURCE_SUBDIR "DISABLE ADDING TO BUILD")
  else()
    set(SOURCE_SUBDIR_ARG)
  endif()
  include(FetchContent)
  FetchContent_Declare(
    rocm-cmake
    GIT_REPOSITORY https://github.com/ROCm/rocm-cmake.git
    GIT_TAG        rocm-6.4.4
    ${SOURCE_SUBDIR_ARG}
  )

  FetchContent_GetProperties(rocm-cmake)
  if(NOT rocm-cmake_POPULATED)
    # rocm-cmake 0.12.0 and higher needs to built from source
    FetchContent_Populate(rocm-cmake)
    message("Populated: ${rocm-cmake_SOURCE_DIR}")
    execute_process(
      WORKING_DIRECTORY ${rocm-cmake_SOURCE_DIR}
      COMMAND ${CMAKE_COMMAND} ${rocm-cmake_SOURCE_DIR} -DCMAKE_INSTALL_PREFIX=.
    )
    execute_process(
      WORKING_DIRECTORY ${rocm-cmake_SOURCE_DIR}
      COMMAND ${CMAKE_COMMAND} --build ${rocm-cmake_SOURCE_DIR} --target install
    )
  endif()

  FetchContent_MakeAvailable(rocm-cmake)
  find_package(ROCmCMakeBuildTools CONFIG REQUIRED NO_DEFAULT_PATH PATHS "${rocm-cmake_SOURCE_DIR}")
else()
  find_package(ROCmCMakeBuildTools 0.7.3 CONFIG REQUIRED PATHS "${ROCM_ROOT}")
endif()

include(ROCMSetupVersion)
include(ROCMCreatePackage)
include(ROCMInstallTargets)
include(ROCMPackageConfigHelpers)
include(ROCMInstallSymlinks)
include(ROCMCheckTargetIds)
include(ROCMClients)
