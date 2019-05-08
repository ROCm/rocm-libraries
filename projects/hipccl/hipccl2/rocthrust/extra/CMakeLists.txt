# MIT License
#
# Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.
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

cmake_minimum_required(VERSION 3.5.1 FATAL_ERROR)

# This project includes tests that should be run after
# rocThrust is installed from package or using `make install`
project(rocThrust_package_install_test CXX)

# CMake modules
list(APPEND CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_SOURCE_DIR}/../cmake
  ${HIP_PATH}/cmake /opt/rocm/hip/cmake # FindHIP.cmake
)

# Verify that hcc compiler is used on ROCM platform
include(VerifyCompiler)
include(DownloadProject)

# Download rocPRIM (only for ROCm platform)
if(HIP_PLATFORM STREQUAL "hcc")
  if(NOT DEFINED rocprim_DIR)
    message(STATUS "Downloading and building rocPRIM.")
    set(rocprim_DIR "${CMAKE_CURRENT_BINARY_DIR}/rocprim" CACHE PATH "")
    download_project(
      PROJ           rocprim
      GIT_REPOSITORY https://github.com/ROCmSoftwarePlatform/rocPRIM.git
      GIT_TAG        develop
      INSTALL_DIR    ${rocprim_DIR}
      CMAKE_ARGS     -DBUILD_TEST=OFF -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
      LOG_DOWNLOAD   TRUE
      LOG_CONFIGURE  TRUE
      LOG_BUILD      TRUE
      LOG_INSTALL    TRUE
      BUILD_PROJECT  TRUE
      ${UPDATE_DISCONNECTED_IF_AVAILABLE}
    )
  endif()
  find_package(rocprim REQUIRED CONFIG PATHS "${rocprim_DIR}")
endif()

# Find rocThrust
find_package(rocthrust REQUIRED CONFIG HINTS ${rocthrust_DIR} PATHS "/opt/rocm/rocthrust")

# Build CXX flags
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")

# AMD targets
set(AMDGPU_TARGETS gfx803;gfx900;gfx906 CACHE STRING "List of specific machine types for library to target")

# Enable testing (ctest)
enable_testing()

function(add_rocthrust_test TEST_NAME TEST_SOURCES)
  list(GET TEST_SOURCES 0 TEST_MAIN_SOURCE)
  get_filename_component(TEST_TARGET ${TEST_MAIN_SOURCE} NAME_WE)
  add_executable(${TEST_TARGET} ${TEST_SOURCES})
  target_link_libraries(${TEST_TARGET}
    PRIVATE
      rocthrust
  )
  foreach(amdgpu_target ${AMDGPU_TARGETS})
    target_link_libraries(${TEST_TARGET}
      PRIVATE
        --amdgpu-target=${amdgpu_target}
    )
  endforeach()
  # Set HIP as device system for Thrust
  target_compile_definitions(${TEST_TARGET}
      PRIVATE
          THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_HIP
  )
  add_test(${TEST_NAME} ${TEST_TARGET})
endfunction()

# rocThrust package test
add_rocthrust_test("test_rocthrust_package" test_rocthrust_package.cpp)
