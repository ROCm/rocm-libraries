# MIT License
#
# Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
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

cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

# This project includes tests that should be run after
# rocPRIM is installed from package or using `make install`
project(rocprim_package_install_test CXX)

# CMake modules
list(APPEND CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake
  ${HIP_PATH}/cmake /opt/rocm/hip/cmake # FindHIP.cmake
)

# Verify that hcc compiler is used on ROCM platform
include(VerifyCompiler)

# Download CUB
include(DownloadProject)
if(HIP_PLATFORM STREQUAL "nvcc")
  if(NOT DEFINED CUB_INCLUDE_DIR)
    download_project(PROJ   cub
             GIT_REPOSITORY https://github.com/NVlabs/cub.git
             GIT_TAG        v1.8.0
             LOG_DOWNLOAD   TRUE
             LOG_CONFIGURE  TRUE
             LOG_BUILD      TRUE
             LOG_INSTALL    TRUE
             BUILD_PROJECT  FALSE
             ${UPDATE_DISCONNECTED_IF_AVAILABLE}
    )
    set(CUB_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/cub-src/ CACHE PATH "")
  endif()
endif()

if(HIP_PLATFORM STREQUAL "hcc")
  # Find rocPRIM
  find_package(rocprim REQUIRED CONFIG HINTS ${rocprim_DIR} PATHS "/opt/rocm/rocprim")
endif()
# Find hipCUB
find_package(hipcub REQUIRED CONFIG HINTS ${hipcub_DIR} PATHS "/opt/rocm/hipcub")

# Build CXX flags
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")

# AMD targets
set(AMDGPU_TARGETS gfx803;gfx900 CACHE STRING "List of specific machine types for library to target")

# Enable testing (ctest)
enable_testing()

# Build
function(add_rocprim_test TEST_NAME TEST_SOURCES)
  list(GET TEST_SOURCES 0 TEST_MAIN_SOURCE)
  get_filename_component(TEST_TARGET ${TEST_MAIN_SOURCE} NAME_WE)
  add_executable(${TEST_TARGET} ${TEST_SOURCES})
  target_link_libraries(${TEST_TARGET}
    PRIVATE
      roc::rocprim_hip
  )
  foreach(amdgpu_target ${AMDGPU_TARGETS})
    target_link_libraries(${TEST_TARGET}
      PRIVATE
        --amdgpu-target=${amdgpu_target}
    )
  endforeach()
  add_test(${TEST_NAME} ${TEST_TARGET})
endfunction()

function(add_hipcub_test TEST_NAME TEST_SOURCES)
  list(GET TEST_SOURCES 0 TEST_MAIN_SOURCE)
  get_filename_component(TEST_TARGET ${TEST_MAIN_SOURCE} NAME_WE)
  if(HIP_PLATFORM STREQUAL "hcc")
    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    target_link_libraries(${TEST_TARGET}
      PRIVATE
        hip::hipcub
    )
    foreach(amdgpu_target ${AMDGPU_TARGETS})
      target_link_libraries(${TEST_TARGET}
        PRIVATE
          --amdgpu-target=${amdgpu_target}
      )
    endforeach()
  else() # CUDA/nvcc
    set_source_files_properties(${TEST_SOURCES}
      PROPERTIES
        CUDA_SOURCE_PROPERTY_FORMAT OBJ
      )
      CUDA_ADD_EXECUTABLE(${TEST_TARGET}
        ${TEST_SOURCES}
        OPTIONS
          --expt-extended-lambda
    )
    target_link_libraries(${TEST_TARGET} hip::hipcub)
    target_include_directories(${TEST_TARGET} SYSTEM PRIVATE ${CUB_INCLUDE_DIR})
  endif()
  add_test(${TEST_NAME} ${TEST_TARGET})
endfunction()

# rocPRIM package test
if(HIP_PLATFORM STREQUAL "hcc")
  add_rocprim_test("test_rocprim_package" test_rocprim_package.cpp)
endif()
# hipCUB package test
add_hipcub_test("test_hipcub_package" test_hipcub_package.cpp)