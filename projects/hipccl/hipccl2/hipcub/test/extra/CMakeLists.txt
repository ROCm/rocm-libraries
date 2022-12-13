# MIT License
#
# Copyright (c) 2017-2020 Advanced Micro Devices, Inc. All rights reserved.
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

cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
cmake_policy(VERSION 3.16...3.21)

# This project includes tests that should be run after
# hipCUB is installed from package or using `make install`
project(hipCUB_package_install_test CXX)

# CMake modules
list(APPEND CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake
  ${HIP_PATH}/cmake /opt/rocm/hip/cmake # FindHIP.cmake
)

# Verify that hip-clang is used on ROCM platform
include(VerifyCompiler)

# Download CUB
include(DownloadProject)
if(HIP_COMPILER STREQUAL "nvcc")
  file(
    DOWNLOAD https://github.com/NVIDIA/cub/archive/1.17.2.zip
    ${CMAKE_CURRENT_BINARY_DIR}/cub-1.17.2.zip
    STATUS cub_download_status LOG cub_download_log
  )
  list(GET cub_download_status 0 cub_download_error_code)
  if(cub_download_error_code)
    message(FATAL_ERROR "Error: downloading "
      "https://github.com/NVIDIA/cub/archive/1.17.2.zip failed "
      "error_code: ${cub_download_error_code} "
      "log: ${cub_download_log} "
    )
  endif()

  execute_process(
    COMMAND ${CMAKE_COMMAND} -E tar xzf ${CMAKE_CURRENT_BINARY_DIR}/cub-1.17.2.zip
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    RESULT_VARIABLE cub_unpack_error_code
  )
  if(cub_unpack_error_code)
    message(FATAL_ERROR "Error: unpacking ${CMAKE_CURRENT_BINARY_DIR}/cub-1.17.2.zip failed")
  endif()
  set(CUB_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/cub-1.17.2/ CACHE PATH "" FORCE)
  message(STATUS "CUB_INCLUDE_DIR: ${CUB_INCLUDE_DIR}")

  file(
    DOWNLOAD https://github.com/NVIDIA/thrust/archive/1.17.2.zip
    ${CMAKE_CURRENT_BINARY_DIR}/thrust-1.17.2.zip
    STATUS thrust_download_status LOG thrust_download_log
  )
  list(GET thrust_download_status 0 thrust_download_error_code)
  if(thrust_download_error_code)
    message(FATAL_ERROR "Error: downloading "
      "https://github.com/NVIDIA/thrust/archive/1.17.2.zip failed "
      "error_code: ${thrust_download_error_code} "
      "log: ${thrust_download_log} "
    )
  endif()

  execute_process(
    COMMAND ${CMAKE_COMMAND} -E tar xzf ${CMAKE_CURRENT_BINARY_DIR}/thrust-1.17.2.zip
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    RESULT_VARIABLE thrust_unpack_error_code
  )
  if(thrust_unpack_error_code)
    message(FATAL_ERROR "Error: unpacking ${CMAKE_CURRENT_BINARY_DIR}/thrust-1.17.2.zip failed")
  endif()
  set(THRUST_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/thrust-1.17.2/ CACHE PATH "" FORCE)
  message(STATUS "THRUST_INCLUDE_DIR: ${THRUST_INCLUDE_DIR}")
endif()

# Download rocPRIM (only for ROCm platform)
if(HIP_COMPILER STREQUAL "clang")
  if(NOT DEFINED rocprim_DIR)
    message(STATUS "Downloading and building rocPRIM.")
    set(rocprim_DIR "${CMAKE_CURRENT_BINARY_DIR}/rocprim" CACHE PATH "")
    download_project(
      PROJ                rocprim
      GIT_REPOSITORY      https://github.com/ROCmSoftwarePlatform/rocPRIM.git
      GIT_TAG             master
      INSTALL_DIR         ${rocprim_DIR}
      CMAKE_ARGS          -DBUILD_TEST=OFF -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
      LOG_DOWNLOAD        TRUE
      LOG_CONFIGURE       TRUE
      LOG_BUILD           TRUE
      LOG_INSTALL         TRUE
      BUILD_PROJECT       TRUE
      UPDATE_DISCONNECTED TRUE # Never update automatically from the remote repository
    )
  endif()
  find_package(rocprim REQUIRED CONFIG PATHS "${rocprim_DIR}")
endif()

# Find hipCUB
find_package(hipcub REQUIRED CONFIG HINTS ${hipcub_DIR} PATHS "/opt/rocm")

# Build CXX flags
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable testing (ctest)
enable_testing()

function(add_hipcub_test TEST_NAME TEST_SOURCES)
  list(GET TEST_SOURCES 0 TEST_MAIN_SOURCE)
  get_filename_component(TEST_TARGET ${TEST_MAIN_SOURCE} NAME_WE)
  add_executable(${TEST_TARGET} ${TEST_SOURCES})

  if(HIP_COMPILER STREQUAL "nvcc")
    set_property(TARGET ${TEST_TARGET} PROPERTY CUDA_STANDARD 14)
    set_source_files_properties(${TEST_SOURCES} PROPERTIES LANGUAGE CUDA)
    target_link_libraries(${TEST_TARGET}
      PRIVATE
        hip::hipcub
    )
    target_include_directories(${TEST_TARGET}
      PRIVATE
        $<BUILD_INTERFACE:${CUB_INCLUDE_DIR}>
        $<BUILD_INTERFACE:${THRUST_INCLUDE_DIR}>
    )
  elseif(HIP_COMPILER STREQUAL "clang")
    target_link_libraries(${TEST_TARGET}
      PRIVATE
        ${hipcub_LIBRARIES} # hip::hipcub
    )
  endif()
  add_test(
    NAME ${TEST_NAME}
    COMMAND ${TEST_TARGET}
  )
endfunction()

# hipCUB package test
add_hipcub_test("test_hipcub_package" test_hipcub_package.cpp)
