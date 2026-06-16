# MIT License
#
# Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
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

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
cmake_policy(VERSION 3.18...3.25)

# This project includes tests that should be run after
# hipCUB is installed from package or using `make install`
project(hipCUB_package_install_test CXX)

# Set the ROCM install directory.
if(WIN32)
  set(ROCM_ROOT "$ENV{HIP_PATH}" CACHE PATH "Root directory of the ROCm installation")
else()
  set(ROCM_ROOT "/opt/rocm" CACHE PATH "Root directory of the ROCm installation")
endif()

# Add hipCUB's CMake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake")

# Find and verify HIP.
include(VerifyCompiler)

# CUB (only for CUDA platform)
if(HIP_COMPILER STREQUAL "nvcc")
  set(CCCL_MINIMUM_VERSION 2.8.2)
  if(NOT DOWNLOAD_CUB)
    find_package(CCCL ${CCCL_MINIMUM_VERSION} CONFIG)
  endif()

  if (NOT CCCL_FOUND)
    message(STATUS "CCCL not found, downloading and extracting CCCL ${CCCL_MINIMUM_VERSION}")
    file(DOWNLOAD https://github.com/NVIDIA/cccl/archive/refs/tags/v${CCCL_MINIMUM_VERSION}.zip
                  ${CMAKE_CURRENT_BINARY_DIR}/cccl-${CCCL_MINIMUM_VERSION}.zip
        STATUS cccl_download_status LOG cccl_download_log)

    list(GET cccl_download_status 0 cccl_download_error_code)
    if(cccl_download_error_code)
      message(FATAL_ERROR "Error: downloading "
              "https://github.com/NVIDIA/cccl/archive/refs/tags/v${CCCL_MINIMUM_VERSION}.zip failed "
              "error_code: ${cccl_download_error_code} "
              "log: ${cccl_download_log}")
    endif()

    if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
      file(ARCHIVE_EXTRACT INPUT ${CMAKE_CURRENT_BINARY_DIR}/cccl-${CCCL_MINIMUM_VERSION}.zip)
    else()
      execute_process(COMMAND "${CMAKE_COMMAND}" -E tar xf ${CMAKE_CURRENT_BINARY_DIR}/cccl-${CCCL_MINIMUM_VERSION}.zip
                      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                      RESULT_VARIABLE cccl_unpack_error_code)
      if(cccl_unpack_error_code)
        message(FATAL_ERROR "Error: unpacking ${CMAKE_CURRENT_BINARY_DIR}/cccl-${CCCL_MINIMUM_VERSION}.zip failed")
      endif()
    endif()

    find_package(CCCL ${CCCL_MINIMUM_VERSION} CONFIG REQUIRED NO_DEFAULT_PATH
                 PATHS ${CMAKE_CURRENT_BINARY_DIR}/cccl-${CCCL_MINIMUM_VERSION})
  endif()
else()
  # rocPRIM (only for ROCm platform)
  if(NOT DEPENDENCIES_FORCE_DOWNLOAD)
    # Add default install location for WIN32 and non-WIN32 as hint
    find_package(rocprim CONFIG REQUIRED PATHS "${ROCM_ROOT}/lib/cmake/rocprim")
  endif()
  if(NOT TARGET roc::rocprim)
    message(STATUS "rocPRIM not found. Fetching...")
    FetchContent_Declare(
      prim
      GIT_REPOSITORY https://github.com/ROCm/rocPRIM.git
      GIT_TAG        develop
    )
    FetchContent_MakeAvailable(prim)
    if(NOT TARGET roc::rocprim)
      add_library(roc::rocprim ALIAS rocprim)
    endif()
  else()
    find_package(rocprim CONFIG REQUIRED)
  endif()
endif()

# Find hipCUB
find_package(hipcub CONFIG REQUIRED PATHS "${ROCM_ROOT}/lib/cmake/hipcub")

# Build CXX flags
if (NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_CXX_STANDARD EQUAL 17)
  message(FATAL_ERROR "Only C++17 is supported")
endif()

# Enable testing (ctest)
enable_testing()

function(add_hipcub_test TEST_NAME TEST_SOURCES)
  list(GET TEST_SOURCES 0 TEST_MAIN_SOURCE)
  get_filename_component(TEST_TARGET ${TEST_MAIN_SOURCE} NAME_WE)
  add_executable(${TEST_TARGET} ${TEST_SOURCES})
  if(HIP_COMPILER STREQUAL "nvcc")
    target_compile_features(${TEST_TARGET} PUBLIC cuda_std_14)
    set_source_files_properties(${TEST_SOURCES} PROPERTIES LANGUAGE CUDA)
    target_link_libraries(${TEST_TARGET} PRIVATE CCCL::CCCL)
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
