# MIT License
#
# Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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

# Tests whether rocRAND and cuRAND generate the same stream of numbers
# when legacy ordering is used.

cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(rocrand_parity_test CXX)

# CMake modules
list(APPEND CMAKE_PREFIX_PATH $ENV{ROCM_PATH} $ENV{ROCM_PATH}/hip)
list(APPEND CMAKE_MODULE_PATH
    $ENV{ROCM_PATH}/lib/cmake/hip
    ${HIP_PATH}/cmake $ENV{ROCM_PATH}/hip/cmake # FindHIP.cmake
)
# Find HIP
if(CMAKE_CXX_COMPILER MATCHES ".*/nvcc$" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    find_package(hip QUIET CONFIG PATHS $ENV{ROCM_PATH})
    if(NOT hip_FOUND)
        find_package(HIP REQUIRED)
        if((HIP_COMPILER STREQUAL "hipcc") AND (HIP_PLATFORM STREQUAL "nvcc"))
        # TODO: The HIP package on NVIDIA platform is incorrect at few versions
        set(HIP_COMPILER "nvcc" CACHE STRING "HIP Compiler" FORCE)
        endif()
    endif()
else()
    find_package(hip REQUIRED CONFIG PATHS $ENV{ROCM_PATH})
endif()

if(CMAKE_VERSION VERSION_LESS 3.17)
    find_package(CUDA REQUIRED)
else()
    find_package(CUDAToolkit)
    set(CUDA_LIBRARIES CUDA::cudart)
    set(CUDA_curand_LIBRARY CUDA::curand)
    set(CUDA_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS})
endif()

if (NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_CXX_STANDARD EQUAL 17)
  message(FATAL_ERROR "Only C++17 is supported")
endif()

# Find rocRAND
find_package(rocrand REQUIRED CONFIG HINTS ${rocrand_DIR} PATHS "$ENV{ROCM_PATH}/rocrand")

# Create an object library for the curand part of the test. A separate object is compiled to
# avoid symbol errors between curand/cuda and rocrand/hip.
add_library(test_rocrand_parity_curand OBJECT parity_curand.cpp)
target_include_directories(test_rocrand_parity_curand PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(test_rocrand_parity_curand ${CUDA_LIBRARIES} ${CUDA_curand_LIBRARY})

# Create an object library for the rocrand part of the test.
add_library(test_rocrand_parity_rocrand OBJECT parity_rocrand.cpp)

target_link_libraries(test_rocrand_parity_rocrand PRIVATE roc::rocrand)
if(HIP_COMPILER STREQUAL "nvcc")
    # Get HIP options
    execute_process(
        COMMAND ${HIP_HIPCONFIG_EXECUTABLE} --cpp_config
        OUTPUT_VARIABLE HIP_CPP_CONFIG
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE
    )
    separate_arguments(HIP_CPP_CONFIG UNIX_COMMAND "${HIP_CPP_CONFIG}")
    target_compile_options(test_rocrand_parity_rocrand PRIVATE ${HIP_CPP_CONFIG})

    target_include_directories(test_rocrand_parity_rocrand PRIVATE ${CUDA_INCLUDE_DIRS})
    target_link_libraries(test_rocrand_parity_rocrand PRIVATE ${CUDA_LIBRARIES})
else()
    target_link_libraries(test_rocrand_parity_rocrand PRIVATE hip::host)
    # HIP on Windows: xhip is required with clang++ to get __half defined
    if (WIN32)
        target_compile_options(test_rocrand_parity_rocrand PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-xhip>")
    endif()
endif()

# Compile the files that invoke each API separately, to avoid symbol namespacing errors.
add_executable(test_rocrand_parity parity.cpp)
target_link_libraries(test_rocrand_parity
    test_rocrand_parity_curand
    test_rocrand_parity_rocrand
)
