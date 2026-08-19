# MIT License
#
# Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

if(HIP_COMPILER STREQUAL "nvcc")
  message(FATAL_ERROR "Configuration tuning is currently supported for AMD GPUs only")
endif()

set(BENCHMARK_TUNING_MIN_GRID_SIZE 32768 CACHE STRING
  "Configurations with fewer total number of threads are omitted")
set(BENCHMARK_TUNING_THREAD_OPTIONS "64, 128, 256, 512, 1024" CACHE STRING
  "Comma-separated list of benchmarked block sizes")
set(BENCHMARK_TUNING_BLOCK_OPTIONS "64, 128, 256, 512, 1024, 2048" CACHE STRING
  "Comma-separated list of benchmarked grid sizes")

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/benchmark_tuning_setup.hpp.in"
  "${CMAKE_BINARY_DIR}/benchmark/benchmark_tuning_setup.hpp")

set(CMAKE_CXX_STANDARD 17)
add_executable(benchmark_rocrand_tuning
  benchmark_tuning.cpp
  benchmark_tuning_lfsr113.cpp
  benchmark_tuning_mrg31k3p.cpp
  benchmark_tuning_mrg32k3a.cpp
  benchmark_tuning_mt19937.cpp
  benchmark_tuning_mtgp32.cpp
  benchmark_tuning_philox.cpp
  benchmark_tuning_threefry2x32_20.cpp
  benchmark_tuning_threefry2x64_20.cpp
  benchmark_tuning_threefry4x32_20.cpp
  benchmark_tuning_threefry4x64_20.cpp
  benchmark_tuning_xorwow.cpp
)
target_link_libraries(benchmark_rocrand_tuning PRIVATE rocrand benchmark::benchmark hip::device)
target_include_directories(benchmark_rocrand_tuning PRIVATE
  "${CMAKE_BINARY_DIR}/benchmark"
  "${PROJECT_SOURCE_DIR}/benchmark"
  "${PROJECT_SOURCE_DIR}/library/src")

if(NOT WIN32)
  foreach(amdgpu_target ${AMDGPU_TARGETS})
      target_compile_options(rocrand PRIVATE --offload-arch=${amdgpu_target})
  endforeach()
endif()
