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

function(add_rocprim_benchmark BENCHMARK_SOURCE)
  get_filename_component(BENCHMARK_TARGET ${BENCHMARK_SOURCE} NAME_WE)
  add_executable(${BENCHMARK_TARGET} ${BENCHMARK_SOURCE})
  target_link_libraries(${BENCHMARK_TARGET}
    PRIVATE
      rocprim_hip
      benchmark::benchmark
  )
  foreach(amdgpu_target ${AMDGPU_TARGETS})
    target_link_libraries(${BENCHMARK_TARGET}
      PRIVATE
        --amdgpu-target=${amdgpu_target}
    )
  endforeach()
  set_target_properties(${BENCHMARK_TARGET}
    PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/benchmark"
  )
endfunction()

# ****************************************************************************
# Benchmarks
# ****************************************************************************

add_rocprim_benchmark(benchmark_block_discontinuity.cpp)
add_rocprim_benchmark(benchmark_block_exchange.cpp)
add_rocprim_benchmark(benchmark_block_histogram.cpp)
add_rocprim_benchmark(benchmark_block_radix_sort.cpp)
add_rocprim_benchmark(benchmark_block_reduce.cpp)
add_rocprim_benchmark(benchmark_block_scan.cpp)
add_rocprim_benchmark(benchmark_block_sort.cpp)
add_rocprim_benchmark(benchmark_device_binary_search.cpp)
add_rocprim_benchmark(benchmark_device_histogram.cpp)
add_rocprim_benchmark(benchmark_device_merge.cpp)
add_rocprim_benchmark(benchmark_device_merge_sort.cpp)
add_rocprim_benchmark(benchmark_device_partition.cpp)
add_rocprim_benchmark(benchmark_device_radix_sort.cpp)
add_rocprim_benchmark(benchmark_device_reduce_by_key.cpp)
add_rocprim_benchmark(benchmark_device_reduce.cpp)
add_rocprim_benchmark(benchmark_device_run_length_encode.cpp)
add_rocprim_benchmark(benchmark_device_scan.cpp)
add_rocprim_benchmark(benchmark_device_select.cpp)
add_rocprim_benchmark(benchmark_device_segmented_radix_sort.cpp)
add_rocprim_benchmark(benchmark_device_segmented_reduce.cpp)
add_rocprim_benchmark(benchmark_device_transform.cpp)
add_rocprim_benchmark(benchmark_warp_reduce.cpp)
add_rocprim_benchmark(benchmark_warp_scan.cpp)
add_rocprim_benchmark(benchmark_warp_sort.cpp)
add_rocprim_benchmark(benchmark_device_memory.cpp)
