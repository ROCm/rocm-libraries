# MIT License
#
# Copyright (c) 2020-2025 Advanced Micro Devices, Inc. All rights reserved.
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

function(add_hipcub_benchmark BENCHMARK_SOURCE)
  get_filename_component(BENCHMARK_TARGET ${BENCHMARK_SOURCE} NAME_WE)
  if(USE_HIPCXX)
    set_source_files_properties(${BENCHMARK_SOURCE} PROPERTIES LANGUAGE HIP)
  endif()
  add_executable(${BENCHMARK_TARGET} ${BENCHMARK_SOURCE})
  target_include_directories(${BENCHMARK_TARGET} SYSTEM BEFORE
    PUBLIC
      "$<BUILD_INTERFACE:${GOOGLEBENCHMARK_ROOT}/include>"
  )
  target_link_libraries(${BENCHMARK_TARGET}
    PRIVATE
      benchmark::benchmark
      hipcub
  )
  if((HIP_COMPILER STREQUAL "nvcc"))
    set_property(TARGET ${BENCHMARK_TARGET} PROPERTY CUDA_STANDARD 17)
    set_source_files_properties(${BENCHMARK_SOURCE} PROPERTIES LANGUAGE CUDA)
    target_compile_options(${BENCHMARK_TARGET}
      PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    )
    target_link_libraries(${BENCHMARK_TARGET}
      PRIVATE
        hipcub_cub
    )
  endif()
  set_target_properties(${BENCHMARK_TARGET}
    PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/benchmark"
  )
  rocm_install(TARGETS ${BENCHMARK_TARGET} COMPONENT benchmarks)
  if (WIN32 AND NOT DEFINED DLLS_COPIED)
    set(DLLS_COPIED "YES")
    set(DLLS_COPIED ${DLLS_COPIED} PARENT_SCOPE)
    # for now adding in all .dll as dependency chain is not cmake based on win32
    file( GLOB third_party_dlls
    LIST_DIRECTORIES ON
    CONFIGURE_DEPENDS
    ${HIP_DIR}/bin/*.dll
    ${CMAKE_SOURCE_DIR}/rtest.*
    )
    foreach( file_i ${third_party_dlls})
      add_custom_command( TARGET ${BENCHMARK_TARGET} POST_BUILD COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different ${file_i} ${PROJECT_BINARY_DIR}/benchmark )
    endforeach( file_i )
  endif()         
endfunction()

# ****************************************************************************
# Benchmarks
# ****************************************************************************
add_hipcub_benchmark(benchmark_block_adjacent_difference.cpp)
add_hipcub_benchmark(benchmark_block_discontinuity.cpp)
add_hipcub_benchmark(benchmark_block_exchange.cpp)
add_hipcub_benchmark(benchmark_block_histogram.cpp)
add_hipcub_benchmark(benchmark_block_merge_sort.cpp)
add_hipcub_benchmark(benchmark_block_radix_sort.cpp)
add_hipcub_benchmark(benchmark_block_radix_rank.cpp)
add_hipcub_benchmark(benchmark_block_reduce.cpp)
add_hipcub_benchmark(benchmark_block_run_length_decode.cpp)
add_hipcub_benchmark(benchmark_block_scan.cpp)
add_hipcub_benchmark(benchmark_block_shuffle.cpp)
add_hipcub_benchmark(benchmark_device_adjacent_difference.cpp)
add_hipcub_benchmark(benchmark_device_batch_copy.cpp)
add_hipcub_benchmark(benchmark_device_batch_memcpy.cpp)
add_hipcub_benchmark(benchmark_device_for.cpp)
add_hipcub_benchmark(benchmark_device_histogram.cpp)
add_hipcub_benchmark(benchmark_device_memory.cpp)
add_hipcub_benchmark(benchmark_device_merge.cpp)
add_hipcub_benchmark(benchmark_device_merge_sort.cpp)
add_hipcub_benchmark(benchmark_device_partition.cpp)
add_hipcub_benchmark(benchmark_device_radix_sort.cpp)
add_hipcub_benchmark(benchmark_device_reduce_by_key.cpp)
add_hipcub_benchmark(benchmark_device_reduce.cpp)
add_hipcub_benchmark(benchmark_device_run_length_encode.cpp)
add_hipcub_benchmark(benchmark_device_scan.cpp)
add_hipcub_benchmark(benchmark_device_segmented_sort.cpp)
add_hipcub_benchmark(benchmark_device_segmented_radix_sort.cpp)
add_hipcub_benchmark(benchmark_device_segmented_reduce.cpp)
add_hipcub_benchmark(benchmark_device_select.cpp)
add_hipcub_benchmark(benchmark_device_spmv.cpp)
add_hipcub_benchmark(benchmark_warp_exchange.cpp)
add_hipcub_benchmark(benchmark_warp_load.cpp)
add_hipcub_benchmark(benchmark_warp_reduce.cpp)
add_hipcub_benchmark(benchmark_warp_scan.cpp)
add_hipcub_benchmark(benchmark_warp_store.cpp)
add_hipcub_benchmark(benchmark_warp_merge_sort.cpp)
