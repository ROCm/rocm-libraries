# MIT License
#
# Copyright (c) 2017-2026 Advanced Micro Devices, Inc. All rights reserved.
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

function(add_example EXAMPLE_SOURCE)
  get_filename_component(EXAMPLE_TARGET ${EXAMPLE_SOURCE} NAME_WE)

  if(USE_HIPCXX)
    set_source_files_properties(${EXAMPLE_SOURCE} PROPERTIES LANGUAGES HIP)
  endif()

  add_executable(${EXAMPLE_TARGET} ${EXAMPLE_SOURCE})

  target_link_libraries(${EXAMPLE_TARGET}
    PRIVATE
    rocprim
  )

  if(USE_HIPCXX)
    target_link_libraries(${EXAMPLE_TARGET}
      PRIVATE
      $<IF:$<LINK_LANGUAGE:HIP>,hip::host,hip::device>
    )
  else()
    target_link_libraries(${EXAMPLE_TARGET}
      PRIVATE
      hip::device)
  endif()

  set_target_properties(${EXAMPLE_TARGET}
    PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/example"
  )

  # Use an EXAMPLE_SOURCE like "rocprim/device/example_device_search.cpp"
  # to create a test named "example.rocprim.device.example_device_search"
  get_filename_component(DIR_PATH "${EXAMPLE_SOURCE}" PATH)
  string(REPLACE "/" "." DIR_PATH_DOTS "${DIR_PATH}")
  set(TEST_NAME "example.${DIR_PATH_DOTS}.${EXAMPLE_TARGET}")
  add_test(NAME ${TEST_NAME} COMMAND ${EXAMPLE_TARGET})

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
      add_custom_command( TARGET ${EXAMPLE_TARGET} POST_BUILD COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different ${file_i} ${PROJECT_BINARY_DIR}/example )
    endforeach( file_i )
  endif()
endfunction()

# ****************************************************************************
# rocPRIM examples
# ****************************************************************************

add_example(extra/example_temporary_storage.cpp)
add_example(rocprim/example_type_traits.cpp)
add_example(rocprim/block/example_block_adjacent_difference.cpp)
add_example(rocprim/block/example_block_discontinuity.cpp)
add_example(rocprim/block/example_block_exchange.cpp)
add_example(rocprim/block/example_block_histogram.cpp)
add_example(rocprim/block/example_block_load.cpp)
add_example(rocprim/block/example_block_radix_rank.cpp)
add_example(rocprim/block/example_block_radix_sort.cpp)
add_example(rocprim/block/example_block_reduce.cpp)
add_example(rocprim/block/example_block_run_length_decode.cpp)
add_example(rocprim/block/example_block_shuffle.cpp)
add_example(rocprim/block/example_block_sort.cpp)
add_example(rocprim/block/example_block_scan.cpp)
add_example(rocprim/block/example_block_store.cpp)
add_example(rocprim/device/example_device_adjacent_difference.cpp)
add_example(rocprim/device/example_device_adjacent_find.cpp)
add_example(rocprim/device/example_device_binary_search.cpp)
add_example(rocprim/device/example_device_copy.cpp)
add_example(rocprim/device/example_device_find_end.cpp)
add_example(rocprim/device/example_device_find_first_of.cpp)
add_example(rocprim/device/example_device_histogram.cpp)
add_example(rocprim/device/example_device_memcpy.cpp)
add_example(rocprim/device/example_device_merge_inplace.cpp)
add_example(rocprim/device/example_device_merge_sort.cpp)
add_example(rocprim/device/example_device_merge.cpp)
add_example(rocprim/device/example_device_nth_element.cpp)
add_example(rocprim/device/example_device_partial_sort.cpp)
add_example(rocprim/device/example_device_partition.cpp)
add_example(rocprim/device/example_device_radix_sort.cpp)
add_example(rocprim/device/example_device_reduce_by_key.cpp)
add_example(rocprim/device/example_device_reduce.cpp)
add_example(rocprim/device/example_device_run_length_encode.cpp)
add_example(rocprim/device/example_device_scan_by_key.cpp)
add_example(rocprim/device/example_device_scan.cpp)
add_example(rocprim/device/example_device_search.cpp)
add_example(rocprim/device/example_device_segmented_radix_sort.cpp)
add_example(rocprim/device/example_device_segmented_reduce.cpp)
add_example(rocprim/device/example_device_segmented_scan.cpp)
add_example(rocprim/device/example_device_select.cpp)
add_example(rocprim/device/example_device_topk.cpp)
add_example(rocprim/device/example_device_transform.cpp)
add_example(rocprim/intrinsics/example_arch.cpp)
add_example(rocprim/types/example_future_value.cpp)
add_example(rocprim/warp/example_warp_exchange.cpp)
add_example(rocprim/warp/example_warp_load.cpp)
add_example(rocprim/warp/example_warp_reduce.cpp)
add_example(rocprim/warp/example_warp_scan.cpp)
add_example(rocprim/warp/example_warp_sort.cpp)
add_example(rocprim/warp/example_warp_store.cpp)
