# MIT License #
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

set(GPU_TEST_TARGETS "" CACHE STRING "List of specific device types to test for (Leave empty for default system device)")

# Gets a test target name based on the first source file.
function(get_hipcub_test_target TEST_SOURCES TEST_TARGET)
  list(GET TEST_SOURCES 0 TEST_MAIN_SOURCE)
  get_filename_component(TEST_TARGET ${TEST_MAIN_SOURCE} NAME_WE)
  set(TEST_TARGET ${TEST_TARGET} PARENT_SCOPE)
endfunction()

function(add_hipcub_test TEST_NAME TEST_SOURCES)
  get_hipcub_test_target(${TEST_SOURCES} TEST_TARGET)
  add_hipcub_test_internal(${TEST_NAME} "${TEST_SOURCES}" ${TEST_TARGET})
endfunction()

function(add_hipcub_test_internal TEST_NAME TEST_SOURCES TEST_TARGET)
  if(USE_HIPCXX)
    set_source_files_properties(${TEST_SOURCES} PROPERTIES LANGUAGE HIP)
  endif()

  add_executable(${TEST_TARGET} ${TEST_SOURCES})
  target_link_libraries(${TEST_TARGET}
    PRIVATE
      GTest::gtest
      GTest::gtest_main
  )

  if(USE_SYSTEM_LIB)
    target_link_libraries(${TEST_TARGET} PRIVATE hip::hipcub)
  else()
    target_link_libraries(${TEST_TARGET} PRIVATE hipcub)
  endif()

  if(HIP_COMPILER STREQUAL "nvcc")
    # For test/hipcub, std::exclusive_scan is used, which is a c++ 17 feature
    target_compile_features(${TEST_TARGET} PUBLIC cuda_std_17)
    set_source_files_properties(${TEST_SOURCES} PROPERTIES LANGUAGE CUDA)
    target_link_libraries(${TEST_TARGET}
      PRIVATE
        hipcub_cub
    )
  endif()

  if(WIN32)
    # Usage of 128-bit integral types (__int128_t and __uint128_t which alignment is 16 bytes)
    # requires an extended alignment support otherwise a static assert will be triggered in
    # functions like std::stable_sort. _ENABLE_EXTENDED_ALIGNED_STORAGE turns it on:
    target_compile_definitions(${TEST_TARGET} PRIVATE _ENABLE_EXTENDED_ALIGNED_STORAGE)
  endif()

  set_target_properties(${TEST_TARGET}
    PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test/hipcub"
  )
  if(GPU_TEST_TARGETS)
    foreach(GPU_TARGET IN LISTS GPU_TEST_TARGETS)
      add_relative_test("${GPU_TARGET}-${TEST_NAME}" ${TEST_TARGET})
      set_tests_properties("${GPU_TARGET}-${TEST_NAME}"
          PROPERTIES
              RESOURCE_GROUPS "1,${GPU_TARGET}:1"
      )
    endforeach()
  else()
      add_relative_test(${TEST_NAME} ${TEST_TARGET})
  endif()

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
      add_custom_command( TARGET ${TEST_TARGET} POST_BUILD COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different ${file_i} ${PROJECT_BINARY_DIR}/test/hipcub )
    endforeach( file_i )
  endif()
endfunction()

# Returns a list of values that match the pattern
# "if ${WORD} == <value>" with optional whitespace between the tokens.
function(get_match_list FILE_STRING WORD MATCH_LIST)
  # collect all substrings that match the pattern
  string(REGEX MATCHALL "${WORD}[ ]*==[ ]*[0-9]*" LINE_MATCHES "${${FILE_STRING}}")
  set(${MATCH_LIST} "")
  # iterate over the substrings, record the values using the same regex
  foreach(LINE IN LISTS LINE_MATCHES)
    string(REGEX MATCH "${WORD}[ ]*==[ ]*([0-9]*)" TMP "${LINE}")
    list(APPEND ${MATCH_LIST} "${CMAKE_MATCH_1}")
  endforeach()
  set(${MATCH_LIST} ${${MATCH_LIST}} PARENT_SCOPE)
endfunction()

# Replacement for add_hipcub_test that splits up test cases allowing them to be compiled in parallel.
# A single .cpp.in file is provided containing valid C++ code with the addition of slice definitions.
# The slice definitions HIPCUB_TEST_SLICE, HIPCUB_TEST_SUITE_SLICE, and HIPCUB_TEST_TYPE_SLICE demarkate
# slices of non-typed tests, typed test suites, and test types respectively. The slice cases must be
# marked with an "(el)if" statement (no "else") that has a unique value to ensure that the correct slice
# gets enabled. This function will generate a separate .cpp file for all non-typed test slices and the
# product of the typed test suites and test types.
#
# This example will generate five files:
#
# #cmakedefine HIPCUB_TEST_SUITE_SLICE @HIPCUB_TEST_SUITE_SLICE@
# #cmakedefine HIPCUB_TEST_TYPE_SLICE  @HIPCUB_TEST_TYPE_SLICE@
# #cmakedefine HIPCUB_TEST_SLICE       @HIPCUB_TEST_SLICE@
#
# #if   HIPCUB_TEST_SLICE == 0
#   DEFINE_NAMED_TEST(TestSuiteName, NonTypedTest)
# #endif
# #if   HIPCUB_TEST_SUITE_SLICE == 0
#   REGISTER_TYPED_TEST(TestSuiteName, TestZero, test_zero)
#   REGISTER_TYPED_TEST_SUITE(TestSuiteName, TestZero)
# #elif HIPCUB_TEST_SUITE_SLICE == 1
#   REGISTER_TYPED_TEST(TestSuiteName, TestOne, test_one)
#   REGISTER_TYPED_TEST(TestSuiteName, TestTwo, test_two)
#   REGISTER_TYPED_TEST_SUITE(TestSuiteName, TestOne, TestTwo)
# #endif
# #if   HIPCUB_TEST_TYPE_SLICE == 0
#   INSTANTIATE_TYPED_TEST(TestSuiteName, double)
# #elif HIPCUB_TEST_TYPE_SLICE == 1
#   INSTANTIATE_TYPED_TEST(TestSuiteName, float)
#   INSTANTIATE_TYPED_TEST(TestSuiteName, int)
# #endif
function(add_hipcub_test_parallel TEST_NAME TEST_SOURCE)
  get_hipcub_test_target(${TEST_SOURCE} TEST_TARGET)

  file(READ ${TEST_SOURCE} FILE_CONTENTS)

  set(SOURCES "")

  # first, handle all non-typed tests
  # disable typed test, generate one file for each non-typed test
  set(HIPCUB_TEST_SUITE_SLICE -1)
  set(HIPCUB_TEST_TYPE_SLICE -1)
  get_match_list(FILE_CONTENTS "HIPCUB_TEST_SLICE" TEST_SLICE_LIST)
  list(LENGTH TEST_SLICE_LIST TEST_SLICE_COUNT)
  if(TEST_SLICE_COUNT EQUAL 0)
    message(VERBOSE "found no non-typed tests for test target ${TEST_TARGET}")
  else()
    message(VERBOSE "found ${TEST_SLICE_COUNT} non-typed test slice(s) for test target ${TEST_TARGET}")
    foreach(HIPCUB_TEST_SLICE IN LISTS TEST_SLICE_LIST)
      set(FILENAME "${TEST_TARGET}.parallel/${TEST_TARGET}_${HIPCUB_TEST_SLICE}.cpp")
      configure_file(${TEST_SOURCE} ${FILENAME} @ONLY)
      list(APPEND SOURCES "${CMAKE_CURRENT_BINARY_DIR}/${FILENAME}")
    endforeach()
  endif()

  # second, handle all typed tests
  # disable non-typed test, generate one file for each test suite and test type pair
  set(HIPCUB_TEST_SLICE -1)
  get_match_list(FILE_CONTENTS "HIPCUB_TEST_SUITE_SLICE" TEST_SUITE_SLICE_LIST)
  list(LENGTH TEST_SUITE_SLICE_LIST TEST_SUITE_SLICE_COUNT)
  get_match_list(FILE_CONTENTS "HIPCUB_TEST_TYPE_SLICE" TEST_TYPE_SLICE_LIST)
  list(LENGTH TEST_TYPE_SLICE_LIST TEST_TYPE_SLICE_COUNT)
  if(TEST_SUITE_SLICE_COUNT EQUAL 0 OR TEST_TYPE_SLICE_COUNT EQUAL 0)
    message(VERBOSE "found no typed tests for test target ${TEST_TARGET}")
  else()
    message(VERBOSE "found ${TEST_SUITE_SLICE_COUNT} test suite slice(s) and \
${TEST_TYPE_SLICE_COUNT} test type slice(s) for test target ${TEST_TARGET}")
    foreach(HIPCUB_TEST_SUITE_SLICE IN LISTS TEST_SUITE_SLICE_LIST)
      foreach(HIPCUB_TEST_TYPE_SLICE IN LISTS TEST_TYPE_SLICE_LIST)
        set(FILENAME "${TEST_TARGET}.parallel/${TEST_TARGET}_typed_${HIPCUB_TEST_SUITE_SLICE}_${HIPCUB_TEST_TYPE_SLICE}.cpp")
        configure_file(${TEST_SOURCE} ${FILENAME} @ONLY)
        list(APPEND SOURCES "${CMAKE_CURRENT_BINARY_DIR}/${FILENAME}")
      endforeach()
    endforeach()
  endif()

  # if no files are generated, nothing is built for the target
  list(LENGTH SOURCES SOURCES_COUNT)
  if(${SOURCES_COUNT} EQUAL 0)
    message(FATAL_ERROR "no .cpp files generated for target ${TEST_TARGET}")
  endif()

  add_hipcub_test_internal(${TEST_NAME} "${SOURCES}" ${TEST_TARGET})
  target_include_directories("${TEST_TARGET}" PRIVATE "../../test/hipcub")
endfunction()

# ****************************************************************************
# Tests
# ****************************************************************************

# HIP basic test, which also checks if there are no linkage problems when there are multiple sources
add_hipcub_test("hipcub.BasicTest" "test_hipcub_basic.cpp;detail/get_hipcub_version.cpp")

# This test checks if hipCUB can compile with the __HIP_NO_HALF_CONVERSIONS__ and __HIP_NO_HALF_OPERATORS__ flags.
add_hipcub_test("hipcub.NoHalfOperators" test_hipcub_no_half_operators.cpp)

add_hipcub_test("hipcub.CachingDeviceAllocator" test_hipcub_caching_device_allocator.cpp)
add_hipcub_test("hipcub.BlockAdjacentDifference" test_hipcub_block_adjacent_difference.cpp)
add_hipcub_test("hipcub.BlockDiscontinuity" test_hipcub_block_discontinuity.cpp)
add_hipcub_test("hipcub.BlockExchange" test_hipcub_block_exchange.cpp)
add_hipcub_test("hipcub.BlockHistogram" test_hipcub_block_histogram.cpp)
add_hipcub_test("hipcub.BlockLoadStore" test_hipcub_block_load_store.cpp)
add_hipcub_test("hipcub.BlockMergeSort" test_hipcub_block_merge_sort.cpp)
add_hipcub_test("hipcub.BlockRadixRank" test_hipcub_block_radix_rank.cpp)
add_hipcub_test("hipcub.BlockRadixSort" test_hipcub_block_radix_sort.cpp)
add_hipcub_test("hipcub.BlockReduce" test_hipcub_block_reduce.cpp)
add_hipcub_test("hipcub.BlockRunLengthDecode" test_hipcub_block_run_length_decode.cpp)
add_hipcub_test("hipcub.BlockScan" test_hipcub_block_scan.cpp)
add_hipcub_test("hipcub.BlockShuffle" test_hipcub_block_shuffle.cpp)
add_hipcub_test("hipcub.DeviceAdjacentDifference" test_hipcub_device_adjacent_difference.cpp)
add_hipcub_test("hipcub.DeviceCopy" test_hipcub_device_copy.cpp)
add_hipcub_test("hipcub.DeviceFor" test_hipcub_device_for.cpp)
add_hipcub_test("hipcub.DeviceHistogram" test_hipcub_device_histogram.cpp)
add_hipcub_test("hipcub.DeviceMemcpy" test_hipcub_device_memcpy.cpp)
add_hipcub_test("hipcub.DeviceMerge" test_hipcub_device_merge.cpp)
add_hipcub_test("hipcub.DeviceMergeSort" test_hipcub_device_merge_sort.cpp)
add_hipcub_test_parallel("hipcub.DeviceRadixSort" test_hipcub_device_radix_sort.cpp.in)
add_hipcub_test("hipcub.DeviceReduce" test_hipcub_device_reduce.cpp)
add_hipcub_test("hipcub.DeviceRunLengthEncode" test_hipcub_device_run_length_encode.cpp)
add_hipcub_test("hipcub.DeviceReduceByKey" test_hipcub_device_reduce_by_key.cpp)
add_hipcub_test("hipcub.DeviceScan" test_hipcub_device_scan.cpp)
add_hipcub_test_parallel("hipcub.DeviceSegmentedRadixSort" test_hipcub_device_segmented_radix_sort.cpp.in)
add_hipcub_test("hipcub.DeviceSegmentedReduce" test_hipcub_device_segmented_reduce.cpp)
add_hipcub_test_parallel("hipcub.DeviceSegmentedSort" test_hipcub_device_segmented_sort.cpp.in)
add_hipcub_test("hipcub.DeviceSelect" test_hipcub_device_select.cpp)
add_hipcub_test("hipcub.DeviceSpmv" test_hipcub_device_spmv.cpp)
add_hipcub_test("hipcub.DeviceTransform" test_hipcub_device_transform.cpp)
add_hipcub_test("hipcub.DevicePartition" test_hipcub_device_partition.cpp)
add_hipcub_test("hipcub.Grid" test_hipcub_grid.cpp)
add_hipcub_test("hipcub.UtilPtx" test_hipcub_util_ptx.cpp)
add_hipcub_test("hipcub.UtilDevice" test_hipcub_util_device.cpp)
add_hipcub_test("hipcub.Vector" test_hipcub_vector.cpp)
add_hipcub_test("hipcub.WarpExchange" test_hipcub_warp_exchange.cpp)
add_hipcub_test("hipcub.WarpLoad" test_hipcub_warp_load.cpp)
add_hipcub_test("hipcub.WarpMergeSort" test_hipcub_warp_merge_sort.cpp)
add_hipcub_test("hipcub.WarpReduce" test_hipcub_warp_reduce.cpp)
add_hipcub_test("hipcub.WarpScan" test_hipcub_warp_scan.cpp)
add_hipcub_test("hipcub.WarpStore" test_hipcub_warp_store.cpp)
add_hipcub_test("hipcub.Iterators" test_hipcub_iterators.cpp)
add_hipcub_test("hipcub.Thread" test_hipcub_thread.cpp)
add_hipcub_test("hipcub.ThreadOperators" test_hipcub_thread_operators.cpp)
add_hipcub_test("hipcub.ThreadSort" test_hipcub_thread_sort.cpp)

add_hipcub_test("hipcub.SinglePassScanOperators" test_hipcub_single_pass_scan_operators.cpp)
