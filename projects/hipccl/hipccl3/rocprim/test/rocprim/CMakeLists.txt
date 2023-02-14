# MIT License
#
# Copyright (c) 2017-2023 Advanced Micro Devices, Inc. All rights reserved.
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

set(AMDGPU_TEST_TARGETS "" CACHE STRING "List of specific device types to test for") # Leave empty for default system device

# Gets a test target name based on the first source file.
function(get_rocprim_test_target TEST_SOURCES TEST_TARGET)
  list(GET TEST_SOURCES 0 TEST_MAIN_SOURCE)
  get_filename_component(TEST_TARGET ${TEST_MAIN_SOURCE} NAME_WE)
  set(TEST_TARGET ${TEST_TARGET} PARENT_SCOPE)
endfunction()

function(add_rocprim_test TEST_NAME TEST_SOURCES)
  get_rocprim_test_target(${TEST_SOURCES} TEST_TARGET)
  add_rocprim_test_internal(${TEST_NAME} "${TEST_SOURCES}" ${TEST_TARGET})
endfunction()

function(add_rocprim_test_internal TEST_NAME TEST_SOURCES TEST_TARGET)
  add_executable(${TEST_TARGET} ${TEST_SOURCES})
  rocm_install(TARGETS ${TEST_TARGET} COMPONENT tests)

  target_include_directories(${TEST_TARGET} SYSTEM BEFORE
    PUBLIC
      $<BUILD_INTERFACE:${COMMON_TEST_HEADER_DIRECTORY}>
  )

  target_link_libraries(${TEST_TARGET}
    PRIVATE
      GTest::GTest
      GTest::Main
  )
  if(NOT USE_HIP_CPU)
    target_link_libraries(${TEST_TARGET}
      PRIVATE
        rocprim_hip
    )
  else()
    target_link_libraries(${TEST_TARGET}
      PRIVATE
        rocprim
        Threads::Threads
        hip_cpu_rt::hip_cpu_rt
    )
    if(STL_DEPENDS_ON_TBB)
      target_link_libraries(${TEST_TARGET}
        PRIVATE
          TBB::tbb
      )
    endif()
  endif()

  target_compile_options(${TEST_TARGET}
    PRIVATE
      $<$<CXX_COMPILER_ID:MSVC>:
        /bigobj # number of sections exceeded object file format limit: compile with /bigobj
      >
  )

  set_target_properties(${TEST_TARGET}
    PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test/rocprim"
  )
  if(AMDGPU_TEST_TARGETS)
    foreach(AMDGPU_TARGET IN LISTS AMDGPU_TEST_TARGETS)
      add_relative_test("${AMDGPU_TARGET}-${TEST_NAME}" ${TEST_TARGET})
      set_tests_properties("${AMDGPU_TARGET}-${TEST_NAME}"
          PROPERTIES
              RESOURCE_GROUPS "1,${AMDGPU_TARGET}:1"
              LABELS "hip;${AMDGPU_TARGET}"
      )
    endforeach()
  else()
      add_relative_test(${TEST_NAME} ${TEST_TARGET})
      set_tests_properties(${TEST_NAME}
          PROPERTIES
              LABELS "hip"
      )
  endif()

  if (WIN32 AND NOT DEFINED DLLS_COPIED_2)
    set(DLLS_COPIED_2 "YES")
    set(DLLS_COPIED_2 ${DLLS_COPIED_2} PARENT_SCOPE)
    # for now adding in all .dll as dependency chain is not cmake based on win32
    file( GLOB third_party_dlls
    LIST_DIRECTORIES ON
    CONFIGURE_DEPENDS
    ${HIP_DIR}/bin/*.dll
    ${CMAKE_SOURCE_DIR}/rtest.*
    )
    foreach( file_i ${third_party_dlls})
      add_custom_command( TARGET ${TEST_TARGET} POST_BUILD COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different ${file_i} ${PROJECT_BINARY_DIR}/test/rocprim )
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

# Replacement for add_rocprim_test that splits up test cases allowing them to be compiled in parallel.
# A single .cpp.in file is provided containing valid C++ code with the addition of slice definitions.
# The slice definitions ROCPRIM_TEST_SLICE, ROCPRIM_TEST_SUITE_SLICE, and ROCPRIM_TEST_TYPE_SLICE demarkate
# slices of non-typed tests, typed test suites, and test types respectively. The slice cases must be
# marked with an "(el)if" statement (no "else") that has a unique value to ensure that the correct slice
# gets enabled. This function will generate a separate .cpp file for all non-typed test slices and the
# product of the typed test suites and test types.
#
# This example will generate five files:
#
# #cmakedefine ROCPRIM_TEST_SUITE_SLICE @ROCPRIM_TEST_SUITE_SLICE@
# #cmakedefine ROCPRIM_TEST_TYPE_SLICE  @ROCPRIM_TEST_TYPE_SLICE@
# #cmakedefine ROCPRIM_TEST_SLICE       @ROCPRIM_TEST_SLICE@
#
# #if   ROCPRIM_TEST_SLICE == 0
#   DEFINE_NAMED_TEST(TestSuiteName, NonTypedTest)
# #endif
# #if   ROCPRIM_TEST_SUITE_SLICE == 0
#   REGISTER_TYPED_TEST(TestSuiteName, TestZero, test_zero)
#   REGISTER_TYPED_TEST_SUITE(TestSuiteName, TestZero)
# #elif ROCPRIM_TEST_SUITE_SLICE == 1
#   REGISTER_TYPED_TEST(TestSuiteName, TestOne, test_one)
#   REGISTER_TYPED_TEST(TestSuiteName, TestTwo, test_two)
#   REGISTER_TYPED_TEST_SUITE(TestSuiteName, TestOne, TestTwo)
# #endif
# #if   ROCPRIM_TEST_TYPE_SLICE == 0
#   INSTANTIATE_TYPED_TEST(TestSuiteName, double)
# #elif ROCPRIM_TEST_TYPE_SLICE == 1
#   INSTANTIATE_TYPED_TEST(TestSuiteName, float)
#   INSTANTIATE_TYPED_TEST(TestSuiteName, int)
# #endif
function(add_rocprim_test_parallel TEST_NAME TEST_SOURCE)
  get_rocprim_test_target(${TEST_SOURCE} TEST_TARGET)

  file(READ ${TEST_SOURCE} FILE_CONTENTS)

  set(SOURCES "")

  # first, handle all non-typed tests
  # disable typed test, generate one file for each non-typed test
  set(ROCPRIM_TEST_SUITE_SLICE -1)
  set(ROCPRIM_TEST_TYPE_SLICE -1)
  get_match_list(FILE_CONTENTS "ROCPRIM_TEST_SLICE" TEST_SLICE_LIST)
  list(LENGTH TEST_SLICE_LIST TEST_SLICE_COUNT)
  if(TEST_SLICE_COUNT EQUAL 0)
    message(VERBOSE "found no non-typed tests for test target ${TEST_TARGET}")
  else()
    message(VERBOSE "found ${TEST_SLICE_COUNT} non-typed test slice(s) for test target ${TEST_TARGET}")
    foreach(ROCPRIM_TEST_SLICE IN LISTS TEST_SLICE_LIST)
      set(FILENAME "${TEST_TARGET}.parallel/${TEST_TARGET}_${ROCPRIM_TEST_SLICE}.cpp")
      configure_file(${TEST_SOURCE} ${FILENAME} @ONLY)
      list(APPEND SOURCES "${CMAKE_CURRENT_BINARY_DIR}/${FILENAME}")
    endforeach()
  endif()

  # second, handle all typed tests
  # disable non-typed test, generate one file for each test suite and test type pair
  set(ROCPRIM_TEST_SLICE -1)
  get_match_list(FILE_CONTENTS "ROCPRIM_TEST_SUITE_SLICE" TEST_SUITE_SLICE_LIST)
  list(LENGTH TEST_SUITE_SLICE_LIST TEST_SUITE_SLICE_COUNT)
  get_match_list(FILE_CONTENTS "ROCPRIM_TEST_TYPE_SLICE" TEST_TYPE_SLICE_LIST)
  list(LENGTH TEST_TYPE_SLICE_LIST TEST_TYPE_SLICE_COUNT)
  if(TEST_SUITE_SLICE_COUNT EQUAL 0 OR TEST_TYPE_SLICE_COUNT EQUAL 0)
    message(VERBOSE "found no typed tests for test target ${TEST_TARGET}")
  else()
    message(VERBOSE "found ${TEST_SUITE_SLICE_COUNT} test suite slice(s) and \
${TEST_TYPE_SLICE_COUNT} test type slice(s) for test target ${TEST_TARGET}")
    foreach(ROCPRIM_TEST_SUITE_SLICE IN LISTS TEST_SUITE_SLICE_LIST)
      foreach(ROCPRIM_TEST_TYPE_SLICE IN LISTS TEST_TYPE_SLICE_LIST)
        set(FILENAME "${TEST_TARGET}.parallel/${TEST_TARGET}_typed_${ROCPRIM_TEST_SUITE_SLICE}_${ROCPRIM_TEST_TYPE_SLICE}.cpp")
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

  set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_CLEAN_FILES "${TEST_TARGET}.parallel")
  add_rocprim_test_internal(${TEST_NAME} "${SOURCES}" ${TEST_TARGET})
  target_include_directories("${TEST_TARGET}" PRIVATE "../../test/rocprim")
endfunction()

# ****************************************************************************
# Tests
# ****************************************************************************

# HIP basic test, which also checks if there are no linkage problems when there are multiple sources
add_rocprim_test("rocprim.basic_test" "test_basic.cpp;detail/get_rocprim_version.cpp")

add_rocprim_test("rocprim.arg_index_iterator" test_arg_index_iterator.cpp)
add_rocprim_test("rocprim.temporary_storage_partitioning" test_temporary_storage_partitioning.cpp)
add_rocprim_test_parallel("rocprim.block_adjacent_difference" test_block_adjacent_difference.cpp.in)
add_rocprim_test("rocprim.block_discontinuity" test_block_discontinuity.cpp)
add_rocprim_test("rocprim.block_exchange" test_block_exchange.cpp)
add_rocprim_test("rocprim.block_histogram" test_block_histogram.cpp)
add_rocprim_test("rocprim.block_load_store" test_block_load_store.cpp)
add_rocprim_test("rocprim.block_sort_merge" test_block_sort_merge.cpp)
add_rocprim_test("rocprim.block_sort_merge_stable" test_block_sort_merge_stable.cpp)
add_rocprim_test_parallel("rocprim.block_radix_rank" test_block_radix_rank.cpp.in)
add_rocprim_test("rocprim.block_radix_sort" test_block_radix_sort.cpp)
add_rocprim_test("rocprim.block_reduce" test_block_reduce.cpp)
add_rocprim_test_parallel("rocprim.block_scan" test_block_scan.cpp.in)
add_rocprim_test("rocprim.block_shuffle" test_block_shuffle.cpp)
add_rocprim_test("rocprim.block_sort_bitonic" test_block_sort_bitonic.cpp)
add_rocprim_test("rocprim.config_dispatch" test_config_dispatch.cpp)
add_rocprim_test("rocprim.constant_iterator" test_constant_iterator.cpp)
add_rocprim_test("rocprim.counting_iterator" test_counting_iterator.cpp)
add_rocprim_test("rocprim.device_binary_search" test_device_binary_search.cpp)
add_rocprim_test("rocprim.device_adjacent_difference" test_device_adjacent_difference.cpp)
add_rocprim_test("rocprim.device_histogram" test_device_histogram.cpp)
add_rocprim_test("rocprim.device_merge" test_device_merge.cpp)
add_rocprim_test("rocprim.device_merge_sort" test_device_merge_sort.cpp)
add_rocprim_test("rocprim.device_partition" test_device_partition.cpp)
add_rocprim_test_parallel("rocprim.device_radix_sort" test_device_radix_sort.cpp.in)
add_rocprim_test("rocprim.device_reduce_by_key" test_device_reduce_by_key.cpp)
add_rocprim_test("rocprim.device_reduce" test_device_reduce.cpp)
add_rocprim_test("rocprim.device_run_length_encode" test_device_run_length_encode.cpp)
add_rocprim_test("rocprim.device_scan" test_device_scan.cpp)
add_rocprim_test_parallel("rocprim.device_segmented_radix_sort" test_device_segmented_radix_sort.cpp.in)
add_rocprim_test("rocprim.device_segmented_reduce" test_device_segmented_reduce.cpp)
add_rocprim_test("rocprim.device_segmented_scan" test_device_segmented_scan.cpp)
add_rocprim_test("rocprim.device_select" test_device_select.cpp)
add_rocprim_test("rocprim.device_transform" test_device_transform.cpp)
add_rocprim_test("rocprim.discard_iterator" test_discard_iterator.cpp)
add_rocprim_test("rocprim.reverse_iterator" test_reverse_iterator.cpp)
if(NOT USE_HIP_CPU)
  add_rocprim_test("rocprim.texture_cache_iterator" test_texture_cache_iterator.cpp)
endif()
add_rocprim_test("rocprim.thread" test_thread.cpp)
add_rocprim_test("rocprim.thread_algos" test_thread_algos.cpp)
add_rocprim_test("rocprim.transform_iterator" test_transform_iterator.cpp)
add_rocprim_test("rocprim.no_half_operators" test_no_half_operators.cpp)
add_rocprim_test("rocprim.intrinsics" test_intrinsics.cpp)
add_rocprim_test("rocprim.warp_exchange" test_warp_exchange.cpp)
add_rocprim_test("rocprim.warp_load" test_warp_load.cpp)
add_rocprim_test("rocprim.warp_reduce" test_warp_reduce.cpp)
add_rocprim_test("rocprim.warp_scan" test_warp_scan.cpp)
add_rocprim_test("rocprim.warp_sort" test_warp_sort.cpp)
add_rocprim_test("rocprim.warp_store" test_warp_store.cpp)
add_rocprim_test("rocprim.zip_iterator" test_zip_iterator.cpp)
