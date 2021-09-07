# ########################################################################
# Copyright 2019 Advanced Micro Devices, Inc.
# ########################################################################

if(NOT CMAKE_VERSION VERSION_LESS 3.12)
  list(JOIN PRNG_SEEDS ", " PRNG_SEEDS_INITIALIZER)
else() # Workaround for not having string(JOIN) and list(JOIN)
  # https://stackoverflow.com/a/31880875/1476661
  function (ListToString result delim)
    list(GET ARGV 2 temp)
    math(EXPR N "${ARGC}-1")
    if((${N} GREATER 3) OR (${N} EQUAL 3))
      foreach(IDX RANGE 3 ${N})
        list(GET ARGV ${IDX} STR)
        set(temp "${temp}${delim}${STR}")
      endforeach()
    endif()
    set(${result} "${temp}" PARENT_SCOPE)
  endfunction(ListToString)
  ListToString(PRNG_SEEDS_INITIALIZER ", " ${PRNG_SEEDS})
  message(${PRNG_SEEDS_INITIALIZER})
endif()

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/test_seed.in.hpp
    ${CMAKE_CURRENT_BINARY_DIR}/test_seed.hpp
)

function(add_relative_test test_name test_target)
    get_target_property(EXE_PATH ${test_target} RUNTIME_OUTPUT_DIRECTORY)
    if(EXE_PATH STREQUAL "EXE_PATH-NOTFOUND")
        set(EXE_PATH ".")
    endif()
    get_filename_component(EXE_PATH "${EXE_PATH}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_BINARY_DIR}")
    get_target_property(EXE_NAME ${test_target} RUNTIME_OUTPUT_NAME)
    if(EXE_NAME STREQUAL "EXE_NAME-NOTFOUND")
        get_target_property(EXE_NAME ${test_target} OUTPUT_NAME)
        if(EXE_NAME STREQUAL "EXE_NAME-NOTFOUND")
            set(EXE_NAME "${test_target}")
        endif()
    endif()
    file(RELATIVE_PATH rel_path "${CMAKE_CURRENT_BINARY_DIR}" "${EXE_PATH}/${EXE_NAME}")
    add_test(NAME "${test_name}" COMMAND "./${rel_path}")
endfunction()

function(add_rocthrust_test TEST)
    set(TEST_SOURCE "test_${TEST}.cpp")
    set(TEST_TARGET "${TEST}.hip")
    add_executable(${TEST_TARGET} ${TEST_SOURCE})
    target_include_directories(${TEST_TARGET} SYSTEM BEFORE
        PUBLIC
            ${GTEST_INCLUDE_DIRS}
            ${CMAKE_CURRENT_BINARY_DIR}
    )
    target_link_libraries(${TEST_TARGET}
        PRIVATE
            rocthrust
            roc::rocprim_hip
            ${GTEST_BOTH_LIBRARIES}
    )
    if (NOT WIN32)
      foreach(amdgpu_target ${AMDGPU_TARGETS})
          target_link_libraries(${TEST_TARGET}
              PRIVATE
                  --amdgpu-target=${amdgpu_target}
          )
      endforeach()
    endif()
    set_target_properties(${TEST_TARGET}
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test/"
    )
    if(AMDGPU_TEST_TARGETS)
    foreach(AMDGPU_TARGET IN LISTS AMDGPU_TEST_TARGETS)
        add_test("${AMDGPU_TARGET}-${TEST_TARGET}" ${TEST_TARGET})
        set_tests_properties("${AMDGPU_TARGET}-${TEST_TARGET}"
            PROPERTIES
                RESOURCE_GROUPS "1,${AMDGPU_TARGET}:1"
                LABELS "hip;${AMDGPU_TARGET}"
        )
    endforeach()
    else()
        add_relative_test(${TEST_TARGET} ${TEST_TARGET})
        set_tests_properties(${TEST_TARGET}
            PROPERTIES
                LABELS "hip"
        )
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
        add_custom_command( TARGET ${TEST_TARGET} POST_BUILD COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different ${file_i} ${PROJECT_BINARY_DIR}/test )
      endforeach( file_i )
    endif()        
endfunction()

# ****************************************************************************
# Tests
# ****************************************************************************

add_rocthrust_test("adjacent_difference")
add_rocthrust_test("advance")
add_rocthrust_test("allocator")
add_rocthrust_test("allocator_aware_policies")
add_rocthrust_test("async_copy")
add_rocthrust_test("async_reduce")
add_rocthrust_test("async_sort")
add_rocthrust_test("async_transform")
add_rocthrust_test("binary_search")
add_rocthrust_test("binary_search_descending")
add_rocthrust_test("binary_search_vector")
add_rocthrust_test("binary_search_vector_descending")
add_rocthrust_test("complex")
add_rocthrust_test("complex_transform")
add_rocthrust_test("constant_iterator")
add_rocthrust_test("copy")
add_rocthrust_test("copy_n")
add_rocthrust_test("count")
add_rocthrust_test("counting_iterator")
add_rocthrust_test("dereference")
add_rocthrust_test("device_delete")
add_rocthrust_test("device_ptr")
add_rocthrust_test("device_reference")
add_rocthrust_test("discard_iterator")
add_rocthrust_test("distance")
add_rocthrust_test("equal")
add_rocthrust_test("fill")
add_rocthrust_test("find")
add_rocthrust_test("for_each")
add_rocthrust_test("gather")
add_rocthrust_test("generate")
add_rocthrust_test("inner_product")
add_rocthrust_test("is_sorted")
add_rocthrust_test("is_partitioned")
add_rocthrust_test("is_sorted_until")
add_rocthrust_test("max_element")
add_rocthrust_test("memory")
add_rocthrust_test("merge")
add_rocthrust_test("merge_by_key")
add_rocthrust_test("min_element")
add_rocthrust_test("minmax_element")
add_rocthrust_test("mismatch")
add_rocthrust_test("mr_disjoint_pool")
add_rocthrust_test("mr_new")
add_rocthrust_test("mr_pool")
add_rocthrust_test("mr_pool_options")
add_rocthrust_test("pair")
add_rocthrust_test("pair_reduce")
add_rocthrust_test("pair_scan")
add_rocthrust_test("pair_sort")
add_rocthrust_test("pair_transform")
add_rocthrust_test("partition")
add_rocthrust_test("partition_point")
add_rocthrust_test("permutation_iterator")
add_rocthrust_test("random")
add_rocthrust_test("reduce")
add_rocthrust_test("reduce_by_key")
add_rocthrust_test("remove")
add_rocthrust_test("replace")
add_rocthrust_test("reverse_iterator")
add_rocthrust_test("set_difference")
add_rocthrust_test("set_difference_by_key")
add_rocthrust_test("set_difference_by_key_descending")
add_rocthrust_test("set_difference_descending")
add_rocthrust_test("set_intersection")
add_rocthrust_test("set_intersection_by_key")
add_rocthrust_test("set_intersection_by_key_descending")
add_rocthrust_test("set_intersection_descending")
add_rocthrust_test("set_intersection_key_value")
add_rocthrust_test("set_symmetric_difference")
add_rocthrust_test("set_symmetric_difference_by_key_descending")
add_rocthrust_test("set_symmetric_difference_by_key")
add_rocthrust_test("shuffle")
add_rocthrust_test("scan")
add_rocthrust_test("scan_by_key")
add_rocthrust_test("scatter")
add_rocthrust_test("sequence")
add_rocthrust_test("stable_sort")
add_rocthrust_test("stable_sort_by_key")
add_rocthrust_test("stable_sort_by_key_large")
add_rocthrust_test("stable_sort_large")
add_rocthrust_test("sort")
add_rocthrust_test("sort_by_key")
add_rocthrust_test("sort_by_key_variable_bits")
add_rocthrust_test("sort_permutation_iterator")
add_rocthrust_test("sort_variables")
add_rocthrust_test("swap_ranges")
add_rocthrust_test("tabulate")
add_rocthrust_test("transform")
add_rocthrust_test("transform_iterator")
add_rocthrust_test("transform_reduce")
add_rocthrust_test("transform_scan")
add_rocthrust_test("tuple")
add_rocthrust_test("tuple_reduce")
add_rocthrust_test("tuple_sort")
add_rocthrust_test("tuple_transform")
add_rocthrust_test("uninitialized_copy")
add_rocthrust_test("uninitialized_fill")
add_rocthrust_test("unique")
add_rocthrust_test("unique_by_key")
add_rocthrust_test("vector")
add_rocthrust_test("vector_allocators")
add_rocthrust_test("vector_insert")
add_rocthrust_test("vector_manipulation")
add_rocthrust_test("zip_iterator")
add_rocthrust_test("zip_iterator_reduce")
add_rocthrust_test("zip_iterator_scan")
add_rocthrust_test("zip_iterator_sort")
add_rocthrust_test("zip_iterator_sort_by_key")
add_rocthrust_test("zip_iterator_reduce_by_key")
