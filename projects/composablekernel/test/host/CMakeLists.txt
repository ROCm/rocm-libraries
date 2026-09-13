# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# A deliberately small collection of whole test executables which exercise
# host-side CK utilities without allocating memory or launching a HIP kernel.
# Keeping this list explicit prevents newly added device tests from silently
# entering CPU-only sanitizer jobs.
set(_CK_HOST_TEST_ROOT "${CMAKE_CURRENT_LIST_DIR}/..")

add_gtest_executable(unit_sequence
    "${_CK_HOST_TEST_ROOT}/util/unit_sequence.cpp")
add_gtest_executable(unit_ford
    "${_CK_HOST_TEST_ROOT}/util/unit_ford.cpp")
add_gtest_executable(unit_sequence_helper
    "${_CK_HOST_TEST_ROOT}/util/unit_sequence_helper.cpp")
add_gtest_executable(unit_tensor_descriptor_functors
    "${_CK_HOST_TEST_ROOT}/util/unit_tensor_descriptor_functors.cpp")
add_gtest_executable(unit_index_expression
    "${_CK_HOST_TEST_ROOT}/util/unit_index_expression.cpp")

add_gtest_executable(test_ck_tile_tuple_apply
    "${_CK_HOST_TEST_ROOT}/ck_tile/container/test_tuple_apply.cpp")
add_gtest_executable(ck_tile_unit_sequence
    "${_CK_HOST_TEST_ROOT}/ck_tile/core/container/unit_sequence.cpp")
add_gtest_executable(test_ck_tile_sequence
    "${_CK_HOST_TEST_ROOT}/ck_tile/utility/test_sequence.cpp")
add_gtest_executable(test_ck_tile_static_ford
    "${_CK_HOST_TEST_ROOT}/ck_tile/utility/test_static_ford.cpp")

set(_CK_HOST_PRINT_TEST_ROOT "${_CK_HOST_TEST_ROOT}/ck_tile/utility/print")
add_gtest_executable(test_print_sequence
    "${_CK_HOST_PRINT_TEST_ROOT}/test_print_sequence.cpp")
add_gtest_executable(test_print_array
    "${_CK_HOST_PRINT_TEST_ROOT}/test_print_array.cpp")
add_gtest_executable(test_print_tuple
    "${_CK_HOST_PRINT_TEST_ROOT}/test_print_tuple.cpp")
add_gtest_executable(test_print_coordinate_transform
    "${_CK_HOST_PRINT_TEST_ROOT}/test_print_coordinate_transform.cpp")
add_gtest_executable(test_print_static_encoding_pattern
    "${_CK_HOST_PRINT_TEST_ROOT}/test_print_static_encoding_pattern.cpp")
add_gtest_executable(test_print_buffer_view
    "${_CK_HOST_PRINT_TEST_ROOT}/test_print_buffer_view.cpp")
add_gtest_executable(test_print_basic_types
    "${_CK_HOST_PRINT_TEST_ROOT}/test_print_basic_types.cpp")

add_gtest_executable(test_custom_type
    "${_CK_HOST_TEST_ROOT}/data_type/test_custom_type.cpp")
add_gtest_executable(test_type_convert_const
    "${_CK_HOST_TEST_ROOT}/data_type/type_convert_const.cpp")

foreach(_ck_host_test
        unit_sequence
        unit_ford
        unit_sequence_helper
        unit_tensor_descriptor_functors
        unit_index_expression
        test_ck_tile_tuple_apply
        ck_tile_unit_sequence
        test_ck_tile_sequence
        test_ck_tile_static_ford
        test_custom_type)
    if(TARGET ${_ck_host_test})
        target_link_libraries(${_ck_host_test} PRIVATE utility)
    endif()
endforeach()
