# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25.2)
project(shared-ctest-optional-args-scope LANGUAGES NONE)

enable_testing()

if(NOT DEFINED ROCM_LIBRARIES_ROOT)
    get_filename_component(ROCM_LIBRARIES_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../.." ABSOLUTE)
endif()

include("${ROCM_LIBRARIES_ROOT}/shared/ctest/TestCategories.cmake")

set(_working_dir "${CMAKE_CURRENT_BINARY_DIR}/staging")
file(MAKE_DIRECTORY "${_working_dir}")

set(_yaml_file "${CMAKE_CURRENT_BINARY_DIR}/test_categories.yaml")
file(WRITE "${_yaml_file}"
[=[
test_categories:
  quick:
    description: Minimal regression suite
    test_patterns:
      - "*"
    labels:
      - quick
execution_settings:
  category_timeouts:
    quick: 30
]=]
)

set(_install_ctest_file "${CMAKE_CURRENT_BINARY_DIR}/install_CTestTestfile.cmake")
file(WRITE "${_install_ctest_file}" "# generated test file\n")

# Assert that a generated file contains an expected regex.
function(_assert_file_contains file_path needle)
    file(READ "${file_path}" contents)
    if(NOT contents MATCHES "${needle}")
        message(FATAL_ERROR "Expected ${file_path} to contain: ${needle}")
    endif()
endfunction()

# Assert that a generated file omits an unexpected regex.
function(_assert_file_omits file_path needle)
    file(READ "${file_path}" contents)
    if(contents MATCHES "${needle}")
        message(FATAL_ERROR "Expected ${file_path} to omit: ${needle}")
    endif()
endfunction()

# Call apply_test_category_labels from a parent scope with its own ARGV4.
function(_call_without_resource_group)
    apply_test_category_labels(
        fake_tests
        "${_yaml_file}"
        "${_working_dir}"
        "${_install_ctest_file}"
    )
endfunction()

_call_without_resource_group(one two three four EXPECT_TARGET)
_assert_file_contains("${CMAKE_CURRENT_BINARY_DIR}/test_categories.cmake" "fake_tests_quick_suite")
_assert_file_omits("${CMAKE_CURRENT_BINARY_DIR}/test_categories.cmake" "EXPECT_TARGET")
_assert_file_omits("${CMAKE_CURRENT_BINARY_DIR}/test_categories.cmake" "RESOURCE_GROUPS")
_assert_file_contains("${_install_ctest_file}" "fake_tests_quick_suite")
_assert_file_omits("${_install_ctest_file}" "EXPECT_TARGET")
_assert_file_omits("${_install_ctest_file}" "RESOURCE_GROUPS")

set(_resource_install_ctest_file "${CMAKE_CURRENT_BINARY_DIR}/resource_install_CTestTestfile.cmake")
file(WRITE "${_resource_install_ctest_file}" "# generated test file\n")

apply_test_category_labels(
    fake_resource_tests
    "${_yaml_file}"
    "${_working_dir}"
    "${_resource_install_ctest_file}"
    gpus
)
_assert_file_contains("${CMAKE_CURRENT_BINARY_DIR}/test_categories.cmake" "fake_resource_tests_gpus_quick_suite")
_assert_file_contains("${CMAKE_CURRENT_BINARY_DIR}/test_categories.cmake" "RESOURCE_GROUPS \\\"1,gpus:1\\\"")
_assert_file_contains("${_resource_install_ctest_file}" "fake_resource_tests_gpus_quick_suite")
_assert_file_contains("${_resource_install_ctest_file}" "RESOURCE_GROUPS \\\"1,gpus:1\\\"")
