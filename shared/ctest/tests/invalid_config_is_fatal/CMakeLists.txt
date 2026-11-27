# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Configuring this project has to fail: the YAML below carries a pattern that the gtest
# parser rejects, and apply_test_category_labels() turns the parser's "invalid config"
# status into a configure error. The registering test (../CMakeLists.txt) passes only when
# the error message appears.

cmake_minimum_required(VERSION 3.25.2)
project(shared-ctest-invalid-config-is-fatal LANGUAGES NONE)

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
    test_patterns:
      - "Suite?.Test"
    labels:
      - quick
]=]
)

apply_test_category_labels(invalid_config_target "${_yaml_file}" "${_working_dir}")

message(STATUS "apply_test_category_labels returned for an invalid YAML")
