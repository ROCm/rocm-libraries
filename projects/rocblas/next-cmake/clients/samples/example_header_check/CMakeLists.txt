# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_executable(rocblas-example-header-check
               ${TEMP_CLIENT_SAMPLES_DIR}/example_header_check.cpp)

target_link_libraries(rocblas-example-header-check PRIVATE sample-helper-cpp)

set_target_properties(
  rocblas-example-header-check PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                          ${PROJECT_BINARY_DIR}/staging)
