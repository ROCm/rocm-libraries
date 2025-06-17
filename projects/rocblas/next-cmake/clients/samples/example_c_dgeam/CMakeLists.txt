# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_executable(rocblas-example-c-dgeam
               ${TEMP_CLIENT_SAMPLES_DIR}/example_c_dgeam.c)

target_link_libraries(rocblas-example-c-dgeam PRIVATE sample-helper-c)

set_target_properties(
  rocblas-example-c-dgeam PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                     ${PROJECT_BINARY_DIR}/staging)
