# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_executable(rocblas-example-sgemm
               ${TEMP_CLIENT_SAMPLES_DIR}/example_sgemm.cpp)

target_link_libraries(rocblas-example-sgemm PRIVATE sample-helper-cpp)

set_target_properties(
  rocblas-example-sgemm PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                   ${PROJECT_BINARY_DIR}/staging)
