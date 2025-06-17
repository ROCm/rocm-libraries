# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_executable(
  rocblas-example-sgemm-multiple-strided-batch
  ${TEMP_CLIENT_SAMPLES_DIR}/example_sgemm_multiple_strided_batch.cpp)

target_link_libraries(rocblas-example-sgemm-multiple-strided-batch
                      PRIVATE sample-helper-cpp)

set_target_properties(
  rocblas-example-sgemm-multiple-strided-batch
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/staging)
