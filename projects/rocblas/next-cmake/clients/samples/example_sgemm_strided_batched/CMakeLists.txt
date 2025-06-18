# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_executable(rocblas-example-sgemm-strided-batched
               ${TEMP_CLIENT_SAMPLES_DIR}/example_sgemm_strided_batched.cpp)

target_link_libraries(rocblas-example-sgemm-strided-batched
                      PRIVATE sample-helper-cpp)

set_target_properties(
  rocblas-example-sgemm-multiple-strided-batch
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/staging)
