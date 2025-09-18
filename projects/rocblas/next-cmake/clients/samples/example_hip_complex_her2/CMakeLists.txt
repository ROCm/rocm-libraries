# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_executable(rocblas-example-hip-complex-her2
               ${TEMP_CLIENT_SAMPLES_DIR}/example_hip_complex_her2.cpp)

target_link_libraries(rocblas-example-hip-complex-her2
                      PRIVATE sample-helper-cpp)

set_target_properties(
  rocblas-example-hip-complex-her2 PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                              ${PROJECT_BINARY_DIR}/staging)
