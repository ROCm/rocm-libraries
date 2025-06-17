# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_executable(rocblas-example-gemv-graph-capture
               ${TEMP_CLIENT_SAMPLES_DIR}/example_gemv_graph_capture.cpp)

target_link_libraries(rocblas-example-gemv-graph-capture
                      PRIVATE sample-helper-cpp)

set_target_properties(
  rocblas-example-gemv-graph-capture PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                ${PROJECT_BINARY_DIR}/staging)
