# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_executable(rocblas-example-scal-template
               ${TEMP_CLIENT_SAMPLES_DIR}/example_scal_template.cpp)

target_link_libraries(rocblas-example-scal-template PRIVATE sample-helper-cpp)

set_target_properties(
  rocblas-example-scal-template PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                           ${PROJECT_BINARY_DIR}/staging)
