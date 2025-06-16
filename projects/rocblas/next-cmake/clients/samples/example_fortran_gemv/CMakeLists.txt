# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_executable(rocblas-example-fortran-gemv
               ${TEMP_CLIENT_SAMPLES_DIR}/example_fortran_gemv.f90)

target_link_libraries(rocblas-example-fortran-gemv PRIVATE sample-helper-f90)

set_target_properties(
  rocblas-example-fortran-gemv PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                          ${PROJECT_BINARY_DIR}/staging)
