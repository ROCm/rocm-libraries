# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_executable(rocblas-example-fortran-scal
               ${TEMP_CLIENT_SAMPLES_DIR}/example_fortran_scal.f90)

target_link_libraries(rocblas-example-fortran-scal PRIVATE sample-helper-f90)

set_target_properties(
  rocblas-example-fortran-scal PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                          ${PROJECT_BINARY_DIR}/staging)
