# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_executable(rocblas-example-solver
               ${TEMP_CLIENT_SAMPLES_DIR}/example_solver_rocblas.cpp)

target_link_libraries(rocblas-example-solver PRIVATE sample-helper-cpp)

# This is over including files, but rocblase_device_malloc.cpp is used in this
# sample and it's the only place it's used anywhere in rocBLAS.
target_include_directories(
  rocblas-example-solver
  PRIVATE ${TEMP_CLIENT_SAMPLES_DIR}/../../library/src/include)

# test deprecated API without warnings until they are removed.  Then this entire example can be deleted TODO
target_compile_options(
  rocblas-example-solver PRIVATE -Wno-deprecated)

set_target_properties(
  rocblas-example-solver PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                    ${PROJECT_BINARY_DIR}/staging)
