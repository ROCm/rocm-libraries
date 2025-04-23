###############################################################################
 #
 # MIT License
 #
 # Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
 #
 # Permission is hereby granted, free of charge, to any person obtaining a copy
 # of this software and associated documentation files (the "Software"), to deal
 # in the Software without restriction, including without limitation the rights
 # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 # copies of the Software, and to permit persons to whom the Software is
 # furnished to do so, subject to the following conditions:
 #
 # The above copyright notice and this permission notice shall be included in
 # all copies or substantial portions of the Software.
 #
 # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 # SOFTWARE.
 #
 ###############################################################################

include( CMakeDependentOption )

cmake_dependent_option( ROCWMMA_VALIDATE_WITH_ROCBLAS "Use rocBLAS for validation" ON "ROCWMMA_BUILD_VALIDATION_TESTS" OFF )
cmake_dependent_option( ROCWMMA_BENCHMARK_WITH_ROCBLAS "Include rocBLAS benchmark performance comparisons" OFF "ROCWMMA_BUILD_BENCHMARK_TESTS" OFF )

set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CMAKE_COMMAND} -E time")
set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "${CMAKE_COMMAND} -E time")

if(ROCWMMA_VALIDATE_WITH_ROCBLAS OR ROCWMMA_BENCHMARK_WITH_ROCBLAS)
  find_package( rocblas REQUIRED PATHS /opt/rocm /opt/rocm/rocblas $ENV{ROCBLAS_DIR} )
  rocm_package_add_dependencies("rocblas >= 2.32.0" COMPONENT tests)
endif()

set(ROCWMMA_TEST_GEMM_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} ${ROCWMMA_TEST_GEMM_INCLUDE_DIRS})
set(ROCWMMA_GEMM_TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# Custom target to build all rocWMMA gemm-validation tests
if(ROCWMMA_BUILD_VALIDATION_TESTS)
  add_custom_target(rocwmma_gemm_tests_validate)
endif()

# Custom target to build all rocWMMA gemm-benchmark tests
if(ROCWMMA_BUILD_BENCHMARK_TESTS)
  add_custom_target(rocwmma_gemm_tests_bench)
endif()

### GEMM tests that have rocBLAS support
# Use rocBLAS for validation
function(add_gemm_validation_test TEST_TARGET TEST_SOURCE)
  list(APPEND TEST_SOURCE ${ARGN})

  # Create target
  add_rocwmma_validation_test(${TEST_TARGET} ${TEST_SOURCE})

  # Add gemm include directory
  target_include_directories(${TEST_TARGET} PRIVATE ${ROCWMMA_TEST_GEMM_INCLUDE_DIRS})

  # Put binary outputs in the same directory
  set_target_properties(${TEST_TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${ROCWMMA_GEMM_TEST_OUTPUT_DIR})

  # Add dependency to custom target
  add_dependencies(rocwmma_gemm_tests_validate ${TEST_TARGET})

  # Link to rocBLAS
  if(ROCWMMA_VALIDATE_WITH_ROCBLAS)
    target_link_libraries(${TEST_TARGET} roc::rocblas)
    target_compile_definitions(${TEST_TARGET} PRIVATE ROCWMMA_VALIDATE_WITH_ROCBLAS)
  endif()
endfunction()

# Include rocBLAS performance benchmark
function(add_gemm_benchmark_test TEST_TARGET TEST_SOURCE)
  list(APPEND TEST_SOURCE ${ARGN})

  # Create target
  add_rocwmma_benchmark_test(${TEST_TARGET} ${TEST_SOURCE})

  # Add gemm include directory
  target_include_directories(${TEST_TARGET} PRIVATE ${ROCWMMA_TEST_GEMM_INCLUDE_DIRS})

  # Put binary outputs in the same directory
  set_target_properties(${TEST_TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${ROCWMMA_GEMM_TEST_OUTPUT_DIR})

  # Add dependency to custom target
  add_dependencies(rocwmma_gemm_tests_bench ${TEST_TARGET})

  # Link to rocBLAS
  if(ROCWMMA_BENCHMARK_WITH_ROCBLAS)
    target_link_libraries(${TEST_TARGET} roc::rocblas)
    target_compile_definitions(${TEST_TARGET} PRIVATE ROCWMMA_BENCHMARK_WITH_ROCBLAS)
  endif()
endfunction()

# Create tests based on config
function(add_gemm_test TEST_TARGET_PREFIX TEST_SOURCE)
  list(APPEND TEST_SOURCE ${ARGN})
  if(ROCWMMA_BUILD_BENCHMARK_TESTS)
    add_gemm_benchmark_test(${TEST_TARGET_PREFIX}-bench ${TEST_SOURCE})
  endif()
  if(ROCWMMA_BUILD_VALIDATION_TESTS)
    add_gemm_validation_test(${TEST_TARGET_PREFIX}-validate ${TEST_SOURCE})
  endif()
endfunction()

# GEMM common test sources
set(GemmCommonSources ${ROCWMMA_COMMON_TEST_SOURCES}
                      ${CMAKE_CURRENT_SOURCE_DIR}/gemm_kernel_base.cpp
                      ${CMAKE_CURRENT_SOURCE_DIR}/gemm_resource.cpp)

# Tests for cooperative kernel classes
add_subdirectory(gemm_PGR1_LB2_MP0_MB_CP)

# Tests for non-cooperative kernel classes
add_subdirectory(gemm_PGR0_LB0_MP0_SB_NC)
add_subdirectory(gemm_PGR0_LB0_MP0_MB_NC)
