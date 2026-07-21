# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(TEMP_CLIENT_SAMPLES_DIR ${TEMP_CLIENTS_SOURCE_DIR}/samples)
set(_CMAKE_CURRENT_SOURCE_DIR ${TEMP_CLIENT_SAMPLES_DIR})

add_library(sample-helper-common INTERFACE)

if(ROCBLAS_ENABLE_BLIS)
  target_link_libraries(sample-helper-common INTERFACE BLIS::BLIS)
endif()
target_link_libraries(sample-helper-common INTERFACE hip::host hip::device
                                                     roc::rocblas)
target_include_directories(
  sample-helper-common
  INTERFACE $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/../include>
            $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/../common>)
set_target_properties(
  sample-helper-common PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                  "${PROJECT_BINARY_DIR}/staging")

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_definitions(sample-helper-common
                             INTERFACE ROCBLAS_INTERNAL_API)
endif()

add_library(sample-helper-cpp OBJECT)

target_compile_features(sample-helper-cpp PUBLIC cxx_std_14)
target_compile_definitions(sample-helper-cpp PUBLIC ROCM_USE_FLOAT16)
target_link_libraries(sample-helper-cpp PUBLIC sample-helper-common)
target_sources(
  sample-helper-cpp
  PRIVATE
    "${_CMAKE_CURRENT_SOURCE_DIR}/../common/client_utility.cpp"
    "${_CMAKE_CURRENT_SOURCE_DIR}/../common/singletons.cpp"
    "${_CMAKE_CURRENT_SOURCE_DIR}/../common/rocblas_random.cpp"
    "${_CMAKE_CURRENT_SOURCE_DIR}/../common/host_alloc.cpp"
    "${_CMAKE_CURRENT_SOURCE_DIR}/../include/client_utility.hpp"
)

add_library(sample-helper-c INTERFACE)

target_compile_definitions(sample-helper-c INTERFACE __HIP_PLATFORM_HCC__)
target_compile_features(sample-helper-c INTERFACE c_std_11)
target_link_libraries(sample-helper-c INTERFACE sample-helper-common)

if(ROCBLAS_ENABLE_ASAN)
  rocblas_target_configure_sanitizers(sample-helper-cpp PUBLIC)
  rocblas_target_configure_sanitizers(sample-helper-c INTERFACE)
endif()

if(ROCBLAS_ENABLE_FORTRAN)
  add_library(sample-helper-f90 INTERFACE)
  # Link this here because we don't need to link against rocblas-clients-common
  # in the samples
  target_link_libraries(
    sample-helper-f90 INTERFACE roc::rocblas
                                rocblas::rocblas-fortran-module
                                rocblas::rocblas-fortran-client
                                hip::device)
  endif()

# Unused examples:
# example_gemm_strided_batched_ex_xdl_math_op.cpp
# example_openmp.cpp

# C examples
add_subdirectory(example_c_dgeam)

# C++ examples
add_subdirectory(example_gemv_graph_capture)
add_subdirectory(example_header_check)
add_subdirectory(example_hip_complex_her2)
add_subdirectory(example_scal_multiple_strided_batch)
add_subdirectory(example_scal_template)
add_subdirectory(example_solver_rocblas)
add_subdirectory(example_sscal)
if(ROCBLAS_ENABLE_TENSILE)
  add_subdirectory(example_user_driven_tuning)
endif()

# C++ L3 examples with or with tensile or source gemm builds
add_subdirectory(example_sgemm)
add_subdirectory(example_sgemm_multiple_strided_batch)
add_subdirectory(example_sgemm_strided_batched)

if(ROCBLAS_ENABLE_FORTRAN)
  # Fortran examples
  add_subdirectory(example_fortran_axpy)
  add_subdirectory(example_fortran_scal)
  add_subdirectory(example_fortran_gemv)
endif()
