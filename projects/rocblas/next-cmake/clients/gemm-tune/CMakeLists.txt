# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(_CMAKE_CURRENT_SOURCE_DIR ${TEMP_CLIENTS_SOURCE_DIR}/benchmarks/gemm_tune)

add_executable(rocblas-gemm-tune)

set_target_properties(
  rocblas-gemm-tune PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                               ${PROJECT_BINARY_DIR}/staging)

target_compile_definitions(
  rocblas-gemm-tune PRIVATE ROCBLAS_BENCH ROCBLAS_NO_DEPRECATED_WARNINGS
                            __HIP_HCC_COMPAT_MODE__=1)

target_link_libraries(rocblas-gemm-tune PRIVATE rocblas-clients-common)

if(ROCBLAS_ENABLE_ASAN)
  rocblas_target_configure_sanitizers(rocblas-gemm-tune PRIVATE)
endif()

add_subdirectory(include)
add_subdirectory(src)
