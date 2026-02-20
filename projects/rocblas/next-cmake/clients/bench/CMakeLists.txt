# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(_CMAKE_CURRENT_SOURCE_DIR ${TEMP_CLIENTS_SOURCE_DIR}/benchmarks)

add_executable(rocblas-bench)

set_target_properties(rocblas-bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                               "${PROJECT_BINARY_DIR}/staging")

target_compile_definitions(rocblas-bench PRIVATE GOOGLE_TEST)

target_link_libraries(rocblas-bench PRIVATE rocblas-clients-common)

if(ROCBLAS_ENABLE_ASAN)
  rocblas_target_configure_sanitizers(rocblas-bench PRIVATE)
endif()

add_subdirectory(include)
add_subdirectory(src)
