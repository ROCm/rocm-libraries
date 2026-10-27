# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_executable(sample_hipblaslt_gemm_w4a16 "${CMAKE_CURRENT_SOURCE_DIR}/sample_hipblaslt_gemm_w4a16.cpp")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    target_compile_options(sample_hipblaslt_gemm_w4a16 PRIVATE -mno-avx)
endif()
target_link_libraries(sample_hipblaslt_gemm_w4a16 PRIVATE sample-helper)
