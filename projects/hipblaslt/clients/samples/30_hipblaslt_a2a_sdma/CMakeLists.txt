# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

option(HIPBLASLT_ENABLE_A2A_SDMA_SAMPLE
    "Build the fused GEMM+A2A SDMA sample (needs hsakmt)" OFF)

if(NOT HIPBLASLT_ENABLE_A2A_SDMA_SAMPLE OR WIN32)
    return()
endif()

if(NOT HIPBLASLT_ENABLE_GEMM_A2A_FUSION)
    message(FATAL_ERROR "HIPBLASLT_ENABLE_A2A_SDMA_SAMPLE requires HIPBLASLT_ENABLE_GEMM_A2A_FUSION=ON")
endif()

add_executable(sample_a2a_sdma
    "${CMAKE_CURRENT_SOURCE_DIR}/sample_a2a_sdma.cpp")
target_link_libraries(sample_a2a_sdma
    PRIVATE sample-helper hsakmt::hsakmt hsa-runtime64::hsa-runtime64)

# Per-source: <SdmaQueue.hpp> lives in the TensileLite client include dir.
set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/sample_a2a_sdma.cpp
    PROPERTIES
        COMPILE_OPTIONS "-I${HIPBLASLT_SDMA_QUEUE_INCLUDE_DIR}"
)
