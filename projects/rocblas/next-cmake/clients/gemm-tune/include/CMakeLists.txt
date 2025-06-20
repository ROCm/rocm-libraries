# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(_CMAKE_CURRENT_SOURCE_DIR
    ${TEMP_CLIENTS_SOURCE_DIR}/benchmarks/gemm_tune/include)

target_sources(rocblas-gemm-tune
               PRIVATE "${_CMAKE_CURRENT_SOURCE_DIR}/../gemm_tuners.hpp")
