# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(_CMAKE_CURRENT_SOURCE_DIR ${TEMP_CLIENTS_SOURCE_DIR}/benchmarks/include)

target_sources(rocblas-bench
               PRIVATE "${_CMAKE_CURRENT_SOURCE_DIR}/../program_options.hpp")
