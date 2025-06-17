# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(_CMAKE_CURRENT_SOURCE_DIR ${TEMP_CLIENTS_SOURCE_DIR}/gtest/include)

target_sources(
  rocblas-test
  PRIVATE "${_CMAKE_CURRENT_SOURCE_DIR}/../blas_ex/blas1_ex_gtest.hpp")
