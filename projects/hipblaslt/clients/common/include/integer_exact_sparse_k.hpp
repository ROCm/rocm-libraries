// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>

// The sparse_k integer_exact pattern keeps each row of A nonzero at one K index in each of
// kIntegerExactSparseKTerms equal stretches of K and at the last K index. Shared by the device
// fills (hipblaslt_init.hpp) and the MX generator (mxDataGen.cpp), which must agree.
constexpr size_t kIntegerExactSparseKTerms = 16;

// True when sparse_k keeps K index k of A's row `row` nonzero. constexpr so device fills can
// call it.
constexpr bool integer_exact_sparse_k_kept(size_t k, size_t K, size_t row)
{
    const size_t width = (K + kIntegerExactSparseKTerms - 1) / kIntegerExactSparseKTerms;
    const size_t s     = width > 0 ? width : 1;
    return k % s == (k / s + row) % s || k + 1 == K;
}
