// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The pooler / unpooler bench instances with A RowMajor, B ColumnMajor and the tdmstaged epilogue.

#include "gemm_pooler_unpooler_bench.hpp"

namespace pooler_bench {

int dispatch_rcr_tdmstaged(const ck_tile::ArgParser& arg,
                           const Shape& shape,
                           const std::string& name)
{
    return dispatch_layout<RowMajor, ColumnMajor, true>(arg, shape, name);
}

} // namespace pooler_bench
