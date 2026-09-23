// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "fmha_fwd.hpp"

ck_tile::index_t
fmha_fwd_block_scale_size_kv(const std::string&, ck_tile::index_t, ck_tile::index_t)
{
    // TDM V128 test kernels only support no_scale.
    return fmha_fwd_largest_n_tile_size;
}
