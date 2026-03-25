// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Shared type definitions for kpack API.

#pragma once

#include <cstdint>

namespace rocm_ck {

/// Fixed-width integer type for kernel index/size arguments.
/// Matches ck_tile::index_t but avoids pulling in CK Tile headers.
using index_t = std::int32_t;

/// 64-bit index type for large strides (batch_stride * nhead can exceed int32).
/// Matches ck_tile::long_index_t.
using long_index_t = std::int64_t;

/// AMD GCN/CDNA wavefront size.
constexpr int warp_size = 64;

} // namespace rocm_ck
