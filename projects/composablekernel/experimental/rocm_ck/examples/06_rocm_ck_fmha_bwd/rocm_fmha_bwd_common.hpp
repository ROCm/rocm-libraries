// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Shared types for all FMHA BWD kernel families (OGradDotO, DqDkDv, ConvertDq).
//
// This header has NO CK Tile dependency. It is included by both host code
// (main.cpp) and device code (.hip files).

#pragma once

#include <rocm_ck/datatype_utils.hpp>
#include <rocm_ck/types.hpp>

#include <cstdint>

namespace rocm_ck {

/// 64-bit index type for large strides (batch_stride × nhead can exceed int32).
/// Matches ck_tile::long_index_t.
using long_index_t = std::int64_t;

/// FMHA attention mode: fixed-length batches vs variable-length groups.
enum class FmhaMode
{
    BATCH,
    GROUP
};

/// Bias type for attention score modification.
/// Values must match ck_tile::BlockAttentionBiasEnum.
enum class FmhaBiasType
{
    NONE,
    ELEMENTWISE,
    ALIBI
};

} // namespace rocm_ck
