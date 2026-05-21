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

// Padding semantics vary per kernel family:
//   OGradDotO / ConvertDQ: bool (pad or no-pad)
//   DqDkDv: int {0=none, 1=small, 8=full vector-aligned}
// The tri-valued int maps to CK Tile's TileFmhaBwdTraits::kPadHeadDimQ/V
// which controls vector load widths. 0 = no padding, 1 = scalar fallback,
// 8 = full 128-bit vector loads with padding. bool is sufficient for
// OGradDotO/ConvertDQ which only need on/off.

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
