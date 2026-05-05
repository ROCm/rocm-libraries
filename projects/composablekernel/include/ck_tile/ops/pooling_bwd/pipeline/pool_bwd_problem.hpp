// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <typename DOutDataType_,
          typename IndexDataType_,
          typename DInDataType_,
          typename BlockShape_,
          bool HasOverlap_>
struct PoolBwdProblem
{
    using DOutDataType  = remove_cvref_t<DOutDataType_>;
    using IndexDataType = remove_cvref_t<IndexDataType_>;
    using DInDataType   = remove_cvref_t<DInDataType_>;
    using BlockShape    = remove_cvref_t<BlockShape_>;

    static constexpr bool kHasOverlap = HasOverlap_;

    static constexpr bool kDInIsFp32OrFp64 =
        std::is_same_v<DInDataType, float> || std::is_same_v<DInDataType, double>;

    using DInAtomicAddPreCast = std::conditional_t<kDInIsFp32OrFp64, DInDataType, float>;

    static constexpr bool kNeedFp32Workspace = kHasOverlap && !kDInIsFp32OrFp64;
};

} // namespace ck_tile
