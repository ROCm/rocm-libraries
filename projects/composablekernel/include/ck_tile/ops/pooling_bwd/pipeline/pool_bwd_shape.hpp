// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <index_t BlockSize_ = 256, index_t VectorSize_ = 4>
struct PoolBwdShape
{
    static constexpr index_t kBlockSize  = BlockSize_;
    static constexpr index_t kVectorSize = VectorSize_;

    static_assert(kBlockSize > 0, "kBlockSize must be positive");
    static_assert(kVectorSize == 1 || kVectorSize == 2 || kVectorSize == 4,
                  "kVectorSize must be 1, 2, or 4");
};

} // namespace ck_tile
