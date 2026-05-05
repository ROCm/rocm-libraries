// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

struct PoolBwdDefaultPolicy
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetVectorSize()
    {
        return Problem::BlockShape::kVectorSize;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetBlockSize()
    {
        return Problem::BlockShape::kBlockSize;
    }
};

} // namespace ck_tile
