// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

// LSE window plus a window for the per-row max of the scores (the max logit), which has the same
// layout as the LSE. The forward pipelines only use the LSE window through store_tile(), so this
// pair can be passed wherever the LSE window is expected.
template <typename LseWindow, typename MaxWindow>
struct FmhaLseMaxWindow
{
    LseWindow lse;
    MaxWindow max;
};

template <typename LseWindow, typename MaxWindow, typename DistributedTensor>
CK_TILE_DEVICE void store_tile(FmhaLseMaxWindow<LseWindow, MaxWindow>& window,
                               const DistributedTensor& tensor)
{
    store_tile(window.lse, tensor);
}

template <typename LseWindow, typename MaxWindow, typename DistributedTensor>
CK_TILE_DEVICE void store_row_max(FmhaLseMaxWindow<LseWindow, MaxWindow>& window,
                                  const DistributedTensor& tensor)
{
    store_tile(window.max, tensor);
}

} // namespace ck_tile
