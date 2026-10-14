// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

// LSE window plus a window for the per-row max logits, which have the same layout as the LSE.
// The forward pipelines only use the LSE window through store_tile(), so this
// pair can be passed wherever the LSE window is expected.
template <typename LseWindow, typename MaxLogitsWindow>
struct FmhaLseMaxLogitsWindow
{
    LseWindow lse;
    MaxLogitsWindow max_logits;
};

template <typename LseWindow, typename MaxLogitsWindow, typename DistributedTensor>
CK_TILE_DEVICE void store_tile(FmhaLseMaxLogitsWindow<LseWindow, MaxLogitsWindow>& window,
                               const DistributedTensor& tensor)
{
    store_tile(window.lse, tensor);
}

template <typename LseWindow, typename MaxLogitsWindow, typename DistributedTensor>
CK_TILE_DEVICE void store_max_logits(FmhaLseMaxLogitsWindow<LseWindow, MaxLogitsWindow>& window,
                                     const DistributedTensor& tensor)
{
    store_tile(window.max_logits, tensor);
}

} // namespace ck_tile
