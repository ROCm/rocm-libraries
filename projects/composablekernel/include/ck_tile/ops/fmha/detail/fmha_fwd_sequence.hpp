// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile::detail {

// Physical starts describe packed storage. Logical lengths take precedence so
// attention never includes padding between sequences. Batch mode passes null
// starts and lengths, and may override its padded length with cumulative lengths.
CK_TILE_HOST_DEVICE constexpr index_t fmha_fwd_sequence_length(bool group,
                                                               index_t batch,
                                                               index_t padded_length,
                                                               const int32_t* lengths,
                                                               const int32_t* cumulative,
                                                               const int32_t* starts)
{
    if(lengths)
        return lengths[batch];
    if(cumulative)
        return cumulative[batch + 1] - cumulative[batch];
    if(group)
        return starts[batch + 1] - starts[batch];
    return padded_length;
}

} // namespace ck_tile::detail
