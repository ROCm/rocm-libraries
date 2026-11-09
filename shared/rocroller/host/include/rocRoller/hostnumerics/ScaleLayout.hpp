// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <optional>
#include <roc/hostnumerics/reshape_permutation.hpp>

namespace rocRoller::hostnumerics
{
    // Scale layout consumed when the kernel skips its lane exchanges. Dimensions
    // count scales; the reduction extent is contiguous in the natural input.
    struct ScaleSwizzle
    {
        size_t tileReduction;
        size_t tileNonReduction;
        size_t subTileReduction;
    };

    // Natural and optional outer-tile shapes are [nonReduction, reduction].
    // The swizzle uses 16-lane groups in a 64-lane wave, with non-reduction
    // width 32 or 64. tileReduction must be a nonzero multiple of 4, and
    // subTileReduction must be nonzero. Width 32 requires subTileReduction 2 or 4.
    // Swizzle and outer tiles must divide their containing shapes; no padding
    // is added. Invalid ranks, factors or divisibility throw std::runtime_error.
    roc::hostnumerics::ReshapePermutation
        ScaleStoragePermutation(const roc::hostnumerics::Shape&                natural,
                                const std::optional<ScaleSwizzle>&             swizzle,
                                const std::optional<roc::hostnumerics::Shape>& tile);
}
