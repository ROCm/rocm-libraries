// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace roc::host_numerics {

// Selects which end of a shape changes fastest when converting between tensor
// coordinates and a logical linear index.
enum class IndexOrder {
    FirstDimensionFastest,
    LastDimensionFastest,
};

}  // namespace roc::host_numerics
