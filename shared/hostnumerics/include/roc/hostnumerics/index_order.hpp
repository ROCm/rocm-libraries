// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace roc::hostnumerics {

// Selects which end of a shape changes fastest when converting between tensor
// coordinates and a logical linear index.
// This order defines enumeration (for example, generation and output sampling),
// independently of physical strides. It does not constrain tensor layouts.
enum class IndexOrder {
    FirstDimensionFastest,
    LastDimensionFastest,
    Count,
};

}  // namespace roc::hostnumerics
