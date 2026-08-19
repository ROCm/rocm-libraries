// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace roc::host_validation {

// Selects which end of a shape changes fastest when converting between tensor
// coordinates and a logical linear index.
enum class IndexOrder {
    FirstDimensionFastest,
    LastDimensionFastest,
};

// Compatibility names retained while generation and comparison consumers
// converge on the component-wide IndexOrder vocabulary.
using LogicalIndexOrder = IndexOrder;
using ComparisonIndexOrder = IndexOrder;

}  // namespace roc::host_validation
