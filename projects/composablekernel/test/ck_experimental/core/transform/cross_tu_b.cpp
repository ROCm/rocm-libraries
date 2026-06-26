// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/// @file
/// @brief TU B of the V4 cross-TU NTTP equality test. Identical to TU A by
///        construction (same buildSampleSplicedGraph() call from same header).

#include "cross_tu_graph.hpp"

namespace cross_tu_v4 {

const int* tuBCanary()
{
    constexpr auto g = buildSampleSplicedGraph();
    return kCanaryFor<g>.data();
}

} // namespace cross_tu_v4
