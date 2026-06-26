// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/// @file
/// @brief TU A of the V4 cross-TU NTTP equality test. Returns the address
///        of kCanaryFor<g> where g = buildSampleSplicedGraph().

#include "cross_tu_graph.hpp"

namespace cross_tu_v4 {

const int* tuACanary()
{
    constexpr auto g = buildSampleSplicedGraph();
    return kCanaryFor<g>.data();
}

} // namespace cross_tu_v4
