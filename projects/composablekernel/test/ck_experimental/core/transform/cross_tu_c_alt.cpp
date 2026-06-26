// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/// @file
/// @brief TU C of the V4 cross-TU NTTP equality test. Builds the ALT
///        spliced graph to prove the test mechanism has discriminating power.

#include "cross_tu_graph.hpp"

namespace cross_tu_v4 {

const int* tuCCanary()
{
    constexpr auto g = buildAltSplicedGraph();
    return kCanaryFor<g>.data();
}

} // namespace cross_tu_v4
