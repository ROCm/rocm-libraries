// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

namespace hipdnn_plugin_sdk::heuristics
{

/// @brief Revision of what this build's published features mean.
///
/// `features_hash` covers feature names, not how their values are computed; the loader
/// refuses a model trained against another revision. Bump it when any published feature
/// changes meaning or value (FLOP/byte convention, operand encoding, renames), not when a
/// feature is added. Models that record no revision are revision 1.
///
/// Revision 2: SDPA backward counts each product on its own head dimension
/// (2·pairs·(3·D_qk + 2·D_v) instead of 2.5x the forward), max-pool backward counts one
/// operation per dy element instead of the window per element, and graph.logical_bytes counts
/// a paged KV cache at the keys the graph declares instead of every block.
inline constexpr int64_t FEATURE_SEMANTICS_REVISION = 2;

} // namespace hipdnn_plugin_sdk::heuristics
