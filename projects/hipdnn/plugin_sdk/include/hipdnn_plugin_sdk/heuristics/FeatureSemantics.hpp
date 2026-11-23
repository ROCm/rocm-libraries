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
inline constexpr int64_t FEATURE_SEMANTICS_REVISION = 1;

} // namespace hipdnn_plugin_sdk::heuristics
