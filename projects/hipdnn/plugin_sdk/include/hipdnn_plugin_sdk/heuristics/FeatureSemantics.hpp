// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

namespace hipdnn_plugin_sdk::heuristics
{

/// @brief The revision of what this build's published features MEAN.
///
/// A UHD's `features_hash` fingerprints its signature -- which names it reads -- but not
/// the C++ that computes the values behind those names. A model trained on `graph.flops`
/// under one FLOP convention and scored under another reads the same name, passes the same
/// hash, and silently mispredicts. This revision is what closes that gap: uhd_gen records
/// it in `trained_against.feature_semantics_revision` when it trains (read from
/// `hipdnn_uhd_features`, never restated in Python), and the loader refuses a model whose
/// recorded revision is not this one.
///
/// Bump it when any published feature changes meaning or value for the same graph and
/// device: a FLOP or byte convention, an operand's encoding (dims order, data-type codes,
/// what `virtual` means), or a feature's name. Adding a new feature name does NOT bump it:
/// no existing model can read a name it was never trained on.
///
/// A document that records no revision was trained against revision 1 -- every model
/// shipped before this constant existed -- so a bump makes all of them refuse, which is
/// the point.
inline constexpr int64_t FEATURE_SEMANTICS_REVISION = 1;

} // namespace hipdnn_plugin_sdk::heuristics
