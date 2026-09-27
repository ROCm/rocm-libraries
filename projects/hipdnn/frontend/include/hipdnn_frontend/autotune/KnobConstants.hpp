// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file KnobConstants.hpp
 * @brief Local definitions of well-known knob names used by the autotune module
 *
 * Provides autotune-local copies of knob name constants to avoid a compile-time
 * dependency on plugin_sdk (which defines the canonical versions in
 * GlobalKnobDefines.hpp). The values must remain in sync with plugin_sdk.
 */

#pragma once

#include <hipdnn_frontend/knob/Knob.hpp>

#include <algorithm>
#include <vector>

namespace hipdnn_frontend::autotune::detail
{

// Name of the benchmarking knob that triggers engine-internal cache priming.
// Managed exclusively by autotune() in EXHAUSTIVE mode; rejected/stripped
// by all add_engine_*() functions.
static constexpr const char* BENCHMARKING_KNOB_NAME = "global.benchmarking";

// True if the engine's knob list contains the benchmarking knob, meaning the
// engine supports exhaustive priming.
inline bool knobsSupportExhaustive(const std::vector<Knob>& knobs)
{
    return std::any_of(knobs.begin(), knobs.end(), [](const Knob& knob) {
        return knob.knobId() == BENCHMARKING_KNOB_NAME;
    });
}

} // namespace hipdnn_frontend::autotune::detail
