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

namespace hipdnn_frontend
{
namespace autotune
{

/// Name of the benchmarking knob that triggers engine-internal cache priming.
/// Managed exclusively by autotune() in EXHAUSTIVE mode; rejected/stripped
/// by all add_engine_*() functions.
static constexpr const char* BENCHMARKING_KNOB_NAME = "global.benchmarking";

} // namespace autotune
} // namespace hipdnn_frontend
