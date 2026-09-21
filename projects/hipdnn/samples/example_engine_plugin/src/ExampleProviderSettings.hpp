// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// TEMPLATE ADAPTATION: Use this struct if your plugin supports knobs. Rename the struct
// and replace the fields with your own settings.
// These fields are populated by your PlanBuilder's initializeExecutionSettings()
// method from engine config knobs, then read by buildPlan() to create operation parameters.

/// Plugin-specific execution settings.
///
/// Holds settings that control execution behavior, populated from
/// engine configuration knobs during initializeExecutionSettings().
struct ExampleProviderSettings
{
    /// Negative slope for leaky ReLU (0.0 = standard ReLU).
    /// Controlled by the "example.relu.negative_slope" knob.
    double reluNegativeSlope = 0.0;

    /// Thread block size for the convolution kernel.
    /// Controlled by the "BLOCK_SIZE" knob.
    int64_t blockSize = 256;
};
