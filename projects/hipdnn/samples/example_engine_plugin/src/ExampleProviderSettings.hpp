// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// TEMPLATE ADAPTATION: Use this struct if your plugin supports knobs. Rename the struct
// and remove the reluNegativeSlope field, then add your own settings fields.
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
};
