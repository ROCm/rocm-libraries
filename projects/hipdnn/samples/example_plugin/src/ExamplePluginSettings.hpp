// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

/// Plugin-specific execution settings.
///
/// Holds settings that control execution behavior, populated from
/// engine configuration knobs during initializeExecutionSettings().
struct ExamplePluginSettings
{
    /// Negative slope for leaky ReLU (0.0 = standard ReLU).
    /// Controlled by the "example.relu.negative_slope" knob.
    double reluNegativeSlope = 0.0;
};
