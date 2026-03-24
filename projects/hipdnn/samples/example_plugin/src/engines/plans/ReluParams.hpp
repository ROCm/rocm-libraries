// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

namespace example_plugin
{

/// Parameters for the ReLU forward plan.
struct ReluParams
{
    int64_t inputUid; // UID for the input tensor.
    int64_t outputUid; // UID for the output tensor.
    int64_t numElements; // Total number of elements in the tensor.
    double negativeSlope; // Slope for negative inputs (0.0 = standard ReLU, >0 = leaky ReLU).
};

} // namespace example_plugin
