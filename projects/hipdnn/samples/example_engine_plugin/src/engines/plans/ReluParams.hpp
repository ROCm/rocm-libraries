// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// TEMPLATE REFERENCE: The approach of using a POD struct to hold the parameters extracted
// from the operation graph by the PlanBuilder is not required as the parameters can
// be passed directly to the Plan instead. The POD struct approach can be useful if there
// are a large number of parameters. Replace with your operation's parameters.

#pragma once

#include <cstdint>

namespace example_provider
{

/// Parameters for the ReLU forward plan.
struct ReluParams
{
    int64_t inputUid; // UID for the input tensor.
    int64_t outputUid; // UID for the output tensor.
    int64_t numElements; // Total number of elements in the tensor.
    double negativeSlope; // Slope for negative inputs (0.0 = standard ReLU, >0 = leaky ReLU).
};

} // namespace example_provider
