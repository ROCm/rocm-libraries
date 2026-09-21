// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// TEMPLATE REFERENCE: Convolution parameter struct

#pragma once

#include <cstdint>

namespace example_provider
{

/// Parameters for the convolution forward plan.
struct ConvFwdParams
{
    int64_t inputUid; // UID for the input tensor.
    int64_t weightUid; // UID for the weight/filter tensor.
    int64_t outputUid; // UID for the output tensor.
    int64_t n; // Batch size.
    int64_t c; // Number of input channels.
    int64_t h; // Input height.
    int64_t w; // Input width.
    int64_t k; // Number of output channels (filters).
    int64_t r; // Filter height.
    int64_t s; // Filter width.
    int64_t outH; // Output height.
    int64_t outW; // Output width.
    int64_t padH; // Vertical padding.
    int64_t padW; // Horizontal padding.
    int64_t strideH; // Vertical stride.
    int64_t strideW; // Horizontal stride.
    int64_t blockSize; // HIP thread block size for the kernel launch.
};

} // namespace example_provider
