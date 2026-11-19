// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include "fft_enums.h"
#include <cstddef>
#include <vector>

namespace reference_fft
{
    // Dimensions use the same row-major ordering as fft_params, with strides in
    // real or complex elements according to the transform's input/output types.
    struct dimension
    {
        std::ptrdiff_t length;
        std::ptrdiff_t input_stride;
        std::ptrdiff_t output_stride;
    };
}
