// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

namespace hip_kernel_provider_common
{

enum MaskType : int
{
    NO_MASK = 0,
    TOP_LEFT_CAUSAL,
    BOTTOM_RIGHT_CAUSAL,
    WINDOW_GENERIC
};

enum RoundingMode : int
{
    RTNE = 0, // Round to Nearest Even (IEEE default)
    RTNA, // Round to Nearest Away from zero
    RTZ // Round toward Zero
};

enum BatchMode : int
{
    BATCH = 0, // All sequences have same length
    GROUP // Variable sequence lengths
};

}
