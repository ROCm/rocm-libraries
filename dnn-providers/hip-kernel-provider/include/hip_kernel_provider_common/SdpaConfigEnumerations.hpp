// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

namespace hip_kernel_provider_common
{

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
