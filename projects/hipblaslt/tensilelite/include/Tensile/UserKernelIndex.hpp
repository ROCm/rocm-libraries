// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace TensileLite
{
    /**
     * Solution index ranges:
     *
     *   [0, 2^30)          shipped library
     *   [2^30, 3 * 2^29)   reserved for JIT-generated kernels
     *   [3 * 2^29, 2^31)   kernels registered at runtime
     *
     * A user kernel index is valid only in the process that assigned it.
     */
    constexpr int UserKernelIndexBase = 3 << 29;

    constexpr bool isUserKernelIndex(int index)
    {
        return index >= UserKernelIndexBase;
    }
} // namespace TensileLite
