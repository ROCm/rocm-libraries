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
     * A local kernel index is valid only in the process that assigned it.
     */
    constexpr int LocalKernelIndexBase = 3 << 29;

    constexpr bool isLocalKernelIndex(int index)
    {
        return index >= LocalKernelIndexBase;
    }
} // namespace TensileLite
