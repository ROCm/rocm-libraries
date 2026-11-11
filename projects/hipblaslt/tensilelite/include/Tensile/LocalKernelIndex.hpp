// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace TensileLite
{
    /**
     * Solution index ranges:
     *
     *   [0, JitKernelIndexBase)                     shipped library
     *   [JitKernelIndexBase, LocalKernelIndexBase)  JIT-generated kernels
     *   [LocalKernelIndexBase, 2^31)                kernels registered at runtime
     *
     * that is, [0, 2^30), [2^30, 3 * 2^29) and [3 * 2^29, 2^31). A library
     * that assigns or routes indices in these ranges takes the bounds from
     * here, so the ranges cannot disagree.
     *
     * A local kernel index is valid only in the process that assigned it.
     */
    constexpr int JitKernelIndexBase   = 1 << 30;
    constexpr int LocalKernelIndexBase = 3 << 29;
    static_assert(0 < JitKernelIndexBase && JitKernelIndexBase < LocalKernelIndexBase,
                  "the shipped, JIT and local index ranges must be in this order");

    constexpr bool isJitKernelIndex(int index)
    {
        return index >= JitKernelIndexBase && index < LocalKernelIndexBase;
    }

    constexpr bool isLocalKernelIndex(int index)
    {
        return index >= LocalKernelIndexBase;
    }
} // namespace TensileLite
