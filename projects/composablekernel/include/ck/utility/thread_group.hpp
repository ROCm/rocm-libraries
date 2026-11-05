// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "get_id.hpp"

namespace ck {

// OpaqueThreadId: hide the thread id from the optimizer (empty volatile asm) so that thread-id
// dependent setup is not hoisted out of enclosing loops. Used by persistent kernels, where such
// hoisting keeps per-thread setup values live across the whole tile loop and causes spilling.
template <index_t ThreadPerBlock, bool OpaqueThreadId = false>
struct ThisThreadBlock
{
    static constexpr index_t kNumThread_ = ThreadPerBlock;

    __device__ static constexpr index_t GetNumOfThread() { return kNumThread_; }

    __device__ static constexpr bool IsBelong() { return true; }

    __device__ static index_t GetThreadId()
    {
        index_t tid = get_thread_local_1d_id();
        if constexpr(OpaqueThreadId)
        {
            asm volatile("" : "+v"(tid));
        }
        return tid;
    }
};

} // namespace ck
