// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"

namespace ck_tile {

/// @brief Completion counters that order work between workgroups of one launch (GPU scope).
///
/// A producer workgroup publishes every global write it made, then adds to a counter. A consumer
/// workgroup waits until the counter reaches a target, then may read those writes. Counters only
/// ever grow: a launch with epoch e waits for e * count, so they are zeroed once at allocation and
/// never reset between launches. Comparisons are wrap-safe while one launch adds less than 2^31.
///
/// Both calls are collective: every thread of the workgroup must call them.
struct megakernel_counter
{
    /// True when @p value has reached @p target, modulo 2^32.
    CK_TILE_HOST_DEVICE static constexpr bool reached(uint32_t value, uint32_t target)
    {
        return static_cast<int32_t>(value - target) >= 0;
    }

    /// Publishes this workgroup's global writes, then adds @p count to @p counter.
    CK_TILE_DEVICE static void signal(uint32_t* counter, uint32_t count = 1)
    {
        // TDM stores are not modelled by the memory legaliser, so a fence does not wait for them.
        s_wait_tensorcnt<0>();
        // Each wave waits for its own stores and writes back what the release needs; the barrier
        // then orders every wave before the one atomic, which releases again at agent scope.
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
        __syncthreads();
        if(threadIdx.x == 0)
        {
            __scoped_atomic_fetch_add(counter, count, __ATOMIC_RELEASE, __MEMORY_SCOPE_DEVICE);
        }
    }

    /// Blocks the workgroup until @p counter reaches @p target, then acquires at agent scope.
    CK_TILE_DEVICE static void wait(const uint32_t* counter, uint32_t target)
    {
        if(is_poll_wave())
        {
            poll(counter, target);
        }
        acquire();
    }

    /// Wave 0 polls; the other waves park at the barrier in acquire() and add no memory traffic.
    /// Several poll() calls may share one acquire().
    CK_TILE_DEVICE static bool is_poll_wave()
    {
        return threadIdx.x < static_cast<uint32_t>(get_warp_size());
    }

    /// Spins the calling wave until @p counter reaches @p target. No ordering by itself.
    CK_TILE_DEVICE static void poll(const uint32_t* counter, uint32_t target)
    {
        while(!reached(load(counter), target))
        {
            __builtin_amdgcn_s_sleep(1);
        }
    }

    /// Joins the workgroup after poll(), then acquires at agent scope in every wave.
    CK_TILE_DEVICE static void acquire()
    {
        __syncthreads();
        // Every wave invalidates its own caches: a later read must not hit a stale line.
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    }

    CK_TILE_DEVICE static uint32_t load(const uint32_t* counter)
    {
        // Every lane loads the same word; readfirstlane keeps the loop condition scalar.
        return __builtin_amdgcn_readfirstlane(__scoped_atomic_load_n(
            const_cast<uint32_t*>(counter), __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE));
    }
};

} // namespace ck_tile
