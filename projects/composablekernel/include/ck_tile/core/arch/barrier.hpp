// SPDX-License-Identifier: MIT
// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"

namespace ck_tile {

template <index_t PhaseWidth>
class LdsAtomicBarrier
{
    static_assert(
        PhaseWidth == 11 || PhaseWidth == 4 || PhaseWidth == 3 || PhaseWidth == 1,
        "from gfx1250 SPG: The pending count and phase fields have configurable width: 21, 28, 29 "
        "or 31 bits.");

    private:
    union BarrierData
    {
        struct
        {
            uint32_t pending_count : (32 - PhaseWidth);
            uint32_t phase : PhaseWidth;
            uint32_t init_count : 16;
            uint32_t zeros : 16;
        };
        uint64_t raw_;
    };

    BarrierData barrier_;

    public:
    CK_TILE_DEVICE LdsAtomicBarrier() = delete;

    CK_TILE_DEVICE void init(uint32_t init_val)
    {
        // Create a local union to construct the value
        BarrierData temp{};
        temp.init_count    = init_val;
        temp.pending_count = init_val;
        temp.phase         = (1 << PhaseWidth) - 1;
        temp.zeros         = 0;
        __atomic_store_n(reinterpret_cast<uint64_t*>(&this->barrier_), temp.raw_, __ATOMIC_RELAXED);
    }

    // Spin until the signaller(s) advance the phase to `phase`.
    //
    // Define CK_TILE_BARRIER_SPIN_LIMIT to bound the spin. That does NOT make a
    // mis-driven barrier correct -- the kernel then proceeds on data that may
    // not have landed and will fail validation -- but it turns a whole-GPU hang
    // into a reported wrong answer, which is debuggable. Leave it undefined for
    // production builds.
    CK_TILE_DEVICE void wait(uint32_t phase)
    {
        phase = phase & ((1 << PhaseWidth) - 1);
#ifdef CK_TILE_BARRIER_SPIN_LIMIT
        uint64_t spins = 0;
#endif
        while(this->barrier_.phase != phase)
        {
#if defined(__gfx125__)
            __builtin_amdgcn_s_sleep(1); // wait for 1-64 clocks
#endif
#ifdef CK_TILE_BARRIER_SPIN_LIMIT
            if(++spins > CK_TILE_BARRIER_SPIN_LIMIT)
                return;
#endif
        }
    }
};

} // namespace ck_tile
