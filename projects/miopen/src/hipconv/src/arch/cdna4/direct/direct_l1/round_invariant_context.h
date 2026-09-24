#pragma once

#include <hip/hip_runtime.h>

#include "magic_division.h"

// RoundInvariantContext: manually spill the grid-decode inputs to LDS.
//
// The decode runs at two points each round (round-top sched and cross-round
// next_sched), so its inputs stay live across the whole compute phase between
// them and spill K128 (already at the 256-VGPR ceiling). Parking them in LDS and
// reloading at each site removes that cross-phase live range. Which values: the
// XCD/wg-derived ones (no kernarg backing) must be parked; the kernarg ones and
// magic_n are parked too because the compiler kept them live otherwise. This
// drops every k_divisible config to ScratchSize 0; the K128 !k_divisible fallback
// still spills 12 B (its writer path has no headroom), accepted.
//
// Two properties make the reload stick:
//   * Store by lane 0 (wave-uniform values), load whole-wave: different lanes, so
//     the compiler cannot forward the store and keep a register copy live.
//   * readfirstlane on each reload pins it back to an SGPR so the decode stays scalar.
// The LDS pointer is deliberately NOT volatile: volatile keeps one live ds-read
// address per slot across the loop (measured far worse); plain reads share a base
// and the store/load lane split already prevents the reload being optimized away.

namespace hipconv::cdna4::direct_l1
{

struct RoundInvariantContext
{
    // XCD/workgroup-derived (no kernarg backing).
    int x_begin;
    int x_count;
    int round_count;
    int wg_id;
    // Per-XCD K-slice geometry: per_xcd blocks starting at global block kslice*per_xcd.
    int per_xcd;
    int kslice;
    int blocks_p;
    int blocks_q;
    int n_blocks;
    // 4th divisor, stored as its two raw words and reconstructed on load.
    MagicDiv magic_n;

    static constexpr int NUM_SLOTS = 11;

    __device__ void store(int* lds) const
    {
        if(threadIdx.x == 0)
        {
            lds[0]  = x_begin;
            lds[1]  = x_count;
            lds[2]  = round_count;
            lds[3]  = wg_id;
            lds[4]  = per_xcd;
            lds[5]  = kslice;
            lds[6]  = blocks_p;
            lds[7]  = blocks_q;
            lds[8]  = n_blocks;
            lds[9]  = static_cast<int>(magic_n.multiplier);
            lds[10] = static_cast<int>(magic_n.shift);
        }
        __syncthreads();
    }

    __device__ static RoundInvariantContext load(int* lds)
    {
        auto rfl = [](int v) { return __builtin_amdgcn_readfirstlane(v); };
        RoundInvariantContext c;
        c.x_begin            = rfl(lds[0]);
        c.x_count            = rfl(lds[1]);
        c.round_count        = rfl(lds[2]);
        c.wg_id              = rfl(lds[3]);
        c.per_xcd            = rfl(lds[4]);
        c.kslice             = rfl(lds[5]);
        c.blocks_p           = rfl(lds[6]);
        c.blocks_q           = rfl(lds[7]);
        c.n_blocks           = rfl(lds[8]);
        c.magic_n.multiplier = static_cast<uint32_t>(rfl(lds[9]));
        c.magic_n.shift      = static_cast<uint32_t>(rfl(lds[10]));
        return c;
    }
};

} // namespace hipconv::cdna4::direct_l1
