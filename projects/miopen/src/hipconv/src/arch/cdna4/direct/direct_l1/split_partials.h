#pragma once

// fp32 partials of a split-K launch (Config::split_k).
//
// A tile's reduction is shared by the S items of its plane (SplitPlan), split s running
// the s-th contiguous range of the plane's live (depth tap, C(64)) steps. Each (item,
// K-block) is a slot, and each (slot, wave) owns one contiguous run of that wave's
// accumulators in register order: fragment f = (p * wave_q16 + q16) * wave_k16 + k16,
// lane l's fp32x4 at element f * WAVE_SIZE + l. A wave's store of one fragment and the
// reduction's load of it are then each 1 KiB contiguous. See
// docs/algorithms/direct/direct-3d.md.

#include "config.h"
#include "types.h"

#include <hip/hip_runtime.h>
#include <cstddef>

namespace hipconv::cdna4::direct_l1
{

template <Config cfg>
struct SplitPartials
{
    static constexpr int frags = cfg.wave_p * cfg.wave_q16 * cfg.wave_k16;

    // fp32x4 elements one (slot, wave) holds.
    static constexpr int wave_vec4 = frags * WAVE_SIZE;

    // Slot index, the grid's X index major and the K-block minor.
    //
    // x is BalancedL2Grid's flattened index: P fastest, then Q, N, and outermost the
    // group axis, whose inner half numbers a column's items.
    __host__ __device__ static size_t slot_index(int g_and_item,
                                                 int n,
                                                 int q,
                                                 int p,
                                                 int k,
                                                 int n_blocks,
                                                 int blocks_q,
                                                 int blocks_p,
                                                 int Kq)
    {
        const size_t x =
            ((static_cast<size_t>(g_and_item) * n_blocks + n) * blocks_q + q) * blocks_p + p;
        return x * Kq + k;
    }

    // First fp32x4 element of (slot, global wave).
    __host__ __device__ static size_t offset_vec4(size_t slot, int global_wave)
    {
        return (slot * cfg.num_waves() + global_wave) * wave_vec4;
    }

    static_assert(cfg.num_waves() * wave_vec4 * 4 == cfg.block_p() * cfg.block_q() * cfg.block_k(),
                  "a tile's waves hold exactly its outputs, one fp32 each");
};

// Workspace bytes of the partials: one fp32 per output of every slot.
inline size_t split_partials_bytes(const Config& cfg, size_t slots)
{
    return slots * cfg.block_p() * cfg.block_q() * cfg.block_k() * sizeof(float);
}

} // namespace hipconv::cdna4::direct_l1
