#pragma once

// A split entry stores partial gradients to a workspace; reduce_partials_cdna4 sums them into dW.
//
// Each nonempty segment stores one partial.
//
// The waves running a cell store its accumulators in MFMA accumulator order, one 1 KiB tile per
// (K16, filter position, C16) block and one 16-byte store per lane, so the store needs no
// transpose. See the Grid section of docs/algorithms/direct/direct-wgrad.md.

#include "bunnies.hpp"
#include "bunnies_cdna4.hpp"
#include "config.h"
#include "epilogue.h"
#include "grid.h"
#include "main_loop.h"
#include "mathutil.h"
#include "types.h"

#include <hip/hip_runtime.h>

#include <cstddef>

namespace hipconv::cdna4::direct_wgrad
{

// The workspace's axes, outermost first, are partial, group, K16, filter position, C16 and lane.
//
// The group is the convolution group, and each lane holds one fp32x4 in accumulator order.
// K16 and C16 index one group's filters and channels. Their counts include the padded channel
// blocks, so a wave past the group's real channel count stores into the padding and the
// reduction drops it.
struct PartialLayout
{
    int groups;
    int k16s;
    int positions;
    int c16s;

    constexpr size_t part_vec4() const
    {
        return static_cast<size_t>(groups) * k16s * positions * c16s * WAVE_SIZE;
    }

    constexpr size_t tile_vec4(int part, int group, int k16, int position, int c16) const
    {
        const size_t row = (static_cast<size_t>(part) * groups + group) * k16s + k16;
        return ((row * positions + position) * c16s + c16) * WAVE_SIZE;
    }
};

constexpr PartialLayout partial_layout(const FlatGrid& grid, int kh, int kw)
{
    return {grid.groups,
            grid.k_blocks() * grid.block_k / MFMA_N,
            kh * kw,
            grid.c_blocks() * grid.block_c / MFMA_M};
}

// The conv kernel stores one partial per segment that holds at least one spatial entry.
constexpr int partial_count(const FlatGrid& grid)
{
    return grid.live_splits();
}

// Store the cell's accumulators to partial `part` in accumulator order.
//
// part is the cell's segment index, which numbers the partials; group is the wave's convolution
// group, and k16 and c16 are the wave's first K16 and C16 blocks within that group. Lane l of a
// 16×16 fp32 accumulator holds channels 4 * (l / 16) to 4 * (l / 16) + 3 of filter l % 16, so the
// reduction reads each lane's store back as four consecutive channels.
//
// The waves_q partitions sharing a channel tile first sum through LDS, still in accumulator
// order, and split the stores between them, so the tile is stored once per cell.
template <Config cfg>
__device__ void run_partial_epilogue(int wave,
                                     int item,
                                     int part,
                                     int group,
                                     int k16,
                                     int c16,
                                     const PartialLayout& layout,
                                     float* __restrict__ partials,
                                     Accumulators<cfg>& acc)
{
    constexpr int blocks      = cfg.wave_c16 * cfg.wave_k16;
    constexpr int block_float = WAVE_SIZE * 4;
    const int lane            = bunnies::lane_id();

    // Retire the main loop's trailing prefetch so that a later cell can reuse the row buffers.
    arch::s_wait_vmcnt<0>();

    auto global_block = [&](int position, int b) {
        const int mb = b / cfg.wave_k16;
        const int nb = b % cfg.wave_k16;
        return reinterpret_cast<fp32x4_t*>(partials) +
               layout.tile_vec4(part, group, k16 + nb, position, c16 + mb) + lane;
    };

    if constexpr(cfg.waves_q == 1)
    {
        bunnies::static_unroll<cfg.kh>([&](auto r) {
            bunnies::static_unroll<cfg.kw>([&](auto s) {
                float* const base =
                    partials + layout.tile_vec4(part, group, k16, r * cfg.kw + s, c16) * 4;
                bunnies::store_tile<arch::global_store_b128>(
                    acc.tile[r][s], base, [&](int mb, int nb, int m, int n) {
                    return (nb * layout.positions * layout.c16s + mb) * block_float + 16 * m +
                           4 * n;
                });
            });
        });
    }
    else
    {
        // Each wave stages its tile here: block b at b * block_float, lane l's fp32x4 at 4 * l.
        __shared__ alignas(16) fp32x4_t staging[cfg.waves() * blocks * WAVE_SIZE];
        fp32x4_t* const group = staging + (wave - item) * blocks * WAVE_SIZE;
        float* const mine     = reinterpret_cast<float*>(group + item * blocks * WAVE_SIZE);

        bunnies::static_unroll<cfg.kh>([&](auto r) {
            bunnies::static_unroll<cfg.kw>([&](auto s) {
                bunnies::store_tile<arch::ds_store_b128>(
                    acc.tile[r][s], mine, [&](int mb, int nb, int m, int n) {
                    return (mb * cfg.wave_k16 + nb) * block_float + 16 * m + 4 * n;
                });

                partition_barrier<cfg>();

                for(int b = item; b < blocks; b += cfg.waves_q)
                {
                    fp32x4_t sum = group[b * WAVE_SIZE + lane];
                    for(int q = 1; q < cfg.waves_q; ++q)
                        sum += group[(q * blocks + b) * WAVE_SIZE + lane];
                    *global_block(r * cfg.kw + s, b) = sum;
                }

                // The next filter position reuses this staging, so all waves finish reading first.
                partition_barrier<cfg>();
            });
        });
    }
}

// A reduction wave group has one wave per C16 block of a C64 block.
constexpr int REDUCE_WAVES = 4;

// A reduction workgroup has REDUCE_GROUPS wave groups when there are REDUCE_GROUPS_FROM_PARTS or
// more partials, and one otherwise.
constexpr int REDUCE_GROUPS            = 4;
constexpr int REDUCE_GROUPS_FROM_PARTS = 8;

// Sum the partials into dW.
//
// Each workgroup takes one (convolution group, K16, filter position, C64) block. Wave w of wave
// group g sums C16 block 4 * c64 + w over partials g, g + Groups, ..., each lane adding its own
// fp32x4, which holds four consecutive channels of one filter. Groups 1 and up then pass their
// sums to group 0 through LDS. The four waves of group 0 write 64 channels of 16
// filters: 128 contiguous bytes per filter of a 16-bit dW, 256 of an fp32 one.
template <typename dw_t, int Groups>
__global__ __launch_bounds__(REDUCE_WAVES* Groups* WAVE_SIZE) void reduce_partials_cdna4(
    const fp32x4_t* __restrict__ partials,
    dw_t* __restrict__ wgrad,
    PartialLayout layout,
    int parts,
    int k_per_group,
    int c_per_group)
{
    // The conv kernel's MFMA builtins limit this kernel to the conv kernel's targets.
    // The reduction itself uses no MFMA.
    if(!(__builtin_amdgcn_is_invocable(__builtin_amdgcn_mfma_f32_16x16x32_f16) &&
         __builtin_amdgcn_is_invocable(__builtin_amdgcn_mfma_f32_16x16x32_bf16)))
        return;

    const int wave  = bunnies::wave_id();
    const int lane  = bunnies::lane_id();
    const int cwave = wave % REDUCE_WAVES;
    const int group = wave / REDUCE_WAVES;

    const int c64s = divup(layout.c16s, REDUCE_WAVES);
    int block      = static_cast<int>(blockIdx.x);
    const int c64  = block % c64s;
    block /= c64s;
    const int position = block % layout.positions;
    block /= layout.positions;
    const int k16        = block % layout.k16s;
    const int conv_group = block / layout.k16s;

    // C16 blocks from layout.c16s up are padding in the last C64 block.
    // Their waves skip the loads and the store but reach the barrier, contributing zero.
    const int c16       = c64 * REDUCE_WAVES + cwave;
    const bool in_range = c16 < layout.c16s;

    fp32x4_t sum = {0.0f, 0.0f, 0.0f, 0.0f};
    if(in_range)
    {
        const fp32x4_t* src = partials + layout.tile_vec4(0, conv_group, k16, position, c16) + lane;
        const size_t stride = layout.part_vec4();
#pragma unroll 4
        for(int p = group; p < parts; p += Groups)
            sum += src[p * stride];
    }

    if constexpr(Groups > 1)
    {
        __shared__ fp32x4_t combine[(Groups - 1) * REDUCE_WAVES * WAVE_SIZE];
        if(group > 0)
            combine[((group - 1) * REDUCE_WAVES + cwave) * WAVE_SIZE + lane] = sum;
        __syncthreads();
        if(group > 0)
            return;
        for(int g = 1; g < Groups; ++g)
            sum += combine[((g - 1) * REDUCE_WAVES + cwave) * WAVE_SIZE + lane];
    }

    const int k = k16 * MFMA_N + lane % 16;
    const int c = c16 * MFMA_M + 4 * (lane / 16);
    if(!in_range || k >= k_per_group || c >= c_per_group)
        return;

    const size_t filter = static_cast<size_t>(conv_group) * k_per_group + k;
    dw_t* const dst     = wgrad + (filter * layout.positions + position) * c_per_group + c;
    if(c_per_group % 4 == 0)
    {
        auto run = packed_convert<dw_t>(sum);
        arch::global_store<4 * sizeof(dw_t)>::store(dst, &run);
    }
    else
    {
        // A channel count that is not a multiple of 4 misaligns the vector store.
        // is_applicable requires an even count for 16-bit operands and allows any count for tf32.
        for(int i = 0; i < 4 && c + i < c_per_group; ++i)
            dst[i] = static_cast<dw_t>(sum[i]);
    }
}

} // namespace hipconv::cdna4::direct_wgrad
