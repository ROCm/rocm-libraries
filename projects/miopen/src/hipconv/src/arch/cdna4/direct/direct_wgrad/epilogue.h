#pragma once

// Epilogue: reduce the register accumulators into the fp32 dW through global atomicAdd.
//
// dW is K x Kh x Kw x C, so C is its contiguous axis while the accumulators hold K across the
// lanes of a wave. A direct write would put every lane on its own cache line, so the reduction
// transposes through LDS, one filter position at a time to keep the staging to one tile per wave.
//
// The waves_q waves sharing a channel tile hold partial gradients for the same dW elements, so
// they sum through the staging and split the drain between them, leaving one atomic per element
// per workgroup. That is the only cross-wave step here and the reason for the barriers.

#include "bunnies.hpp"
#include "bunnies_cdna4.hpp"
#include "config.h"
#include "main_loop.h"
#include "operand_loader.h"

#include "hipconv/conv2d_params.hpp"
#include <hip/hip_runtime.h>

namespace hipconv::cdna4::direct_wgrad
{

// The dW tensor as the epilogue sees it, K x Kh x Kw x C.
//
// filter_end stops at the end of the wave's own group, not of the tensor. RowLoader guards
// nothing on the channel axis, so a K block wider than its group reads the next group's filters
// and the MFMA multiplies them against this group's input channels; bounding on the tensor adds
// one group's gradient into the next group's filters. The wave's own group rather than the
// tile's, because a tile covering waves_g groups holds a wave of each.
struct WgradTensorPars
{
    int filter_end;
    int chans; // one group's channel count, dW's contiguous extent and the tile's channel bound
};

// One wave's staging tile: its own filter position's C x K, in dW's k-major order.
//
// Rotating each k row by 4 channels makes both accesses conflict-free with no padding. The write
// is b128 and its 8-lane phase carries eight consecutive k at one channel, which the rotation
// puts on 8 disjoint groups of 4 banks; the read is b32 and takes 32 channels of one k per half
// wave, whose two k rows would otherwise land on the same 32 banks. The rotation is an XOR by a
// multiple of 4, so every lane's 16-byte write stays 16-byte aligned.
template <Config cfg>
struct StagingLayout
{
    static constexpr int cols        = cfg.wave_c();
    static constexpr int rows        = cfg.wave_k();
    static constexpr int size_floats = rows * cols;

    // Distinct rotations, one per 4-channel group in a row.
    static constexpr int rotations = cols / 4;

    static_assert(cols % 4 == 0, "a lane stages 4 consecutive channels");
    static_assert((cols & (cols - 1)) == 0, "the rotation is an XOR, so cols must be a power of 2");

    __device__ static int offset(int k, int c) { return k * cols + (c ^ (4 * (k % rotations))); }
};

// A rendezvous between the waves sharing a channel tile, and nothing where a wave owns it alone.
//
// phase_barrier rather than __syncthreads, which would lower to a vmcnt(0) over the atomics this
// loop is issuing. The caller supplies the s_wait_lgkmcnt.
template <Config cfg>
__device__ inline void partition_barrier()
{
    if constexpr(cfg.waves_q > 1)
        phase_barrier();
}

// Add this workgroup's partial gradient into dW.
//
// c_local is the wave's channel origin within its own group and k_base its filter origin within
// the workgroup's block; c_origin and k_origin are the block's origin within the tensor, c_origin
// inside its group and k_origin across all of them. item is the wave's spatial partition, which
// decides both the staging tile it writes and the share of the drain it takes.
//
// The bounds test carries the whole channel guard. RowLoader does not guard the channel axis, so
// a lane whose eight-channel group runs past the real C or K reads the next pixel's leading
// channels, and those products land in accumulator rows and columns that only this test drops.
//
// acc is read-only here; the reference is non-const because bunnies::store_tile takes its tile
// by mutable reference, even though it only reads it.
template <Config cfg>
__device__ void run_epilogue(int wave,
                             int item,
                             int lane,
                             int c_local,
                             int k_base,
                             float* __restrict__ wgrad,
                             const WgradTensorPars& pars,
                             int c_origin,
                             int k_origin,
                             Accumulators<cfg>& acc)
{
    using Layout = StagingLayout<cfg>;

    // k rows one wave drains per round: 64 lanes over a row of wave_c channels.
    constexpr int rows_per_round = WAVE_SIZE / Layout::cols;
    static_assert(WAVE_SIZE % Layout::cols == 0);

    // The waves sharing this channel tile are a contiguous run of staging tiles, because the
    // wave decode puts the item fastest, so the group starts at wave - item.
    __shared__ alignas(16) float staging[cfg.waves() * Layout::size_floats];
    float* const group = staging + (wave - item) * Layout::size_floats;
    float* const tile  = group + item * Layout::size_floats;

    // The main loop exits with its trailing prefetch still moving; retired here so that a later
    // cell can reuse the row buffers.
    arch::s_wait_vmcnt<0>();

    // Filter position (r, s) occupies dW row r * kw + s of every filter.
    const int krs_stride = cfg.kh * cfg.kw * pars.chans;

    // Where this lane reads on the drain, and the dW element it lands on.
    const int drain_c  = lane % Layout::cols;
    const int drain_k  = lane / Layout::cols;
    const int chan     = c_origin + c_local + drain_c;
    const bool chan_ok = chan < pars.chans;

    bunnies::static_unroll<cfg.kh>([&](auto r) {
        bunnies::static_unroll<cfg.kw>([&](auto s) {
            // One b128 store per block covers the lane's whole 4-channel run from item 0's
            // offset, because the staging layout's XOR is an add on a 4-aligned base.
            bunnies::store_tile<arch::ds_store_b128>(
                acc.tile[r][s], tile, [&](int mb, int nb, int m, int n) {
                const int c = mb * MFMA_M + m;
                const int k = nb * MFMA_N + n;
                return Layout::offset(k, c);
            });

            // LDS ops retire out of order, so the wait makes the reads below see the writes, and
            // the barrier makes the other partitions' tiles visible.
            arch::s_wait_lgkmcnt<0>();
            partition_barrier<cfg>();

            float* dst = wgrad + (r * cfg.kw + s) * pars.chans + chan;

            // Each partition drains every waves_q'th round, so the group covers the tile
            // once between them and dW sees one atomic per element.
            for(int k0 = item * rows_per_round; k0 < Layout::rows;
                k0 += cfg.waves_q * rows_per_round)
            {
                const int k_local = k0 + drain_k;
                const int filter  = k_origin + k_base + k_local;
                if(!chan_ok || filter >= pars.filter_end)
                    continue;

                // waves_q reads over waves_q times fewer rounds, so summing the partitions costs
                // the LDS traffic of a single-partition drain.
                const int offset = Layout::offset(k_local, drain_c);
                float sum        = group[offset];
                for(int q = 1; q < cfg.waves_q; ++q)
                    sum += group[q * Layout::size_floats + offset];

                // 32 consecutive channels of one filter: a 128-byte run, one cache line.
                atomicAdd(dst + static_cast<size_t>(filter) * krs_stride, sum);
            }

            // The next filter position writes these same addresses, so the group has to be
            // done reading them first.
            arch::s_wait_lgkmcnt<0>();
            partition_barrier<cfg>();
        });
    });
}

} // namespace hipconv::cdna4::direct_wgrad
