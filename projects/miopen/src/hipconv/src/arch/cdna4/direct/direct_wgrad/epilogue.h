#pragma once

// This epilogue writes dW for configs whose grid does not split the spatial axis.
//
// It stores fp32, or the operand type when the operands are 16-bit. Configs that split the spatial
// axis store partial gradients instead; see partials.h.
//
// dW is K × Kh × Kw × C with C contiguous, but the accumulators hold K across a wave's lanes, so a
// direct store would put every lane on its own cache line. The epilogue therefore transposes
// through LDS, one filter position at a time, with one staging tile per wave.
//
// The waves_q waves that share one K × C tile hold partial gradients of the same dW elements. Each
// stages its own, and the drain adds the waves_q staging tiles and divides the rows among those
// waves, so each element is stored once.

#include "bunnies.hpp"
#include "bunnies_cdna4.hpp"
#include "config.h"
#include "main_loop.h"
#include "operand_loader.h"

#include "hipconv/conv_params.hpp"
#include <hip/hip_runtime.h>

namespace hipconv::cdna4::direct_wgrad
{

// A wave stores only the filters below filter_end and the channels below chans.
//
// filter_end is the end of the wave's own group. RowLoader does not guard the channel axis, so a
// K block wider than a group reads the next group's filters, and a bound at the tensor's end
// would store those products over that group's dW. A tile covering waves_g groups holds waves of
// each, so the bound is per wave.
struct WgradTensorPars
{
    int filter_end;
    int chans; // one group's channel count, dW's contiguous extent and the tile's channel bound
};

// One wave's staging tile holds a filter position's K × C block, channels contiguous as in dW.
//
// Each row XORs its 16-byte chunk index with the index of its 32-dword ds_write_b128 bank period,
// modulo the row's chunk count. The direct kernel's Swizzle<TileSizeC> applies the same rule to
// 16-bit rows. The swizzle keeps both the b128 staging write and the b128 drain read free of bank
// conflicts, and keeps a 4-channel run contiguous and aligned.
template <Config cfg>
struct StagingLayout
{
    static constexpr int cols        = cfg.wave_c();
    static constexpr int rows        = cfg.wave_k();
    static constexpr int size_floats = rows * cols;

    static constexpr int chunks          = cols / 4;
    static constexpr int rows_per_period = cols >= 32 ? 1 : 32 / cols;

    static_assert(cols % 4 == 0, "a lane stages 4 consecutive channels");
    static_assert((cols & (cols - 1)) == 0, "the swizzle is an XOR, so cols must be a power of 2");

    __device__ static int offset(int k, int c)
    {
        return k * cols + (c ^ (4 * (k / rows_per_period % chunks)));
    }
};

// partition_barrier orders staging writes before other lanes' reads, and those reads before reuse.
//
// At waves_q > 1 the partitions read each other's tiles and need a workgroup barrier; at
// waves_q == 1 a wave reads only its own. See
// docs/algorithms/details/direct/direct-wgrad-epilogue-lds-ordering.md.
template <Config cfg>
__device__ inline void partition_barrier()
{
    if constexpr(cfg.waves_q > 1)
        __syncthreads();
    else
        __syncwarp();
}

// Store this workgroup's gradient tile to dW.
//
// The grid leaves the spatial axis unsplit and the launch does not zero dW, so the stores have to
// reach every element; the tiles and the drain cover the tensor, and the guards below drop only
// padding.
//
// c_local is the wave's channel offset from c_origin and k_base its filter offset from k_origin;
// c_origin is the block's channel origin within its group, and k_origin its filter origin across
// all groups. item is the wave's spatial partition: the wave stages into tile item and drains
// rounds item, item + waves_q, and so on, where a round is rows_per_round rows of the tile.
//
// The drain's chan_ok and filter_end test is the only channel guard. RowLoader does not guard the
// channel axis, so a lane whose eight-channel group runs past the real C or K reads the next
// pixel's leading channels, and those products land in accumulator rows and columns that only
// this test drops.
//
// acc is read-only here; the reference is non-const because bunnies::store_tile takes its tile by
// mutable reference.
//
// dw_t is float or, for 16-bit operands, the operand type, which the store rounds to once.
template <Config cfg, typename dw_t>
__device__ void run_epilogue(int wave,
                             int item,
                             int lane,
                             int c_local,
                             int k_base,
                             dw_t* __restrict__ wgrad,
                             const WgradTensorPars& pars,
                             int c_origin,
                             int k_origin,
                             Accumulators<cfg>& acc)
{
    // A store is correct only where each dW element has one writer.
    static_assert(cfg.no_splits);

    using Layout = StagingLayout<cfg>;

    // A lane drains 4 consecutive channels, so a row of wave_c channels spans wave_c / 4 lanes.
    constexpr int lanes_per_row  = Layout::cols / 4;
    constexpr int rows_per_round = WAVE_SIZE / lanes_per_row;
    static_assert(WAVE_SIZE % lanes_per_row == 0 && Layout::rows % rows_per_round == 0);

    // The staging tiles of the waves_q waves sharing a K × C tile start at tile wave - item.
    // wave_slot makes item the fastest-varying index, so those tiles are contiguous.
    __shared__ alignas(16) float staging[cfg.waves() * Layout::size_floats];
    float* const partition_tiles = staging + (wave - item) * Layout::size_floats;
    float* const tile            = partition_tiles + item * Layout::size_floats;

    // Wait for the main loop's trailing prefetch, so that the next cell can reuse the row buffers.
    arch::s_wait_vmcnt<0>();

    const int krs_stride = cfg.kh * cfg.kw * pars.chans;

    const int drain_c  = 4 * (lane % lanes_per_row);
    const int drain_k  = lane / lanes_per_row;
    const int chan     = c_origin + c_local + drain_c;
    const bool chan_ok = chan < pars.chans;

    // A channel count that is not a multiple of 4 misaligns the vector store.
    // is_applicable requires an even count for 16-bit operands and allows any count for tf32.
    const bool whole_runs = pars.chans % 4 == 0;

    bunnies::static_unroll<cfg.kh>([&](auto r) {
        bunnies::static_unroll<cfg.kw>([&](auto s) {
            bunnies::store_tile<arch::ds_store_b128>(
                acc.tile[r][s], tile, [&](int mb, int nb, int m, int n) {
                const int c = mb * MFMA_M + m;
                const int k = nb * MFMA_N + n;
                return Layout::offset(k, c);
            });

            partition_barrier<cfg>();

            dw_t* dst = wgrad + (r * cfg.kw + s) * pars.chans + chan;

            // The partitions together store each dW element of the tile once.
            for(int k0 = item * rows_per_round; k0 < Layout::rows;
                k0 += cfg.waves_q * rows_per_round)
            {
                const int k_local = k0 + drain_k;
                const int filter  = k_origin + k_base + k_local;
                if(!chan_ok || filter >= pars.filter_end)
                    continue;

                // A lane sums waves_q tiles per round over 1 / waves_q of the rounds.
                // The sum therefore costs the LDS reads of a single-partition drain.
                const int offset = Layout::offset(k_local, drain_c);
                fp32x4_t sum;
                arch::ds_load_b128::load(partition_tiles + offset, &sum);
                for(int q = 1; q < cfg.waves_q; ++q)
                {
                    fp32x4_t part;
                    arch::ds_load_b128::load(partition_tiles + q * Layout::size_floats + offset,
                                             &part);
                    sum += part;
                }

                // A row's lanes store wave_c consecutive channels of one filter.
                // At wave_c 32 that is a whole 128-byte line of fp32, half a line of fp16/bf16.
                dw_t* const out = dst + static_cast<size_t>(filter) * krs_stride;
                if(whole_runs)
                {
                    auto run = packed_convert<dw_t>(sum);
                    arch::global_store<4 * sizeof(dw_t)>::store(out, &run);
                }
                else
                {
                    for(int i = 0; i < 4 && chan + i < pars.chans; ++i)
                        out[i] = static_cast<dw_t>(sum[i]);
                }
            }

            // The next filter position reuses this staging, so all lanes finish their reads first.
            partition_barrier<cfg>();
        });
    });
}

} // namespace hipconv::cdna4::direct_wgrad
