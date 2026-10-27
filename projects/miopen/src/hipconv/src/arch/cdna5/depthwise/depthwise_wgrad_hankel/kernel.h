#pragma once

// CDNA5 depthwise weights-gradient using the 16x16x32 wave32 WMMA.

#include <cstdlib>
#include "bunnies.hpp"
#include "bunnies_mi400.hpp"
#include "config_matcher.hpp"
#include "config_table.h"
#include "wave.h"
#include "delta_issuer.h"
#include "depthwise_conv_kernel.h"
#include "detail.h"
#include "reduction.hpp"
#include "hip_util.h"
#include "input_issuer.h"
#include "launch_params.h"
#include "lds.h"
#include "mathutil.h"
#include "tdm_desc.h"
#include "types.h"

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <limits>
#include <string>

namespace hipconv::cdna5::depthwise_wgrad_hankel
{

using namespace hipconv;
using bunnies::TdmDesc;

// Rows of H a workgroup owns. H is split until the grid is a couple of blocks a
// CU, since unchunked a wave's serial TDM-to-WMMA chain overlaps with nothing.
inline int chunk_rows(const Config& cfg, const ConvParams& par)
{
    const int floor_rows = cfg.min_rows_per_chunk;
    if(par.h <= 0 || par.n <= 0)
        return floor_rows;
    const int c_units = divup(par.c, min(par.c, cfg.staged_channels()));
    // Counted before the batch is folded: letting it see the smaller grid would
    // make it split H finer, which puts back the prologues the folding removes.
    const long long base_wgs = (long long)c_units * divup(par.q, cfg.q_per_wave()) * par.n;
    if(base_wgs <= 0)
        return floor_rows;
    const long long target      = 2LL * cu_count();
    const long long want_chunks = (target + base_wgs - 1) / base_wgs;
    const int chunks            = (int)std::min<long long>(par.h, std::max(1LL, want_chunks));
    return std::max(floor_rows, divup(par.h, chunks));
}

// The partial each block writes for the reduce kernel to sum, sized by the grid.
// Counted by the config's own q per wave, which is the grid's y, and by the
// images it folds, which divide the grid's z.
inline size_t
partition_workspace_size(const ConvParams& par, int rows_per_chunk, int q_per_wave, int n_per_block)
{
    constexpr size_t max_workspace = 1ull << 30;
    const size_t q_tiles           = (size_t)divup(par.q, q_per_wave);
    const size_t chunks            = (size_t)divup(par.h, rows_per_chunk);
    const size_t num_partitions    = (size_t)divup(par.n, n_per_block) * q_tiles * chunks;
    if(num_partitions <= 1)
        return 0;
    const size_t dw_elems = (size_t)par.k * par.kh * par.kw;
    if(num_partitions > (size_t)INT_MAX ||
       dw_elems > std::numeric_limits<size_t>::max() / sizeof(float) / num_partitions)
        return std::numeric_limits<size_t>::max();
    const size_t bytes = num_partitions * dw_elems * sizeof(float);
    return bytes <= max_workspace ? bytes : std::numeric_limits<size_t>::max();
}

// fp32 roundings on the longest path from a product to one dW element.
inline size_t accumulation_depth(const ConvParams& par, const Config& cfg)
{
    const int rows          = chunk_rows(cfg, par);
    const size_t partitions = (size_t)divup(par.n, cfg.n_per_block) *
                              (size_t)divup(par.q, cfg.q_per_wave()) * (size_t)divup(par.h, rows);
    // Both reduce levels are serial from zero: ceil(P/W) - 1 per wave, then W - 1 across waves.
    const size_t reduce =
        partitions > 1 ? (partitions + REDUCE_WAVES - 1) / REDUCE_WAVES + REDUCE_WAVES - 2 : 0;
    return 31 + (size_t)rows * cfg.n_per_block + reduce;
}

// One block: the staged buffers, the descriptors, the prologue that fills them,
// the row loop each wave runs with its role, and the epilogue's engine store.
template <Config cfg, DataType DT, bool PITCH_ALIGNED>
__device__ void conv2d_depthwise_wgrad_hankel_nhwc_impl(const ToType<DT>* __restrict__ input,
                                                        const ToType<DT>* __restrict__ delta,
                                                        float* __restrict__ wgrad,
                                                        int C,
                                                        int hi,
                                                        int wi,
                                                        int ho,
                                                        int wo,
                                                        int py,
                                                        int px,
                                                        int rows_per_chunk,
                                                        int num_chunks,
                                                        int N)
{
    using ElemT  = ToType<DT>;
    namespace bn = bunnies;

    constexpr int CH            = cfg.channels_per_wmma();
    constexpr int STAGE_C       = cfg.staged_channels();
    constexpr int QW            = cfg.q_per_wave();
    constexpr StageGeometry GEO = stage_geometry(cfg);
    constexpr int PF            = cfg.prefetch_depth;
    static_assert(QW == 32 || QW == 16);
    static_assert(CH * cfg.kh <= 16 && CH * cfg.kw <= 16);
    static_assert(PF >= 2, "a spare slot is what the first step issues into");

    static_assert(cfg.waves_per_wg > 1, "the two TDM streams need a wave each");
    constexpr int INPUT_ISSUER = 1;
    constexpr int DELTA_ISSUER = 0;

    const int wave_id = bn::wave_id();
    const int lane    = bn::lane_id();
    const int stage_w = min(STAGE_C, C);
    const int c_base  = static_cast<int>(blockIdx.x) * stage_w;
    static_assert((INPUT_ISSUER ^ DELTA_ISSUER) & 1,
                  "the two engines are reachable from waves of opposite parity");
    static_assert(INPUT_ISSUER < cfg.waves_per_wg && DELTA_ISSUER < cfg.waves_per_wg,
                  "both issuers have to be waves the block has");
    const int wave_c           = wave_id * cfg.channels_per_wave();
    constexpr int PASS_THREADS = cfg.block_size();
    const int pass_tid         = (int)threadIdx.x;
    const int q_base           = static_cast<int>(blockIdx.y) * QW;
    const int n_group          = static_cast<int>(blockIdx.z) / num_chunks;
    const int chunk            = static_cast<int>(blockIdx.z) - n_group * num_chunks;
    const int n_first          = n_group * cfg.n_per_block;
    const int n_last           = min(N, n_first + cfg.n_per_block);
    const int y_begin          = chunk * rows_per_chunk;
    const int y_end            = min(hi, y_begin + rows_per_chunk);
    if(c_base >= C)
        return;

    extern __shared__ __align__(TDM_LDS_ALIGN) unsigned char smem[];
    unsigned char* input_stage = smem;
    unsigned char* delta_pool  = input_stage + PF * GEO.input_stage_slot_bytes;

    const int pitch = min(stage_w, C - c_base);

    const int input_x0           = cfg.stride * q_base - px;
    const int input_left         = min(GEO.input_cols, max(0, -input_x0));
    const int input_valid_x      = input_x0 + input_left;
    const int input_avail        = max(0, wi - input_valid_x);
    const int input_tile_cols    = min(GEO.input_cols - input_left, input_avail);
    constexpr bool IN_PAD_WANTED = cfg.a_from_tr16();
    const bool input_padded      = IN_PAD_WANTED && pitch >= 8 && (pitch & (pitch - 1)) == 0;
    const int input_row          = pitch + (input_padded ? IN_PAD : 0);
    const int align_cols         = max(1, TDM_LDS_ALIGN / (input_row * (int)sizeof(ElemT)));
    const int halo_cols          = align_up(input_left, align_cols);
    const int col_off            = halo_cols - input_left;
    TdmDesc input_tdm;
    input_tdm.init((unsigned)sizeof(ElemT),
                   (unsigned)pitch,
                   (unsigned)pitch,
                   (unsigned long long)C,
                   input_padded,
                   input_padded ? (unsigned)(31 - __builtin_clz((unsigned)(pitch / 4))) : 0u,
                   input_padded ? (unsigned)(IN_PAD * (int)sizeof(ElemT) / 4 - 1) : 0u);
    input_tdm.set_dim1((unsigned)input_avail, (unsigned)input_tile_cols);

    TdmDesc delta_tdm;
    delta_tdm.init(
        (unsigned)sizeof(ElemT), (unsigned)pitch, (unsigned)pitch, (unsigned long long)C);
    const int delta_avail = max(0, wo - q_base);
    const int delta_cols  = min(QW, delta_avail);
    delta_tdm.set_dim1((unsigned)delta_avail, (unsigned)delta_cols);
    delta_tdm.set_tile_dim2(1);

    // A zero outer extent makes the engine fill a tile rather than read it, so a
    // staged buffer is cleared by one issue instead of by the block walking it.
    auto zero_tile = [](TdmDesc& d, unsigned long long base, unsigned lds_off, int cols) {
        d.set_dim1(0u, (unsigned)cols);
        d.arm(base, lds_off);
        d.fire_load();
    };

    // Armed and fired apart, the step's barrier going between: the addressing
    // lands in the barrier's free window and the transfer starts below its wait.
    auto arm_input_row = [&](unsigned long long addr, int slot) {
        input_tdm.arm(
            addr,
            (unsigned)(slot * GEO.input_stage_slot_bytes + halo_cols * input_row * sizeof(ElemT)));
    };
    auto fire_input_row = [&]() { input_tdm.fire_load(); };

    // EXISTS is the third mode's per-load flag: at zero the tile is filled with
    // zeroes instead of read, which is how a row off the image is asked for.
    auto arm_delta = [&]<bool EXISTS = true>(unsigned long long addr, int slot) {
        delta_tdm.set_dim2(EXISTS ? 1u : 0u);
        delta_tdm.arm(addr, (unsigned)(delta_pool - smem + slot * GEO.delta_row_bytes));
    };
    auto fire_delta = [&]() { delta_tdm.fire_load(); };

    // Consecutive pool slots against consecutive image rows in one issue. The LDS
    // step is a pool row either way; the global step is zero where the passes are
    // the zero-filled ones, since an absent row is the same row refused.
    auto batch_delta = [&](unsigned long long global_inc_bytes, unsigned passes) {
        delta_tdm.set_iterate((unsigned)(GEO.delta_row_bytes / sizeof(ElemT)),
                              global_inc_bytes / sizeof(ElemT),
                              passes);
    };

    using ArmInputFn   = decltype(arm_input_row);
    using FireInputFn  = decltype(fire_input_row);
    using ArmDeltaFn   = decltype(arm_delta);
    using FireDeltaFn  = decltype(fire_delta);
    using BatchDeltaFn = decltype(batch_delta);
    InputIssuer<cfg, ArmInputFn, FireInputFn> input_role{
        arm_input_row, fire_input_row, y_begin, y_end};
    DeltaIssuer<cfg, ArmDeltaFn, FireDeltaFn, BatchDeltaFn> del_role{
        arm_delta, fire_delta, batch_delta, y_begin, y_end, py, ho};

    // Zeroing the margins, once each: every load after this is clipped to the
    // columns the image has, so nothing writes there again. The engine fills the
    // right halo and the delta tail itself, on the opening pass of prime() and
    // window(), by being asked for the row whole; the left halo is the columns
    // before global_addr, which no extent reaches, so the block writes those.
    // Counts are the rung's rather than the block's to keep them constant, and
    // overrunning a margin is free -- past it are the columns the engine lays
    // down before anything reads them.
    constexpr int ZERO_EL   = 16 / (int)sizeof(ElemT);
    constexpr int ROW_MAX   = cfg.staged_channels() + IN_PAD;
    constexpr int PAD_EL    = TDM_LDS_ALIGN / (int)sizeof(ElemT);
    constexpr int STEP_MAX  = ROW_MAX > PAD_EL ? ROW_MAX : PAD_EL;
    constexpr int HALO_VECS = divup((cfg.kw - 1) * ROW_MAX + STEP_MAX, ZERO_EL);
    static_assert(HALO_VECS * 16 <= GEO.input_stage_slot_bytes,
                  "the bound has to stay inside the slot it writes");
    if(halo_cols > 0)
    {
        static_for<PF>([&]<int I>() {
            const unsigned slot =
                (unsigned)(uintptr_t)(input_stage + I * GEO.input_stage_slot_bytes);
            for(int v = pass_tid; v < HALO_VECS; v += PASS_THREADS)
                ds_store(slot + (unsigned)v * 16u, dword4{0u, 0u, 0u, 0u});
        });
    }
    // prime() reaches every slot but the spare one, whose first write is the
    // first step's and is already clipped, so its right halo is laid here.
    constexpr int SLOT_VECS = GEO.input_stage_slot_bytes / 16;
    if(input_tile_cols < GEO.input_cols - input_left)
    {
        const unsigned slot =
            (unsigned)(uintptr_t)(input_stage + (PF - 1) * GEO.input_stage_slot_bytes);
        const int right0 = (halo_cols + input_tile_cols) * input_row / ZERO_EL;
        for(int v = right0 + pass_tid; v < SLOT_VECS; v += PASS_THREADS)
            ds_store(slot + (unsigned)v * 16u, dword4{0u, 0u, 0u, 0u});
    }
    // A stride-2 pool is three rows deeper than its opening pass reaches, and
    // the slot that pass misses is the one the first steady fetch writes.
    if constexpr(!cfg.b_rotates())
    {
        if(wave_id == DELTA_ISSUER && delta_cols < QW)
        {
            static_for<GEO.delta_pool_rows>([&]<int I>() {
                zero_tile(delta_tdm,
                          (unsigned long long)reinterpret_cast<uintptr_t>(delta),
                          (unsigned)(delta_pool - smem + I * GEO.delta_row_bytes),
                          QW);
            });
            delta_tdm.set_dim1((unsigned)delta_avail, (unsigned)delta_cols);
            __builtin_amdgcn_s_wait_tensorcnt(0);
        }
    }
    __syncthreads();

    Wave<cfg, DT, PITCH_ALIGNED> wave{input_stage,
                                      delta_pool,
                                      lane,
                                      wave_c,
                                      c_base,
                                      col_off,
                                      pitch,
                                      py,
                                      y_begin,
                                      y_end,
                                      C,
                                      input_row};
    static_assert(cfg.b_in_reg(), "the opening fetches are part of the issued run");

    // The images the block folds into one accumulator. Everything above is laid
    // down once; the window belongs to an image and is laid again for each.
#pragma unroll 1
    for(int img = n_first; img < n_last; ++img)
    {
        wave.reset_image();
        if(wave_id == INPUT_ISSUER)
        {
            input_role.open(
                (unsigned long long)reinterpret_cast<uintptr_t>(input) +
                    (((unsigned long long)img * hi * wi + (unsigned long long)y_begin * wi +
                      (unsigned long long)input_valid_x) *
                         C +
                     c_base) *
                        sizeof(ElemT),
                (unsigned long long)wi * C * sizeof(ElemT));
            // Each slot's first write, so it asks for the staged row whole and
            // lets the engine fill what the image has not; the clipped loads of
            // the steady loop leave that margin alone from here on.
            input_tdm.set_dim1((unsigned)input_avail, (unsigned)(GEO.input_cols - input_left));
            input_role.prime();
            input_tdm.set_dim1((unsigned)input_avail, (unsigned)input_tile_cols);
        }

        if(wave_id == DELTA_ISSUER)
        {
            del_role.open(
                (unsigned long long)reinterpret_cast<uintptr_t>(delta) +
                    (((unsigned long long)img * ho * wo + (unsigned long long)q_base) * C +
                     c_base) *
                        sizeof(ElemT),
                (unsigned long long)wo * C * sizeof(ElemT));
            // The pool's first pass over each slot, asked for whole: what the
            // image has not is past tensor_dim1, so the engine fills it, and the
            // clipped loads below leave it alone from then on.
            if constexpr(cfg.b_rotates())
                delta_tdm.set_dim1((unsigned)delta_avail, (unsigned)QW);
            del_role.window();
        }
        __syncthreads();

        wave.prime_b();
        __syncthreads();
        if(wave_id == DELTA_ISSUER)
        {
            // window() and this between them reach every slot, so the pool is
            // whole and the tile narrows to the columns that are read.
            del_role.open_fetch();
            if constexpr(cfg.b_rotates())
                delta_tdm.set_dim1((unsigned)delta_avail, (unsigned)delta_cols);
        }

        if(wave_id == INPUT_ISSUER)
        {
            wave.run(input_role);
            input_role.drain();
        }
        else if(wave_id == DELTA_ISSUER)
        {
            wave.run(del_role);
            del_role.drain();
        }
        else
        {
            NoIssue role;
            wave.run(role);
        }
        // Every wave is done with the pool and both engines are back, so the next
        // image may lay its window in the same slots.
        __syncthreads();
    }

    __syncthreads();
    wave.store();
    __syncthreads();

    const size_t partition      = (size_t)blockIdx.z * gridDim.y + blockIdx.y;
    const unsigned run          = (unsigned)max(0, min(STAGE_C, C - c_base)) * cfg.kh * cfg.kw;
    constexpr unsigned PER_128B = 128 / sizeof(float);
    const unsigned head         = run / 2 / PER_128B * PER_128B;
    const bool split            = head >= 2 * PER_128B && run - head >= 2 * PER_128B;
    const unsigned mine         = wave_id == DELTA_ISSUER ? (split ? head : run) : run - head;
    if((wave_id == DELTA_ISSUER || (split && wave_id == INPUT_ISSUER)) && mine > 0)
    {
        const unsigned off = wave_id == DELTA_ISSUER ? 0u : head;
        TdmDesc out_tdm{};
        out_tdm.init((unsigned)sizeof(float), mine, mine, (unsigned long long)mine);
        out_tdm.set_dim1(1u, 1u);
        out_tdm.store((unsigned long long)reinterpret_cast<uintptr_t>(
                          wgrad + partition * (size_t)C * cfg.kh * cfg.kw +
                          (size_t)c_base * cfg.kh * cfg.kw + off),
                      (unsigned)(input_stage - smem) + off * (unsigned)sizeof(float));
        __builtin_amdgcn_s_wait_tensorcnt(0);
    }
}

// The body is guarded rather than written straight into the entry point, this
// being compiled for every target the build enumerates.
template <Config cfg, DataType DT, bool PITCH_ALIGNED>
__global__ __launch_bounds__(cfg.block_size()) void conv2d_depthwise_wgrad_hankel_nhwc_cdna5(
    const ToType<DT>* __restrict__ input,
    const ToType<DT>* __restrict__ delta,
    float* __restrict__ wgrad,
    int C,
    int hi,
    int wi,
    int ho,
    int wo,
    int py,
    int px,
    int rows_per_chunk,
    int num_chunks,
    int N)
{
    if(__builtin_amdgcn_is_invocable(__builtin_amdgcn_wmma_f32_16x16x32_f16) &&
       __builtin_amdgcn_is_invocable(__builtin_amdgcn_wmma_f32_16x16x32_bf16) &&
       __builtin_amdgcn_is_invocable(__builtin_amdgcn_ds_load_tr16_b128_v8i16) &&
       __builtin_amdgcn_is_invocable(__builtin_amdgcn_tensor_load_to_lds) &&
       __builtin_amdgcn_is_invocable(__builtin_amdgcn_tensor_store_from_lds) &&
       __builtin_amdgcn_is_invocable(__builtin_amdgcn_s_wait_tensorcnt) &&
       __builtin_amdgcn_is_invocable(__builtin_amdgcn_permlane16_var) &&
       __builtin_amdgcn_is_invocable(__builtin_amdgcn_s_barrier_signal) &&
       __builtin_amdgcn_is_invocable(__builtin_amdgcn_s_barrier_wait))
    {
        conv2d_depthwise_wgrad_hankel_nhwc_impl<cfg, DT, PITCH_ALIGNED>(
            input, delta, wgrad, C, hi, wi, ho, wo, py, px, rows_per_chunk, num_chunks, N);
    }
}

// The dispatch, plus the reduce that sums the partials where the grid made more
// than one.
template <Config cfg>
void launch_impl(const LaunchParams& lp,
                 const ConvParams& par,
                 const void* in,
                 const void* wei,
                 void* out,
                 void* workspace,
                 hipStream_t stream)
{
    const size_t dw_elems         = (size_t)par.k * par.kh * par.kw;
    const unsigned num_partitions = lp.grid.y * lp.grid.z;
    float* main_out =
        num_partitions > 1 ? static_cast<float*>(workspace) : static_cast<float*>(out);

    // ALIGNED says every block's pitch is a multiple of the transpose group.
    auto launch = [&]<DataType DT, bool ALIGNED>() {
        using ElemT = ToType<DT>;
        conv2d_depthwise_wgrad_hankel_nhwc_cdna5<cfg, DT, ALIGNED>
            <<<lp.grid, lp.block_size, lp.dynamic_shared_bytes, stream>>>(
                static_cast<const ElemT*>(in),
                static_cast<const ElemT*>(wei),
                main_out,
                par.c,
                par.h,
                par.w,
                par.p,
                par.q,
                par.pad_h,
                par.pad_w,
                chunk_rows(cfg, par),
                divup(par.h, chunk_rows(cfg, par)),
                par.n);
    };
    // A pitch is the staged width or the C left past it, so both have to divide.
    auto dispatch = [&]<DataType DT>() {
        if constexpr(cfg.staged_channels() % TR_GROUP == 0)
        {
            if(par.c % TR_GROUP == 0)
            {
                launch.template operator()<DT, true>();
                return;
            }
        }
        launch.template operator()<DT, false>();
    };
    if(par.input_type == DataType::bf16)
        dispatch.template operator()<DataType::bf16>();
    else
        dispatch.template operator()<DataType::fp16>();

    if(num_partitions > 1)
    {
        using ::hipconv::cdna5::REDUCE_COLS;
        using ::hipconv::cdna5::REDUCE_WAVES;
        const int rpass =
            ::hipconv::cdna5::reduce_partitions_per_pass((int)num_partitions, LDS_BYTES);
        const unsigned rgrid = (unsigned)((dw_elems + REDUCE_COLS - 1) / REDUCE_COLS);
        ::hipconv::cdna5::
            conv2d_wgrad_reduce_tdm_cdna5<<<rgrid,
                                            REDUCE_COLS * REDUCE_WAVES,
                                            ::hipconv::cdna5::reduce_tdm_lds_bytes(rpass),
                                            stream>>>(static_cast<float*>(out),
                                                      static_cast<const float*>(workspace),
                                                      (int)num_partitions,
                                                      (unsigned long long)dw_elems,
                                                      rpass);
    }
}

// One rung of the table: a filter size, a stride and a tile, with the grid and
// the workspace that shape implies.
class Depthwise_Wgrad_Hankel_ConvKernel : public DepthwiseWgradConvKernel
{
public:
    constexpr Depthwise_Wgrad_Hankel_ConvKernel(const Config& cfg, LaunchFn launch_fn)
        : DepthwiseWgradConvKernel(launch_fn)
        , cfg_(cfg)
    {
    }

    KVDescriptor config_descriptor() const override { return ConfigMatcher(cfg_); }

    bool is_applicable(const ConvParams& par) const override
    {
        if(!DepthwiseWgradConvKernel::is_applicable(par))
            return false;
        if(par.kh != par.kw)
            return false;
        if(partition_workspace_size(
               par, chunk_rows(cfg_, par), cfg_.q_per_wave(), cfg_.n_per_block) ==
           std::numeric_limits<size_t>::max())
            return false;
        return par.kh == 3 || par.kh == 5 || par.kh == 7 || par.kh == 9 || par.kh == 11;
    }

    bool is_valid_config(const ConvParams& par) const override
    {
        return par.direction == cfg_.direction && par.kh == cfg_.kh && par.kw == cfg_.kw &&
               par.stride_h == cfg_.stride && par.stride_w == cfg_.stride;
    }

    LaunchParams get_launch_params(const ConvParams& par) const override
    {
        LaunchParams lp;
        lp.grid                 = dim3(divup(par.c, min(par.c, cfg_.staged_channels())),
                       divup(par.q, cfg_.q_per_wave()),
                       divup(par.n, cfg_.n_per_block) * divup(par.h, chunk_rows(cfg_, par)));
        lp.block_size           = dim3(cfg_.block_size(), 1, 1);
        lp.dynamic_shared_bytes = stage_geometry(cfg_).block_bytes;
        return lp;
    }

    size_t get_workspace_size(const ConvParams& par) const override
    {
        return partition_workspace_size(
            par, chunk_rows(cfg_, par), cfg_.q_per_wave(), cfg_.n_per_block);
    }

    // Which tile the shape gets: what the tile fills rather than what it stages,
    // widest dim0 first, then the grid it leaves a CU, then widest q.
    float get_weighted_throughput_index(const ConvParams& par) const override
    {
        const int eff_c        = min(cfg_.staged_channels(), par.c);
        const int eff_q        = min(cfg_.q_per_wave(), par.q);
        const long long blocks = (long long)divup(par.c, min(par.c, cfg_.staged_channels())) *
                                 divup(par.q, cfg_.q_per_wave()) * divup(par.n, cfg_.n_per_block) *
                                 divup(par.h, chunk_rows(cfg_, par));
        const long long waves = blocks * cfg_.waves_per_wg / cu_count();
        const int filled      = (int)(waves < 16 ? waves : 16);
        return (float)(eff_c * 65536 - (cfg_.staged_channels() - eff_c) * 8192 + filled * 256 +
                       eff_q * 8 - (cfg_.q_per_wave() - eff_q));
    }

    // Supplies the blocked accumulation depth; the default would use all N*P*Q
    // products.
    void get_tolerance(const ConvParams& par, float& atol, float& rtol) const override
    {
        get_mixed_precision_tolerance(par, accumulation_depth(par, cfg_), atol, rtol);
    }

private:
    const Config& cfg_;
};

} // namespace hipconv::cdna5::depthwise_wgrad_hankel
