// Weights-gradient (wgrad) kernel for grouped convolution, 4 channels per group,
// simulated TF32. Operands are stored as fp32 in HBM and split into (big, small)
// BF16 pairs feeding three 4x4x4 BF16 MFMAs; the fp32 accumulator is reduced in
// LDS and combined across workgroups with atomicAdd. Ported from the original
// grouped_4c_wgrad_tf32.h into the ConvKernel one-TU-per-kernel form.

#include "conv_kernel_table.h"
#include "grouped_conv_kernel.h"
#include "matrix_layout.h"
#include "swizzle.h"
#include "detail.h"
#include "types.h"
#include "mfma_dispatch.h"
#include "mathutil.h"
#include "launch_params.h"
#include "transpose_lds_layout.h"
#include "hip_util.h"
#include "memory.h"
#include "packed_ops.h"
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>

namespace hipconv::cdna4
{
namespace grouped_4c_wgrad_tf32
{

using namespace hipconv;
using namespace hipconv::cdna4;

// CDNA4 has 64 threads per wave.
constexpr int WAVE_SIZE = 64;

// There are four channels per channel-group.
constexpr int GROUP_SIZE = 4;

// Every wave processes 16 channel-groups.
constexpr int GROUPS_PER_WAVE = 16;

// Kernel configuration (i.e., tiling). Same shape as the FP16/BF16 sibling:
// the 2D wave grid (waves_g x waves_q) is the 4c-specific layout used by all
// 4c wgrad variants.
struct Config
{
    // Number of waves tiled along the channel-groups dimension.
    int waves_g;

    // Number of waves tiled along the q-dimension (i.e., output columns).
    // These waves compute partial sums for the same set of weight gradients
    // and must be reduced in LDS during the epilogue.
    int waves_q;

    // Number of output columns per wave divided by 4
    // (mfma_4x4x4_16b K-reduces 4 columns per instruction).
    int wave_q4 = 8;

    // Convolution kernel height ..
    int kh = 3;

    // .. and width.
    int kw = 3;

    Direction direction = Direction::Wgrad;

    constexpr int waves() const { return waves_g * waves_q; }
    constexpr int block_groups() const { return waves_g * GROUPS_PER_WAVE; }
    constexpr int block_c() const { return block_groups() * GROUP_SIZE; }
    constexpr int block_q() const { return waves_q * wave_q4 * 4; }
    constexpr int threads() const { return waves() * WAVE_SIZE; }
};

constexpr Config configs[] = {
    // The smallest configuration is often the fastest.
    // Increasing occupancy is more effective than decreasing global atomicAdd
    // here (the kernel is memory-bound).
    {.waves_g = 1, .waves_q = 4, .wave_q4 = 1},
};

constexpr int NUM_CONFIGS = sizeof(configs) / sizeof(configs[0]);

template <Config cfg>
__device__ void conv2d_grouped_4c_wgrad_tf32_cdna4_nhwc_impl(const float* __restrict__ input,
                                                             const float* __restrict__ delta,
                                                             float* __restrict__ wgrad,
                                                             int N,
                                                             int groups,
                                                             int /*c_per_group*/,
                                                             int /*k_per_group*/,
                                                             int hi,
                                                             int wi,
                                                             int ho,
                                                             int wo,
                                                             int /*fy*/,
                                                             int /*fx*/,
                                                             int /*sy*/,
                                                             int /*sx*/,
                                                             int /*dy*/,
                                                             int /*dx*/,
                                                             int py,
                                                             int px)
{
    // Type setup. LDS holds BF16 (the operand type fed to the 4x4x4 MFMA),
    // even though the kernel signature is fp32: input/delta are split into
    // (big, small) BF16 pairs on the way in, and the fp32 accumulator stays
    // in registers.
    //   ResultLayout<4, 4, 16, float>     - lane distribution of the fp32x4
    //     accumulator under mfma_4x4x4_16b batch=16.
    //   TransposeLayout<4, 4, 16>         - lane decoding for ds_read_tr16_b64.
    //   Sw = SwizzleT_wgrad<BLOCK_C>      - wgrad-specific uint2 swizzle that
    //     is bank-free under the 4 x × 8 c4 phase model of ds_read_tr16_b64;
    //     reused for both BF16 pair LDS halves at uint2 (= bf16x4) granularity.
    //     See swizzle.h for the derivation.
    using datatypex4_t    = bf16x4_t;
    using ResultLayout    = MatrixLayout<4, 4, 16, float>;
    using TransposeLayout = TransposeLDSLayout<4, 4, 16>;
    using Sw              = SwizzleT_wgrad<cfg.block_c()>;

    constexpr int BLOCK_C  = cfg.block_c();
    constexpr int BLOCK_C4 = BLOCK_C / 4;

    constexpr int WAVE_Q_COLS   = cfg.wave_q4 * 4;
    constexpr int BLOCK_Q_TOTAL = cfg.waves_q * WAVE_Q_COLS;
    constexpr int BLOCK_W_TOTAL = BLOCK_Q_TOTAL + (cfg.kw - 1); // +(kw-1) padding for the W slide

    // Compute-stage LDS geometry. There are FOUR independent regions:
    //   input_big   /  input_small : BF16 halves of the fp32 input  tile (W-wide)
    //   delta_big   /  delta_small : BF16 halves of the fp32 delta tile (Q-wide)
    // Each region is double-buffered (NUM_*_LDS_BUFFERS = 2). Sizes are in
    // uint2 (= 1 bf16x4 = 4 channels) units, half the per-lane width of the
    // fp16/bf16 wgrad uint4-based layout - so each lane stores 1 bf16x4 per
    // pass into each of the two halves (vs 1 uint4 = 8 channels in fp16/bf16).
    constexpr int NUM_INPUT_LDS_BUFFERS    = 2;
    constexpr int NUM_DELTA_LDS_BUFFERS    = 2;
    constexpr int INPUT_LDS_BUFFER_SIZE_U2 = BLOCK_C4 * BLOCK_W_TOTAL;
    constexpr int DELTA_LDS_BUFFER_SIZE_U2 = BLOCK_C4 * BLOCK_Q_TOTAL;

    // Two halves per region => *2 in the byte count.
    constexpr int COMPUTE_LDS_BYTES =
        (int)sizeof(bf16x4_t) * (NUM_INPUT_LDS_BUFFERS * INPUT_LDS_BUFFER_SIZE_U2 * 2 +
                                 NUM_DELTA_LDS_BUFFERS * DELTA_LDS_BUFFER_SIZE_U2 * 2);

    // Staging LDS geometry (4c-specific, identical to fp16/bf16 wgrad).
    //   NATURAL_K_STRIDE  = kh*kw*GROUP_SIZE (= 36 for 3x3x4)
    //   STAGING_K_STRIDE  = NATURAL_K_STRIDE + 1 (= 37) - +1 padding breaks the
    //     32-bank alignment in the cross-slot reduce of phase 2.
    //   STAGING_SLOT_SIZE = NUM_K * STAGING_K_STRIDE - one slot per wave_q.
    constexpr int WEIGHTS_PER_WG   = cfg.block_groups() * GROUP_SIZE * cfg.kh * cfg.kw * GROUP_SIZE;
    constexpr int NATURAL_K_STRIDE = cfg.kh * cfg.kw * GROUP_SIZE;
    constexpr int STAGING_K_STRIDE = NATURAL_K_STRIDE + 1;
    constexpr int NUM_K            = cfg.block_groups() * GROUP_SIZE;
    constexpr int STAGING_SLOT_SIZE = NUM_K * STAGING_K_STRIDE;
    constexpr int STAGING_LDS_BYTES = cfg.waves_q * STAGING_SLOT_SIZE * (int)sizeof(float);
    // Compute and staging LDS reuse the same allocation (epilogue happens after
    // the compute loop completes), so only the larger of the two is needed.
    constexpr int TOTAL_LDS_BYTES =
        COMPUTE_LDS_BYTES > STAGING_LDS_BYTES ? COMPUTE_LDS_BYTES : STAGING_LDS_BYTES;

    __shared__ char lds_raw[TOTAL_LDS_BYTES];

    // LDS region pointers: layout is [input_big | input_small | delta_big | delta_small],
    // each spanning NUM_*_LDS_BUFFERS double-buffer slots of *_LDS_BUFFER_SIZE_U2 bf16x4 each.
    bf16x4_t* input_big_lds   = reinterpret_cast<bf16x4_t*>(lds_raw);
    bf16x4_t* input_small_lds = input_big_lds + NUM_INPUT_LDS_BUFFERS * INPUT_LDS_BUFFER_SIZE_U2;
    bf16x4_t* delta_big_lds   = input_small_lds + NUM_INPUT_LDS_BUFFERS * INPUT_LDS_BUFFER_SIZE_U2;
    bf16x4_t* delta_small_lds = delta_big_lds + NUM_DELTA_LDS_BUFFERS * DELTA_LDS_BUFFER_SIZE_U2;

    const int tid  = threadIdx.x;
    const int wave = tid / WAVE_SIZE;
    const int lane = tid % WAVE_SIZE;

    // Wave mapping: waves_g waves along groups, waves_q waves along q
    // (2D wave grid - same as fp16/bf16 wgrad).
    const int wave_g        = wave % cfg.waves_g;
    const int wave_q        = wave / cfg.waves_g;
    const int wave_q_offset = wave_q * WAVE_Q_COLS;

    const int block_group_idx = blockIdx.x;
    const int block_q         = static_cast<int>(blockIdx.y) * BLOCK_Q_TOTAL;
    const int block_group     = block_group_idx * cfg.block_groups();

    // Channel offset in c4 (= 4-channel block) units. fp16/bf16 wgrad uses
    // c8 here because its HBM/LDS unit is uint4 (8 BF16 channels); TF32 uses
    // uint4 fp32 (= 4 channels) on the HBM side too, but the LDS stride is
    // uint2 = 1 bf16x4 = 4 BF16 channels, so c4 is the right unit.
    const int block_c4 = block_group * GROUP_SIZE / 4;

    const int C  = groups * GROUP_SIZE;
    const int C4 = C / 4;

    // Accumulators: one fp32x4 per (r, s) filter position. Each fp32x4 covers
    // 16 groups x (4 lanes x 4 elements) = 64 weight gradients under the
    // batch=16 4x4x4 MFMA geometry.
    constexpr auto Zero = fp32x4_t{0.f, 0.f, 0.f, 0.f};
    fp32x4_t acc[cfg.kh * cfg.kw];
    for(int i = 0; i < cfg.kh * cfg.kw; i++)
        acc[i] = Zero;

    // BF16 pair (big, small) for one 4x4 tile fed to mfma_4x4x4_16b.
    // big   = top 16 bits of the fp32 value (BF16 round-toward-zero)
    // small = residual (low 16 bits expressed as a BF16), so big + small ~= fp32.
    // The dot product is expanded as 3 BF16 MFMAs (see main loop).
    // Delta register circular buffer: kh rows x wave_q4 Q-tiles, each entry
    // holding a (big, small) pair (vs a single bf16x4_t in fp16/bf16 wgrad).
    bf16_pair_x4 delta_regs[cfg.kh][cfg.wave_q4];

    constexpr int rsrc_data_format = 1 << 15;

    auto input_bytes = static_cast<size_t>(N) * hi * wi * C * sizeof(float);
    auto input_rsrc  = __builtin_amdgcn_make_buffer_rsrc(
        const_cast<float*>(input), 0, input_bytes, rsrc_data_format);

    auto delta_bytes = static_cast<size_t>(N) * ho * wo * C * sizeof(float);
    auto delta_rsrc  = __builtin_amdgcn_make_buffer_rsrc(
        const_cast<float*>(delta), 0, delta_bytes, rsrc_data_format);

    // Multi-pass loading setup. Unit is uint2 (= bf16x4) on the LDS side
    // and uint4 (= fp32x4) on the HBM side; the row width in both units is
    // BLOCK_C4 (= BLOCK_C / 4 channels).
    constexpr int INPUT_LOAD_PASSES = divup(BLOCK_W_TOTAL * BLOCK_C4, cfg.threads());
    constexpr int DELTA_LOAD_PASSES = divup(BLOCK_Q_TOTAL * BLOCK_C4, cfg.threads());

    // TransposeLDSLayout<4,4,16>:
    //   row(lane, 0) = (lane/4)%4  -- q-column within 4-col tile
    //   col(lane) = 0              -- always 0 (M=4, reads 4 consecutive elements)
    //   batch(lane) = ((lane/16)*4 + lane%4) % 16  -- group index (0..15)
    // lane_c4 then anchors the lane to its (wave_g, group) c4-column in LDS.
    const int lane_row0  = TransposeLayout::row(lane, 0);
    const int lane_batch = TransposeLayout::batch(lane);
    const int lane_c4    = wave_g * GROUPS_PER_WAVE + lane_batch;

    // Load one tile of delta from LDS to registers as a (big, small) pair.
    // The same swizzled offset is used for both halves; one ds_read_tr16_b64
    // per half. fp16/bf16 wgrad does this with a single ds_read.
    auto load_delta_q4_lds = [&](int buf, int q4) -> bf16_pair_x4 {
        int off     = Sw::offset_uint2(wave_q_offset + q4 * 4 + lane_row0, lane_c4);
        auto* big_p = reinterpret_cast<int16x4_t*>(delta_big_lds + buf * DELTA_LDS_BUFFER_SIZE_U2);
        auto* small_p =
            reinterpret_cast<int16x4_t*>(delta_small_lds + buf * DELTA_LDS_BUFFER_SIZE_U2);
        bf16_pair_x4 r;
        r.big = __builtin_bit_cast(bf16x4_t, __builtin_amdgcn_ds_read_tr16_b64_v4i16(&big_p[off]));
        r.small =
            __builtin_bit_cast(bf16x4_t, __builtin_amdgcn_ds_read_tr16_b64_v4i16(&small_p[off]));
        return r;
    };

    // Load one delta row (all wave_q4 tiles) from LDS to the register buffer.
    auto load_delta_lds = [&](int buf, int slot) {
        for(int wq = 0; wq < cfg.wave_q4; wq++)
            delta_regs[slot][wq] = load_delta_q4_lds(buf, wq);
    };

    // Load one input column-tile from LDS to registers as a (big, small) pair.
    // Mirrors load_delta_q4_lds for the input region.
    auto load_input_lds = [&](int buf, int col) -> bf16_pair_x4 {
        int off     = Sw::offset_uint2(wave_q_offset + col + lane_row0, lane_c4);
        auto* big_p = reinterpret_cast<int16x4_t*>(input_big_lds + buf * INPUT_LDS_BUFFER_SIZE_U2);
        auto* small_p =
            reinterpret_cast<int16x4_t*>(input_small_lds + buf * INPUT_LDS_BUFFER_SIZE_U2);
        bf16_pair_x4 r;
        r.big = __builtin_bit_cast(bf16x4_t, __builtin_amdgcn_ds_read_tr16_b64_v4i16(&big_p[off]));
        r.small =
            __builtin_bit_cast(bf16x4_t, __builtin_amdgcn_ds_read_tr16_b64_v4i16(&small_p[off]));
        return r;
    };

    const int block_n = blockIdx.z;
    {
        const size_t input_n_offset = (size_t)block_n * hi * wi * C4;
        const size_t delta_n_offset = (size_t)block_n * ho * wo * C4;

        // Per-pass load info, invariant across rows. Each pass is one
        // contiguous-tid uint2 slot in LDS; we recover its (col, c4_idx)
        // logical coordinates via Sw::x_uint2 / c4_uint2 (the inverse of
        // Sw::offset_uint2) so the corresponding HBM offset can be precomputed.
        // global_offset = input_bytes is the OOB sentinel (raw_buffer_load
        // returns zero for out-of-bound offsets).
        uint32_t input_global_offsets[INPUT_LOAD_PASSES];
        bool input_active[INPUT_LOAD_PASSES];
        bf16x4_t* input_big_addrs[INPUT_LOAD_PASSES];
        bf16x4_t* input_small_addrs[INPUT_LOAD_PASSES];
        for(int pass = 0; pass < INPUT_LOAD_PASSES; pass++)
        {
            int lds_idx_u2          = tid + pass * cfg.threads();
            input_active[pass]      = (lds_idx_u2 < BLOCK_W_TOTAL * BLOCK_C4);
            input_big_addrs[pass]   = &input_big_lds[lds_idx_u2];
            input_small_addrs[pass] = &input_small_lds[lds_idx_u2];
            if(input_active[pass])
            {
                int col        = Sw::x_uint2(lds_idx_u2);
                int c4_idx     = Sw::c4_uint2(lds_idx_u2);
                int global_col = (block_q - px) + col;
                if(global_col >= 0 && global_col < wi)
                    // 16 = sizeof(uint4) = 1 fp32x4 = 4 channels worth of bytes.
                    input_global_offsets[pass] =
                        16 * (input_n_offset + (size_t)global_col * C4 + block_c4 + c4_idx);
                else
                    input_global_offsets[pass] = static_cast<uint32_t>(input_bytes);
            }
        }

        uint32_t delta_global_offsets[DELTA_LOAD_PASSES];
        bool delta_active[DELTA_LOAD_PASSES];
        bf16x4_t* delta_big_addrs[DELTA_LOAD_PASSES];
        bf16x4_t* delta_small_addrs[DELTA_LOAD_PASSES];
        for(int pass = 0; pass < DELTA_LOAD_PASSES; pass++)
        {
            int lds_idx_u2          = tid + pass * cfg.threads();
            delta_active[pass]      = (lds_idx_u2 < BLOCK_Q_TOTAL * BLOCK_C4);
            delta_big_addrs[pass]   = &delta_big_lds[lds_idx_u2];
            delta_small_addrs[pass] = &delta_small_lds[lds_idx_u2];
            if(delta_active[pass])
            {
                int col      = Sw::x_uint2(lds_idx_u2);
                int c4_idx   = Sw::c4_uint2(lds_idx_u2);
                int global_q = block_q + col;
                if(global_q >= 0 && global_q < wo)
                    delta_global_offsets[pass] =
                        16 * (delta_n_offset + (size_t)global_q * C4 + block_c4 + c4_idx);
                else
                    delta_global_offsets[pass] = static_cast<uint32_t>(delta_bytes);
            }
        }

        const size_t input_row_stride = (size_t)wi * C4 * sizeof(uint4);
        const size_t delta_row_stride = (size_t)wo * C4 * sizeof(uint4);

        // Register-side staging for the fp32 data fetched in load_*_global
        // before it is converted and committed to LDS by commit_*_lds.
        // Indexed by [ping-pong buf][pass] just like the LDS double buffer.
        fp32x4_t input_pending[NUM_INPUT_LDS_BUFFERS][INPUT_LOAD_PASSES];
        fp32x4_t delta_pending[NUM_DELTA_LDS_BUFFERS][DELTA_LOAD_PASSES];

        // Issue an asynchronous HBM->VGPR raw load for one input row.
        // Unlike fp16/bf16 wgrad's buffer_load_lds, this does not write to LDS
        // directly: the fp32 value sits in input_pending until commit_input_lds
        // is invoked after wait_vmcnt<0>. Splitting the transfer this way lets
        // us insert the fp32 -> BF16 pair conversion while preserving the
        // prefetch-overlap pattern.
        auto load_input_global = [&](int buf, int y) {
            for(int pass = 0; pass < INPUT_LOAD_PASSES; pass++)
            {
                if(input_active[pass])
                {
                    uint32_t global_offset = (y >= 0 && y < hi)
                                                 ? input_global_offsets[pass] + y * input_row_stride
                                                 : static_cast<uint32_t>(input_bytes);
                    auto raw =
                        __builtin_amdgcn_raw_buffer_load_b128(input_rsrc, global_offset, 0, 0);
                    input_pending[buf][pass] = __builtin_bit_cast(fp32x4_t, raw);
                }
            }
        };

        // Issue an asynchronous HBM->VGPR raw load for one delta row.
        // Mirrors load_input_global for the delta tensor.
        auto load_delta_global = [&](int buf, int p) {
            for(int pass = 0; pass < DELTA_LOAD_PASSES; pass++)
            {
                if(delta_active[pass])
                {
                    uint32_t global_offset = (p >= 0 && p < ho)
                                                 ? delta_global_offsets[pass] + p * delta_row_stride
                                                 : static_cast<uint32_t>(delta_bytes);
                    auto raw =
                        __builtin_amdgcn_raw_buffer_load_b128(delta_rsrc, global_offset, 0, 0);
                    delta_pending[buf][pass] = __builtin_bit_cast(fp32x4_t, raw);
                }
            }
        };

        // Commit pending fp32 data to LDS as a (big, small) BF16 pair.
        // Must be called only after the corresponding load_input_global has
        // retired (wait_vmcnt<0>). Two ds_write_b64 - one per LDS half.
        auto commit_input_lds = [&](int buf) {
            for(int pass = 0; pass < INPUT_LOAD_PASSES; pass++)
            {
                if(input_active[pass])
                {
                    auto pair = fp32x4_to_bf16_pair(input_pending[buf][pass]);
                    *(input_big_addrs[pass] + buf * INPUT_LDS_BUFFER_SIZE_U2)   = pair.big;
                    *(input_small_addrs[pass] + buf * INPUT_LDS_BUFFER_SIZE_U2) = pair.small;
                }
            }
        };

        // Commit pending fp32 delta data to LDS as a (big, small) BF16 pair.
        auto commit_delta_lds = [&](int buf) {
            for(int pass = 0; pass < DELTA_LOAD_PASSES; pass++)
            {
                if(delta_active[pass])
                {
                    auto pair = fp32x4_to_bf16_pair(delta_pending[buf][pass]);
                    *(delta_big_addrs[pass] + buf * DELTA_LDS_BUFFER_SIZE_U2)   = pair.big;
                    *(delta_small_addrs[pass] + buf * DELTA_LDS_BUFFER_SIZE_U2) = pair.small;
                }
            }
        };

        // Prologue: pre-fill the delta circular buffer with the kh-1 rows
        // that precede the first input row (negative y, fed via raw_buffer_load
        // OOB sentinel for the rows above the image). Same pattern as
        // fp16/bf16 wgrad, plus an explicit commit_delta_lds because TF32
        // needs the fp32 -> BF16 pair conversion before the LDS write.
        for(int r = 0; r < cfg.kh - 1; r++)
        {
            int p = py - (cfg.kh - 1) + r;
            load_delta_global(0, p);
            wait_vmcnt<0>();
            commit_delta_lds(0);
            __syncthreads();
            load_delta_lds(0, r);
            __syncthreads();
        }

        // Issue the first input row load and the first main-loop delta load
        // (both async). Their commits happen at the top of the first iteration.
        load_input_global(0, 0);
        load_delta_global(0, py);

        int tic = 1;
        int toc = 0;

        // Process a full kh-block of rows.
        // Per iteration: wait for the previous HBM loads to retire, commit
        // the fp32 -> BF16 pairs to LDS, kick off the next async HBM loads,
        // load the next delta slot from LDS into registers, then run the
        // 3-term BF16 MFMA expansion for every (S, Q4, R) in the kh-block.
        auto process_kh_block = [&](int y_base) {
            static_for<cfg.kh>([&]<int Y_LOCAL>() {
                int y = y_base + Y_LOCAL;

                wait_vmcnt<0>();
                commit_input_lds(toc);
                commit_delta_lds(toc);
                __syncthreads();

                if((y + 1) < hi)
                {
                    load_input_global(tic, y + 1);
                    load_delta_global(tic, y + 1 + py);
                }

                constexpr int DELTA_SLOT = (cfg.kh - 1 + Y_LOCAL) % cfg.kh;
                load_delta_lds(toc, DELTA_SLOT);

                static_for<cfg.kw>([&]<int S>() {
                    static_for<cfg.wave_q4>([&]<int Q4>() {
                        auto input_reg = load_input_lds(toc, Q4 * 4 + S);
                        static_for<cfg.kh>([&]<int R>() {
                            constexpr int SLOT = (cfg.kh - 1 + Y_LOCAL - R + cfg.kh) % cfg.kh;
                            const auto& dt     = delta_regs[SLOT][Q4];
                            // 3-term TF32 expansion (big*big + big*small +
                            // small*big, small*small dropped) lives inside the
                            // mfma_4x4x4_16b(bf16_pair_x4, bf16_pair_x4, fp32x4)
                            // overload in mfma_dispatch.h, shared with
                            // grouped_4c.h fprop/dgrad.
                            acc[R * cfg.kw + S] =
                                mfma_4x4x4_16b(input_reg, dt, acc[R * cfg.kw + S]);
                        });
                    });
                });

                tic ^= 1;
                toc ^= 1;
            });
        };

        int y_base = 0;
        for(; y_base + cfg.kh <= hi; y_base += cfg.kh)
            process_kh_block(y_base);

        // Remainder: leftover rows that don't fill a full kh-block. Same body
        // as process_kh_block but guarded by Y_LOCAL >= num_remainder.
        {
            int num_remainder = hi - y_base;
            static_for<cfg.kh>([&]<int Y_LOCAL>() {
                if(Y_LOCAL >= num_remainder)
                    return;
                int y = y_base + Y_LOCAL;

                wait_vmcnt<0>();
                commit_input_lds(toc);
                commit_delta_lds(toc);
                __syncthreads();

                if((y + 1) < hi)
                {
                    load_input_global(tic, y + 1);
                    load_delta_global(tic, y + 1 + py);
                }

                constexpr int DELTA_SLOT = (cfg.kh - 1 + Y_LOCAL) % cfg.kh;
                load_delta_lds(toc, DELTA_SLOT);

                static_for<cfg.kw>([&]<int S>() {
                    static_for<cfg.wave_q4>([&]<int Q4>() {
                        auto input_reg = load_input_lds(toc, Q4 * 4 + S);
                        static_for<cfg.kh>([&]<int R>() {
                            constexpr int SLOT = (cfg.kh - 1 + Y_LOCAL - R + cfg.kh) % cfg.kh;
                            const auto& dt     = delta_regs[SLOT][Q4];
                            acc[R * cfg.kw + S] =
                                mfma_4x4x4_16b(input_reg, dt, acc[R * cfg.kw + S]);
                        });
                    });
                });

                tic ^= 1;
                toc ^= 1;
            });
        }
    }

    // Epilogue: LDS-staged coalesced atomicAdd with wave-q reduction.
    //
    // Phase 1: Each wave writes its accumulators to its own staging slot in LDS.
    // Phase 2: Reduce across waves_q to produce a single set of weight gradients.
    // Phase 3: Coalesced atomicAdd from LDS slot 0 to global memory.
    //
    // Identical to fp16/bf16 wgrad - the BF16-pair scheme only affects the
    // compute loop above; by the time we reach here the accumulators are
    // already plain fp32x4. The compute and staging LDS regions reuse the
    // same allocation (TOTAL_LDS_BYTES is the max of the two).
    float* staging_lds = reinterpret_cast<float*>(lds_raw);

    // ResultLayout<4,4,16,float>:
    //   outer(lane) = lane % 4        -> k within group
    //   inner(lane, idx) = idx        -> c within group
    //   batch(lane) = (lane/4) % 16   -> group index
    // k_in_tile is this lane's K-row index within the workgroup's staging slot.
    const int lane_k    = ResultLayout::outer(lane);
    const int lane_grp  = ResultLayout::batch(lane);
    const int k_in_tile = (wave_g * GROUPS_PER_WAVE + lane_grp) * GROUP_SIZE + lane_k;

    wait_vmcnt<0>();
    __syncthreads();

    // Phase 1: each wave writes its 9 acc[] tiles into its own staging slot
    // (slot index = wave_q). The padded STAGING_K_STRIDE (= NATURAL_K_STRIDE+1)
    // breaks the 32-bank alignment that the natural row length (= 36) would
    // otherwise hit on the cross-slot reads in phase 2.
    for(int r = 0; r < cfg.kh; r++)
        for(int s = 0; s < cfg.kw; s++)
            for(int idx = 0; idx < 4; idx++)
            {
                int c_local    = ResultLayout::inner(lane, idx);
                int lds_offset = wave_q * STAGING_SLOT_SIZE + k_in_tile * STAGING_K_STRIDE +
                                 r * cfg.kw * GROUP_SIZE + s * GROUP_SIZE + c_local;
                staging_lds[lds_offset] = acc[r * cfg.kw + s][idx];
            }
    __syncthreads();

    // Phase 2: reduce across waves_q (iterate over the padded layout). Each
    // i-value is owned by exactly one thread for both reading and writing,
    // so there is no race on staging_lds[i].
    for(int i = tid; i < STAGING_SLOT_SIZE; i += cfg.threads())
    {
        float sum = 0.f;
        for(int wq = 0; wq < cfg.waves_q; wq++)
            sum += staging_lds[wq * STAGING_SLOT_SIZE + i];
        // Each i-value is owned by one thread for reading and writing,
        // so there is no race condition here.
        staging_lds[i] = sum;
    }
    __syncthreads();

    // Phase 3: coalesced atomicAdd from slot 0 to global memory. Indices walk
    // the natural KRSC layout; we remap to the padded staging offset on read.
    int global_k_base = block_group * GROUP_SIZE;
    float* wgrad_tile = wgrad + (size_t)global_k_base * cfg.kh * cfg.kw * GROUP_SIZE;

    for(int i = tid; i < WEIGHTS_PER_WG; i += cfg.threads())
    {
        int k   = i / NATURAL_K_STRIDE;
        int rem = i % NATURAL_K_STRIDE;
        atomicAdd(&wgrad_tile[i], staging_lds[k * STAGING_K_STRIDE + rem]);
    }
}

template <Config cfg>
__global__ void conv2d_grouped_4c_wgrad_tf32_nhwc_cdna4(const float* __restrict__ input,
                                                        const float* __restrict__ delta,
                                                        float* __restrict__ wgrad,
                                                        int N,
                                                        int groups,
                                                        int c_per_group,
                                                        int k_per_group,
                                                        int hi,
                                                        int wi,
                                                        int ho,
                                                        int wo,
                                                        int fy,
                                                        int fx,
                                                        int sy,
                                                        int sx,
                                                        int dy,
                                                        int dx,
                                                        int py,
                                                        int px)
{
    if(__builtin_amdgcn_is_invocable(__builtin_amdgcn_mfma_f32_4x4x4f16) &&
       __builtin_amdgcn_is_invocable(__builtin_amdgcn_mfma_f32_4x4x4bf16_1k) &&
       __builtin_amdgcn_is_invocable(__builtin_amdgcn_raw_ptr_buffer_load_lds) &&
       __builtin_amdgcn_is_invocable(__builtin_amdgcn_ds_read_tr16_b64_v4i16))
    {
        conv2d_grouped_4c_wgrad_tf32_cdna4_nhwc_impl<cfg>(input,
                                                          delta,
                                                          wgrad,
                                                          N,
                                                          groups,
                                                          c_per_group,
                                                          k_per_group,
                                                          hi,
                                                          wi,
                                                          ho,
                                                          wo,
                                                          fy,
                                                          fx,
                                                          sy,
                                                          sx,
                                                          dy,
                                                          dx,
                                                          py,
                                                          px);
    }
}

template <Config cfg>
void launch_impl(const LaunchParams& lp,
                 const Conv2dParams& par,
                 const void* in,
                 const void* wei,
                 void* out,
                 void* /*workspace*/,
                 hipStream_t stream)
{
    auto wgts_bytes = sizeof(float) * par.k * par.kh * par.kw * par.channels_per_group();
    HIP_CHECK(hipMemsetAsync(out, 0, wgts_bytes, stream));
    conv2d_grouped_4c_wgrad_tf32_nhwc_cdna4<cfg>
        <<<lp.grid, lp.block_size, 0, stream>>>(static_cast<const float*>(in),
                                                static_cast<const float*>(wei),
                                                static_cast<float*>(out),
                                                par.n,
                                                par.groups,
                                                par.channels_per_group(),
                                                par.filters_per_group(),
                                                par.h,
                                                par.w,
                                                par.p,
                                                par.q,
                                                par.kh,
                                                par.kw,
                                                par.stride_h,
                                                par.stride_w,
                                                par.dilation_h,
                                                par.dilation_w,
                                                par.pad_h,
                                                par.pad_w);
}

class Grouped_4C_Wgrad_TF32_ConvKernel : public GroupedWgradConvKernel
{
public:
    constexpr Grouped_4C_Wgrad_TF32_ConvKernel(const Config& cfg, LaunchFn launch_fn)
        : GroupedWgradConvKernel(launch_fn)
        , cfg_(cfg)
    {
    }

    // This kernel is tf32-only: fp32 input/dY operands, fp32 dW result.
    bool is_applicable(const Conv2dParams& par) const override
    {
        if(par.input_type != DataType::tf32)
            return false;
        if(par.weight_type != DataType::tf32)
            return false;
        if(par.output_grad_type() != DataType::fp32)
            return false;
        if(par.direction != Direction::Wgrad)
            return false;
        if(par.order != TensorOrder::NHWC)
            return false;
        if(par.weight_grad_type != DataType::fp32)
            return false;
        if(par.kh != 3 || par.kw != 3)
            return false;
        if(par.k != par.c)
            return false;
        if(par.channels_per_group() != group_channels())
            return false;
        if(par.c % group_channels() != 0)
            return false;
        if(par.stride_h != 1 || par.stride_w != 1)
            return false;
        if(par.dilation_h != 1 || par.dilation_w != 1)
            return false;
        if(par.pad_h > par.kh - 1 || par.pad_w > par.kw - 1)
            return false;
        // Buffer load instructions use 32-bit offsets and the buffer descriptor's
        // NUM_RECORDS field is 32 bits. Reject tensors >= 4 GB. (fp32 operands are
        // 4x the byte count of fp16/bf16 wgrad, so this matters even more here.)
        Conv2dSize sz(par);
        if(sz.input_bytes() > INT32_MAX)
            return false;
        if(sz.output_grad_bytes() > INT32_MAX)
            return false;
        return true;
    }

    bool is_valid_config(const Conv2dParams& par) const override
    {
        if(par.direction != cfg_.direction)
            return false;
        if((par.groups % cfg_.block_groups()) != 0)
            return false;
        // Reject wider Q tiling when the selected block width would be at least twice par.q.
        if(cfg_.wave_q4 > 1 && cfg_.block_q() >= 2 * par.q)
            return false;
        return true;
    }

    LaunchParams get_launch_params(const Conv2dParams& par) const override
    {
        auto blocks_q = divup(par.q, cfg_.block_q());
        auto blocks_c = divup(par.groups, cfg_.block_groups());

        LaunchParams launch;
        launch.grid       = dim3(blocks_c, blocks_q, par.n);
        launch.block_size = dim3(cfg_.threads(), 1, 1);
        return launch;
    }

protected:
    int group_channels() const override { return 4; }

private:
    const Config& cfg_;
};

HIPCONV_DEFINE_KERNEL_TABLE(Grouped_4C_Wgrad_TF32_ConvKernel);

} // namespace grouped_4c_wgrad_tf32
} // namespace hipconv::cdna4

// Host-only: the device linker must not see this struct or its function pointers.
#ifndef __HIP_DEVICE_COMPILE__
HIPCONV_EXPORT_KERNEL_TABLE(grouped_4c_wgrad_tf32_cdna4_kernels,
                            hipconv::cdna4::grouped_4c_wgrad_tf32);
#endif
