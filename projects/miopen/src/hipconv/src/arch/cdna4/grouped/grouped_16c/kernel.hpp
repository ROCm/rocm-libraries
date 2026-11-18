#include "config.hpp"
#include "config_table.hpp"
#include "config_desc.hpp"

// CDNA4 grouped convolution kernel, 16 channels per group.
//
// This unified TU subsumes the original grouped_16c (fp16/bf16) and
// grouped_16c_tf32 (Phase 1) variants behind a single GroupedDataTraits<DT>
// abstraction (mirroring grouped_4c.cpp). The kernel body is parameterised by
// DataType DT; fp16, bf16 and tf32 share the same control flow with two
// structural `if constexpr` branches:
//   1) Dgrad weight prologue: fp16/bf16 use ds_read_tr16_b64 (hardware
//      (k <-> c) transpose); tf32 has no ds_read_tr32 builtin and uses 4
//      scalar fp32 reads per filter tap to perform the same transpose in
//      software (amortised in the prologue only).
//   2) Output flush stage: fp16/bf16 pack fp32x4 to element_t via
//      packed_convert and store one uint2 per lane; tf32 keeps the fp32x4
//      accumulator and stores it as raw uint4.
//
// TF32 (simulated) follows the same caller-side bf16-pair pre-split contract
// used in grouped_4c.cpp: the input main loop calls fp32xN_to_bf16_pair once
// per (Y_LOCAL, S) tap so the inner R loop can call
// mfma_16x16x16(bf16_pair_x4, bf16_pair_x4, fp32x4_t) without re-splitting on
// every MFMA.

#include "kv_descriptor.h"
#include "grouped_conv_kernel.h"
#include "matrix_layout.h"
#include "swizzle.h"
#include "detail.h"
#include "types.h"
#include "../mfma_dispatch.h"
#include "../depth_taps.h"
#include "mathutil.h"
#include "launch_params.h"
#include "transpose_lds_layout.h"
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

namespace hipconv::cdna4::grouped_16c
{

using namespace hipconv;
using namespace hipconv::cdna4;

// ----------------------------------------------------------------------------
// Per-dtype traits: type aliases for the 4-wide operand vector, the value fed
// to mfma_16x16x16, the LDS access unit, and the swizzle layout, plus the
// to_mfma_operand / zero_operand helpers that absorb the fp32 -> bf16-pair
// split for tf32 while remaining identity for fp16/bf16.
// ----------------------------------------------------------------------------

template <DataType DT>
struct GroupedDataTraits;

template <>
struct GroupedDataTraits<DataType::fp16>
{
    using element_t      = fp16_t;
    using operand_t      = fp16x4_t;
    using mfma_operand_t = fp16x4_t;
    using lds_unit_t     = uint2;
    template <int C>
    using Sw                             = SwizzleT<C>;
    static constexpr int CH_PER_UINT4    = 8;
    static constexpr bool dgrad_use_tr16 = true;
    static constexpr bool needs_lds_pack = true;
    static __device__ __forceinline__ mfma_operand_t to_mfma_operand(operand_t raw) { return raw; }
    static __device__ __forceinline__ mfma_operand_t zero_operand()
    {
        return mfma_operand_t{0, 0, 0, 0};
    }
};

template <>
struct GroupedDataTraits<DataType::bf16>
{
    using element_t      = bf16_t;
    using operand_t      = bf16x4_t;
    using mfma_operand_t = bf16x4_t;
    using lds_unit_t     = uint2;
    template <int C>
    using Sw                             = SwizzleT<C>;
    static constexpr int CH_PER_UINT4    = 8;
    static constexpr bool dgrad_use_tr16 = true;
    static constexpr bool needs_lds_pack = true;
    static __device__ __forceinline__ mfma_operand_t to_mfma_operand(operand_t raw) { return raw; }
    static __device__ __forceinline__ mfma_operand_t zero_operand()
    {
        return mfma_operand_t{bf16_t(0), bf16_t(0), bf16_t(0), bf16_t(0)};
    }
};

template <>
struct GroupedDataTraits<DataType::tf32>
{
    using element_t      = fp32_t;
    using operand_t      = fp32x4_t;
    using mfma_operand_t = bf16_pair_x4;
    using lds_unit_t     = uint4;
    template <int C>
    using Sw                             = SwizzleT_fp32<C>;
    static constexpr int CH_PER_UINT4    = 4;
    static constexpr bool dgrad_use_tr16 = false;
    static constexpr bool needs_lds_pack = false;
    static __device__ __forceinline__ mfma_operand_t to_mfma_operand(operand_t raw)
    {
        return fp32xN_to_bf16_pair(raw);
    }
    static __device__ __forceinline__ mfma_operand_t zero_operand()
    {
        return fp32xN_to_bf16_pair(fp32x4_t{0.f, 0.f, 0.f, 0.f});
    }
};


template <Config cfg, DataType DT>
__device__ void conv2d_grouped_16c_cdna4_nhwc_impl(const ToType<DT>* __restrict__ in,
                                                   const ToType<DT>* __restrict__ wei,
                                                   double alpha,
                                                   double beta,
                                                   ToType<DT>* __restrict__ out,
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
                                                   int px,
                                                   DepthParams dp)
{
    using Traits         = GroupedDataTraits<DT>;
    using element_t      = typename Traits::element_t;
    using operand_t      = typename Traits::operand_t;
    using mfma_operand_t = typename Traits::mfma_operand_t;
    using lds_unit_t     = typename Traits::lds_unit_t;
    using Sw             = typename Traits::template Sw<cfg.block_c()>;
    using OperandLayout  = MatrixLayout<16, 16, 1, element_t>;
    using ResultLayout   = MatrixLayout<16, 16, 1, float>;
    using int16x4_t      = __attribute__((ext_vector_type(4))) short;

    // Channels per uint4 HBM/LDS transfer: 8 for fp16/bf16, 4 for tf32.
    constexpr int CH_PER_UINT4 = Traits::CH_PER_UINT4;
    // Number of uint4 slots per group along C (2 for fp16/bf16, 4 for tf32).
    constexpr int GROUP_UINT4 = cfg.group_size / CH_PER_UINT4;
    // Bytes per LDS read unit (uint2=8B for fp16/bf16, uint4=16B for tf32).
    constexpr int LDS_UNIT_PER_UINT4 = (int)(sizeof(uint4) / sizeof(lds_unit_t));

    static_assert(sizeof(uint4) / sizeof(element_t) == CH_PER_UINT4,
                  "CH_PER_UINT4 must match element size");

    // Number of input columns loaded by each workgroup (output columns plus halo).
    // For dilation > 1, the dilated input is wider but we load only the real (non-zero)
    // compact columns.
    constexpr int BLOCK_W = divup(BLOCK_Q + (cfg.kw - 1), cfg.dilation);

    // uint4 vectors per channels fiber.
    constexpr int BLOCK_C_UINT4 = cfg.block_c() / CH_PER_UINT4;

    // Each wave processes a single group of channels.
    constexpr int BLOCK_GROUPS = cfg.waves_per_wg;

    // Output block width after subsampling for stride > 1.
    constexpr int OUTPUT_BLOCK_Q = BLOCK_Q / cfg.stride;

    // Vectors to store (uint4 elements per workgroup along output channels).
    constexpr int STORE_VECS = OUTPUT_BLOCK_Q * BLOCK_C_UINT4;

    constexpr bool ONE_WAVE = cfg.block_size() == WAVE_SIZE;

    constexpr int NUM_INPUT_LDS_BUFFERS = 2;

    // Size of one input LDS buffer in uint4 units.
    constexpr int INPUT_LDS_BUFFER_UINT4 = BLOCK_C_UINT4 * BLOCK_W;

    // Same buffer size expressed in lds_unit_t units. The main-loop read
    // pointer indexes in lds_unit_t; the pre-computed input_lds_offsets[] are
    // also in lds_unit_t units.
    constexpr int INPUT_LDS_BUFFER_LDS_UNIT = INPUT_LDS_BUFFER_UINT4 * LDS_UNIT_PER_UINT4;

    // Size of LDS buffer for outputs (uint4 units).
    constexpr int OUTPUT_LDS_BUFFER_UINT4 = BLOCK_C_UINT4 * OUTPUT_BLOCK_Q;

    // Weight LDS: each (k_local, khw) row holds group_size element_t channels =
    // GROUP_UINT4 uint4 slots.
    constexpr int WEIGHT_LDS_SIZE_UINT4 =
        BLOCK_GROUPS * cfg.group_size * cfg.kh * cfg.kw * GROUP_UINT4;

    // Multi-pass HBM->LDS loading. Each thread issues one uint4 per pass.
    // fp16/bf16 always degenerates to LOAD_PASSES == 1; tf32 generally needs 2
    // passes because every uint4 holds only 4 channels.
    constexpr int LOAD_PASSES = (BLOCK_W * BLOCK_C_UINT4 + cfg.block_size() - 1) / cfg.block_size();

    // Unified LDS: large enough for either (input double-buffer + output) or (weight prologue).
    constexpr int IO_LDS_SIZE =
        NUM_INPUT_LDS_BUFFERS * INPUT_LDS_BUFFER_UINT4 + OUTPUT_LDS_BUFFER_UINT4;
    __shared__ uint4 lds_buf[maximum(WEIGHT_LDS_SIZE_UINT4, IO_LDS_SIZE)];
    uint4* input_lds  = lds_buf;
    uint4* output_lds = lds_buf + NUM_INPUT_LDS_BUFFERS * INPUT_LDS_BUFFER_UINT4;

    const int tid  = threadIdx.x;
    const int wave = tid / WAVE_SIZE;
    const int lane = tid % WAVE_SIZE;

    // Workgroup coordinates.
    const int block_q_n_idx   = blockIdx.x;
    const int block_n_mod_idx = block_q_n_idx % cfg.n_fold;
    const int block_q_idx     = block_q_n_idx / cfg.n_fold;
    const int block_group_idx = blockIdx.y;
    // Constant for a conv2d, which is one plane per image.
    const int d_in  = cfg.kd == 1 ? 1 : dp.d_in;
    const int d_out = cfg.kd == 1 ? 1 : dp.d_out;
    // Workgroups n_fold apart in x share an XCD, so a plane block spreads its planes across
    // XCDs. With fewer planes than n_fold, the spare slots take row bands of the planes.
    const int planes     = N * d_out;
    const bool fold_rows = planes < cfg.n_fold;
    const int h_split    = fold_rows ? cfg.n_fold / planes : 1;
    const int band       = fold_rows ? block_n_mod_idx / planes : 0;
    const int block_n_idx =
        fold_rows ? block_n_mod_idx - band * planes : blockIdx.z * cfg.n_fold + block_n_mod_idx;
    if(band >= h_split || block_n_idx >= planes)
        return;

    // block_n is the output plane; a conv3d reads the input planes in `taps`.
    const int block_n = block_n_idx;
    const int image_n = cfg.kd == 1 ? block_n : block_n / d_out;
    const int kd      = cfg.kd == 1 ? 1 : dp.kd;
    const auto taps   = live_depth_taps<cfg.kd, cfg.direction == Direction::Dgrad>(
        cfg.kd == 1 ? 0 : block_n % d_out, dp);
    const int block_q       = block_q_idx * BLOCK_Q;
    const int block_group   = block_group_idx * BLOCK_GROUPS;
    const int block_k       = block_group * cfg.group_size;
    const int block_c_uint4 = block_group * GROUP_UINT4; // in uint4 units

    // Base pointer for this batch image in NHWC layout (all in uint4 units).
    const int C       = groups * cfg.group_size;
    const int C_UINT4 = C / CH_PER_UINT4;

    const size_t offset_block = (size_t)image_n * d_in * hi * wi * C_UINT4;
    const size_t wi_stride    = (size_t)wi * C_UINT4;
    const size_t wo_stride    = (size_t)wo * C_UINT4;

    // Effective (dilated) input height: for dilation > 1, the kernel iterates over
    // hi_eff virtual rows but only loads real data for rows divisible by dilation.
    // The band computes output rows [p_lo, p_hi), counted at stride 1, from input rows
    // [y_begin, y_end) as a plane of its own: hi_eff, py_band and ho_band describe it and the
    // input and output bases start at its first rows. y_begin is a multiple of kh, which the
    // accumulator ring needs, and of the dgrad upsample factor, so it is a real row.
    const int hi_eff_plane = (cfg.dilation > 1) ? (hi - 1) * cfg.dilation + 1 : hi;
    const int p_total      = (cfg.stride == 2) ? hi_eff_plane + 2 * py - (cfg.kh - 1) : ho;
    const int p_band =
        ((p_total + h_split - 1) / h_split + cfg.stride - 1) / cfg.stride * cfg.stride;
    const int p_lo = band * p_band;
    const int p_hi = min(p_total, p_lo + p_band);
    if(p_lo >= p_hi)
        return;
    const int y_begin = max(p_lo - py, 0) / (cfg.kh * cfg.dilation) * (cfg.kh * cfg.dilation);
    const int y_end   = min(hi_eff_plane, p_hi - py + cfg.kh - 1);
    const int hi_eff  = (y_end - y_begin - 1) / cfg.dilation * cfg.dilation + 1;
    const int py_band = y_begin + py - p_lo;
    const int ho_band = (p_hi - p_lo + cfg.stride - 1) / cfg.stride;

    // The descriptor spans the workgroup's image from the band's first row, or with per_row
    // each load's spans its row.
    const auto row_bytes = static_cast<uint32_t>(wi_stride * sizeof(uint4));
    const uint32_t oob = cfg.per_row ? row_bytes : (d_in * hi - y_begin / cfg.dilation) * row_bytes;
    const uint4* image_in = reinterpret_cast<const uint4*>(in) + offset_block +
                            (size_t)(y_begin / cfg.dilation) * wi_stride;
    constexpr int rsrc_data_format = 1 << 15;
    const auto image_rsrc =
        __builtin_amdgcn_make_buffer_rsrc(const_cast<uint4*>(image_in), 0, oob, rsrc_data_format);
    struct RowLoad
    {
        __amdgpu_buffer_rsrc_t rsrc;
        uint32_t offset;
    };
    auto row_load = [&](int row) -> RowLoad {
        if constexpr(cfg.per_row)
            return {__builtin_amdgcn_make_buffer_rsrc(
                        const_cast<uint4*>(image_in + (size_t)row * wi_stride),
                        0,
                        row_bytes,
                        rsrc_data_format),
                    0};
        else
            return {image_rsrc, row * row_bytes};
    };

    // For dilation > 1, compute the first compact column that maps to LDS col 0.
    int compact_base = 0;
    if constexpr(cfg.dilation > 1)
    {
        int ds = block_q - px; // dilated start position
        // ceil(ds / dilation): positive branch uses standard ceiling formula,
        // negative branch relies on C++ truncation-toward-zero giving ceiling.
        compact_base = (ds >= 0) ? (ds + cfg.dilation - 1) / cfg.dilation : ds / cfg.dilation;
    }

    // Map threads to LDS input buffer addresses across LOAD_PASSES passes.
    bool load_active[LOAD_PASSES];
    uint4* store_input_lds_pass[LOAD_PASSES];
    uint32_t input_voffset_pass[LOAD_PASSES];
    for(int p = 0; p < LOAD_PASSES; p++)
    {
        const int slot          = tid + p * cfg.block_size();
        load_active[p]          = (slot < BLOCK_W * BLOCK_C_UINT4);
        store_input_lds_pass[p] = &input_lds[slot];

        int col = Sw::x(slot);
        int c_uint4_p;
        if constexpr(Traits::needs_lds_pack)
            c_uint4_p = Sw::c8(slot);
        else
            c_uint4_p = Sw::c4(slot);
        int gcol_p;
        if constexpr(cfg.dilation > 1)
            gcol_p = compact_base + col;
        else
            gcol_p = (block_q - px) + col;
        if(load_active[p] && 0 <= gcol_p && gcol_p < wi)
        {
            input_voffset_pass[p] = sizeof(uint4) * (gcol_p * C_UINT4 + block_c_uint4 + c_uint4_p);
        }
        else
        {
            input_voffset_pass[p] = oob;
        }
    }

    // Each wave computes one group of channels.
    auto wave_group = wave;

    // Load weights from global memory through LDS into registers.
    // weights_reg holds mfma_operand_t: fp16x4/bf16x4 for fp16/bf16 (identity),
    // bf16_pair_x4 for tf32 (caller-side fp32 -> bf16 pair split done in this
    // prologue so the K loop reuses the pre-split operands).
    mfma_operand_t weights_reg[cfg.kd * cfg.kh * cfg.kw];
    {
        auto weight_bytes = static_cast<size_t>(groups * cfg.group_size) * kd * cfg.kh * cfg.kw *
                            cfg.group_size * sizeof(element_t);
        auto weight_rsrc = __builtin_amdgcn_make_buffer_rsrc(
            const_cast<element_t*>(wei), 0, weight_bytes, rsrc_data_format);
        uint32_t voffset_base = block_k * kd * cfg.kh * cfg.kw * cfg.group_size * sizeof(element_t);

        // A conv3d stages one depth tap at a time, so LDS keeps its conv2d size.
        constexpr int TAP_UINT4 = cfg.kh * cfg.kw * GROUP_UINT4;
        static_for<cfg.kd>([&]<int J>() {
            if constexpr(J > 0)
                sync_before_lds_dma<ONE_WAVE>();
            for(int j = tid; j < WEIGHT_LDS_SIZE_UINT4; j += cfg.block_size())
            {
#ifdef __gfx950__
                __builtin_amdgcn_raw_ptr_buffer_load_lds(
                    weight_rsrc,
                    &lds_buf[j],
                    16,
                    voffset_base +
                        weight_tap_offset<TAP_UINT4, cfg.kd>(j, taps.tap[J], kd) * sizeof(uint4),
                    0,
                    0,
                    0);
#endif
            }

            wait_vmcnt<0>();
            __syncthreads();

            if constexpr(cfg.direction == Direction::Dgrad)
            {
                if constexpr(Traits::dgrad_use_tr16)
                {
                    // fp16/bf16 path: ds_read_tr16_b64 performs the (k <-> c)
                    // transpose in hardware.
                    const int k_stride = cfg.kh * cfg.kw * cfg.group_size;
                    auto* wei_grp_base = reinterpret_cast<element_t*>(lds_buf) +
                                         wave_group * cfg.group_size * k_stride;

                    using TransposeLayout = TransposeLDSLayout<16, 16>;
                    const int k           = TransposeLayout::row(lane);
                    const int c           = TransposeLayout::col(lane);

#pragma unroll
                    for(int khw = 0; khw < cfg.kh * cfg.kw; khw++)
                    {
                        auto* addr = reinterpret_cast<int16x4_t*>(wei_grp_base + k * k_stride +
                                                                  khw * cfg.group_size + c);

                        int16x4_t r = __builtin_amdgcn_ds_read_tr16_b64_v4i16(addr);
                        weights_reg[J * cfg.kh * cfg.kw + khw] =
                            Traits::to_mfma_operand(__builtin_bit_cast(operand_t, r));
                    }
                }
                else
                {
                    // tf32 path: software (k <-> c) transpose with 4 scalar fp32
                    // reads per filter tap (36 total for kh*kw=9, amortised over
                    // the entire main loop).
                    const element_t* wei_elem = reinterpret_cast<const element_t*>(lds_buf);
                    const int k_stride        = cfg.kh * cfg.kw * cfg.group_size;
                    const int base_k          = wave_group * cfg.group_size;
                    const int c_lane          = lane % 16;
                    const int k_base          = (lane / 16) * 4;

#pragma unroll
                    for(int khw = 0; khw < cfg.kh * cfg.kw; khw++)
                    {
                        operand_t w_raw;
                        for(int i = 0; i < 4; i++)
                        {
                            int k_global = base_k + k_base + i;
                            int idx      = k_global * k_stride + khw * cfg.group_size + c_lane;
                            w_raw[i]     = wei_elem[idx];
                        }
                        weights_reg[J * cfg.kh * cfg.kw + khw] = Traits::to_mfma_operand(w_raw);
                    }
                }
            }
            else
            {
                // Fprop: linear weight read in MFMA operand layout.
                auto lane_k    = OperandLayout::outer(lane);
                auto lane_c4_w = OperandLayout::inner(lane) / 4;
                auto k_local   = wave_group * cfg.group_size + lane_k;
                // weight LDS in lds_unit_t units: each (k_local, khw) tap has
                // GROUP_UINT4 * LDS_UNIT_PER_UINT4 lds_unit_t slots.
                constexpr int GROUP_LDS_UNIT = GROUP_UINT4 * LDS_UNIT_PER_UINT4;
                auto* weights_lds            = reinterpret_cast<const lds_unit_t*>(lds_buf) +
                                    k_local * (cfg.kh * cfg.kw * GROUP_LDS_UNIT) + lane_c4_w;
#pragma unroll
                for(int khw = 0; khw < cfg.kh * cfg.kw; khw++)
                {
                    operand_t w_raw = *(const operand_t*)(weights_lds + khw * GROUP_LDS_UNIT);
                    weights_reg[J * cfg.kh * cfg.kw + khw] = Traits::to_mfma_operand(w_raw);
                }
            }
        });
    }

    // Ensure all threads have finished reading weights from LDS before
    // the input buffer_load_lds overwrites the same lds_buf region.
    sync_before_lds_dma<ONE_WAVE>();

    // Issue a multi-pass async input load for dilated row index y_local.
    auto issue_input_load = [&](int y_local, int slot, int tic_phase) {
        if(y_local < 0 || y_local >= hi_eff)
            return;
        if constexpr(cfg.dilation > 1)
        {
            if((y_local % cfg.dilation) != 0)
                return;
        }
        const int y_global = (cfg.dilation > 1) ? (y_local / cfg.dilation) : y_local;
        const RowLoad row  = row_load(taps.plane[slot] * hi + y_global);
        for(int p = 0; p < LOAD_PASSES; p++)
        {
            if(load_active[p])
            {
                __builtin_amdgcn_raw_ptr_buffer_load_lds(row.rsrc,
                                                         store_input_lds_pass[p] +
                                                             tic_phase * INPUT_LDS_BUFFER_UINT4,
                                                         16,
                                                         input_voffset_pass[p] + row.offset,
                                                         0,
                                                         0,
                                                         0);
            }
        }
    };

    // Prologue: seed buffer 0 (toc init = 0).
    if(taps.count > 0)
        issue_input_load(0, 0, /*tic_phase=*/0);

    // Map lanes to MFMA result matrix coordinates.
    const int lane_q  = ResultLayout::outer(lane);
    const int lane_c4 = ResultLayout::inner(lane) / 4;

    // Map threads to output addresses for transferring through LDS to global memory.
    // Each lane writes one uint2 (4 fp16/bf16 channels) or one uint4 (4 fp32 channels).
    const bool store_active      = (tid < STORE_VECS);
    const uint4* load_output_lds = nullptr;
    uint4* store_output_global   = nullptr;
    if(store_active)
    {
        int c_uint4;
        int col;
        if constexpr(Traits::needs_lds_pack)
        {
            c_uint4 = tid % BLOCK_C_UINT4;
            col     = tid / BLOCK_C_UINT4;
        }
        else
        {
            // SwizzleT_fp32::c4 has a `(4*x) % C4` correction that does not
            // vanish for waves_per_wg ∈ {3, 5, 6, 7}, so use the swizzle
            // inverses rather than naive modulo.
            col     = Sw::x(tid);
            c_uint4 = Sw::c4(tid);
        }
        if constexpr(cfg.stride == 2)
        {
            const int q_out = (block_q / 2) + col;
            load_output_lds = &output_lds[Sw::offset_uint4(col, c_uint4)];
            if(q_out < wo)
            {
                const int K_UINT4   = C / CH_PER_UINT4;
                store_output_global = reinterpret_cast<uint4*>(out) +
                                      ((size_t)block_n * ho + p_lo / cfg.stride) * wo * K_UINT4 +
                                      (size_t)q_out * K_UINT4 + block_c_uint4 + c_uint4;
            }
        }
        else
        {
            const int q     = block_q + col;
            load_output_lds = &output_lds[Sw::offset_uint4(col, c_uint4)];
            if(q < wo)
            {
                const int K_UINT4   = C / CH_PER_UINT4;
                store_output_global = reinterpret_cast<uint4*>(out) +
                                      ((size_t)block_n * ho + p_lo / cfg.stride) * wo * K_UINT4 +
                                      (size_t)q * K_UINT4 + block_c_uint4 + c_uint4;
            }
        }
    }

    // Pre-compute the kw input LDS offsets (lds_unit_t units) for this thread.
    auto input_lds_offset_fn = [](int x, int c4) constexpr -> int {
        if constexpr(Traits::needs_lds_pack)
            return Sw::offset_uint2(x, c4);
        else
            return Sw::offset_uint4(x, c4);
    };

    constexpr int LANE_GROUP_C4 = cfg.group_size / 4; // 4: c4 strips per group
    int input_lds_offsets[cfg.kw];
    bool input_lds_mask[cfg.kw]; // only meaningful for dilation > 1
    static_for<cfg.kw>([&]<int S>() {
        if constexpr(cfg.dilation > 1)
        {
            int dilated_global = (block_q - px) + lane_q + S;
            // C++ % truncates toward zero, so -1 % 2 == -1 (not 0); == 0 is correct.
            bool is_real = (dilated_global % cfg.dilation == 0);
            int lds_col  = is_real ? (dilated_global / cfg.dilation - compact_base) : 0;
            input_lds_offsets[S] =
                input_lds_offset_fn(lds_col, wave_group * LANE_GROUP_C4 + lane_c4);
            input_lds_mask[S] = is_real;
        }
        else
        {
            input_lds_offsets[S] =
                input_lds_offset_fn(lane_q + S, wave_group * LANE_GROUP_C4 + lane_c4);
            input_lds_mask[S] = true;
        }
    });

    // Pre-compute the output LDS swizzle offset (thread-constant).
    int output_lds_offset;
    bool output_lane_active;
    if constexpr(cfg.stride == 2)
    {
        output_lane_active = (lane_q % 2 == 0);
        output_lds_offset  = input_lds_offset_fn(lane_q / 2, wave_group * LANE_GROUP_C4 + lane_c4);
    }
    else
    {
        output_lane_active = true;
        output_lds_offset  = input_lds_offset_fn(lane_q, wave_group * LANE_GROUP_C4 + lane_c4);
    }

    // Circular buffer of accumulators.
    constexpr auto Zero = fp32x4_t{0.f, 0.f, 0.f, 0.f};
    fp32x4_t acc[cfg.kh];
    for(int i = 0; i < cfg.kh; i++)
        acc[i] = Zero;

    int tic = 1;
    int toc = 0;

    // Flush one accumulator slot to global memory via the LDS staging area.
    //   1. fp16/bf16: packed_convert<element_t>(slot) + uint2 store;
    //      tf32: raw uint4 store (accumulator already fp32x4 = uint4).
    //   2. __syncthreads to commit the LDS staging area.
    //   3. Coalesced HBM uint4 store via *load_output_lds.
    auto flush_p = [&](fp32x4_t slot, int p_out) {
        if constexpr(cfg.stride == 2)
        {
            if(!(p_out >= 0 && (p_out % 2) == 0 && (p_out / 2) < ho_band))
                return;
        }
        else
        {
            if(!(p_out >= 0 && p_out < ho_band))
                return;
        }

        if constexpr(Traits::needs_lds_pack)
        {
            auto out_reg           = packed_convert<element_t>(slot);
            auto* store_output_lds = reinterpret_cast<lds_unit_t*>(output_lds);
            if constexpr(cfg.stride == 2)
            {
                if(output_lane_active)
                    store_output_lds[output_lds_offset] =
                        *reinterpret_cast<const lds_unit_t*>(&out_reg);
            }
            else
            {
                store_output_lds[output_lds_offset] =
                    *reinterpret_cast<const lds_unit_t*>(&out_reg);
            }
        }
        else
        {
            if constexpr(cfg.stride == 2)
            {
                if(output_lane_active)
                    output_lds[output_lds_offset] = *reinterpret_cast<const uint4*>(&slot);
            }
            else
            {
                output_lds[output_lds_offset] = *reinterpret_cast<const uint4*>(&slot);
            }
        }

        __syncthreads();

        if(store_output_global)
        {
            const int wo_row                        = (cfg.stride == 2) ? (p_out / 2) : p_out;
            store_output_global[wo_row * wo_stride] = *load_output_lds;
        }
    };

    // Main loop iterates over input rows (dilated rows for dilation > 1).
    for(int y_base = 0; y_base + cfg.kh <= hi_eff; y_base += cfg.kh)
    {
        static_for<cfg.kh>([&]<int Y_LOCAL>() {
            int y = y_base + Y_LOCAL;
            static_for<cfg.kd>([&]<int T>() {
                if(T >= taps.count)
                    return;
                wait_vmcnt<0>();
                __syncthreads();

                if(T + 1 < taps.count)
                    issue_input_load(y, T + 1, tic);
                else if(y + 1 < hi_eff)
                    issue_input_load(y + 1, 0, tic);

                static_for<cfg.kw>([&]<int S>() {
                    const lds_unit_t* lds_ptr = reinterpret_cast<const lds_unit_t*>(input_lds) +
                                                toc * INPUT_LDS_BUFFER_LDS_UNIT +
                                                input_lds_offsets[S];
                    operand_t input_raw = *(const operand_t*)lds_ptr;

                    if constexpr(cfg.dilation > 1)
                    {
                        bool row_is_real = (y % cfg.dilation == 0);
                        if(!row_is_real || !input_lds_mask[S])
                            input_raw = operand_t{0, 0, 0, 0};
                    }

                    // Per-S input split (caller-side pre-split contract): for tf32
                    // this is one fp32 -> bf16-pair conversion reused across the R
                    // taps; for fp16/bf16 it is the identity.
                    mfma_operand_t input_reg = Traits::to_mfma_operand(input_raw);

                    static_for<cfg.kh>([&]<int R>() {
                        constexpr int p_idx = (Y_LOCAL - R + cfg.kh) % cfg.kh;
                        if constexpr(cfg.direction == Direction::Dgrad)
                            acc[p_idx] = mfma_16x16x16(
                                weights_reg[T * cfg.kh * cfg.kw + (cfg.kh - 1 - R) * cfg.kw +
                                            (cfg.kw - 1 - S)],
                                input_reg,
                                acc[p_idx]);
                        else
                            acc[p_idx] =
                                mfma_16x16x16(weights_reg[T * cfg.kh * cfg.kw + R * cfg.kw + S],
                                              input_reg,
                                              acc[p_idx]);
                    });
                });

                tic ^= 1;
                toc ^= 1;
            });
            // With no live tap nothing above synced, and flush_p reuses the output LDS.
            if(cfg.kd > 1 && taps.count == 0)
                __syncthreads();

            constexpr int P_FLUSH = (Y_LOCAL + 1) % cfg.kh;
            int p_out             = y + py_band - (cfg.kh - 1);
            flush_p(acc[P_FLUSH], p_out);
            acc[P_FLUSH] = Zero;
        });
    }

    // Remainder: the hi_eff % kh leftover rows.
    {
        int y_rem_base = (hi_eff / cfg.kh) * cfg.kh;
        static_for<cfg.kh>([&]<int Y_LOCAL>() {
            if(Y_LOCAL >= hi_eff % cfg.kh)
                return;
            int y = y_rem_base + Y_LOCAL;
            static_for<cfg.kd>([&]<int T>() {
                if(T >= taps.count)
                    return;
                wait_vmcnt<0>();
                __syncthreads();

                if(T + 1 < taps.count)
                    issue_input_load(y, T + 1, tic);
                else if(y + 1 < hi_eff)
                    issue_input_load(y + 1, 0, tic);

                static_for<cfg.kw>([&]<int S>() {
                    const lds_unit_t* lds_ptr = reinterpret_cast<const lds_unit_t*>(input_lds) +
                                                toc * INPUT_LDS_BUFFER_LDS_UNIT +
                                                input_lds_offsets[S];
                    operand_t input_raw = *(const operand_t*)lds_ptr;

                    if constexpr(cfg.dilation > 1)
                    {
                        bool row_is_real = (y % cfg.dilation == 0);
                        if(!row_is_real || !input_lds_mask[S])
                            input_raw = operand_t{0, 0, 0, 0};
                    }

                    mfma_operand_t input_reg = Traits::to_mfma_operand(input_raw);

                    static_for<cfg.kh>([&]<int R>() {
                        constexpr int p_idx = (Y_LOCAL - R + cfg.kh) % cfg.kh;
                        if constexpr(cfg.direction == Direction::Dgrad)
                            acc[p_idx] = mfma_16x16x16(
                                weights_reg[T * cfg.kh * cfg.kw + (cfg.kh - 1 - R) * cfg.kw +
                                            (cfg.kw - 1 - S)],
                                input_reg,
                                acc[p_idx]);
                        else
                            acc[p_idx] =
                                mfma_16x16x16(weights_reg[T * cfg.kh * cfg.kw + R * cfg.kw + S],
                                              input_reg,
                                              acc[p_idx]);
                    });
                });

                tic ^= 1;
                toc ^= 1;
            });
            // With no live tap nothing above synced, and flush_p reuses the output LDS.
            if(cfg.kd > 1 && taps.count == 0)
                __syncthreads();

            constexpr int P_FLUSH = (Y_LOCAL + 1) % cfg.kh;
            int p_out             = y + py_band - (cfg.kh - 1);
            flush_p(acc[P_FLUSH], p_out);
            acc[P_FLUSH] = Zero;
        });
    }

    // Tail: flush output rows whose last input contribution would land at y >= hi_eff.
    {
        const int tail_end = ho_band * cfg.stride;
        for(int p_out = hi_eff - cfg.kh + 1 + py_band; p_out < tail_end; p_out++)
        {
            if constexpr(cfg.stride == 2)
            {
                if(p_out % 2 != 0 || p_out / 2 >= ho_band)
                    continue;
            }

            __syncthreads(); // separate prior LDS reads from this iteration's writes
            int p_idx = (p_out - py_band + cfg.kh) % cfg.kh;
            fp32x4_t slot;
            dispatch<cfg.kh>(p_idx, [&]<int P>() {
                slot   = acc[P];
                acc[P] = Zero;
            });
            flush_p(slot, p_out);
        }
    }
}

// Compute grouped convolution with stride 1 or 2 and group size 16.
//
// This kernel assumes the number of input and output channels is the same.
//
// Input layout: N hi wi C_UINT4
// Weights layout: K kh kw GROUP_UINT4
// Output layout: N ho wo K_UINT4
template <Config cfg, DataType DT>
// A conv3d tile holds kd taps of weights; let small workgroups claim the registers.
__global__
__launch_bounds__(cfg.kd == 1 ? 1024 : cfg.block_size()) void conv2d_grouped_16c_nhwc_cdna4(
    const ToType<DT>* __restrict__ in,
    const ToType<DT>* __restrict__ wei,
    double alpha,
    double beta,
    ToType<DT>* __restrict__ out,
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
    int px,
    DepthParams dp)
{
    if(__builtin_amdgcn_is_invocable(__builtin_amdgcn_mfma_f32_16x16x16f16) &&
       __builtin_amdgcn_is_invocable(__builtin_amdgcn_mfma_f32_16x16x16bf16_1k) &&
       __builtin_amdgcn_is_invocable(__builtin_amdgcn_raw_ptr_buffer_load_lds) &&
       __builtin_amdgcn_is_invocable(__builtin_amdgcn_ds_read_tr16_b64_v4i16))
    {
        conv2d_grouped_16c_cdna4_nhwc_impl<cfg, DT>(in,
                                                    wei,
                                                    alpha,
                                                    beta,
                                                    out,
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
                                                    px,
                                                    dp);
    }
}

// tf32 doubles the weight registers; these tiles would spill them to scratch.
constexpr bool tf32_fits(const Config& cfg)
{
    return cfg.waves_per_wg <= (cfg.kd <= 3 ? 8 : 4);
}

template <Config cfg>
void launch_impl(const LaunchParams& lp,
                 const Device&,
                 const ConvParams& par,
                 const void* in,
                 const void* wei,
                 void* out,
                 void* /*workspace*/,
                 hipStream_t stream)
{
    auto typed_launch = [&]<DataType DT>() {
        using dtype = ToType<DT>;
        auto view   = SizeView<cfg.direction>(par);
        conv2d_grouped_16c_nhwc_cdna4<cfg, DT>
            <<<lp.grid, lp.block_size, lp.dynamic_shared_bytes, stream>>>(
                static_cast<const dtype*>(in),
                static_cast<const dtype*>(wei),
                1.0,
                0.0,
                static_cast<dtype*>(out),
                par.n,
                par.groups,
                par.channels_per_group(),
                par.filters_per_group(),
                view.h(),
                view.w(),
                view.p(),
                view.q(),
                par.kh,
                par.kw,
                par.stride_h,
                par.stride_w,
                par.dilation_h,
                par.dilation_w,
                view.pad_h(),
                view.pad_w(),
                DepthParams::of(par));
    };
    if(par.input_type == DataType::bf16)
        typed_launch.template operator()<DataType::bf16>();
    else if(par.input_type == DataType::tf32)
    {
        if constexpr(tf32_fits(cfg))
            typed_launch.template operator()<DataType::tf32>();
    }
    else
        typed_launch.template operator()<DataType::fp16>();
}

class Grouped_16C_ConvKernel : public GroupedConvKernel
{
public:
    constexpr Grouped_16C_ConvKernel(const Config& cfg, LaunchFn launch_fn)
        : GroupedConvKernel(launch_fn)
        , cfg_(cfg)
    {
    }

    bool is_valid_config(const Device& dev, const ConvParams& par) const override
    {
        if(par.direction != cfg_.direction)
            return false;
        if(depth_tap_capacity(par) != cfg_.kd)
            return false;
        if(cfg_.per_row != grouped_image_past_2gib(par))
            return false;
        if(par.input_type == DataType::tf32 && !tf32_fits(cfg_))
            return false;

        if(cfg_.direction == Direction::Dgrad)
        {
            // Dgrad swaps stride <-> dilation relative to forward params
            if(par.dilation_h != cfg_.stride || par.dilation_w != cfg_.stride)
                return false;
            if(par.stride_h != cfg_.dilation || par.stride_w != cfg_.dilation)
                return false;
        }
        else
        {
            if(par.stride_h != cfg_.stride || par.stride_w != cfg_.stride)
                return false;
            if(par.dilation_h != cfg_.dilation || par.dilation_w != cfg_.dilation)
                return false;
        }

        if((par.groups % cfg_.waves_per_wg) != 0)
            return false;

        // TF32 doubles the LDS footprint; gate off waves_per_wg=16 configs.
        if(par.input_type == DataType::tf32 && cfg_.waves_per_wg > 8)
            return false;

        if(get_launch_params(dev, par).grid.z > max_grid_z)
            return false;
        return true;
    }

    LaunchParams get_launch_params(const Device& /*dev*/, const ConvParams& par) const override
    {
        const int out_q          = (cfg_.direction == Direction::Dgrad) ? par.w : par.q;
        const int output_block_q = BLOCK_Q / cfg_.stride;
        auto blocks_w            = divup(out_q, output_block_q);
        auto blocks_w_n          = blocks_w * cfg_.n_fold;
        auto blocks_c            = divup(par.c, cfg_.block_c());
        auto blocks_n_fold =
            divup(par.n * (cfg_.direction == Direction::Dgrad ? par.d : par.e), cfg_.n_fold);

        LaunchParams launch;
        launch.grid       = dim3(blocks_w_n, blocks_c, blocks_n_fold);
        launch.block_size = dim3(cfg_.block_size(), 1, 1);
        return launch;
    }
    bool supports_dims(int dims) const override { return dims == 2 || dims == 3; }
    KVDescriptor config_descriptor() const override { return config_fields(cfg_); }

protected:
    int group_channels() const override { return 16; }

private:
    const Config& cfg_;
};


} // namespace hipconv::cdna4::grouped_16c
