#include "config.hpp"
#include "config_table.hpp"
#include "config_desc.hpp"

// CDNA4 grouped convolution kernel, 8 channels per group.
//
// This unified TU subsumes the original grouped_8c (fp16/bf16) and
// grouped_8c_tf32 (Phase 1) variants behind a single GroupedDataTraits<DT>
// abstraction (mirroring grouped_4c.cpp). The kernel body is parameterised by
// DataType DT; fp16, bf16 and tf32 share the same control flow with only two
// structural branches:
//   1) Output flush stage: fp16/bf16 pack the fp32x4 accumulator slice to
//      element_t via packed_convert and store one uint2 per lane; tf32 keeps
//      the fp32x4 accumulator and stores it as raw uint4 (no packed_convert).
//   2) Input LDS load width: fp16/bf16 reads 8 channels = 1 uint4 per lane;
//      tf32 reads 8 channels = 2 uint4 per lane (via fp32x8_t reinterpret).
//      Both modes are subsumed by reading sizeof(operand_t) bytes per lane and
//      passing the result through Traits::to_mfma_operand.
//
// TF32 (simulated): A and B are stored as fp32 in HBM, but the MFMA path
// decomposes each fp32 operand into a (big, small) BF16 pair and runs three
// BF16 MFMAs (big*big + big*small + small*big). The split is performed once
// per Y_LOCAL in the input main loop and once per filter tap in the weight
// prologue via fp32xN_to_bf16_pair; the main loop calls the overload
// mfma_16x16x32(bf16_pair_x8, bf16_pair_x8, fp32x4_t) in mfma_dispatch.h that
// expands to the three BF16 MFMAs above. This keeps the per-MFMA-call cost
// identical to FP16/BF16 8c and avoids repeating the fp32 -> bf16 split inside
// the inner loops.

#include "direction_field.h"
#include "kv_descriptor.h"
#include "grouped_conv_kernel.h"
#include "../grouped_8c_transforms.h"
#include "matrix_layout.h"
#include "swizzle.h"
#include "detail.h"
#include "types.h"
#include "../mfma_dispatch.h"
#include "../depth_taps.h"
#include "mathutil.h"
#include "launch_params.h"
#include "packed_ops.h"
#include "memory.h"
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>

namespace hipconv::cdna4::grouped_8c
{

using namespace hipconv;
using namespace hipconv::cdna4;

// ----------------------------------------------------------------------------
// Per-dtype traits: type aliases for the 8-wide operand vector, the value fed
// to mfma_16x16x32, the LDS access unit for input reads, and the swizzle
// layout, plus to_mfma_operand / zero_operand helpers that absorb the fp32 ->
// bf16-pair split for tf32 while remaining identity for fp16/bf16. Everything
// is __forceinline__ so the SASS for fp16/bf16 stays equivalent to the
// original hand-written `*(const datatypex8_t*)lds_ptr` reads.
// ----------------------------------------------------------------------------

template <DataType DT>
struct GroupedDataTraits;

template <>
struct GroupedDataTraits<DataType::fp16>
{
    using element_t      = fp16_t;
    using operand_t      = fp16x8_t;
    using mfma_operand_t = fp16x8_t;
    template <int C>
    using Sw                             = SwizzleT<C>;
    static constexpr int CH_PER_UINT4    = 8;
    static constexpr bool needs_lds_pack = true;
    static __device__ __forceinline__ mfma_operand_t to_mfma_operand(operand_t raw) { return raw; }
    static __device__ __forceinline__ mfma_operand_t zero_operand()
    {
        return mfma_operand_t{0, 0, 0, 0, 0, 0, 0, 0};
    }
};

template <>
struct GroupedDataTraits<DataType::bf16>
{
    using element_t      = bf16_t;
    using operand_t      = bf16x8_t;
    using mfma_operand_t = bf16x8_t;
    template <int C>
    using Sw                             = SwizzleT<C>;
    static constexpr int CH_PER_UINT4    = 8;
    static constexpr bool needs_lds_pack = true;
    static __device__ __forceinline__ mfma_operand_t to_mfma_operand(operand_t raw) { return raw; }
    static __device__ __forceinline__ mfma_operand_t zero_operand()
    {
        return mfma_operand_t{
            bf16_t(0),
            bf16_t(0),
            bf16_t(0),
            bf16_t(0),
            bf16_t(0),
            bf16_t(0),
            bf16_t(0),
            bf16_t(0),
        };
    }
};

template <>
struct GroupedDataTraits<DataType::tf32>
{
    using element_t      = fp32_t;
    using operand_t      = fp32x8_t;
    using mfma_operand_t = bf16_pair_x8;
    template <int C>
    using Sw                             = SwizzleT_fp32<C>;
    static constexpr int CH_PER_UINT4    = 4;
    static constexpr bool needs_lds_pack = false;
    static __device__ __forceinline__ mfma_operand_t to_mfma_operand(operand_t raw)
    {
        return fp32xN_to_bf16_pair(raw);
    }
    static __device__ __forceinline__ mfma_operand_t zero_operand()
    {
        return fp32xN_to_bf16_pair(fp32x8_t{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f});
    }
};


template <Config cfg, DataType DT>
__device__ void conv2d_grouped_8c_cdna4_nhwc_impl(const ToType<DT>* __restrict__ in,
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
    using namespace grouped_8c_transforms;
    using Traits         = GroupedDataTraits<DT>;
    using element_t      = typename Traits::element_t;
    using operand_t      = typename Traits::operand_t;
    using mfma_operand_t = typename Traits::mfma_operand_t;
    using Sw             = typename Traits::template Sw<cfg.block_c()>;

    // MFMA 16x16x32: A/B operands are 8-wide (8 fp16/bf16 channels or 8 fp32
    // packed into a bf16 pair); result is fp32x4. OperandLayout is
    // parameterised by element_t but the Toeplitz GT helpers are dtype
    // independent (only per-lane fragment widths matter).
    using OperandLayout = MatrixLayout<16, 32, 1, element_t>;
    using ResultLayout  = MatrixLayout<16, 16, 1, float>;

    // Channels per uint4 HBM/LDS transfer: 8 for fp16/bf16, 4 for tf32.
    constexpr int CH_PER_UINT4 = Traits::CH_PER_UINT4;
    // Number of uint4 slots per group along C (1 for fp16/bf16, 2 for tf32).
    constexpr int GROUP_UINT4 = cfg.group_size / CH_PER_UINT4;

    static_assert(sizeof(uint4) / sizeof(element_t) == CH_PER_UINT4,
                  "CH_PER_UINT4 must match element size");

    // Number of input columns loaded by each workgroup (output columns plus halo).
    constexpr int BLOCK_W = BLOCK_Q + (cfg.kw - 1);

    // uint4 vectors per channels fiber (BLOCK_C / CH_PER_UINT4).
    constexpr int BLOCK_C_UINT4 = cfg.block_c() / CH_PER_UINT4;

    // Each wave processes a single group of channels.
    constexpr int BLOCK_GROUPS = cfg.waves_per_wg;

    // Output block width after subsampling for stride > 1.
    constexpr int OUTPUT_BLOCK_Q = BLOCK_Q / cfg.stride;

    // Vectors to store (output uint4 elements per workgroup).
    constexpr int STORE_VECS = OUTPUT_BLOCK_Q * BLOCK_C_UINT4;

    constexpr bool ONE_WAVE = cfg.block_size() == WAVE_SIZE;

    constexpr int NUM_INPUT_LDS_BUFFERS = 2;

    constexpr int INPUT_LDS_BUFFER_UINT4  = BLOCK_C_UINT4 * BLOCK_W;
    constexpr int OUTPUT_LDS_BUFFER_UINT4 = BLOCK_C_UINT4 * OUTPUT_BLOCK_Q;

    // Weight LDS: each (k, r, s) tap is group_size element_t channels =
    // group_size * sizeof(element_t) bytes = GROUP_UINT4 uint4 slots.
    constexpr int WEIGHT_LDS_SIZE_UINT4 =
        BLOCK_GROUPS * cfg.group_size * cfg.kh * cfg.kw * GROUP_UINT4;

    // Multi-pass HBM->LDS loading. Each thread issues one uint4 per pass.
    // fp16/bf16 always degenerates to LOAD_PASSES == 1 (block_size covers the
    // whole row); tf32 generally needs 2 passes because every uint4 holds only
    // 4 channels. With LOAD_PASSES being a constexpr loop bound the plain for
    // body is fully unrolled, matching the original inline single-pass loader
    // for fp16/bf16.
    constexpr int LOAD_PASSES = (BLOCK_W * BLOCK_C_UINT4 + cfg.block_size() - 1) / cfg.block_size();

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

    // Dgrad stride 2 runs over dY upsampled by UPS: a virtual hi_eff-row grid whose real
    // samples sit at multiples of UPS, the rest zero.
    constexpr int UPS = cfg.dilation;
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

    // Map threads to LDS input buffer addresses across LOAD_PASSES passes.
    // Each pass writes block_size() uint4's; passes whose linearised slot
    // overflows BLOCK_W*BLOCK_C_UINT4 mark their lanes inactive.
    bool load_active[LOAD_PASSES];
    uint4* store_input_lds_pass[LOAD_PASSES];
    for(int p = 0; p < LOAD_PASSES; p++)
    {
        const int slot          = tid + p * cfg.block_size();
        load_active[p]          = (slot < BLOCK_W * BLOCK_C_UINT4);
        store_input_lds_pass[p] = &input_lds[slot];
    }

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

    // Per-pass voffset using SwizzleT::x and the channel-block index (c8 for
    // fp16/bf16, c4 for tf32) inverse functions on the linearised slot.
    uint32_t input_voffset_pass[LOAD_PASSES];
    for(int p = 0; p < LOAD_PASSES; p++)
    {
        const int slot = tid + p * cfg.block_size();
        const int col  = Sw::x(slot);
        int c_uint4_p;
        if constexpr(Traits::needs_lds_pack)
            c_uint4_p = Sw::c8(slot);
        else
            c_uint4_p = Sw::c4(slot);
        const int global_col = (block_q - px) + col;
        const bool zero_col  = (UPS > 1) && ((global_col % UPS) != 0);
        const int dY_col     = (UPS > 1) ? (global_col / UPS) : global_col;
        if(load_active[p] && 0 <= global_col && !zero_col && dY_col < wi)
        {
            input_voffset_pass[p] = sizeof(uint4) * (dY_col * C_UINT4 + block_c_uint4 + c_uint4_p);
        }
        else
        {
            input_voffset_pass[p] = oob;
        }
    }

    auto wave_group = wave;

    // Load weights from global memory through LDS into registers.
    // weights_reg holds mfma_operand_t: fp16x8/bf16x8 for fp16/bf16 (identity
    // conversion), bf16_pair_x8 for tf32 (caller-side fp32 -> bf16 pair split
    // computed once in this prologue so the K loop reuses the pre-split
    // operands without recomputing).
    mfma_operand_t weights_reg[cfg.kd * cfg.kh];
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

            // Map lane to GT matrix position.
            int row = lane % 16;
            int g   = lane / 16;

            if constexpr(cfg.direction == Direction::Dgrad)
            {
                // Dgrad path: software (k <-> c) transpose with scalar reads. Both
                // dtype branches index lds_buf reinterpreted as element_t.
                int s_dgrad = cfg.kw - 1 - g + GT::q(row);
                int c_out   = GT::k(row);

                if(GT::filter_is_zero(row, g * 2))
                {
#pragma unroll
                    for(int r = 0; r < cfg.kh; r++)
                        weights_reg[J * cfg.kh + r] = Traits::zero_operand();
                }
                else
                {
                    const element_t* wei_elem = reinterpret_cast<const element_t*>(lds_buf);
                    int base_k                = wave_group * cfg.group_size;
#pragma unroll
                    for(int r = 0; r < cfg.kh; r++)
                    {
                        operand_t w_raw;
                        for(int k = 0; k < cfg.group_size; k++)
                        {
                            int idx = (base_k + k) * cfg.kh * cfg.kw * cfg.group_size +
                                      r * cfg.kw * cfg.group_size + s_dgrad * cfg.group_size +
                                      c_out;
                            w_raw[k] = wei_elem[idx];
                        }
                        weights_reg[J * cfg.kh + r] = Traits::to_mfma_operand(w_raw);
                    }
                }
            }
            else
            {
                int k_val = GT::k(row);
                int s_val = GT::s(row, g * 2);

                if(GT::filter_is_zero(row, g * 2))
                {
#pragma unroll
                    for(int r = 0; r < cfg.kh; r++)
                        weights_reg[J * cfg.kh + r] = Traits::zero_operand();
                }
                else
                {
                    int k_within_wg = wave_group * cfg.group_size + k_val;
#pragma unroll
                    for(int r = 0; r < cfg.kh; r++)
                    {
                        // Each tap is group_size element_t channels = GROUP_UINT4
                        // uint4 slots in LDS. For fp16/bf16 GROUP_UINT4 = 1, so the
                        // uint4 index equals offset_in_taps; for tf32 it is 2.
                        int offset_in_taps = k_within_wg * cfg.kh * cfg.kw + r * cfg.kw + s_val;
                        const operand_t* tap_ptr = reinterpret_cast<const operand_t*>(
                            &lds_buf[offset_in_taps * GROUP_UINT4]);
                        weights_reg[J * cfg.kh + r] = Traits::to_mfma_operand(*tap_ptr);
                    }
                }
            }
        });
    }

    // Ensure all threads have finished reading weights from LDS before
    // the input buffer_load_lds overwrites the same lds_buf region.
    sync_before_lds_dma<ONE_WAVE>();

    // Issue an async input load for input row y_local, splitting work over
    // LOAD_PASSES passes. fp16/bf16 has LOAD_PASSES == 1 so the for-loop
    // unrolls to a single buffer_load_lds, matching the original inline path.
    // Upsample-zero rows load nothing; their MFMAs are skipped via row_is_real.
    auto row_is_real      = [&](int y_local) { return UPS == 1 || (y_local % UPS) == 0; };
    auto issue_input_load = [&](int y_local, int slot, int tic_phase) {
        if(y_local < 0 || y_local >= hi_eff || !row_is_real(y_local))
            return;
        const RowLoad row = row_load(taps.plane[slot] * hi + y_local / UPS);
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

    // Initial async input load -> buffer 0. The main loop reads from `toc`
    // (init 0) and writes the next row into `tic` (init 1), so the prologue
    // must seed buffer 0 to match the very first MFMA's input.
    if(taps.count > 0)
        issue_input_load(0, 0, /*tic_phase=*/0);

    // Map threads to output addresses for transferring through LDS to global
    // memory. For fp16/bf16 each lane writes one uint2 (4 fp16/bf16 channels);
    // for tf32 each lane writes one uint4 (4 fp32). The store_active counter
    // and the c_uint4 mapping share the same unit (uint4) per HBM row.
    const bool store_active      = (tid < STORE_VECS);
    const uint4* load_output_lds = nullptr;
    uint4* store_output_global   = nullptr;
    if(store_active)
    {
        // For fp16/bf16, BLOCK_C_UINT4 == waves_per_wg, so tid % BLOCK_C_UINT4
        // == tid % waves_per_wg matches the original Sw::c8 inversion via
        // direct modulo. For tf32 the swizzle inverse is non-trivial because
        // c4 has a (4*x) % C4 term; we use the inverse functions for both.
        int c_uint4;
        int col;
        if constexpr(Traits::needs_lds_pack)
        {
            c_uint4 = tid % BLOCK_C_UINT4;
            col     = tid / BLOCK_C_UINT4;
        }
        else
        {
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

    // Pre-compute the input LDS offset (uint4 units) for this thread. For
    // fp16/bf16 each lane reads 1 uint4 = 8 channels; for tf32 each lane reads
    // 2 contiguous uint4 via a single fp32x8_t reinterpret. Both modes index
    // input_lds at this uint4 base.
    const int input_x          = 2 * (lane % 16) + lane / 16;
    const int input_lds_offset = Sw::offset_uint4(input_x, wave_group * GROUP_UINT4);

    // Pre-compute the output LDS swizzle offset (thread-constant). The result
    // matrix row encodes (q_tile, k) via the GT structure.
    const int result_n        = ResultLayout::outer(lane);
    const int result_row      = ResultLayout::inner(lane, 0);
    const int c4_within_group = result_row / 4 % 2;
    bool output_lane_active;
    int output_col;
    if constexpr(cfg.stride == 2)
    {
        // With stride=2, only q=0 outputs are valid (q=1 is discarded).
        output_lane_active = (GT::q(result_row) == 0);
        output_col         = result_n;
    }
    else
    {
        output_lane_active = true;
        output_col         = 2 * result_n + GT::q(result_row);
    }
    // fp16/bf16: 4 packed channels per lane = uint2 unit, c4_in_group selects
    // the 4-channel half. tf32: 4 fp32 per lane = uint4 unit, same c4 index.
    int output_lds_offset;
    if constexpr(Traits::needs_lds_pack)
        output_lds_offset =
            Sw::offset_uint2(output_col, wave_group * GROUP_UINT4 * 2 + c4_within_group);
    else
        output_lds_offset =
            Sw::offset_uint4(output_col, wave_group * GROUP_UINT4 + c4_within_group);

    // Circular buffer of accumulators.
    constexpr auto Zero = fp32x4_t{0.f, 0.f, 0.f, 0.f};
    fp32x4_t acc[cfg.kh];
    for(int i = 0; i < cfg.kh; i++)
        acc[i] = Zero;

    int tic = 1;
    int toc = 0;

    // Flush one accumulator slot to global memory via the LDS staging area.
    //   1. fp16/bf16 path: packed_convert<element_t>(slot) + uint2 store;
    //      tf32 path: raw uint4 store (accumulator is already fp32x4 = uint4).
    //   2. __syncthreads to commit the LDS staging area.
    //   3. Coalesced HBM uint4 store via *load_output_lds.
    //
    // `slot` is the accumulator value to flush (caller takes care of how to
    // obtain it -- direct acc[P_FLUSH] in main/remainder loops, or via
    // dispatch<cfg.kh>(p_idx) in the tail loop). `p_out` is the "stride=1
    // P-coordinate"; the lambda divides by stride internally for stride==2.
    //
    // PRECONDITION: p_out, cfg.stride, ho are workgroup-uniform so the OOB
    // return and embedded __syncthreads stay convergent. The acc slot is NOT
    // cleared here -- callers must run `acc[P_FLUSH] = Zero;` after this call
    // (the clear has to run even when the OOB guard skips the flush so the
    // next ring slot is fresh).
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
            auto* store_output_lds = reinterpret_cast<uint2*>(output_lds);
            if constexpr(cfg.stride == 2)
            {
                if(output_lane_active)
                    store_output_lds[output_lds_offset] = *reinterpret_cast<const uint2*>(&out_reg);
            }
            else
            {
                store_output_lds[output_lds_offset] = *reinterpret_cast<const uint2*>(&out_reg);
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

    // Main loop iterates over input rows.
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

                // Load input operand B and convert to mfma_operand_t (identity for
                // fp16/bf16, fp32xN_to_bf16_pair for tf32). The conversion is the
                // caller-side pre-split done once per Y_LOCAL, so the R-loop reuses
                // the prebuilt operand without recomputing.
                operand_t input_raw = *reinterpret_cast<const operand_t*>(
                    &input_lds[toc * INPUT_LDS_BUFFER_UINT4 + input_lds_offset]);
                mfma_operand_t input_reg = Traits::to_mfma_operand(input_raw);

                // Accumulate: R-loop only (S is embedded in the Toeplitz structure).
                if(row_is_real(y))
                {
                    static_for<cfg.kh>([&]<int R>() {
                        constexpr int p_idx = (Y_LOCAL - R + cfg.kh) % cfg.kh;
                        if constexpr(cfg.direction == Direction::Dgrad)
                            acc[p_idx] = mfma_16x16x32(
                                weights_reg[T * cfg.kh + cfg.kh - 1 - R], input_reg, acc[p_idx]);
                        else
                            acc[p_idx] =
                                mfma_16x16x32(weights_reg[T * cfg.kh + R], input_reg, acc[p_idx]);
                    });
                }

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

                operand_t input_raw = *reinterpret_cast<const operand_t*>(
                    &input_lds[toc * INPUT_LDS_BUFFER_UINT4 + input_lds_offset]);
                mfma_operand_t input_reg = Traits::to_mfma_operand(input_raw);

                if(row_is_real(y))
                {
                    static_for<cfg.kh>([&]<int R>() {
                        constexpr int p_idx = (Y_LOCAL - R + cfg.kh) % cfg.kh;
                        if constexpr(cfg.direction == Direction::Dgrad)
                            acc[p_idx] = mfma_16x16x32(
                                weights_reg[T * cfg.kh + cfg.kh - 1 - R], input_reg, acc[p_idx]);
                        else
                            acc[p_idx] =
                                mfma_16x16x32(weights_reg[T * cfg.kh + R], input_reg, acc[p_idx]);
                    });
                }

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

template <Config cfg, DataType DT>
// A conv3d tile holds kd taps of weights; let small workgroups claim the registers.
__global__
__launch_bounds__(cfg.kd == 1 ? 1024 : cfg.block_size()) void conv2d_grouped_8c_nhwc_cdna4(
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
    if(__builtin_amdgcn_is_invocable(__builtin_amdgcn_mfma_f32_16x16x32_f16) &&
       __builtin_amdgcn_is_invocable(__builtin_amdgcn_mfma_f32_16x16x32_bf16) &&
       __builtin_amdgcn_is_invocable(__builtin_amdgcn_raw_ptr_buffer_load_lds))
    {
        conv2d_grouped_8c_cdna4_nhwc_impl<cfg, DT>(in,
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
        conv2d_grouped_8c_nhwc_cdna4<cfg, DT>
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
        typed_launch.template operator()<DataType::tf32>();
    else
        typed_launch.template operator()<DataType::fp16>();
}

class Grouped_8C_ConvKernel : public GroupedConvKernel
{
public:
    constexpr Grouped_8C_ConvKernel(const Config& cfg, LaunchFn launch_fn)
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

        // Dgrad carries the forward stride as the dY upsample factor (cfg.dilation).
        const int fwd_stride = (cfg_.direction == Direction::Dgrad) ? cfg_.dilation : cfg_.stride;
        if(par.stride_h != fwd_stride || par.stride_w != fwd_stride)
            return false;
        if(par.dilation_h != 1 || par.dilation_w != 1)
            return false;

        // Require that the number of groups per work-group evenly divides the total
        // number of groups.
        if((par.groups % cfg_.waves_per_wg) != 0)
            return false;

        // TF32 doubles the LDS footprint per (x, c) cell. The waves_per_wg=16
        // configs would exceed LDS capacity for tf32; gate them off for that dtype.
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
    int group_channels() const override { return 8; }

private:
    const Config& cfg_;
};


} // namespace hipconv::cdna4::grouped_8c
