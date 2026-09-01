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
namespace grouped_4c
{

using namespace hipconv;
using namespace hipconv::cdna4;

// ----------------------------------------------------------------------------
// Per-dtype traits: type aliases for the operand vector, the value fed to
// mfma_4x4x4_16b, the LDS access unit, and the swizzle layout, plus the
// to_mfma_operand / zero_operand helpers that absorb the fp32 -> bf16-pair
// split for tf32 while remaining identity for fp16/bf16. Everything is
// __forceinline__ so the SASS for fp16/bf16 stays equivalent to the original
// hand-written `*(const operand_t*)lds_ptr` reads.
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
        return mfma_operand_t{0, 0, 0, 0};
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
        return fp32x4_to_bf16_pair(raw);
    }
    static __device__ __forceinline__ mfma_operand_t zero_operand()
    {
        return fp32x4_to_bf16_pair(fp32x4_t{0.f, 0.f, 0.f, 0.f});
    }
};

// 64 threads per wave.
constexpr int WAVE_SIZE = 64;

// Block output is 16 columns wide.
constexpr int WARP_Q = 4;

// 4 channels per group.
constexpr int GROUP_SIZE = 4;

// Kernel configuration parameters.

struct Config
{
    int waves_c64;

    int waves_q4;

    int kh = 3;
    int kw = 3;

    int n_fold = 8;

    int stride = 1;

    int dilation = 1;

    Direction direction = Direction::Fprop;

    constexpr int num_waves() const { return waves_c64 * waves_q4; }

    constexpr int block_c() const { return waves_c64 * 64; }

    constexpr int block_q() const { return waves_q4 * 4; }

    constexpr int block_groups() const { return waves_c64 * 16; }

    constexpr int block_size() const { return num_waves() * WAVE_SIZE; }
};

// All instantiated configurations. The first valid config is expected to be the fastest.
// Order is shared across fp16 / bf16 / tf32; is_valid_config picks the entry whose
// stride / dilation / direction match par.
constexpr Config configs[] = {
    {.waves_c64 = 2, .waves_q4 = 8, .direction = Direction::Dgrad},
    {.waves_c64 = 2, .waves_q4 = 4, .direction = Direction::Dgrad},
    {.waves_c64 = 2, .waves_q4 = 2, .direction = Direction::Dgrad},
    {.waves_c64 = 2, .waves_q4 = 1, .direction = Direction::Dgrad},
    {.waves_c64 = 1, .waves_q4 = 1, .direction = Direction::Dgrad},
    {.waves_c64 = 2, .waves_q4 = 8, .dilation = 2, .direction = Direction::Dgrad},
    {.waves_c64 = 2, .waves_q4 = 4, .dilation = 2, .direction = Direction::Dgrad},
    {.waves_c64 = 2, .waves_q4 = 2, .dilation = 2, .direction = Direction::Dgrad},
    {.waves_c64 = 2, .waves_q4 = 1, .dilation = 2, .direction = Direction::Dgrad},
    {.waves_c64 = 1, .waves_q4 = 1, .dilation = 2, .direction = Direction::Dgrad},
    {.waves_c64 = 2, .waves_q4 = 8},
    {.waves_c64 = 2, .waves_q4 = 4},
    {.waves_c64 = 2, .waves_q4 = 2},
    {.waves_c64 = 2, .waves_q4 = 1},
    {.waves_c64 = 1, .waves_q4 = 1},
    {.waves_c64 = 2, .waves_q4 = 8, .stride = 2},
    {.waves_c64 = 2, .waves_q4 = 4, .stride = 2},
    {.waves_c64 = 2, .waves_q4 = 2, .stride = 2},
    {.waves_c64 = 2, .waves_q4 = 1, .stride = 2},
    {.waves_c64 = 1, .waves_q4 = 1, .stride = 2},
};

constexpr int NUM_CONFIGS = sizeof(configs) / sizeof(configs[0]);

template <Config cfg, DataType DT>
__device__ void conv2d_grouped_4c_cdna4_nhwc_impl(const ToType<DT>* __restrict__ in,
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
                                                  int px)
{
    using Traits         = GroupedDataTraits<DT>;
    using element_t      = typename Traits::element_t;
    using operand_t      = typename Traits::operand_t;
    using mfma_operand_t = typename Traits::mfma_operand_t;
    using lds_unit_t     = typename Traits::lds_unit_t;
    using Sw             = typename Traits::template Sw<cfg.block_c()>;

    // We will use the 4x4x4 matrix, batch size 16 matrix multiply instruction.
    constexpr int MFMA_M     = 4;
    constexpr int MFMA_K     = 4;
    constexpr int MFMA_N     = 4;
    constexpr int MFMA_BATCH = 16;
    using OperandLayout      = MatrixLayout<MFMA_M, MFMA_K, MFMA_BATCH, element_t>;
    using ResultLayout       = MatrixLayout<MFMA_M, MFMA_N, MFMA_BATCH, float>;

    constexpr int GROUP_SIZE_4 = GROUP_SIZE / 4; // 1

    // Number of input columns loaded by each workgroup (output columns plus halo).
    // For dilation > 1, the dilated input is wider but we load only the real (non-zero)
    // compact columns.
    constexpr int BLOCK_W = divup(cfg.block_q() + (cfg.kw - 1), cfg.dilation);

    // Channels carried per uint4 HBM transfer: 8 for fp16/bf16, 4 for tf32.
    constexpr int CH_PER_UINT4 = Traits::CH_PER_UINT4;
    // uint4 vectors per channels fiber.
    constexpr int BLOCK_C_UINT4 = cfg.block_c() / CH_PER_UINT4;

    // Output block width after subsampling for stride > 1.
    constexpr int OUTPUT_BLOCK_Q = cfg.block_q() / cfg.stride;

    // Vectors to store (output uint4 elements per workgroup).
    constexpr int STORE_VECS = OUTPUT_BLOCK_Q * BLOCK_C_UINT4;

    constexpr int NUM_INPUT_LDS_BUFFERS = 2;

    // Size of one input LDS buffer in uint4 units.
    constexpr int INPUT_LDS_BUFFER_UINT4 = BLOCK_C_UINT4 * BLOCK_W;

    // Same buffer size expressed in lds_unit_t units (uint2 for fp16/bf16,
    // uint4 for tf32). The main-loop read pointer indexes in this unit; the
    // pre-computed input_lds_offsets[] are also in this unit. The factor
    // LDS_UNIT_PER_UINT4 = sizeof(uint4) / sizeof(lds_unit_t) is 2 for the
    // fp16/bf16 path and 1 for tf32.
    constexpr int LDS_UNIT_PER_UINT4        = (int)(sizeof(uint4) / sizeof(lds_unit_t));
    constexpr int INPUT_LDS_BUFFER_LDS_UNIT = INPUT_LDS_BUFFER_UINT4 * LDS_UNIT_PER_UINT4;

    // Multi-pass HBM->LDS loading. Each thread issues one uint4 per pass.
    // fp16/bf16 always degenerates to LOAD_PASSES == 1 (block_size covers the
    // whole row); tf32 generally needs 2 passes because every uint4 holds only
    // 4 channels. With LOAD_PASSES being a constexpr loop bound the plain
    // for body is fully unrolled, matching the original inline single-pass
    // loader for fp16/bf16.
    constexpr int LOAD_PASSES = divup(INPUT_LDS_BUFFER_UINT4, cfg.block_size());

    // Size of LDS buffer for outputs (in uint4 units).
    constexpr int OUTPUT_LDS_BUFFER_UINT4 = BLOCK_C_UINT4 * OUTPUT_BLOCK_Q;

    // Size of LDS buffer for weight staging in uint4 units. The total weight
    // bytes are kh*kw * block_groups * GROUP_SIZE filters * GROUP_SIZE
    // channels * sizeof(element_t); divide by sizeof(uint4) = 16 to get uint4
    // count.
    constexpr int WEIGHT_LDS_UINT4 = cfg.kh * cfg.kw * cfg.block_groups() * GROUP_SIZE *
                                     GROUP_SIZE * (int)sizeof(element_t) / (int)sizeof(uint4);

    // LDS buffer for staging loads from global memory.
    __shared__ uint4 input_lds[NUM_INPUT_LDS_BUFFERS * INPUT_LDS_BUFFER_UINT4];

    // LDS buffer for weight staging (prologue) and output staging (main loop).
    __shared__ uint4 output_lds[maximum(WEIGHT_LDS_UINT4, OUTPUT_LDS_BUFFER_UINT4)];

    const int tid  = threadIdx.x;
    const int wave = tid / WAVE_SIZE;
    const int lane = tid % WAVE_SIZE;

    // Workgroup coordinates.
    const int block_q_n_idx   = blockIdx.x;
    const int block_n_mod_idx = block_q_n_idx % cfg.n_fold;
    const int block_q_idx     = block_q_n_idx / cfg.n_fold;
    const int block_group_idx = blockIdx.y;
    const int block_n_div_idx = blockIdx.z;
    const int block_n_idx     = block_n_div_idx * cfg.n_fold + block_n_mod_idx;
    if(block_n_idx >= N)
        return;

    // real input index on Global Memory
    const int block_n       = block_n_idx;
    const int block_q       = block_q_idx * cfg.block_q();
    const int block_group   = block_group_idx * cfg.block_groups();
    const int block_k       = block_group * GROUP_SIZE;
    const int block_c_uint4 = block_k / CH_PER_UINT4;

    // Base pointer for this batch image in NHWC layout (all in uint4 units).
    const int C       = groups * GROUP_SIZE;
    const int C_UINT4 = C / CH_PER_UINT4;

    const size_t offset_block = (size_t)block_n * hi * wi * C_UINT4;
    const size_t wi_stride    = (size_t)wi * C_UINT4;
    const size_t wo_stride    = (size_t)wo * C_UINT4;

    // Effective (dilated) input height: for dilation > 1, the kernel iterates over
    // hi_eff virtual rows but only loads real data for rows divisible by dilation.
    const int hi_eff = (cfg.dilation > 1) ? (hi - 1) * cfg.dilation + 1 : hi;

    // Create the input buffer resource.
    auto input_bytes               = static_cast<size_t>(N) * hi * wi * C * sizeof(element_t);
    constexpr int rsrc_data_format = 1 << 15;
    auto input_rsrc                = __builtin_amdgcn_make_buffer_rsrc(
        const_cast<element_t*>(in), 0, input_bytes, rsrc_data_format);

    // For dilation > 1, compute the first compact column that maps to LDS col 0.
    int compact_base = 0;
    if constexpr(cfg.dilation > 1)
    {
        int ds = block_q - px; // dilated start position
        // ceil(ds / dilation): positive branch uses standard ceiling formula,
        // negative branch relies on C++ truncation-toward-zero giving ceiling.
        compact_base = (ds >= 0) ? (ds + cfg.dilation - 1) / cfg.dilation : ds / cfg.dilation;
    }

    // Pre-compute per-pass (col_p, c_uint4_p) and the global HBM voffset for
    // each load pass. Each pass writes at most one uint4 per thread; the OOB
    // sentinel input_bytes makes buffer-load-lds return zero, which is the
    // natural SAME-padding behaviour.
    uint32_t input_voffset_base[LOAD_PASSES];
    bool load_active[LOAD_PASSES];
    int load_lds_slot[LOAD_PASSES];

    for(int p = 0; p < LOAD_PASSES; p++)
    {
        int j            = tid + p * cfg.block_size();
        load_active[p]   = (j < INPUT_LDS_BUFFER_UINT4);
        load_lds_slot[p] = j;
        if(load_active[p])
        {
            int col_p = Sw::x(j);
            // SwizzleT::c8 returns the 8-channel block index; SwizzleT_fp32::c4
            // returns the 4-channel block index. Both correspond to "the
            // channel-block index per uint4" for their dtype.
            int c_uint4_p;
            if constexpr(Traits::needs_lds_pack)
                c_uint4_p = Sw::c8(j);
            else
                c_uint4_p = Sw::c4(j);
            int gcol_p;
            if constexpr(cfg.dilation > 1)
                gcol_p = compact_base + col_p;
            else
                gcol_p = (block_q - px) + col_p;
            if(0 <= gcol_p && gcol_p < wi)
            {
                input_voffset_base[p] = sizeof(uint4) * (offset_block + (size_t)gcol_p * C_UINT4 +
                                                         block_c_uint4 + c_uint4_p);
            }
            else
            {
                input_voffset_base[p] = input_bytes; // OOB -> hardware returns 0
            }
        }
        else
        {
            input_voffset_base[p] = input_bytes;
        }
    }

    //     N        = uint4_off / (hi * wi * C_UINT4)         = block_n
    //     W        = (uint4_off / C_UINT4) mod wi            = gcol_p
    //     C_start  = (uint4_off mod C_UINT4) * CH_PER_UINT4  = (block_c_uint4 + c_uint4_p) *
    //     CH_PER_UINT4

    // Load weights from global memory through LDS into registers.
    // Global layout: wei[k][kh*kw] with GROUP_SIZE element_t per filter
    // position. The total byte count is block_k filters * kh*kw spatial *
    // GROUP_SIZE channels * sizeof(element_t); we load with uint4 strides.
    mfma_operand_t weights_reg[cfg.kh * cfg.kw];
    {
        auto weight_bytes =
            static_cast<size_t>(C) * cfg.kh * cfg.kw * GROUP_SIZE * sizeof(element_t);
        auto weight_rsrc = __builtin_amdgcn_make_buffer_rsrc(
            const_cast<element_t*>(wei), 0, weight_bytes, rsrc_data_format);
        uint32_t voffset_base =
            block_k * cfg.kh * cfg.kw * GROUP_SIZE * (uint32_t)sizeof(element_t);

        for(int j = tid; j < WEIGHT_LDS_UINT4; j += cfg.block_size())
        {
#ifdef __gfx950__
            __builtin_amdgcn_raw_ptr_buffer_load_lds(
                weight_rsrc, &output_lds[j], 16, voffset_base + j * sizeof(uint4), 0, 0, 0);
#endif
        }
    }

    // Load first chunk of inputs asynchronously into buffer 0.
    for(int p = 0; p < LOAD_PASSES; p++)
    {
        if(load_active[p])
        {
#ifdef __gfx950__
            __builtin_amdgcn_raw_ptr_buffer_load_lds(
                input_rsrc, &input_lds[load_lds_slot[p]], 16, input_voffset_base[p], 0, 0, 0);
#endif
        }
    }

    // Each wave computes 16 groups of channels and 4 columns of the output tensor.
    auto wave_c64 = wave % cfg.waves_c64;
    auto wave_q4  = wave / cfg.waves_c64;

    // Map lanes to MFMA matrix coordinates, incorporating the wave's Q offset.
    const int thread_q   = wave_q4 * WARP_Q + ResultLayout::outer(lane);
    const int lane_c4    = ResultLayout::inner(lane) / 4;
    const int lane_batch = ResultLayout::batch(lane);

    // Map threads to output addresses for transferring through LDS to global memory.
    // Each lane writes one uint4 (4 output channels). Sw::offset_uint4 has a
    // consistent meaning across both swizzle classes: its second argument is
    // the channel-block index per uint4.
    const bool store_active      = (tid < STORE_VECS);
    const uint4* load_output_lds = nullptr;
    uint4* store_output_global   = nullptr;
    if(store_active)
    {
        const int c_uint4 = tid % BLOCK_C_UINT4;
        const int col     = tid / BLOCK_C_UINT4;
        if constexpr(cfg.stride == 2)
        {
            const int q_out = (block_q / 2) + col;
            load_output_lds = &output_lds[Sw::offset_uint4(col, c_uint4)];
            if(q_out < wo)
            {
                const int K_UINT4   = C / CH_PER_UINT4;
                store_output_global = reinterpret_cast<uint4*>(out) +
                                      (size_t)block_n * ho * wo * K_UINT4 +
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
                                      (size_t)block_n * ho * wo * K_UINT4 + (size_t)q * K_UINT4 +
                                      block_c_uint4 + c_uint4;
            }
        }
    }

    {
        // Wait for all buffer_load_lds operations (weights + first input chunk)
        // to complete before reading from LDS.
        wait_vmcnt<0>();
        __syncthreads();

        if constexpr(cfg.direction == Direction::Dgrad)
        {
            if constexpr(Traits::dgrad_use_tr16)
            {
                // fp16/bf16 path: ds_read_tr16_b64 performs the (k <-> c)
                // transpose in hardware. Each lane reads one int16x4 per
                // filter position, bit-casts to operand_t and folds through
                // Traits::to_mfma_operand (identity for fp16/bf16, so this is
                // a zero-cost wrapper).
                using TransposeLayout = TransposeLDSLayout<4, 4, 16>;
                const int tr_batch    = TransposeLayout::batch(lane);
                const int tr_row      = TransposeLayout::row(lane);
                int filter_local      = wave_c64 * 64 + tr_batch * GROUP_SIZE + tr_row;
                auto* weight_lds      = reinterpret_cast<element_t*>(output_lds);
                const int khw_stride  = GROUP_SIZE; // 4 element_t per filter position

                for(int khw = 0; khw < cfg.kh * cfg.kw; khw++)
                {
                    auto* addr = reinterpret_cast<int16x4_t*>(
                        &weight_lds[filter_local * cfg.kh * cfg.kw * khw_stride +
                                    khw * khw_stride]);
                    int16x4_t r      = __builtin_amdgcn_ds_read_tr16_b64_v4i16(addr);
                    weights_reg[khw] = Traits::to_mfma_operand(__builtin_bit_cast(operand_t, r));
                }
            }
            else
            {
                // tf32 path: 32-bit fp32 has no ds_read_tr* builtin, so we do a
                // software (k <-> c) transpose with 4 scalar fp32 reads per
                // filter position (4 reads * 9 khw = 36 ds_read_b32 per lane,
                // all amortised over the full main loop). The 180-degree
                // filter rotation required by dgrad is applied at the MFMA
                // call site below, so weights_reg is loaded in natural
                // (r = khw/kw, s = khw%kw) order to share the layout with
                // fprop.
                const int lane_c_target = OperandLayout::outer(lane);
                const int lane_b        = OperandLayout::batch(lane);
                const int k_base_local  = wave_c64 * 64 + lane_b * GROUP_SIZE;
                auto* weight_lds_f      = reinterpret_cast<const float*>(output_lds);

                for(int khw = 0; khw < cfg.kh * cfg.kw; khw++)
                {
                    fp32x4_t v;
                    for(int kk = 0; kk < 4; kk++)
                    {
                        v[kk] = weight_lds_f[((k_base_local + kk) * cfg.kh * cfg.kw + khw) *
                                                 GROUP_SIZE +
                                             lane_c_target];
                    }
                    weights_reg[khw] = Traits::to_mfma_operand(v);
                }
            }
        }
        else
        {
            // Fprop: linear weight read in MFMA operand layout.
            auto lane_k      = OperandLayout::outer(lane);
            auto lane_b      = OperandLayout::batch(lane);
            int filter_local = wave_c64 * 64 + lane_b * GROUP_SIZE + lane_k;
            auto* weight_lds = reinterpret_cast<const lds_unit_t*>(output_lds);
            for(int khw = 0; khw < cfg.kh * cfg.kw; khw++)
            {
                auto* lds_ptr    = &weight_lds[filter_local * cfg.kh * cfg.kw + khw];
                weights_reg[khw] = Traits::to_mfma_operand(*(const operand_t*)lds_ptr);
            }
        }
    }

    // Pre-compute the kw input LDS offsets (in lds_unit_t units) for this thread.
    // lane_q, wave_group, and lane_c4 are constant per thread; hoisting avoids
    // recomputing the swizzle (which contains %) inside the hot loop.
    //
    // SwizzleT::offset_uint2 returns a uint2 offset and SwizzleT_fp32::offset_uint4
    // returns a uint4 offset; the two are the natural read unit of their dtype
    // (8 vs 16 bytes per lane). A single `if constexpr` lambda picks the right
    // function once per thread.
    auto input_lds_offset_fn = [](int x, int c4) constexpr -> int {
        if constexpr(Traits::needs_lds_pack)
            return Sw::offset_uint2(x, c4);
        else
            return Sw::offset_uint4(x, c4);
    };

    int input_lds_offsets[cfg.kw];
    bool input_lds_mask[cfg.kw];
    static_for<cfg.kw>([&]<int S>() {
        if constexpr(cfg.dilation > 1)
        {
            int dilated_global = (block_q - px) + thread_q + S;
            bool is_real       = (dilated_global % cfg.dilation == 0);
            int lds_col        = is_real ? (dilated_global / cfg.dilation - compact_base) : 0;
            input_lds_offsets[S] =
                input_lds_offset_fn(lds_col, wave_c64 * 16 + lane_batch * GROUP_SIZE_4 + lane_c4);
            input_lds_mask[S] = is_real;
        }
        else
        {
            //         offset = SwizzleT<64>::offset_uint2(lane%4 + S, lane/4)
            //               = (lane%4+S)·16 + ((lane%4+S+(lane/4)/2)%8)·2 + (lane/ 4)%2
            input_lds_offsets[S] = input_lds_offset_fn(
                thread_q + S, wave_c64 * 16 + lane_batch * GROUP_SIZE_4 + lane_c4);
            input_lds_mask[S] = true;
        }
    });

    // Pre-compute the output LDS swizzle offset (thread-constant).
    //       baseline (stride=1): output_lds_offset = SwizzleT::offset_uint2(thread_q,
    int output_lds_offset;
    bool output_lane_active;
    if constexpr(cfg.stride == 2)
    {
        output_lane_active = (thread_q % 2 == 0);
        output_lds_offset =
            input_lds_offset_fn(thread_q / 2, wave_c64 * 16 + lane_batch * GROUP_SIZE_4 + lane_c4);
    }
    else
    {
        output_lane_active = true;
        output_lds_offset =
            input_lds_offset_fn(thread_q, wave_c64 * 16 + lane_batch * GROUP_SIZE_4 + lane_c4);
    }

    // Circular buffer of accumulators.
    constexpr auto Zero = fp32x4_t{0.f, 0.f, 0.f, 0.f};
    fp32x4_t acc[cfg.kh];
    for(int i = 0; i < cfg.kh; i++)
        acc[i] = Zero;

    int tic = 1;
    int toc = 0;

    // Flush one accumulator slot to global memory via the LDS staging area.
    //   1. Convert fp32x4 -> narrow element_t and write to output_lds
    //      (fp16/bf16 path); or store as raw uint4 (tf32 path).
    //   2. __syncthreads to commit the LDS staging area.
    //   3. Coalesced HBM store via *load_output_lds.
    //
    // `slot` is the accumulator value to flush (caller takes care of how
    // to obtain it -- direct acc[P_FLUSH] in main/remainder loops, or via
    // dispatch<cfg.kh>(p_idx) in the tail loop where the slot index is
    // runtime). `p_out` is the "stride=1 P-coordinate"; the lambda divides
    // by stride internally for stride==2.
    //
    // PRECONDITION: p_out, cfg.stride, ho are workgroup-uniform, so the
    // OOB return and the embedded __syncthreads stay convergent across the
    // workgroup. All current callers satisfy this (p_out is derived from
    // y_base/y_rem_base + Y_LOCAL + py - kh + 1, all uniform).
    //
    // NOTE: The acc slot is NOT cleared here -- main/remainder callers must
    // do `acc[P_FLUSH] = Zero;` after this call (the clear has to run even
    // when the OOB guard skips the flush so the next ring slot is fresh).
    // The tail caller pre-clears inside its dispatch lambda.
    auto flush_p = [&](fp32x4_t slot, int p_out) {
        // Stride-dependent OOB guard. Each instantiation keeps only one
        // branch after if-constexpr fold.
        if constexpr(cfg.stride == 2)
        {
            if(!(p_out >= 0 && (p_out % 2) == 0 && (p_out / 2) < ho))
                return;
        }
        else
        {
            if(!(p_out >= 0 && p_out < ho))
                return;
        }

        // Stage to LDS. fp16/bf16 packs fp32x4 -> narrow element_t; tf32's
        // accumulator is already uint4-sized so it stores directly.
        if constexpr(Traits::needs_lds_pack)
        {
            auto out_reg           = packed_convert<element_t>(slot);
            auto* store_output_lds = reinterpret_cast<lds_unit_t*>(output_lds);
            if constexpr(cfg.stride == 2)
            {
                // Only even-thread_q lanes carry valid data for stride=2.
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

        // Coalesced HBM write. wo_row maps stride=1 p_out back to the real
        // HO coordinate (divide by 2 for stride=2).
        if(store_output_global)
        {
            const int wo_row                        = (cfg.stride == 2) ? (p_out / 2) : p_out;
            store_output_global[wo_row * wo_stride] = *load_output_lds;
        }
    };

    // Issue a multi-pass async input load for dilated row index y_local.
    // For dilation>1 the kernel iterates over hi_eff virtual rows; only rows
    // where y_local % dilation == 0 correspond to real data, and the HBM
    // offset is (y_local / dilation) * wi_stride. For dilation=1 this
    // collapses to the trivial "load row y_local" path. fp16/bf16 has
    // LOAD_PASSES == 1 so the plain for unrolls to a single buffer_load_lds.
    auto issue_input_load = [&](int y_local) {
        if(y_local < 0 || y_local >= hi_eff)
            return;
        if constexpr(cfg.dilation > 1)
        {
            if((y_local % cfg.dilation) != 0)
                return;
        }
        const int y_global    = (cfg.dilation > 1) ? (y_local / cfg.dilation) : y_local;
        const auto row_offset = (size_t)y_global * wi_stride * sizeof(uint4);
        for(int p = 0; p < LOAD_PASSES; p++)
        {
            if(load_active[p])
            {
                __builtin_amdgcn_raw_ptr_buffer_load_lds(
                    input_rsrc,
                    &input_lds[tic * INPUT_LDS_BUFFER_UINT4 + load_lds_slot[p]],
                    16,
                    input_voffset_base[p] + row_offset,
                    0,
                    0,
                    0);
            }
        }
    };

    // Main loop iterates over input rows (dilated rows for dilation > 1).
    for(int y_base = 0; y_base + cfg.kh <= hi_eff; y_base += cfg.kh)
    {
        // Inner loop unrolls the main loop by a factor of kh, the number of filter rows.
        // static_for makes the loop indices compile-time constants.

        static_for<cfg.kh>([&]<int Y_LOCAL>() {
            wait_vmcnt<0>();
            __syncthreads();

            int y = y_base + Y_LOCAL;
            if(y + 1 < hi_eff)
                issue_input_load(y + 1);

            static_for<cfg.kw>([&]<int S>() {
                const lds_unit_t* lds_ptr = reinterpret_cast<const lds_unit_t*>(input_lds) +
                                            toc * INPUT_LDS_BUFFER_LDS_UNIT + input_lds_offsets[S];
                auto input_reg = Traits::to_mfma_operand(*(const operand_t*)lds_ptr);

                if constexpr(cfg.dilation > 1)
                {
                    bool row_is_real = (y % cfg.dilation == 0);
                    if(!row_is_real || !input_lds_mask[S])
                        input_reg = Traits::zero_operand();
                }

                static_for<cfg.kh>([&]<int R>() {
                    // Compute the position of the output row in the circular buffer of
                    // accumulators. This is a constexpr because of the Y_LOCAL and R
                    // are compile-time constants. Therefore acc[p_idx] compiles to a
                    // register address.
                    constexpr int p_idx = (Y_LOCAL - R + cfg.kh) % cfg.kh;
                    if constexpr(cfg.direction == Direction::Dgrad)
                        acc[p_idx] = mfma_4x4x4_16b(
                            weights_reg[(cfg.kh - 1 - R) * cfg.kw + (cfg.kw - 1 - S)],
                            input_reg,
                            acc[p_idx]);
                    else
                        acc[p_idx] =
                            mfma_4x4x4_16b(weights_reg[R * cfg.kw + S], input_reg, acc[p_idx]);
                });
            });

            tic ^= 1;
            toc ^= 1;

            constexpr int P_FLUSH = (Y_LOCAL + 1) % cfg.kh;
            int p_out             = y + py - (cfg.kh - 1);
            flush_p(acc[P_FLUSH], p_out);
            acc[P_FLUSH] = Zero;
        });
    }

    // Remainder: the hi_eff % kh leftover rows share the same structure as the main
    // loop — y_base is a multiple of kh so Y_LOCAL = y % kh, giving constexpr P.
    {
        int y_rem_base = (hi_eff / cfg.kh) * cfg.kh;
        static_for<cfg.kh>([&]<int Y_LOCAL>() {
            if(Y_LOCAL >= hi_eff % cfg.kh)
                return;
            int y = y_rem_base + Y_LOCAL;

            wait_vmcnt<0>();
            __syncthreads();

            if(y + 1 < hi_eff)
                issue_input_load(y + 1);

            static_for<cfg.kw>([&]<int S>() {
                const lds_unit_t* lds_ptr = reinterpret_cast<const lds_unit_t*>(input_lds) +
                                            toc * INPUT_LDS_BUFFER_LDS_UNIT + input_lds_offsets[S];
                auto input_reg = Traits::to_mfma_operand(*(const operand_t*)lds_ptr);

                if constexpr(cfg.dilation > 1)
                {
                    bool row_is_real = (y % cfg.dilation == 0);
                    if(!row_is_real || !input_lds_mask[S])
                        input_reg = Traits::zero_operand();
                }

                static_for<cfg.kh>([&]<int R>() {
                    constexpr int p_idx = (Y_LOCAL - R + cfg.kh) % cfg.kh;
                    if constexpr(cfg.direction == Direction::Dgrad)
                        acc[p_idx] = mfma_4x4x4_16b(
                            weights_reg[(cfg.kh - 1 - R) * cfg.kw + (cfg.kw - 1 - S)],
                            input_reg,
                            acc[p_idx]);
                    else
                        acc[p_idx] =
                            mfma_4x4x4_16b(weights_reg[R * cfg.kw + S], input_reg, acc[p_idx]);
                });
            });

            tic ^= 1;
            toc ^= 1;

            constexpr int P_FLUSH = (Y_LOCAL + 1) % cfg.kh;
            int p_out             = y + py - (cfg.kh - 1);
            flush_p(acc[P_FLUSH], p_out);
            acc[P_FLUSH] = Zero;
        });
    }

    // Flush output rows whose last input contribution (at filter row kh-1) would land at
    // y >= hi_eff (out of bounds). These rows were never flushed by the main/remainder loops.
    // An output row p_out is last touched at input row y = p_out + kh-1 - py; it needs
    // flushing when that y >= hi_eff, i.e. p_out >= hi_eff - kh + 1 + py.
    //
    // For stride=2 the loop iterates over stride-1 output rows and subsamples.
    {
        const int ho_stride1 = hi_eff + 2 * py - (cfg.kh - 1);
        const int tail_end   = (cfg.stride == 2) ? ho_stride1 : ho;
        for(int p_out = hi_eff - cfg.kh + 1 + py; p_out < tail_end; p_out++)
        {
            if constexpr(cfg.stride == 2)
            {
                if(p_out % 2 != 0 || p_out / 2 >= ho)
                    continue;
            }

            __syncthreads(); // separate prior LDS reads from this iteration's writes
            int p_idx = (p_out - py + cfg.kh) % cfg.kh;
            fp32x4_t slot;
            // Slot index is runtime here, so we use dispatch<cfg.kh> to recover
            // a constexpr P for the acc[] access and pre-clear in one step.
            dispatch<cfg.kh>(p_idx, [&]<int P>() {
                slot   = acc[P];
                acc[P] = Zero;
            });
            flush_p(slot, p_out);
        }
    }
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
template <Config cfg, DataType DT>
__global__ void conv2d_grouped_4c_nhwc_cdna4(const ToType<DT>* __restrict__ in,
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
                                             int px)
{
    if(__builtin_amdgcn_is_invocable(__builtin_amdgcn_mfma_f32_4x4x4f16) &&
       __builtin_amdgcn_is_invocable(__builtin_amdgcn_mfma_f32_4x4x4bf16_1k) &&
       __builtin_amdgcn_is_invocable(__builtin_amdgcn_raw_ptr_buffer_load_lds) &&
       __builtin_amdgcn_is_invocable(__builtin_amdgcn_ds_read_tr16_b64_v4i16))
    {
        conv2d_grouped_4c_cdna4_nhwc_impl<cfg, DT>(in,
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
    auto typed_launch = [&]<DataType DT>() {
        using dtype = ToType<DT>;
        auto view   = SizeView<cfg.direction>(par);
        conv2d_grouped_4c_nhwc_cdna4<cfg, DT>
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
                view.pad_w());
    };
    if(par.input_type == DataType::bf16)
        typed_launch.template operator()<DataType::bf16>();
    else if(par.input_type == DataType::tf32)
        typed_launch.template operator()<DataType::tf32>();
    else
        typed_launch.template operator()<DataType::fp16>();
}

class Grouped_4C_ConvKernel : public GroupedConvKernel
{
public:
    constexpr Grouped_4C_ConvKernel(const Config& cfg, LaunchFn launch_fn)
        : GroupedConvKernel(launch_fn)
        , cfg_(cfg)
    {
    }

    bool is_valid_config(const Conv2dParams& par) const override
    {
        if(par.direction != cfg_.direction)
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

        if((par.groups % cfg_.block_groups()) != 0)
            return false;

        const int out_q          = (par.direction == Direction::Dgrad) ? par.w : par.q;
        const int output_block_q = cfg_.block_q() / cfg_.stride;
        if(out_q < output_block_q && cfg_.waves_q4 > 1)
            return false;

        return true;
    }

    LaunchParams get_launch_params(const Conv2dParams& par) const override
    {
        // Compute the grid size.
        // For Dgrad the output is the input gradient (width = par.w, not par.q).
        const int out_q          = (cfg_.direction == Direction::Dgrad) ? par.w : par.q;
        const int output_block_q = cfg_.block_q() / cfg_.stride;
        auto blocks_w            = divup(out_q, output_block_q);
        auto blocks_w_n          = blocks_w * cfg_.n_fold;
        auto blocks_c            = divup(par.c, cfg_.block_c());
        auto blocks_n_fold       = divup(par.n, cfg_.n_fold);

        LaunchParams launch;
        launch.grid       = dim3(blocks_w_n, blocks_c, blocks_n_fold);
        launch.block_size = dim3(cfg_.block_size(), 1, 1);
        return launch;
    }

protected:
    int group_channels() const override { return 4; }

private:
    const Config& cfg_;
};

HIPCONV_DEFINE_KERNEL_TABLE(Grouped_4C_ConvKernel);

} // namespace grouped_4c
} // namespace hipconv::cdna4

// Host-only: the device linker must not see this struct or its function pointers.
#ifndef __HIP_DEVICE_COMPILE__
HIPCONV_EXPORT_KERNEL_TABLE(grouped_4c_cdna4_kernels, hipconv::cdna4::grouped_4c);
#endif
