#pragma once
// Weights-gradient (wgrad) for depthwise convolution, on batched 4x4x4 MFMA.
//
// docs/algorithms/toeplitz/depthwise-wgrad-hankel-cdna4.md is this kernel: the
// channel-as-MFMA-batch tile, the stride-2 phase split, and the two-step row staging.
// docs/algorithms/toeplitz/toeplitz-wgrad.md is the Hankel construction underneath it.
//
// Kernel body and per-config launch_impl<cfg> live here; the host-safe Config type
// and configs[] table live in config_table.h, which hipconv_autoshard reads to
// generate the launch_impl<> instantiations and the kernel-span export across
// depthwise_wgrad_hankel_shard*.cpp.
//
// dW[c][r][s] = sum_{n,p,q} delta[n][p][q][c] * input[n][p*S + r - py][q*S + s - px][c]
//
// One matrix product per input row y and channel c, summed over y, over q tiles
// and over the batch, with S the stride, q0 the tile's first output column and k
// the column within it:
//
//   A[s][k] = input[y][S*(q0 + k) + s - px] (Hankel: shift on the row index)
//   B[k][r] = delta[(y + py - r)/S][q0 + k] (kh delta rows as the columns)
//   C[s][r] = sum_k A[s][k] B[k][r]         = dW[c][r][s] restricted to q0..q0+K

#include <hip/hip_runtime.h>

#include "../../bunnies_cdna4.hpp"
#include "../../grouped/depth_taps.h"
#include "../../persistent_grid.h"
#include "config_desc.h"
#include "config_table.h"
#include "conv_kernel.h"
#include "depthwise_conv_kernel.h"
#include "detail.h"
#include "hip_util.h"
#include "launch_params.h"
#include "mathutil.h"
#include "matrix_layout.h"
#include "memory.h"
#include "packed_ops.h"
#include "swizzle.h"
#include "transpose_lds_layout.h"
#include "types.h"

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace hipconv::cdna4
{
namespace depthwise_wgrad_hankel
{

using namespace hipconv;

using arch = bunnies::arch_cdna4;

// Workgroups contributing to the same tap: the q tiles times the batch-by-height chunks. The
// x dimension owns disjoint channels, so a partition needs a slot rather than an atomic.
inline size_t partition_count(const dim3& grid)
{
    return (size_t)grid.y * grid.z;
}

// Channel blocks, q tiles, then batch by row chunks. This is the work, not the launch:
// the selection gates below read it to see how much parallelism a config exposes.
//
// A conv3d's delta planes are images, and z repeats them once per depth tap, tap-major, so a
// workgroup's units all share one tap and its partial is that tap's dW slice.
inline dim3 grid_for(const Config& cfg, const ConvParams& par)
{
    const DepthParams dp = DepthParams::of(par);
    return dim3(divup(par.c, BLOCK_C),
                divup(par.q, cfg.block_q()),
                dp.kd * par.n * dp.d_out * divup(par.h, cfg.rows_per_chunk));
}

inline int workgroups(const Config& cfg, const ConvParams& par)
{
    const dim3 g = grid_for(cfg, par);
    return (int)(g.x * g.y * g.z);
}

// Workgroups the fold leaves standing. Not the 2 per CU the selection gate calls "fills the
// machine": LDS keeps 5 of these resident per CU, and folding to the gate's floor costs the
// residency rather than the partials it saves -- measured -37% on n256 h32 w32 c128.
constexpr int FOLD_MIN_WORKGROUPS_PER_CU = 8;

// The grid actually launched: z folds to FOLD_MIN_WORKGROUPS_PER_CU per CU and the kernel
// strides the rest, which keeps the reduce's one-slot-per-z traffic off the batch size.
// Deliberately not what the gates read -- fold first and every config looks big enough.
inline dim3 launch_grid_for(const Config& cfg, const ConvParams& par, int cu_count)
{
    dim3 grid        = grid_for(cfg, par);
    const int kd     = DepthParams::of(par).kd;
    const int planes = (int)(grid.x * grid.y) * kd;
    const int target = maximum(1, FOLD_MIN_WORKGROUPS_PER_CU * cu_count / maximum(1, planes));
    grid.z           = (unsigned)(minimum((int)grid.z / kd, target) * kd);
    return grid;
}

// Whether a shape can use a config at all, ignoring how well it fills the machine.
inline bool tile_fits(const Config& cfg, const ConvParams& par)
{
    if(par.direction != cfg.direction)
        return false;
    if(compiled_filter(par.kh, par.kw) != FilterShape{cfg.kh, cfg.kw})
        return false;
    // One stride for both axes: the operands are decimated by a single constant.
    if(par.stride_h != par.stride_w || par.stride_h != cfg.stride)
        return false;
    // The two tables are partitioned by the operand's type, not its width: sizeof is 4 for fp32
    // as well as tf32.
    if((par.input_type == DataType::tf32) != cfg.is_tf32())
        return false;
    // Channel alignment, for either staging path: a lane's access spans chan_vec channels
    // of one pixel, so it stays inside the pixel exactly when the count divides C. The
    // wide path's uint4 is the chan_vec == 8 case of the same test.
    if(par.c % cfg.chan_vec != 0)
        return false;
    // A q tile wider than the image only idles lanes; a row chunk taller than the
    // image only costs workgroups.
    if(cfg.q_tiles > 4 && cfg.block_q() >= 2 * par.q)
        return false;
    if(cfg.rows_per_chunk > 8 && cfg.rows_per_chunk >= 2 * par.h)
        return false;
    // The grid's counts are int, and a conv3d multiplies them by its planes and depth taps.
    const DepthParams dp = DepthParams::of(par);
    const int64_t z =
        (int64_t)dp.kd * par.n * std::max(dp.d_in, dp.d_out) * divup(par.h, cfg.rows_per_chunk);
    if((int64_t)divup(par.c, BLOCK_C) * divup(par.q, cfg.block_q()) * z > INT32_MAX)
        return false;
    return true;
}

// A block's logical place in the grid, once the channel pair has been re-keyed.
struct BlockIndex
{
    int c_block, q_tile, z;
};

// When C is not a multiple of BLOCK_C the sibling channel blocks own the two halves of
// one 128 B line, and consecutive blocks go to different XCDs, so each half is fetched
// behind its own L2. Re-key the pair onto one XCD, adjacent in its dispatch order.
__device__ inline BlockIndex xcd_paired_index(int C)
{
    const int gx = static_cast<int>(gridDim.x);
    const int gy = static_cast<int>(gridDim.y);
    const int gz = static_cast<int>(gridDim.z);
    const int x  = static_cast<int>(blockIdx.x);
    const int y  = static_cast<int>(blockIdx.y);
    const int z  = static_cast<int>(blockIdx.z);

    // An aligned C splits no line, so leave those grids in dispatch order. The re-key is
    // only a bijection when the pair's stride divides the extent it is taken out of.
    if(C % BLOCK_C == 0 || gx < 2 || (gy * gz) % persistent::NUM_XCD != 0)
        return {x, y, z};

    constexpr int XCD = persistent::NUM_XCD;
    const int flat    = x + gx * (y + gy * z);
    const int rest    = (flat / (XCD * gx)) * XCD + flat % XCD;
    return {(flat / XCD) % gx, rest % gy, rest / gy};
}

// fp32 roundings on the longest path from a product to one dW element.
//
// The blocking leaves two levels: the mma chain one workgroup runs over all the units folded
// onto it, and the reduce's running sum over the partitions. Counting them the way
// docs/algorithms/details/direct/direct-wgrad-tolerance.md derives puts the depth orders of
// magnitude below the N*P*Q products the default tolerance would charge.
inline size_t accumulation_depth(const Config& cfg, const ConvParams& par, int cu_count)
{
    const dim3 grid    = launch_grid_for(cfg, par, cu_count);
    const size_t parts = partition_count(grid) / DepthParams::of(par).kd;
    const size_t units = divup((int)grid_for(cfg, par).z, (int)grid.z);

    // Every K step is charged a rounding: v_mfma_f32_4x4x4's exact block is not measured, the
    // way the 16x16's is. It is 3 against a chain in the hundreds, so measuring it would not
    // move the bound.
    constexpr size_t within_mfma = MFMA_K - 1;
    // One mma per q tile per row of the chunk, each stepping srcC, for every unit the
    // workgroup folds. Rows whose tap reads the zero slot add nothing and cannot round, so
    // this is an upper bound.
    const size_t chain = (size_t)cfg.q_tiles * std::min(cfg.rows_per_chunk, par.h) * units;
    return within_mfma + chain + (parts - 1);
}

// The largest grid any config that fits this shape can reach.
inline int most_workgroups(const ConvParams& par)
{
    int most = 0;
    for(const Config& cfg : configs)
        if(tile_fits(cfg, par))
            most = std::max(most, workgroups(cfg, par));
    return most;
}

template <Config cfg, DataType DT, bool PARTITIONED, DataType DW, bool CONV3D>
__device__ void conv2d_depthwise_wgrad_hankel_nhwc_impl(const ToType<DT>* __restrict__ input,
                                                        const ToType<DT>* __restrict__ delta,
                                                        ToType<DW>* __restrict__ wgrad,
                                                        int N,
                                                        int C,
                                                        int hi,
                                                        int wi,
                                                        int ho,
                                                        int wo,
                                                        int fy,
                                                        int fx,
                                                        int py,
                                                        int px,
                                                        DepthParams dp)
{
    // A 1-D config is a size bucket: the layer's filter (fh, fw) fits inside cfg.kh x cfg.kw.
    constexpr bool BUCKETED = (cfg.kh == 1 || cfg.kw == 1);
    const int fh            = BUCKETED ? fy : cfg.kh;
    const int fw            = BUCKETED ? fx : cfg.kw;

    // Partials stay fp32 for the fold; only a store into the caller's dW narrows.
    static_assert(!PARTITIONED || DW == DataType::fp32, "partials are fp32");

    using datatype_t = ToType<DT>;
    using dw_t       = ToType<DW>;
    // tf32 is split into its bf16 pair on the way into LDS, one plane each: the transpose
    // read and the window perms only come in 16-bit forms.
    constexpr bool is_tf32 = cfg.is_tf32();
    constexpr int PLANES   = is_tf32 ? 2 : 1;
    using lds_t            = std::conditional_t<is_tf32, bf16_t, datatype_t>;
    using ldsx4_t         = std::conditional_t<is_tf32 || DT == DataType::bf16, bf16x4_t, fp16x4_t>;
    using int16x4_t       = __attribute__((ext_vector_type(4))) short;
    using uint32x2_t      = __attribute__((ext_vector_type(2))) unsigned int;
    using TransposeLayout = TransposeLDSLayout<4, 4, 16>;
    // The transpose read's phase is 4 x by 8 c4, which is the spread SwizzleT_wgrad is
    // bank-free on; SwizzleT is built for the untransposed b128 read and is 2-way here.
    using Sw = SwizzleT_wgrad<BLOCK_C>;

    // MFMA operands as bunnies matrices: a TAPS x TAPS block per channel, batched
    // over the CHAN_PER_WAVE channels a wave covers. A tf32 operand is the split pair,
    // and B loads it one plane at a time as the 16-bit plane_b.
    constexpr auto plane_fmt =
        (is_tf32 || DT == DataType::bf16) ? bunnies::fpfmt::e8m7 : bunnies::fpfmt::e5m10;
    constexpr auto compute_fmt = is_tf32 ? bunnies::fpfmt::e8m10_e8m7x2split : plane_fmt;
    using mat_a   = arch::matrix<compute_fmt, TAPS, MFMA_K, bunnies::use::A, CHAN_PER_WAVE>;
    using mat_b   = arch::matrix<compute_fmt, MFMA_K, TAPS, bunnies::use::B, CHAN_PER_WAVE>;
    using plane_b = arch::matrix<plane_fmt, MFMA_K, TAPS, bunnies::use::B, CHAN_PER_WAVE>;
    using mat_acc =
        arch::matrix<bunnies::fpfmt::e8m23, TAPS, TAPS, bunnies::use::Acc, CHAN_PER_WAVE>;

    // Tile geometry. S splits a row by column residue so the decimated operand is
    // again a Hankel matrix.
    constexpr int BLOCK_Q = cfg.block_q();
    constexpr int STRIDE  = cfg.main_stride();
    constexpr int S       = cfg.stride;

    // Tap groups tiling the filter, one accumulator each: group (RGI, SGI) holds
    // r = RGI * TAPS + lane and s = SGI * TAPS + the accumulator index.
    constexpr int RG        = divup(cfg.kh, TAPS);
    constexpr int SG        = divup(cfg.kw, TAPS);
    constexpr int STAP_SPAN = std::min(TAPS, cfg.kw);
    constexpr int RTAP_SPAN = std::min(TAPS, cfg.kh);

    // The A window: words a lane's tap groups reach, and the aligned b128 reads and
    // phase pitch covering them. PRESHIFT is set when a lane out-reaches one perm.
    constexpr int SG_WORDS  = TAPS / (2 * S);
    constexpr bool PRESHIFT = (STAP_SPAN - 1) / S > 2;
    constexpr int WIN_WORDS = 2 * (cfg.q_tiles - 1) + (SG - 1) * SG_WORDS + 3;
    constexpr int WIN_RAW   = WIN_WORDS + (PRESHIFT ? 1 : 0);
    constexpr int WIN_READS = divup(WIN_RAW, 4);
    constexpr int HALF      = 8 * WIN_READS;
    constexpr int IN_STRIDE = S * HALF;

    // Columns the taps read against the wider span the perms touch, which the
    // transpose must cover in whole 4-column groups so no MFMA reads uninitialized LDS.
    constexpr int REAL_COLS        = BLOCK_Q + (cfg.kw - 1) / S;
    constexpr int TAP_COLS         = 2 * WIN_RAW;
    constexpr int GROUPS_PER_PHASE = divup(TAP_COLS, 4);
    constexpr int IN_X_GROUPS_USED = S * GROUPS_PER_PHASE;
    constexpr int IN_X_GROUPS      = divup(IN_X_GROUPS_USED, WAVES_C) * WAVES_C;
    constexpr int COLS_PER_PHASE   = 4 * GROUPS_PER_PHASE;
    static_assert(4 * GROUPS_PER_PHASE <= HALF, "a phase must hold its own groups");

    // Staging tile, carried at main_in's pitch: the transpose writes staging column t
    // to operand column t, so the two layouts have to agree.
    constexpr int STAGE_W           = IN_STRIDE;
    constexpr int STAGE_IN_UINT4    = STAGE_W * BLOCK_C8;
    constexpr int STAGE_DELTA_UINT4 = BLOCK_Q * BLOCK_C8;

    // Delta rows resident at once, one per tap phase, plus a slot of zeros so one
    // address select stands in for masking off the taps with no row this row.
    constexpr int DELTA_RING  = divup(cfg.kh, S);
    constexpr int ZERO_SLOT   = DELTA_RING;
    constexpr int DELTA_SLOTS = DELTA_RING + (S > 1 ? 1 : 0);

    // tf32's register path stores one element per row, which the plain pitches fold onto a
    // few banks, so its rows are skewed. A skew moves a row by whole reads only: 16 B for the
    // b128 window, 8 B for the B tile.
    constexpr int IN_SKEW       = is_tf32 ? 8 : 0;
    constexpr int DELTA_SKEW    = is_tf32 ? 4 : 0;
    constexpr int MAIN_IN_ELEMS = BLOCK_C * IN_STRIDE + IN_SKEW * ((BLOCK_C - 1) / 8);

    auto in_at    = [](int c) { return c * IN_STRIDE + IN_SKEW * (c / 8); };
    auto delta_at = [](int c) { return c * STRIDE + DELTA_SKEW * (c / 16); };

    // A slot pad of 16 puts neighbouring slots 8 dwords apart, which the B tile's
    // ds_read2_b64 needs to cover all banks. S = 2 kh 9 also reads slots four apart, which
    // 16 puts on one bank, so it takes the least-conflicted pad measured per tile width.
    constexpr int SLOT_PAD = !is_tf32                ? 12
                             : S == 2 && cfg.kh == 9 ? (cfg.q_tiles == 8 ? 12 : 28)
                                                     : 16;
    static_assert(SLOT_PAD >= DELTA_SKEW * ((BLOCK_C - 1) / 16), "a slot holds its skewed rows");
    constexpr int SLOT_ELEMS       = BLOCK_C * STRIDE + SLOT_PAD;
    constexpr int MAIN_DELTA_ELEMS = DELTA_SLOTS * SLOT_ELEMS;

    // Rounded to whole waves, so the lanes past the tile have somewhere valid to
    // drop their out-of-range fetch.
    constexpr int STAGE_IN_ALLOC    = divup(STAGE_IN_UINT4, WAVE_SIZE) * WAVE_SIZE;
    constexpr int STAGE_DELTA_ALLOC = divup(STAGE_DELTA_UINT4, WAVE_SIZE) * WAVE_SIZE;

    constexpr int DEPTH = cfg.stage_depth;

    // Narrow C: a uint4 would straddle two pixels, so this path moves NVEC channels a lane
    // register-to-LDS and needs no staging buffers. The LDS side still writes one element
    // per row of main_in, consecutive channels being IN_STRIDE apart there.
    constexpr bool NARROW = cfg.chan_vec < 8;
    constexpr int NVEC    = NARROW ? cfg.chan_vec : 1;
    static_assert(!is_tf32 || cfg.chan_vec < 8,
                  "tf32 forms its bf16 pair in a register and cannot take the wide DMA path");
    using narrow_pack_t  = bunnies::packed_type<NVEC*(int)sizeof(datatype_t)>;
    using narrow_elems_t = std::array<datatype_t, NVEC>;

    constexpr int NARROW_IN_ELEMS     = S * COLS_PER_PHASE * BLOCK_C;
    constexpr int NARROW_IN_GROUPS    = NARROW_IN_ELEMS / NVEC;
    constexpr int NARROW_IN_PASSES    = NARROW ? divup(NARROW_IN_GROUPS, cfg.threads()) : 0;
    constexpr int NARROW_DELTA_ELEMS  = BLOCK_Q * BLOCK_C;
    constexpr int NARROW_DELTA_GROUPS = NARROW_DELTA_ELEMS / NVEC;
    constexpr int NARROW_DELTA_PASSES = NARROW ? divup(NARROW_DELTA_GROUPS, cfg.threads()) : 0;
    constexpr int STAGE_IN_TOTAL      = NARROW ? 1 : DEPTH * STAGE_IN_ALLOC;
    constexpr int STAGE_DELTA_TOTAL   = NARROW ? 1 : DEPTH * STAGE_DELTA_ALLOC;
    constexpr int SINK_TOTAL          = NARROW ? 1 : WAVE_SIZE;
    // A channel tile is a whole number of vectors, so no group straddles the tile either.
    static_assert(BLOCK_C % cfg.chan_vec == 0, "a channel vector has to divide the tile");
    static_assert(8 % NVEC == 0, "a vector's rows share one skew, so its stores sit a pitch apart");

    // A register array is only addressable by a compile-time index, so the narrow path
    // dispatches its buffer on a static_for over DEPTH rather than indexing with buf. It
    // never touches stage_in or stage_delta, which is why those stay at a dummy element.

    // Channel-major operand tiles, main_in[pl][c][x] and main_delta[pl][p % kh][c][x].
    // Neither is double buffered: the second copy costs a workgroup of occupancy, measured -11%.
    __shared__ uint4 stage_in[STAGE_IN_TOTAL];
    __shared__ uint4 stage_delta[STAGE_DELTA_TOTAL];
    __shared__ __align__(16) lds_t main_in[PLANES * MAIN_IN_ELEMS];
    __shared__ __align__(8) lds_t main_delta[PLANES * MAIN_DELTA_ELEMS];
    // Destination for a wave that owns no slot in a pass, so load counts stay equal.
    __shared__ uint4 load_sink[SINK_TOTAL];

    const int tid    = threadIdx.x;
    const int wave   = tid / WAVE_SIZE;
    const int lane   = tid % WAVE_SIZE;
    const int wave_u = __builtin_amdgcn_readfirstlane(wave);

    const BlockIndex bidx  = xcd_paired_index(C);
    const int block_c_base = bidx.c_block * BLOCK_C;
    const int block_q      = bidx.q_tile * BLOCK_Q;

    // The (image, row chunk) units this workgroup owns. z strides the grid rather than
    // taking one apiece, so the reduce reads one slot per workgroup, not one per unit.
    //
    // N counts delta planes, and z is tap-major: the first z_tap z-slots run depth tap 0.
    const int z_units = N * divup(hi, cfg.rows_per_chunk);
    // A conv2d is compiled without the depth decode, which costs it ~1% on short rows.
    constexpr bool planar = !CONV3D;
    const int z_tap       = planar ? gridDim.z : gridDim.z / dp.kd;
    const int tap         = planar ? 0 : bidx.z / z_tap;

    // Re-keyed per unit in the loop below, which the loaders read through their captures.
    int block_n = 0;
    int y0      = 0;
    int y1      = 0;

    const int C8 = C / 8;

    // MFMA operand roles: A supplies row m = lane % 4, B column n = lane % 4. Lanes
    // past the filter extent are clamped, since the epilogue drops their cells.
    const int lane_tap  = lane % TAPS;
    const int lane_s    = std::min(lane_tap, STAP_SPAN - 1);
    const int lane_r    = std::min(lane_tap, RTAP_SPAN - 1);
    const int lane_chan = wave * CHAN_PER_WAVE + (lane / TAPS) % CHAN_PER_WAVE;

    // A tap group adds a whole multiple of TAPS to the tap, so the phase a lane
    // reads is the same in every group. in_at is written out at 64 bits, since calling it
    // reorders the 16-bit kernels' schedule.
    const lds_t* a_base = main_in + (size_t)lane_chan * IN_STRIDE +
                          (size_t)(IN_SKEW * (lane_chan / 8)) + (size_t)(lane_s % S) * HALF;

    // The window offset splits into whole words shifted out once per row and a residue
    // a perm applies; a funnel shift cannot, its distance being taken modulo 32.
    const int lane_off     = lane_s / S;
    const int lane_word    = PRESHIFT ? lane_off / 2 : 0;
    const int lane_res     = lane_off - 2 * lane_word;
    const uint32_t pre_sel = 0x03020100u + 0x01010101u * (4 * lane_word);
    const uint32_t res_sel = 0x03020100u + 0x01010101u * (2 * lane_res);

    // Transpose-read coordinates: lane L receives LDS column L for four consecutive
    // x, so the destination channel is the lane index itself.
    const int lane_row0 = TransposeLayout::row(lane, 0);
    const int lane_c4   = TransposeLayout::batch(lane);

    bunnies::reg_tile<mat_acc, RG, SG> acc{};

    // One row per descriptor, rebased on each load's own n and row. Buffer offsets are 32
    // bits and NUM_RECORDS is too, so spanning an image would cap H * W * C at 2 GB;
    // rebasing moves the image and row strides into the 64-bit base and leaves the offsets
    // to cover a single row, which is what one load actually reaches. A row off the image
    // gets NUM_RECORDS 0; a column off it, the row size as its offset.
    const int64_t input_row_elems = static_cast<int64_t>(wi) * C;
    const int64_t delta_row_elems = static_cast<int64_t>(wo) * C;
    const auto input_row_bytes    = static_cast<uint32_t>(input_row_elems * sizeof(datatype_t));
    const auto delta_row_bytes    = static_cast<uint32_t>(delta_row_elems * sizeof(datatype_t));
    const datatype_t* input_image = input;
    const datatype_t* delta_image = delta;
    auto row_rsrc = [](const datatype_t* image, int64_t row_elems, int row, bool row_ok) {
        return arch::make_buffer(image + (row_ok ? row * row_elems : 0), row_ok ? row_elems : 0);
    };

    // The narrow path has its own plan below and never issues these, so the WIDE_
    // counts fall to zero and it builds none of them.
    constexpr int INPUT_LOAD_PASSES = divup(STAGE_IN_UINT4, cfg.threads());
    constexpr int DELTA_LOAD_PASSES = divup(STAGE_DELTA_UINT4, cfg.threads());
    constexpr int WIDE_IN_PASSES    = NARROW ? 0 : INPUT_LOAD_PASSES;
    constexpr int WIDE_DELTA_PASSES = NARROW ? 0 : DELTA_LOAD_PASSES;

    // Per-pass load plan. The global offset is row-invariant, so the main loop only
    // adds the row stride; a pass whose slots all lie past the tile targets the sink.
    uint32_t input_global_offsets[INPUT_LOAD_PASSES];
    uint4* input_lds_addrs[INPUT_LOAD_PASSES];
    int input_buf_stride[INPUT_LOAD_PASSES];
    for(int pass = 0; pass < WIDE_IN_PASSES; pass++)
    {
        int lds_idx = tid + pass * cfg.threads();
        // Wave-uniform: buffer_load_lds derives every lane's LDS address from one
        // wave base, so the choice of destination cannot vary within a wave.
        bool wave_live = (wave * WAVE_SIZE + pass * cfg.threads()) < STAGE_IN_UINT4;
        bool active    = wave_live && lds_idx < STAGE_IN_UINT4;

        input_lds_addrs[pass]      = wave_live ? &stage_in[lds_idx] : &load_sink[lane];
        input_buf_stride[pass]     = wave_live ? STAGE_IN_ALLOC : 0;
        input_global_offsets[pass] = input_row_bytes;
        if(active)
        {
            int col    = Sw::x(lds_idx);
            int c8_idx = Sw::c8(lds_idx);
            // This is where the phase split happens: the staging column names a phase
            // and a position in it, and the fetch walks the row with stride S.
            int phase      = col / HALF;
            int u          = col - phase * HALF;
            int global_col = (S * block_q - px) + S * u + phase;
            int global_c8  = block_c_base / 8 + c8_idx;
            input_global_offsets[pass] =
                (u < REAL_COLS && global_col >= 0 && global_col < wi && global_c8 < C8)
                    ? sizeof(uint4) * ((size_t)global_col * C8 + global_c8)
                    : input_row_bytes;
        }
    }

    uint32_t delta_global_offsets[DELTA_LOAD_PASSES];
    uint4* delta_lds_addrs[DELTA_LOAD_PASSES];
    int delta_buf_stride[DELTA_LOAD_PASSES];
    for(int pass = 0; pass < WIDE_DELTA_PASSES; pass++)
    {
        int lds_idx    = tid + pass * cfg.threads();
        bool wave_live = (wave * WAVE_SIZE + pass * cfg.threads()) < STAGE_DELTA_UINT4;
        bool active    = wave_live && lds_idx < STAGE_DELTA_UINT4;

        delta_lds_addrs[pass]      = wave_live ? &stage_delta[lds_idx] : &load_sink[lane];
        delta_buf_stride[pass]     = wave_live ? STAGE_DELTA_ALLOC : 0;
        delta_global_offsets[pass] = delta_row_bytes;
        if(active)
        {
            int col                    = Sw::x(lds_idx);
            int c8_idx                 = Sw::c8(lds_idx);
            int global_q               = block_q + col;
            int global_c8              = block_c_base / 8 + c8_idx;
            delta_global_offsets[pass] = (global_q < wo && global_c8 < C8)
                                             ? sizeof(uint4) * ((size_t)global_q * C8 + global_c8)
                                             : delta_row_bytes;
        }
    }

    // Narrow staging plan: lane owns one channel of one operand column. At one element
    // per lane the store address does what the transpose would, so no staging buffers.
    uint32_t narrow_in_offsets[NARROW_IN_PASSES == 0 ? 1 : NARROW_IN_PASSES];
    int narrow_in_lds[NARROW_IN_PASSES == 0 ? 1 : NARROW_IN_PASSES];
    uint32_t narrow_delta_offsets[NARROW_DELTA_PASSES == 0 ? 1 : NARROW_DELTA_PASSES];
    int narrow_delta_lds[NARROW_DELTA_PASSES == 0 ? 1 : NARROW_DELTA_PASSES];
    if constexpr(NARROW)
    {
        // A group is NVEC channels of one column. c_g and C are both multiples of NVEC, so a
        // group is wholly inside the pixel or wholly outside: one range test still covers it.
        constexpr int GROUPS_PER_COL = BLOCK_C / NVEC;
        for(int pass = 0; pass < NARROW_IN_PASSES; pass++)
        {
            const int g   = tid + pass * cfg.threads();
            const int col = g / GROUPS_PER_COL;
            const int ch  = (g - col * GROUPS_PER_COL) * NVEC;
            // Same column map as the wide staging fetch, so the phases land apart
            // exactly as the transpose would have left them.
            const int phase      = S == 1 ? 0 : col / COLS_PER_PHASE;
            const int u          = col - phase * COLS_PER_PHASE;
            const int global_col = (S * block_q - px) + S * u + phase;
            const int c_g        = block_c_base + ch;
            const bool ok        = g < NARROW_IN_GROUPS && u < REAL_COLS && global_col >= 0 &&
                            global_col < wi && c_g < C;
            narrow_in_offsets[pass] =
                ok ? sizeof(datatype_t) * (uint32_t)((size_t)global_col * C + c_g)
                   : input_row_bytes;
            narrow_in_lds[pass] = in_at(ch) + phase * HALF + u;
        }
        for(int pass = 0; pass < NARROW_DELTA_PASSES; pass++)
        {
            const int g   = tid + pass * cfg.threads();
            const int x   = g / GROUPS_PER_COL;
            const int ch  = (g - x * GROUPS_PER_COL) * NVEC;
            const int c_g = block_c_base + ch;
            const bool ok = g < NARROW_DELTA_GROUPS && block_q + x < wo && c_g < C;
            narrow_delta_offsets[pass] =
                ok ? sizeof(datatype_t) * (uint32_t)((size_t)(block_q + x) * C + c_g)
                   : delta_row_bytes;
            narrow_delta_lds[pass] = delta_at(ch) + x;
        }
    }

    narrow_pack_t in_regs[DEPTH][NARROW_IN_PASSES == 0 ? 1 : NARROW_IN_PASSES];
    narrow_pack_t delta_regs[DEPTH][NARROW_DELTA_PASSES == 0 ? 1 : NARROW_DELTA_PASSES];

    // Every wave issues every pass, so vmcnt means the same thing in all of them; rows
    // and lanes past the tile fetch beyond NUM_RECORDS, which returns zero.
    auto load_input_global = [&](int buf, int y) {
        const bool row_ok = y >= 0 && y < hi;
        const auto rsrc   = row_rsrc(input_image, input_row_elems, y, row_ok);
        if constexpr(NARROW)
        {
            // buf is workgroup-uniform, so this is one scalar compare ahead of the loads,
            // which still issue a row early and stay in flight across sync_staged.
            static_for<DEPTH>([&]<int B>() {
                if(buf != B)
                    return;
                static_for<NARROW_IN_PASSES>([&]<int P>() {
                    arch::buffer_load<NVEC * sizeof(datatype_t)>::load(
                        rsrc, &in_regs[B][P], narrow_in_offsets[P], 0);
                });
            });
        }
        else
        {
            for(int pass = 0; pass < INPUT_LOAD_PASSES; pass++)
            {
                arch::buffer_load_lds<16>::load(rsrc,
                                                input_lds_addrs[pass] +
                                                    buf * input_buf_stride[pass],
                                                input_global_offsets[pass],
                                                0);
            }
        }
    };

    auto load_delta_global = [&](int buf, int p) {
        const bool row_ok = p >= 0 && p < ho;
        const auto rsrc   = row_rsrc(delta_image, delta_row_elems, p, row_ok);
        if constexpr(NARROW)
        {
            static_for<DEPTH>([&]<int B>() {
                if(buf != B)
                    return;
                static_for<NARROW_DELTA_PASSES>([&]<int P>() {
                    arch::buffer_load<NVEC * sizeof(datatype_t)>::load(
                        rsrc, &delta_regs[B][P], narrow_delta_offsets[P], 0);
                });
            });
        }
        else
        {
            for(int pass = 0; pass < DELTA_LOAD_PASSES; pass++)
            {
                arch::buffer_load_lds<16>::load(rsrc,
                                                delta_lds_addrs[pass] +
                                                    buf * delta_buf_stride[pass],
                                                delta_global_offsets[pass],
                                                0);
            }
        }
    };

    // Loads in flight per staged row, so the pipeline can wait for the oldest
    // row while the DEPTH - 1 rows behind it stay outstanding.
    constexpr int DELTA_PASSES  = NARROW ? NARROW_DELTA_PASSES : DELTA_LOAD_PASSES;
    constexpr int LOADS_PER_ROW = (NARROW ? NARROW_IN_PASSES : INPUT_LOAD_PASSES) + DELTA_PASSES;

    // The row loop synchronises with a bare s_barrier rather than __syncthreads, whose
    // fence lowers to vmcnt(0) and would drain the staging pipeline every row.

    // Both operands drain together: vmcnt is per-wave, so a wave that waits only for
    // its own load learns nothing about the neighbours filling the rest of the tile.
    auto sync_staged = [] { wait_mem<(DEPTH - 1) * LOADS_PER_ROW, 0>(); };
    // The transposed operands are visible; the staging loads keep flying.
    auto sync_transposed = [] { wait_lgkmcnt<0>(); };

    // [x][c] -> [c][x]: each ds_read_tr16_b64 hands lane L the four x at column L, and
    // consecutive groups go to consecutive waves, so a wave's groups are WAVES_C apart.
    constexpr int X_GROUPS    = BLOCK_Q / 4;
    constexpr int X_PASSES    = X_GROUPS / WAVES_C;
    constexpr int IN_X_PASSES = IN_X_GROUPS / WAVES_C;
    static_assert(X_GROUPS % WAVES_C == 0, "a transpose pass must cover whole waves");

    // Both transposes unroll over a compile-time pass index: `wave` is not provably
    // uniform, so a runtime loop recomputes every LDS address, 25% of the cycles.

    // The column a transpose group covers. The phases sit back to back at HALF pitch,
    // so a phase's groups are consecutive while their columns are not.
    static_assert(S <= 2, "the phase of a group is derived from a single compare");
    auto group_x = [](int g) {
        if constexpr(S == 1)
            return 4 * g;
        else
        {
            const int phase = g >= GROUPS_PER_PHASE;
            return phase * HALF + 4 * (g - phase * GROUPS_PER_PHASE);
        }
    };

    auto transpose_input = [&](int buf) {
        if constexpr(NARROW)
        {
            // Only the last pass can hold lanes past the tile, so only it is checked.
            static_for<DEPTH>([&]<int B>() {
                if(buf != B)
                    return;
                static_for<NARROW_IN_PASSES>([&]<int P>() {
                    if constexpr((P + 1) * cfg.threads() > NARROW_IN_GROUPS)
                        if(tid + P * cfg.threads() >= NARROW_IN_GROUPS)
                            return;
                    // A vector's channels are consecutive rows of main_in, so they land
                    // IN_STRIDE apart and this side stays one element per store.
                    const auto v = __builtin_bit_cast(narrow_elems_t, in_regs[B][P]);
                    static_for<NVEC>([&]<int V>() {
                        const int slot = narrow_in_lds[P] + V * IN_STRIDE;
                        if constexpr(is_tf32)
                        {
                            const bf16_pair pair          = fp32_to_bf16_pair(v[V]);
                            main_in[slot]                 = pair.big;
                            main_in[MAIN_IN_ELEMS + slot] = pair.small;
                        }
                        else
                            main_in[slot] = v[V];
                    });
                });
            });
        }
        else
        {
            auto* src = reinterpret_cast<int16x4_t*>(stage_in + buf * STAGE_IN_ALLOC);
            int16x4_t v[IN_X_PASSES];
            static_for<IN_X_PASSES>([&]<int G>() {
                // Only the last pass can hold waves without real columns, and taking the
                // wave index through readfirstlane keeps the check scalar.
                if constexpr((G + 1) * WAVES_C > IN_X_GROUPS_USED)
                    if(G * WAVES_C + wave_u >= IN_X_GROUPS_USED)
                        return;
                const int xg = group_x(wave + G * WAVES_C);
                arch::ds_read_b64_tr_b16::load(&src[Sw::offset_uint2(xg + lane_row0, lane_c4)],
                                               &v[G]);
            });
            static_for<IN_X_PASSES>([&]<int G>() {
                if constexpr((G + 1) * WAVES_C > IN_X_GROUPS_USED)
                    if(G * WAVES_C + wave_u >= IN_X_GROUPS_USED)
                        return;
                const int xg = group_x(wave + G * WAVES_C);
                arch::ds_store_b64::store(&main_in[in_at(lane) + xg], &v[G]);
            });
        }
    };

    auto transpose_delta = [&](int buf, int slot) {
        if constexpr(NARROW)
        {
            static_for<DEPTH>([&]<int B>() {
                if(buf != B)
                    return;
                static_for<NARROW_DELTA_PASSES>([&]<int P>() {
                    if constexpr((P + 1) * cfg.threads() > NARROW_DELTA_GROUPS)
                        if(tid + P * cfg.threads() >= NARROW_DELTA_GROUPS)
                            return;
                    const auto v = __builtin_bit_cast(narrow_elems_t, delta_regs[B][P]);
                    static_for<NVEC>([&]<int V>() {
                        const int at = slot * SLOT_ELEMS + narrow_delta_lds[P] + V * STRIDE;
                        if constexpr(is_tf32)
                        {
                            const bf16_pair pair              = fp32_to_bf16_pair(v[V]);
                            main_delta[at]                    = pair.big;
                            main_delta[MAIN_DELTA_ELEMS + at] = pair.small;
                        }
                        else
                            main_delta[at] = v[V];
                    });
                });
            });
        }
        else
        {
            auto* src = reinterpret_cast<int16x4_t*>(stage_delta + buf * STAGE_DELTA_ALLOC);
            int16x4_t v[X_PASSES];
            static_for<X_PASSES>([&]<int G>() {
                const int xg = (wave + G * WAVES_C) * 4;
                arch::ds_read_b64_tr_b16::load(&src[Sw::offset_uint2(xg + lane_row0, lane_c4)],
                                               &v[G]);
            });
            static_for<X_PASSES>([&]<int G>() {
                const int xg = (wave + G * WAVES_C) * 4;
                arch::ds_store_b64::store(&main_delta[slot * SLOT_ELEMS + delta_at(lane) + xg],
                                          &v[G]);
            });
        }
    };

    // Delta rows live in a ring keyed by the absolute row index, so the slot a tap
    // reads is fixed by the row it pairs with and no data ever moves.
    auto ring_slot = [&](int p) { return ((p % DELTA_RING) + DELTA_RING) % DELTA_RING; };

    // The freshest delta row an input row pairs with. Only every S-th input row
    // brings a new one; y + py is never negative, so the division floors.
    auto delta_row = [&](int y) { return (y + py) / S; };

    // Zero the slot the mismatched-phase taps read. One write per thread, made
    // visible by the barrier the prologue's first row already carries.
    if constexpr(S > 1)
    {
        for(int i = tid; i < SLOT_ELEMS; i += cfg.threads())
            static_for<PLANES>([&]<int PL>() {
                main_delta[PL * MAIN_DELTA_ELEMS + ZERO_SLOT * SLOT_ELEMS + i] =
                    static_cast<lds_t>(0.f);
            });
        wait_lgkmcnt<0>();
        __builtin_amdgcn_s_barrier();
    }

    // Stage row y, or issue the same loads out of range when y is past the chunk: the
    // vmcnt wait is a compile-time count, so every row has to issue alike.
    auto stage_row = [&](int b, int y) {
        const bool live = y < y1;
        load_input_global(b, live ? y : hi);
        // Only every S-th row brings a delta row the ring does not already hold; the
        // rows in between still issue the load, out of range.
        const bool fresh = live && (y + py) % S == 0;
        load_delta_global(b, fresh ? delta_row(y) : ho);
    };

    constexpr int PRE       = DELTA_RING - (S == 1 ? 1 : 0);
    constexpr int PRE_WAVES = divup(PRE, DEPTH);

    // Every unit runs its rows into the same accumulators, so the workgroup stores one
    // partial for all of them. The stride leaves them disjoint but not contiguous.
    for(int unit = planar ? bidx.z : bidx.z % z_tap; unit < z_units; unit += z_tap)
    {
        block_n          = unit % N;
        const int chunk  = unit / N;
        int64_t in_plane = block_n;
        if constexpr(!planar)
        {
            // The input plane this delta plane meets through the tap; none in the depth padding.
            const int od = block_n % dp.d_out;
            const int id = od * dp.stride - dp.pad + tap * dp.dilation;
            if(id < 0 || id >= dp.d_in)
                continue;
            in_plane = (int64_t)(block_n / dp.d_out) * dp.d_in + id;
        }
        y0          = chunk * cfg.rows_per_chunk;
        y1          = std::min(hi, y0 + cfg.rows_per_chunk);
        input_image = input + in_plane * hi * input_row_elems;
        delta_image = delta + static_cast<int64_t>(block_n) * ho * delta_row_elems;

        // The unit before left loads for rows past its chunk in flight, and its readers have
        // to be done before these slots are refilled.
        wait_mem<0, 0>();
        __builtin_amdgcn_s_barrier();

        // Prologue: fill the ring with the rows the chunk starts out needing -- at stride 1
        // the loop brings in the newest itself -- in waves of DEPTH, one staging buffer each.
        static_for<PRE_WAVES>([&]<int WV>() {
            constexpr int LO  = WV * DEPTH;
            constexpr int CNT = std::min(DEPTH, PRE - LO);
            for(int i = 0; i < CNT; i++)
                load_delta_global(i, delta_row(y0) - DELTA_RING + 1 + LO + i);
            static_for<CNT>([&]<int I>() {
                wait_mem<(CNT - 1 - I) * DELTA_PASSES, 0>();
                __builtin_amdgcn_s_barrier();
                transpose_delta(I, ring_slot(delta_row(y0) - DELTA_RING + 1 + LO + I));
                wait_lgkmcnt<0>();
                __builtin_amdgcn_s_barrier();
            });
        });

        // Prime the pipeline: DEPTH - 1 rows are in flight before the first MFMA, so
        // every row's loads get that many rows of work to hide behind.
        for(int i = 0; i < DEPTH - 1; i++)
            stage_row(i, y0 + i);

        int buf = 0;
        // The ring indices below are recomputed from y, not carried across the loop.
        // Carrying them drops a signed modulo per row and measured 0.4% slower.
        for(int y = y0; y < y1; y++)
        {
            stage_row((buf + DEPTH - 1) % DEPTH, y + DEPTH - 1);

            sync_staged();
            __builtin_amdgcn_s_barrier();

            transpose_input(buf);
            // Uniform across the workgroup, so the rows that bring no new delta row
            // skip the transpose outright rather than masking it off.
            if(S == 1 || (y + py) % S == 0)
                transpose_delta(buf, ring_slot(delta_row(y)));

            sync_transposed();
            __builtin_amdgcn_s_barrier();

            // Every operand read issues before the first MFMA consumes one, so the row pays
            // one LDS latency rather than one per tile -- a third of the stall cycles.

            // The r tap of group RGI pairs this row with delta row (y + py - r) / S, which
            // exists only when S divides it; the other phases read the zero slot.
            int d_slot[RG];
            static_for<RG>([&]<int RGI>() {
                const int num = y + py - (RGI * TAPS + lane_r);
                d_slot[RGI]   = (S == 1 || num % S == 0) ? ring_slot(num / S) : ZERO_SLOT;
            });

            // One window per plane; the B tile holds both planes.
            __attribute__((aligned(16))) uint32_t win[PLANES][4 * WIN_READS];
            bunnies::reg_tile<mat_b, RG, cfg.q_tiles> b;
            static_for<PLANES>([&]<int PL>() {
                static_for<WIN_READS>([&]<int J>() {
                    arch::ds_load_b128::load(a_base + PL * MAIN_IN_ELEMS + J * 8, &win[PL][4 * J]);
                });
                // Column c of the batched B operand carries channel c / TAPS, so the map
                // turns the operand's own coordinate into the channel's row.
                bunnies::load_tile<arch::ds_load_b64, plane_b, /*Small=*/PL == 1>(
                    b, main_delta, [&](int rg, int t, int, int c) {
                    return PL * MAIN_DELTA_ELEMS + d_slot[rg] * SLOT_ELEMS +
                           delta_at(wave * CHAN_PER_WAVE + c / TAPS) + t * MFMA_K;
                });
                // Shift the window down to a residue the perms can reach, in place and
                // ascending so each word is read before it is overwritten.
                if constexpr(PRESHIFT)
                    static_for<WIN_WORDS>([&]<int J>() {
                        win[PL][J] = __builtin_amdgcn_perm(win[PL][J + 1], win[PL][J], pre_sel);
                    });
            });
            static_for<SG>([&]<int SGI>() {
                static_for<cfg.q_tiles>([&]<int T>() {
                    // Both halves of the A fragment are the window taken this lane's tap
                    // late, so each is one byte permute of an adjacent register pair.
                    constexpr int K = 2 * T + SGI * SG_WORDS;
                    mat_a af;
                    static_for<PLANES>([&]<int PL>() {
                        uint32x2_t a = {
                            __builtin_amdgcn_perm(win[PL][K + 1], win[PL][K], res_sel),
                            __builtin_amdgcn_perm(win[PL][K + 2], win[PL][K + 1], res_sel)};
                        *bunnies::block_storage<PL == 1>(af) = __builtin_bit_cast(ldsx4_t, a);
                    });
                    // One A fragment feeds every r-tap group, so it is built once here
                    // rather than per accumulator.
                    static_for<RG>([&]<int RGI>() {
                        arch::mma<>::wmma(
                            acc.block(RGI, SGI), af, b.block(RGI, T), acc.block(RGI, SGI));
                    });
                });
            });

            buf = (buf + 1) % DEPTH;
        }
    }

    // Epilogue: exactly one lane of one workgroup owns each (partition, c, r, s), so every
    // element below is stored once and the reduce, not the memory system, sums them.
    const int c_global = block_c_base + lane_chan;
    if(c_global >= C)
        return;

    dw_t* dst = wgrad;
    if constexpr(PARTITIONED)
    {
        const int partition = bidx.z * gridDim.y + bidx.q_tile;
        dst += (size_t)partition * C * fh * fw;
    }

    // A tap's 16 lanes hold 16 consecutive channels, so channel-major partials put its store
    // on one 64 B sector. Only a lone partition writes the caller's dW, which keeps [c][r][s].
    const size_t tap_stride = PARTITIONED ? (size_t)C : 1;

    dw_t* dw = dst + (PARTITIONED ? (size_t)c_global : (size_t)c_global * fh * fw);
    static_for<RG>([&]<int RGI>() {
        // The taps a group holds past the filter carry a delta row this row never
        // paired with, so they are dropped rather than stored.
        const int r = RGI * TAPS + lane_tap;
        if(r >= fh)
            return;
        static_for<SG>([&]<int SGI>() {
            static_for<TAPS>([&]<int I>() {
                if constexpr(SGI * TAPS + I < cfg.kw)
                    if(SGI * TAPS + I < fw)
                        dw[(r * fw + SGI * TAPS + I) * tap_stride] =
                            static_cast<dw_t>(acc.block(RGI, SGI).data[I]);
            });
        });
    });
}

// Fold the per-partition partials into the caller's dW, transposed back to [c][r][s].
//
// SLICES waves split the partition axis and meet in LDS, one element per lane: dW is too
// small to fill the machine off the element axis alone, and folding the slices in a second
// kernel instead costs more in launch than the bytes it saves.
template <int SLICES, DataType DW, bool CONV3D>
__global__ __launch_bounds__(SLICES* WAVE_SIZE) void conv2d_depthwise_wgrad_hankel_reduce_cdna4(
    ToType<DW>* __restrict__ wgrad,
    const float* __restrict__ partials,
    int dw_elems,
    int partitions,
    int C,
    int taps)
{
    __shared__ float slice_sum[SLICES][WAVE_SIZE];

    const int lane  = threadIdx.x % WAVE_SIZE;
    const int slice = threadIdx.x / WAVE_SIZE;
    const int i     = blockIdx.x * WAVE_SIZE + lane;
    // y is the depth tap; dw_elems and partitions are per tap, each tap's contiguous.
    const int td = blockIdx.y;
    if constexpr(CONV3D)
        partials += (size_t)td * partitions * dw_elems;

    // Several partition reads in flight at once. One accumulator walked without unrolling
    // puts a memory round trip between each add, which measured 10.3 us against 1.2 MB.
    constexpr int U = 4;
    float acc[U]    = {};
    if(i < dw_elems)
    {
        int p = slice;
        for(; p + (U - 1) * SLICES < partitions; p += U * SLICES)
        {
#pragma unroll
            for(int u = 0; u < U; u++)
                acc[u] += partials[(size_t)(p + u * SLICES) * dw_elems + i];
        }
        for(; p < partitions; p += SLICES)
            acc[0] += partials[(size_t)p * dw_elems + i];
    }

    float sum = 0.f;
#pragma unroll
    for(int u = 0; u < U; u++)
        sum += acc[u];

    slice_sum[slice][lane] = sum;
    __syncthreads();

    if(slice != 0 || i >= dw_elems)
        return;
#pragma unroll
    for(int s = 1; s < SLICES; s++)
        sum += slice_sum[s][lane];

    // The reads ran in the partials' own order, so the transpose costs only this one store.
    const int tap  = i / C;
    const size_t c = i - tap * C;
    if constexpr(CONV3D)
        wgrad[(c * gridDim.y + td) * taps + tap] = static_cast<ToType<DW>>(sum);
    else
        wgrad[c * taps + tap] = static_cast<ToType<DW>>(sum);
}

// Waves splitting the partition axis above, at the most a 1024-thread group can carry. The
// fold reads partials the main kernel has just evicted from L2, so it is latency-exposed and
// wants every wave it can get: 16 over 8 is worth 11.4% where the fold owns most of a run.
constexpr int REDUCE_SLICES = 16;

// Without the bound the compiler sizes registers for a 1024-thread group and caps
// this kernel at 128 VGPRs, which spills the 9x9 and 11x11 accumulators to scratch.
template <Config cfg, DataType DT, bool PARTITIONED, DataType DW, bool CONV3D>
__global__ __launch_bounds__(cfg.threads()) void conv2d_depthwise_wgrad_hankel_nhwc_cdna4(
    const ToType<DT>* __restrict__ input,
    const ToType<DT>* __restrict__ delta,
    ToType<DW>* __restrict__ wgrad,
    int N,
    int C,
    int hi,
    int wi,
    int ho,
    int wo,
    int fy,
    int fx,
    int py,
    int px,
    DepthParams dp)
{
    if(__builtin_amdgcn_is_invocable(__builtin_amdgcn_mfma_f32_4x4x4f16) &&
       __builtin_amdgcn_is_invocable(__builtin_amdgcn_mfma_f32_4x4x4bf16_1k) &&
       __builtin_amdgcn_is_invocable(__builtin_amdgcn_raw_ptr_buffer_load_lds) &&
       __builtin_amdgcn_is_invocable(__builtin_amdgcn_ds_read_tr16_b64_v4i16))
    {
        conv2d_depthwise_wgrad_hankel_nhwc_impl<cfg, DT, PARTITIONED, DW, CONV3D>(
            input, delta, wgrad, N, C, hi, wi, ho, wo, fy, fx, py, px, dp);
    }
}

template <Config cfg>
void launch_impl(const LaunchParams& lp,
                 const Device&,
                 const ConvParams& par,
                 const void* in,
                 const void* wei,
                 void* out,
                 void* workspace,
                 hipStream_t stream)
{
    const DepthParams dp = DepthParams::of(par);
    const int dw_elems   = par.k * par.kh * par.kw;
    const size_t parts   = partition_count(lp.grid);

    auto typed_launch =
        [&]<DataType DT, bool PARTITIONED, DataType DW, bool CONV3D = false>(ToType<DW>* dst) {
        using dtype = ToType<DT>;
        conv2d_depthwise_wgrad_hankel_nhwc_cdna4<cfg, DT, PARTITIONED, DW, CONV3D>
            <<<lp.grid, lp.block_size, 0, stream>>>(static_cast<const dtype*>(in),
                                                    static_cast<const dtype*>(wei),
                                                    dst,
                                                    par.n * dp.d_out,
                                                    par.c,
                                                    par.h,
                                                    par.w,
                                                    par.p,
                                                    par.q,
                                                    par.kh,
                                                    par.kw,
                                                    par.pad_h,
                                                    par.pad_w,
                                                    dp);
    };
    // No zeroing on either path: every element the reduce or the caller reads was stored
    // rather than accumulated into, which is also what lets the cast to DW ride on that store.
    auto dispatch = [&]<DataType DT, DataType DW>() {
        // A conv3d always folds through partials, one set per depth tap; squares only.
        if(par.dims == 3)
        {
            if constexpr(cfg.kh == cfg.kw)
            {
                auto* partials = static_cast<float*>(workspace);
                typed_launch.template operator()<DT, true, DataType::fp32, true>(partials);
                conv2d_depthwise_wgrad_hankel_reduce_cdna4<REDUCE_SLICES, DW, true>
                    <<<dim3(divup(dw_elems, WAVE_SIZE), dp.kd),
                       REDUCE_SLICES * WAVE_SIZE,
                       0,
                       stream>>>(static_cast<ToType<DW>*>(out),
                                 partials,
                                 dw_elems,
                                 (int)(parts / dp.kd),
                                 par.c,
                                 par.kh * par.kw);
            }
        }
        else if(parts > 1)
        {
            auto* partials = static_cast<float*>(workspace);
            typed_launch.template operator()<DT, /*PARTITIONED=*/true, DataType::fp32>(partials);

            conv2d_depthwise_wgrad_hankel_reduce_cdna4<REDUCE_SLICES, DW, false>
                <<<divup(dw_elems, WAVE_SIZE), REDUCE_SLICES * WAVE_SIZE, 0, stream>>>(
                    static_cast<ToType<DW>*>(out),
                    partials,
                    dw_elems,
                    (int)parts,
                    par.c,
                    par.kh * par.kw);
        }
        else
        {
            typed_launch.template operator()<DT, /*PARTITIONED=*/false, DW>(
                static_cast<ToType<DW>*>(out));
        }
    };
    // A 4-byte config instantiates only tf32, a 2-byte one only fp16 and bf16.
    if constexpr(cfg.elem_bytes == 4)
    {
        dispatch.template operator()<DataType::tf32, DataType::fp32>();
    }
    else
    {
        // is_applicable admits dW in fp32 or in the input's own type, nothing else.
        const bool narrow_dw = par.weight_grad_type != DataType::fp32;
        if(par.input_type == DataType::bf16)
            narrow_dw ? dispatch.template operator()<DataType::bf16, DataType::bf16>()
                      : dispatch.template operator()<DataType::bf16, DataType::fp32>();
        else
            narrow_dw ? dispatch.template operator()<DataType::fp16, DataType::fp16>()
                      : dispatch.template operator()<DataType::fp16, DataType::fp32>();
    }
}

class Depthwise_Wgrad_Hankel_ConvKernel : public DepthwiseConvKernel
{
public:
    constexpr Depthwise_Wgrad_Hankel_ConvKernel(const Config& cfg, LaunchFn launch_fn)
        : DepthwiseConvKernel(launch_fn)
        , cfg_(cfg)
    {
    }

    std::string_view name() const override { return "depthwise_wgrad_hankel"; }

    KVDescriptor config_descriptor() const override { return ConfigMatcher(cfg_); }

    bool supports_dims(int dims) const override { return dims == 2 || dims == 3; }

    // Does not chain to DepthwiseConvKernel::is_applicable. That base serves the
    // fprop/dgrad families and requires output == input, which never holds once
    // the result is an fp32 weight gradient.
    bool is_applicable(const Device& /*dev*/, const ConvParams& par) const override
    {
        if(par.direction != Direction::Wgrad)
            return false;
        if(par.input_type != DataType::fp16 && par.input_type != DataType::bf16 &&
           par.input_type != DataType::tf32)
            return false;
        // A tf32 x comes with the fp32 dY its forward pass writes.
        const DataType want_grad =
            par.input_type == DataType::tf32 ? DataType::fp32 : par.input_type;
        if(par.output_grad_type() != want_grad)
            return false;
        // fp32 accumulation either way; a narrow dW is rounded once, at the final store.
        // tf32 takes fp32 dW only.
        const bool dw_ok =
            par.weight_grad_type == DataType::fp32 ||
            (par.input_type != DataType::tf32 && par.weight_grad_type == par.input_type);
        if(!dw_ok)
            return false;
        if(par.order != TensorOrder::NHWC)
            return false;
        // Decimation keeps the Hankel structure only for an integer phase count, and the
        // config table carries kernels for stride 1 and 2 alone.
        if(par.stride_h != par.stride_w)
            return false;
        if(par.stride_h != 1 && par.stride_h != 2)
            return false;
        if(par.dilation_h != 1 || par.dilation_w != 1)
            return false;
        if(par.pad_h > par.kh - 1 || par.pad_w > par.kw - 1)
            return false;
        // Buffer offsets and NUM_RECORDS are both 32 bits, but each descriptor is
        // based at one row, so only a single row has to be addressable.
        const int64_t elem    = (int64_t)sizeof_data_type(par.input_type);
        const int64_t in_row  = (int64_t)par.w * par.c * elem;
        const int64_t out_row = (int64_t)par.q * par.k * elem;
        if(in_row > INT32_MAX || out_row > INT32_MAX)
            return false;
        return true;
    }

    bool is_valid_config(const Device& dev, const ConvParams& par) const override
    {
        if(!tile_fits(cfg_, par))
            return false;
        if(par.dims == 3 && cfg_.kh != cfg_.kw)
            return false;
        if(launch_grid_for(cfg_, par, dev.cu_count).z > max_grid_z)
            return false;

        // Of the configs that fit, keep only those whose grid can fill the machine:
        // configs[] is widest-tile-first, so the first survivor is the widest that does.
        const int wg = workgroups(cfg_, par);
        // Two workgroups per CU, not the four this tile can keep resident: at four the
        // choice flips one step too early and costs 8% on n32 h32 c128.
        if(wg >= 2 * dev.cu_count)
            return true;
        // No config fills the machine, so the most-split one is the least bad.
        return wg >= most_workgroups(par);
    }

    LaunchParams get_launch_params(const Device& dev, const ConvParams& par) const override
    {
        LaunchParams launch;
        launch.grid       = launch_grid_for(cfg_, par, dev.cu_count);
        launch.block_size = dim3(cfg_.threads(), 1, 1);
        return launch;
    }

    // Supplies the blocked accumulation depth; the default would use all N*P*Q products.
    void
    get_tolerance(const Device& dev, const ConvParams& par, float& atol, float& rtol) const override
    {
        get_mixed_precision_tolerance(par, accumulation_depth(cfg_, par, dev.cu_count), atol, rtol);
    }

    // One slot per partition, so this grows with the grid; a lone conv2d partition stores
    // straight into the caller's dW and needs none.
    size_t get_workspace_size(const Device& dev, const ConvParams& par) const override
    {
        const size_t parts = partition_count(get_launch_params(dev, par).grid);
        if(parts <= 1 && par.dims != 3)
            return 0;
        return parts * par.k * par.kh * par.kw * sizeof(float);
    }

private:
    const Config& cfg_;
};

} // namespace depthwise_wgrad_hankel
} // namespace hipconv::cdna4
