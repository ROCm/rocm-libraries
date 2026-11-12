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
inline dim3 grid_for(const Config& cfg, const ConvParams& par)
{
    return dim3(divup(par.c, BLOCK_C),
                divup(par.q, cfg.block_q()),
                par.n * divup(par.h, cfg.rows_per_chunk));
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
inline dim3 launch_grid_for(const Config& cfg, const ConvParams& par)
{
    dim3 grid        = grid_for(cfg, par);
    const int planes = (int)(grid.x * grid.y);
    const int target = maximum(1, FOLD_MIN_WORKGROUPS_PER_CU * cu_count() / maximum(1, planes));
    grid.z           = (unsigned)minimum((int)grid.z, target);
    return grid;
}

// Whether a shape can use a config at all, ignoring how well it fills the machine.
inline bool tile_fits(const Config& cfg, const ConvParams& par)
{
    if(par.direction != cfg.direction)
        return false;
    if(par.kh != cfg.kh || par.kw != cfg.kw)
        return false;
    // One stride for both axes: the operands are decimated by a single constant.
    if(par.stride_h != par.stride_w || par.stride_h != cfg.stride)
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
// docs/algorithms/direct/direct-wgrad-tolerance.md derives puts the depth orders of
// magnitude below the N*P*Q products the default tolerance would charge.
inline size_t accumulation_depth(const Config& cfg, const ConvParams& par)
{
    const dim3 grid    = launch_grid_for(cfg, par);
    const size_t parts = partition_count(grid);
    const size_t units = divup((int)grid_for(cfg, par).z, (int)grid.z);

    // Every K step is charged a rounding: v_mfma_f32_4x4x4's exact block is not measured, the
    // way the 16x16's is. It is 3 against a chain in the hundreds, so measuring it would not
    // move the bound.
    constexpr size_t within_mfma = MFMA_K - 1;
    // One mma per q tile per row of the chunk, each stepping srcC, for every unit the
    // workgroup folds. Rows whose tap reads the zero plane add nothing and cannot round, so
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

template <Config cfg, DataType DT, bool PARTITIONED, DataType DW>
__device__ void conv2d_depthwise_wgrad_hankel_nhwc_impl(const ToType<DT>* __restrict__ input,
                                                        const ToType<DT>* __restrict__ delta,
                                                        ToType<DW>* __restrict__ wgrad,
                                                        int N,
                                                        int C,
                                                        int hi,
                                                        int wi,
                                                        int ho,
                                                        int wo,
                                                        int py,
                                                        int px)
{
    // Partials stay fp32 for the fold; only a store into the caller's dW narrows.
    static_assert(!PARTITIONED || DW == DataType::fp32, "partials are fp32");

    using datatype_t      = ToType<DT>;
    using dw_t            = ToType<DW>;
    using datatypex4_t    = std::conditional_t<DT == DataType::bf16, bf16x4_t, fp16x4_t>;
    using int16x4_t       = __attribute__((ext_vector_type(4))) short;
    using uint32x2_t      = __attribute__((ext_vector_type(2))) unsigned int;
    using TransposeLayout = TransposeLDSLayout<4, 4, 16>;
    // The transpose read's phase is 4 x by 8 c4, which is the spread SwizzleT_wgrad is
    // bank-free on; SwizzleT is built for the untransposed b128 read and is 2-way here.
    using Sw = SwizzleT_wgrad<BLOCK_C>;

    // MFMA operands as bunnies matrices: a TAPS x TAPS block per channel, batched
    // over the CHAN_PER_WAVE channels a wave covers.
    constexpr auto half_fmt = (DT == DataType::bf16) ? bunnies::fpfmt::e8m7 : bunnies::fpfmt::e5m10;
    using mat_a             = arch::matrix<half_fmt, TAPS, MFMA_K, bunnies::use::A, CHAN_PER_WAVE>;
    using mat_b             = arch::matrix<half_fmt, MFMA_K, TAPS, bunnies::use::B, CHAN_PER_WAVE>;
    using mat_acc =
        arch::matrix<bunnies::fpfmt::e8m23, TAPS, TAPS, bunnies::use::Acc, CHAN_PER_WAVE>;

    // Tile geometry. S splits a row by column residue so the decimated operand is
    // again a Hankel matrix, and PLANE pads main_delta's slots off the bank period.
    constexpr int BLOCK_Q = cfg.block_q();
    constexpr int STRIDE  = cfg.main_stride();
    constexpr int S       = cfg.stride;
    constexpr int PLANE   = BLOCK_C * STRIDE + 12;

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
    constexpr int MAIN_IN_ELEMS     = BLOCK_C * IN_STRIDE;

    // Delta rows resident at once, one per tap phase, plus a plane of zeros so one
    // address select stands in for masking off the taps with no row this row.
    constexpr int DELTA_RING       = divup(cfg.kh, S);
    constexpr int ZERO_SLOT        = DELTA_RING;
    constexpr int DELTA_PLANES     = DELTA_RING + (S > 1 ? 1 : 0);
    constexpr int MAIN_DELTA_ELEMS = DELTA_PLANES * PLANE;

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
    using narrow_pack_t   = bunnies::packed_type<NVEC*(int)sizeof(datatype_t)>;
    using narrow_elems_t  = std::array<datatype_t, NVEC>;

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

    // A register array is only addressable by a compile-time index, so the narrow path
    // dispatches its buffer on a static_for over DEPTH rather than indexing with buf. It
    // never touches stage_in or stage_delta, which is why those stay at a dummy element.

    // Channel-major operand tiles, main_in[c][x] and main_delta[p % kh][c][x]. Neither
    // is double buffered: the second copy costs a workgroup of occupancy, measured -11%.
    __shared__ uint4 stage_in[STAGE_IN_TOTAL];
    __shared__ uint4 stage_delta[STAGE_DELTA_TOTAL];
    __shared__ __align__(16) datatype_t main_in[MAIN_IN_ELEMS];
    __shared__ __align__(8) datatype_t main_delta[MAIN_DELTA_ELEMS];
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
    const int z_units = N * divup(hi, cfg.rows_per_chunk);

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
    // reads is the same in every group.
    const datatype_t* a_base =
        main_in + (size_t)lane_chan * IN_STRIDE + (size_t)(lane_s % S) * HALF;

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

    // One image per descriptor, rebased on each unit's own n. Buffer offsets are 32 bits
    // and NUM_RECORDS is too, so spanning the whole tensor would cap N * H * W * C at 2 GB;
    // rebasing moves the batch stride into the 64-bit base and leaves the offsets to cover
    // a single image, which is what the loads below actually reach.
    auto input_elems = static_cast<int64_t>(hi) * wi * C;
    auto input_bytes = input_elems * sizeof(datatype_t);
    auto delta_elems = static_cast<int64_t>(ho) * wo * C;
    auto delta_bytes = delta_elems * sizeof(datatype_t);
    auto input_rsrc  = arch::make_buffer(input, input_elems);
    auto delta_rsrc  = arch::make_buffer(delta, delta_elems);

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
        input_global_offsets[pass] = static_cast<uint32_t>(input_bytes);
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
                    : static_cast<uint32_t>(input_bytes);
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
        delta_global_offsets[pass] = static_cast<uint32_t>(delta_bytes);
        if(active)
        {
            int col                    = Sw::x(lds_idx);
            int c8_idx                 = Sw::c8(lds_idx);
            int global_q               = block_q + col;
            int global_c8              = block_c_base / 8 + c8_idx;
            delta_global_offsets[pass] = (global_q < wo && global_c8 < C8)
                                             ? sizeof(uint4) * ((size_t)global_q * C8 + global_c8)
                                             : static_cast<uint32_t>(delta_bytes);
        }
    }

    // Rows are addressed in whole pixels either way, but only the wide path can
    // count them in uint4: a narrow row is 2 * wi * C bytes, which is not one.
    const size_t input_row_stride =
        NARROW ? (size_t)wi * C * sizeof(datatype_t) : (size_t)wi * C8 * sizeof(uint4);
    const size_t delta_row_stride =
        NARROW ? (size_t)wo * C * sizeof(datatype_t) : (size_t)wo * C8 * sizeof(uint4);

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
                   : static_cast<uint32_t>(input_bytes);
            narrow_in_lds[pass] = ch * IN_STRIDE + phase * HALF + u;
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
                   : static_cast<uint32_t>(delta_bytes);
            narrow_delta_lds[pass] = ch * STRIDE + x;
        }
    }

    narrow_pack_t in_regs[DEPTH][NARROW_IN_PASSES == 0 ? 1 : NARROW_IN_PASSES];
    narrow_pack_t delta_regs[DEPTH][NARROW_DELTA_PASSES == 0 ? 1 : NARROW_DELTA_PASSES];

    // Every wave issues every pass, so vmcnt means the same thing in all of them; rows
    // and lanes past the tile fetch beyond NUM_RECORDS, which returns zero.
    auto load_input_global = [&](int buf, int y) {
        const bool row_ok = y >= 0 && y < hi;
        if constexpr(NARROW)
        {
            // buf is workgroup-uniform, so this is one scalar compare ahead of the loads,
            // which still issue a row early and stay in flight across sync_staged.
            static_for<DEPTH>([&]<int B>() {
                if(buf != B)
                    return;
                static_for<NARROW_IN_PASSES>([&]<int P>() {
                    uint32_t off = row_ok ? narrow_in_offsets[P] + y * input_row_stride
                                          : static_cast<uint32_t>(input_bytes);
                    arch::buffer_load<NVEC * sizeof(datatype_t)>::load(
                        input_rsrc, &in_regs[B][P], off, 0);
                });
            });
        }
        else
        {
            for(int pass = 0; pass < INPUT_LOAD_PASSES; pass++)
            {
                uint32_t off = row_ok ? input_global_offsets[pass] + y * input_row_stride
                                      : static_cast<uint32_t>(input_bytes);
                arch::buffer_load_lds<16>::load(
                    input_rsrc, input_lds_addrs[pass] + buf * input_buf_stride[pass], off, 0);
            }
        }
    };

    auto load_delta_global = [&](int buf, int p) {
        const bool row_ok = p >= 0 && p < ho;
        if constexpr(NARROW)
        {
            static_for<DEPTH>([&]<int B>() {
                if(buf != B)
                    return;
                static_for<NARROW_DELTA_PASSES>([&]<int P>() {
                    uint32_t off = row_ok ? narrow_delta_offsets[P] + p * delta_row_stride
                                          : static_cast<uint32_t>(delta_bytes);
                    arch::buffer_load<NVEC * sizeof(datatype_t)>::load(
                        delta_rsrc, &delta_regs[B][P], off, 0);
                });
            });
        }
        else
        {
            for(int pass = 0; pass < DELTA_LOAD_PASSES; pass++)
            {
                uint32_t off = row_ok ? delta_global_offsets[pass] + p * delta_row_stride
                                      : static_cast<uint32_t>(delta_bytes);
                arch::buffer_load_lds<16>::load(
                    delta_rsrc, delta_lds_addrs[pass] + buf * delta_buf_stride[pass], off, 0);
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
                    static_for<NVEC>(
                        [&]<int V>() { main_in[narrow_in_lds[P] + V * IN_STRIDE] = v[V]; });
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
                arch::ds_store_b64::store(&main_in[lane * IN_STRIDE + xg], &v[G]);
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
                        main_delta[slot * PLANE + narrow_delta_lds[P] + V * STRIDE] = v[V];
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
                arch::ds_store_b64::store(&main_delta[slot * PLANE + lane * STRIDE + xg], &v[G]);
            });
        }
    };

    // Delta rows live in a ring keyed by the absolute row index, so the slot a tap
    // reads is fixed by the row it pairs with and no data ever moves.
    auto ring_slot = [&](int p) { return ((p % DELTA_RING) + DELTA_RING) % DELTA_RING; };

    // The freshest delta row an input row pairs with. Only every S-th input row
    // brings a new one; y + py is never negative, so the division floors.
    auto delta_row = [&](int y) { return (y + py) / S; };

    // Zero the plane the mismatched-phase taps read. One write per thread, made
    // visible by the barrier the prologue's first row already carries.
    if constexpr(S > 1)
    {
        for(int i = tid; i < PLANE; i += cfg.threads())
            main_delta[ZERO_SLOT * PLANE + i] = static_cast<datatype_t>(0.f);
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
    for(int unit = bidx.z; unit < z_units; unit += gridDim.z)
    {
        block_n         = unit % N;
        const int chunk = unit / N;
        y0              = chunk * cfg.rows_per_chunk;
        y1              = std::min(hi, y0 + cfg.rows_per_chunk);
        input_rsrc =
            arch::make_buffer(input + static_cast<int64_t>(block_n) * input_elems, input_elems);
        delta_rsrc =
            arch::make_buffer(delta + static_cast<int64_t>(block_n) * delta_elems, delta_elems);

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
            // exists only when S divides it; the other phases read the zero plane.
            int d_slot[RG];
            static_for<RG>([&]<int RGI>() {
                const int num = y + py - (RGI * TAPS + lane_r);
                d_slot[RGI]   = (S == 1 || num % S == 0) ? ring_slot(num / S) : ZERO_SLOT;
            });

            __attribute__((aligned(16))) uint32_t win[4 * WIN_READS];
            bunnies::reg_tile<mat_b, RG, cfg.q_tiles> b;
            static_for<WIN_READS>(
                [&]<int J>() { arch::ds_load_b128::load(a_base + J * 8, &win[4 * J]); });
            // Column c of the batched B operand carries channel c / TAPS, so the map
            // turns the operand's own coordinate into the channel's row.
            bunnies::load_tile<arch::ds_load_b64>(b, main_delta, [&](int rg, int t, int, int c) {
                return d_slot[rg] * PLANE + (wave * CHAN_PER_WAVE + c / TAPS) * STRIDE + t * MFMA_K;
            });
            // Shift the window down to a residue the perms can reach, in place and
            // ascending so each word is read before it is overwritten.
            if constexpr(PRESHIFT)
                static_for<WIN_WORDS>(
                    [&]<int J>() { win[J] = __builtin_amdgcn_perm(win[J + 1], win[J], pre_sel); });
            static_for<SG>([&]<int SGI>() {
                static_for<cfg.q_tiles>([&]<int T>() {
                    // Both halves of the A fragment are the window taken this lane's tap
                    // late, so each is one byte permute of an adjacent register pair.
                    constexpr int K = 2 * T + SGI * SG_WORDS;
                    uint32x2_t a    = {__builtin_amdgcn_perm(win[K + 1], win[K], res_sel),
                                       __builtin_amdgcn_perm(win[K + 2], win[K + 1], res_sel)};
                    mat_a af{__builtin_bit_cast(datatypex4_t, a)};
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
        dst += (size_t)partition * C * cfg.kh * cfg.kw;
    }

    // A tap's 16 lanes hold 16 consecutive channels, so channel-major partials put its store
    // on one 64 B sector. Only a lone partition writes the caller's dW, which keeps [c][r][s].
    const size_t tap_stride = PARTITIONED ? (size_t)C : 1;

    dw_t* dw = dst + (PARTITIONED ? (size_t)c_global : (size_t)c_global * cfg.kh * cfg.kw);
    static_for<RG>([&]<int RGI>() {
        // The taps a group holds past the filter carry a delta row this row never
        // paired with, so they are dropped rather than stored.
        const int r = RGI * TAPS + lane_tap;
        if(r >= cfg.kh)
            return;
        static_for<SG>([&]<int SGI>() {
            static_for<TAPS>([&]<int I>() {
                if constexpr(SGI * TAPS + I < cfg.kw)
                    dw[(r * cfg.kw + SGI * TAPS + I) * tap_stride] =
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
template <int SLICES, DataType DW>
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
    const int tap                             = i / C;
    wgrad[(size_t)(i - tap * C) * taps + tap] = static_cast<ToType<DW>>(sum);
}

// Waves splitting the partition axis above, at the most a 1024-thread group can carry. The
// fold reads partials the main kernel has just evicted from L2, so it is latency-exposed and
// wants every wave it can get: 16 over 8 is worth 11.4% where the fold owns most of a run.
constexpr int REDUCE_SLICES = 16;

// Without the bound the compiler sizes registers for a 1024-thread group and caps
// this kernel at 128 VGPRs, which spills the 9x9 and 11x11 accumulators to scratch.
template <Config cfg, DataType DT, bool PARTITIONED, DataType DW>
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
    int py,
    int px)
{
    if(__builtin_amdgcn_is_invocable(__builtin_amdgcn_mfma_f32_4x4x4f16) &&
       __builtin_amdgcn_is_invocable(__builtin_amdgcn_mfma_f32_4x4x4bf16_1k) &&
       __builtin_amdgcn_is_invocable(__builtin_amdgcn_raw_ptr_buffer_load_lds) &&
       __builtin_amdgcn_is_invocable(__builtin_amdgcn_ds_read_tr16_b64_v4i16))
    {
        conv2d_depthwise_wgrad_hankel_nhwc_impl<cfg, DT, PARTITIONED, DW>(
            input, delta, wgrad, N, C, hi, wi, ho, wo, py, px);
    }
}

template <Config cfg>
void launch_impl(const LaunchParams& lp,
                 const ConvParams& par,
                 const void* in,
                 const void* wei,
                 void* out,
                 void* workspace,
                 hipStream_t stream)
{
    const int dw_elems = par.k * par.kh * par.kw;
    const size_t parts = partition_count(lp.grid);

    auto typed_launch = [&]<DataType DT, bool PARTITIONED, DataType DW>(ToType<DW>* dst) {
        using dtype = ToType<DT>;
        conv2d_depthwise_wgrad_hankel_nhwc_cdna4<cfg, DT, PARTITIONED, DW>
            <<<lp.grid, lp.block_size, 0, stream>>>(static_cast<const dtype*>(in),
                                                    static_cast<const dtype*>(wei),
                                                    dst,
                                                    par.n,
                                                    par.c,
                                                    par.h,
                                                    par.w,
                                                    par.p,
                                                    par.q,
                                                    par.pad_h,
                                                    par.pad_w);
    };
    // No zeroing on either path: every element the reduce or the caller reads was stored
    // rather than accumulated into, which is also what lets the cast to DW ride on that store.
    auto dispatch = [&]<DataType DT, DataType DW>() {
        if(parts > 1)
        {
            auto* partials = static_cast<float*>(workspace);
            typed_launch.template operator()<DT, /*PARTITIONED=*/true, DataType::fp32>(partials);

            conv2d_depthwise_wgrad_hankel_reduce_cdna4<REDUCE_SLICES, DW>
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
    // is_applicable admits dW in fp32 or in the input's own type, nothing else.
    const bool narrow_dw = par.weight_grad_type != DataType::fp32;
    if(par.input_type == DataType::bf16)
        narrow_dw ? dispatch.template operator()<DataType::bf16, DataType::bf16>()
                  : dispatch.template operator()<DataType::bf16, DataType::fp32>();
    else
        narrow_dw ? dispatch.template operator()<DataType::fp16, DataType::fp16>()
                  : dispatch.template operator()<DataType::fp16, DataType::fp32>();
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

    // Does not chain to DepthwiseConvKernel::is_applicable. That base serves the
    // fprop/dgrad families and requires output == input, which never holds once
    // the result is an fp32 weight gradient.
    bool is_applicable(const ConvParams& par) const override
    {
        if(par.direction != Direction::Wgrad)
            return false;
        if(par.input_type != DataType::fp16 && par.input_type != DataType::bf16)
            return false;
        if(par.output_grad_type() != par.input_type)
            return false;
        // fp32 accumulation either way; a narrow dW is rounded once, at the final store.
        if(par.weight_grad_type != DataType::fp32 && par.weight_grad_type != par.input_type)
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
        // based at one image, so only a single n has to be addressable. The bound is
        // INT32_MAX rather than UINT32_MAX because a row-sink offset adds a whole
        // image's worth of row strides on top before going out of range.
        const int64_t elem      = par.input_type == DataType::fp32 ? 4 : 2;
        const int64_t in_image  = (int64_t)par.h * par.w * par.c * elem;
        const int64_t out_image = (int64_t)par.p * par.q * par.k * elem;
        if(in_image > INT32_MAX || out_image > INT32_MAX)
            return false;
        return true;
    }

    bool is_valid_config(const ConvParams& par) const override
    {
        if(!tile_fits(cfg_, par))
            return false;

        // Of the configs that fit, keep only those whose grid can fill the machine:
        // configs[] is widest-tile-first, so the first survivor is the widest that does.
        const int wg = workgroups(cfg_, par);
        // Two workgroups per CU, not the four this tile can keep resident: at four the
        // choice flips one step too early and costs 8% on n32 h32 c128.
        if(wg >= 2 * cu_count())
            return true;
        // No config fills the machine, so the most-split one is the least bad.
        return wg >= most_workgroups(par);
    }

    LaunchParams get_launch_params(const ConvParams& par) const override
    {
        LaunchParams launch;
        launch.grid       = launch_grid_for(cfg_, par);
        launch.block_size = dim3(cfg_.threads(), 1, 1);
        return launch;
    }

    // Supplies the blocked accumulation depth; the default would use all N*P*Q products.
    void get_tolerance(const ConvParams& par, float& atol, float& rtol) const override
    {
        get_mixed_precision_tolerance(par, accumulation_depth(cfg_, par), atol, rtol);
    }

    // One slot per partition, so this grows with the grid; a lone partition stores straight
    // into the caller's dW and needs none.
    size_t get_workspace_size(const ConvParams& par) const override
    {
        const size_t parts = partition_count(get_launch_params(par).grid);
        if(parts <= 1)
            return 0;
        return parts * par.k * par.kh * par.kw * sizeof(float);
    }

private:
    const Config& cfg_;
};

} // namespace depthwise_wgrad_hankel
} // namespace hipconv::cdna4
