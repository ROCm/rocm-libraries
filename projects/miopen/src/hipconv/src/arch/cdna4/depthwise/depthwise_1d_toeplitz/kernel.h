#pragma once
// CDNA4 depthwise convolution -- 1D column-Toeplitz + row-streaming kernel.
//
// Kernel body and per-config launch_impl<cfg> live here; the host-safe Config type and
// configs[] table live in config_table.h, which hipconv_autoshard reads to generate the
// launch_impl<> instantiations and kernel-span export across depthwise_1d_toeplitz_shard*.cpp.
//
// Depthwise specialization of grouped_8c: that kernel runs an F(2,3) horizontal Toeplitz
// on a dense 16x16x32 MFMA with an 8x8 channel-mixing block g[k][c]. Depthwise uses a
// DIAGONAL g (g[k][c] = w[k] iff c==k), so one MFMA still yields 8 channels x 32 columns
// and the off-diagonal work is discarded for free (depthwise is memory-bound). Only the
// weight prologue differs (a one-hot diagonal tap); the rest is inherited.
//
// Scope: kw in {3,5,7,9,11}; kh is the vertical loop bound. Fprop stride 1/2; Dgrad =
// rot180 correlation of dY (stride 2 runs it over dY upsampled 2x via cfg.dilation). 5x5
// is also served by depthwise_2d_toeplitz; this path wins large-spatial.

#include "conv_kernel.h"
#include "config_table.h"
#include "depthwise_conv_kernel.h"
#include "../../bunnies_cdna4.hpp"
#include "../../grouped/grouped_8c_transforms.h"
#include "matrix_layout.h"
#include "swizzle.h"
#include "detail.h"
#include "types.h"
#include "mathutil.h"
#include "launch_params.h"
#include "packed_ops.h"
#include "memory.h"
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <cstddef>

namespace hipconv::cdna4
{
namespace depthwise_1d_toeplitz
{

using namespace hipconv;

using arch = bunnies::arch_cdna4;

// Scatter one element per lane into a lane-chosen register slot.
//
// Builds the one-hot diagonal weight operand: each lane passing `pred` issues one coalesced
// load routed to `slot(mb,nb,lane)`, and the rest keep their pre-zeroed contents -- which
// avoids load_sparse's per-slot scan over an operand whose live slot is known in closed form.
template <typename LoadInst,
          bunnies::reg_tile_concept RegTile,
          typename MemT,
          typename SlotMap,
          typename OffsetMap,
          typename Pred>
__device__ void load_scatter(RegTile& rt, MemT* base, SlotMap&& slot, OffsetMap&& off, Pred&& pred)
{
    using elem_t   = typename RegTile::matrix::base_storage_t;
    const int lane = bunnies::lane_id();
#pragma unroll
    for(int mb = 0; mb < RegTile::row_blocks; ++mb)
    {
#pragma unroll
        for(int nb = 0; nb < RegTile::col_blocks; ++nb)
        {
            if(pred(mb, nb, lane))
            {
                elem_t v;
                LoadInst::load(base + off(mb, nb, lane), &v);
                rt.block(mb, nb).data[slot(mb, nb, lane)] = v;
            }
        }
    }
}

template <Config cfg, DataType DT>
__device__ void conv2d_depthwise_1d_toeplitz_nhwc_impl(const ToType<DT>* __restrict__ in,
                                                       const ToType<DT>* __restrict__ wei,
                                                       ToType<DT>* __restrict__ out,
                                                       int N,
                                                       int C,
                                                       int hi,
                                                       int wi,
                                                       int ho,
                                                       int wo,
                                                       int py,
                                                       int px)
{
    using namespace grouped_8c_transforms;
    using element_t = ToType<DT>;
    using Sw        = SwizzleT<cfg.block_c()>;

    // One 16x16x32 MFMA per horizontal tap.
    //
    // A = one-hot diagonal weights (16x32), B = streamed input (32x16), Acc = fp32 ring
    // (16x16); layouts match the raw fp16x8/fp32x4 vectors, so these wrappers are zero-cost.
    constexpr auto half_fmt = (DT == DataType::bf16) ? bunnies::fpfmt::e8m7 : bunnies::fpfmt::e5m10;
    using mat_a             = arch::matrix<half_fmt, 16, 32, bunnies::use::A>;
    using mat_b             = arch::matrix<half_fmt, 32, 16, bunnies::use::B>;
    using mat_acc           = arch::matrix<bunnies::fpfmt::e8m23, 16, 16, bunnies::use::Acc>;
    // acc narrowed to element_t for the LDS staging store
    using mat_out = arch::matrix<half_fmt, 16, 16, bunnies::use::Acc>;

    using ResultLayout = MatrixLayout<16, 16, 1, float>;

    constexpr int CH_PER_UINT4 = 8;
    constexpr int GROUP_UINT4  = cfg.group_size / CH_PER_UINT4; // == 1

    constexpr int BLOCK_W = BLOCK_Q + (cfg.kw - 1);
    // Dense horizontal passes: one MFMA contracts 4 input positions (K==32); a tile
    // spans kw+1 positions, so ceil((kw+1)/4) passes (kw=3 -> 1, kw=5/7 -> 2).
    constexpr int KWP = ((cfg.kw + 1) + 3) / 4;

    // Structured-sparse (smfmac) path: the one-hot diagonal fits 2:4 sparsity trivially.
    //
    // One tap per 4-wide K group. An smfmac contracts 8 positions (K==64), halving the
    // pass count (KSP == ceil(KWP/2)). Enabled only where it wins (KWP>=2, fp16/bf16);
    // kw=3 stays dense (a K==64 pass would waste half its contraction).
    constexpr bool USE_SPARSE = (KWP >= 2) && (DT == DataType::fp16 || DT == DataType::bf16);
    // Sparse passes: each smfmac spans 8 input positions (== 2 dense passes).
    constexpr int KSP = (KWP + 1) / 2;
    // Weights are staged 1:4 (one tap per group of 4) and cast to the 2:4 form smfmac
    // consumes; see bunnies_cdna4.hpp.
    using smat_a_1of4 =
        arch::sparse_matrix<half_fmt, 16, 64, bunnies::use::A, bunnies::sparsity::n1of4>;
    using smat_a_2of4 =
        arch::sparse_matrix<half_fmt, 16, 64, bunnies::use::A, bunnies::sparsity::n2of4>;
    using smat_b = arch::matrix<half_fmt, 64, 16, bunnies::use::B>;

    constexpr int BLOCK_C_UINT4  = cfg.block_c() / CH_PER_UINT4;
    constexpr int BLOCK_GROUPS   = cfg.waves_per_wg;
    constexpr int OUTPUT_BLOCK_Q = BLOCK_Q / cfg.stride;
    constexpr int STORE_VECS     = OUTPUT_BLOCK_Q * BLOCK_C_UINT4;

    // Batch-folded W tiling: the 32-wide output tile splits into F sub-images.
    //
    // Each sub-image is W_SUB = BLOCK_Q/F columns from a different (strided) batch image
    // with its own SEG_W-wide swizzled LDS segment + halo, so the Toeplitz never crosses
    // an image boundary. F == 1 is the original single-segment path.
    constexpr int F         = cfg.w_fold;
    constexpr int W_SUB     = cfg.w_sub();           // outputs per sub-image (BLOCK_Q / F)
    constexpr int SUBTILES  = cfg.subtiles();        // MFMA tile-columns per sub-image (16 / F)
    constexpr int SEG_W     = W_SUB + (cfg.kw - 1);  // input columns per segment
    constexpr int SEG_UINT4 = BLOCK_C_UINT4 * SEG_W; // uint4 per segment (== per image)

    // Input ring depth (cfg.lds_buffers): prefetch PREFETCH_AHEAD rows into distinct
    // LDS buffers so compute hides each row's load latency.
    //
    // Triple (3) keeps 2 rows in flight; double (2) keeps 1, trading latency hiding for
    // 33% less input LDS (higher occupancy). Tuned per shape.
    constexpr int NUM_INPUT_LDS_BUFFERS   = cfg.lds_buffers;
    constexpr int PREFETCH_AHEAD          = NUM_INPUT_LDS_BUFFERS - 1; // rows in flight
    constexpr int INPUT_LDS_BUFFER_UINT4  = F * SEG_UINT4; // == BLOCK_C_UINT4 * BLOCK_W for F==1
    constexpr int OUTPUT_LDS_BUFFER_UINT4 = BLOCK_C_UINT4 * OUTPUT_BLOCK_Q;
    // Load passes per segment (one buffer resource / image each).
    constexpr int LOAD_PASSES = (SEG_UINT4 + cfg.block_size() - 1) / cfg.block_size();
    // Narrow path streams one channel per lane, walking BLOCK_W * group_size elements.
    constexpr int NARROW_LOAD_ELEMS = BLOCK_W * cfg.group_size;
    constexpr int NARROW_LOAD_PASSES =
        (NARROW_LOAD_ELEMS + cfg.block_size() - 1) / cfg.block_size();
    constexpr int IO_LDS_SIZE =
        NUM_INPUT_LDS_BUFFERS * INPUT_LDS_BUFFER_UINT4 + OUTPUT_LDS_BUFFER_UINT4;

    __shared__ uint4 lds_buf[IO_LDS_SIZE];
    uint4* input_lds  = lds_buf;
    uint4* output_lds = lds_buf + NUM_INPUT_LDS_BUFFERS * INPUT_LDS_BUFFER_UINT4;

    const int tid  = threadIdx.x;
    const int wave = bunnies::wave_id();
    const int lane = bunnies::lane_id();

    const int block_q_n_idx = blockIdx.x;
    // n_fold packs eff_n_fold n-groups into blockIdx.x beside the w-tiles, capped at
    // n_groups so a small folded batch leaves no empty x-slots. Matches get_launch_params.
    const int n_groups        = (N + F - 1) / F;
    const int eff_n_fold      = (cfg.n_fold < n_groups) ? cfg.n_fold : n_groups;
    const int block_n_mod_idx = block_q_n_idx % eff_n_fold;
    const int block_q_idx     = block_q_n_idx / eff_n_fold;
    const int block_group_idx = blockIdx.y;
    // gridDim.z packs (n-group-fold-div, h-chunk).
    //
    // The batch is enumerated in n_groups slots of F images each; a workgroup owns one
    // group and packs its F images (strided by n_groups). F == 1 reduces to one image
    // per workgroup.
    const int block_z_idx     = blockIdx.z;
    const int blocks_n_fold   = (n_groups + eff_n_fold - 1) / eff_n_fold;
    const int block_g_div_idx = block_z_idx % blocks_n_fold;
    const int h_chunk_idx     = block_z_idx / blocks_n_fold;
    const int block_g_idx     = block_g_div_idx * eff_n_fold + block_n_mod_idx;
    if(block_g_idx >= n_groups)
        return;

    // Sub-image s (s in [0,F)) of this tile is batch image block_g_idx + s*n_groups.
    const int block_n       = block_g_idx; // sub-image 0 (== the whole image for F==1)
    const int block_q       = block_q_idx * W_SUB;
    const int block_group   = block_group_idx * BLOCK_GROUPS;
    const int block_k       = block_group * cfg.group_size;
    const int block_c_uint4 = block_group * GROUP_UINT4;

    const int C_UINT4 = C / CH_PER_UINT4;

    // Input-upsample factor (Dgrad stride-2 => 2, else 1): a dense stride-1 conv over a
    // virtual grid whose real (dY) samples sit at multiples of UPS, the rest zero.
    constexpr int UPS = cfg.dilation;
    const int hi_eff  = (UPS > 1) ? ((hi - 1) * UPS + 1) : hi;

    // H-tiling: split the serial vertical stream into num_h_chunks spans of kh-row
    // blocks (one workgroup each, packed into gridDim.z) so parallelism scales with H.
    //
    // Chunk>0 re-streams the preceding kh-block as warm-up to seed the acc ring, with
    // flushes gated to owned rows (each output written once); the last chunk also runs
    // the remainder and tail flush. num_h_chunks == 1 is the original whole-image stream.
    const int n_full_blocks  = hi_eff / cfg.kh;
    const int num_h_chunks   = gridDim.z / blocks_n_fold;
    const bool is_last_chunk = (h_chunk_idx == num_h_chunks - 1);
    // Floor sizing, last chunk absorbs the leftover blocks; every chunk is non-empty
    // (num_h_chunks <= n_full_blocks), so exactly one chunk owns the remainder + tail.
    const int chunk_blocks = n_full_blocks / num_h_chunks;
    const int own_blk_lo   = h_chunk_idx * chunk_blocks;
    const int own_blk_hi   = is_last_chunk ? n_full_blocks : (own_blk_lo + chunk_blocks);
    const int own_y_lo     = own_blk_lo * cfg.kh; // first owned input row
    const int stream_y_lo  = (own_blk_lo > 0) ? (own_blk_lo - 1) * cfg.kh : 0; // + warm-up
    const int main_y_hi    = own_blk_hi * cfg.kh; // main-loop end (exclusive)
    // Highest input row this chunk will touch (for prefetch bounds).
    const int end_y = is_last_chunk ? hi_eff : main_y_hi;

    const size_t wi_stride = (size_t)wi * C_UINT4;
    const size_t wo_stride = (size_t)wo * C_UINT4;

    const int wave_group = wave;

    // Weight prologue -- depthwise diagonal.
    //
    // Each lane's 8-wide operand is one-hot, carrying its single tap w[global_ch][r][s_val]
    // at c==k_val, so the g-block is diagonal. For KWP>1, pass ph covers input positions
    // [4*ph, 4*ph+4) at tap column s_val = 4*ph + gg - q (out-of-range taps stay zero).
    // Issued up front so the weight loads overlap the input-load setup + first DMA below.
    bunnies::reg_tile<mat_a, KWP, cfg.kh> weights_reg{};        // dense: off-diagonal stays 0
    bunnies::reg_tile<smat_a_2of4, KSP, cfg.kh> sweights_reg{}; // sparse: compressed diagonal + idx
    if constexpr(!USE_SPARSE)
    {
        // Each lane owns channel k_val = GT::k(row) at input position gg; for pass ph its
        // live tap column is s_val = 4*ph + gg - q_val (out-of-range or past-C lanes stay
        // zero). One coalesced load per (ph, r) routed into the lane's diagonal slot k_val
        // -- load_scatter avoids load_sparse's per-slot scan over the one-hot operand.
        const int row   = lane % 16;
        const int gg    = lane / 16; // input position within the pass (0..3)
        const int k_val = GT::k(row);
        const int q_val = GT::q(row);
        const int gch   = block_k + wave_group * cfg.group_size + k_val;
        load_scatter<arch::global_load<sizeof(element_t)>>(
            weights_reg,
            const_cast<element_t*>(wei),
            [&](int, int, int) { return k_val; }, // one-hot slot: the lane's own channel
            [&](int ph, int r, int) -> size_t {
            const int s_val = (4 * ph + gg) - q_val;
            // Dgrad correlates dY with the rot180 filter: flip the taps
            // (SizeView already swaps spatial dims and remaps pad to (k-1)-pad).
            const int r_w = (cfg.direction == Direction::Dgrad) ? (cfg.kh - 1 - r) : r;
            const int s_w = (cfg.direction == Direction::Dgrad) ? (cfg.kw - 1 - s_val) : s_val;
            return ((size_t)gch * cfg.kh + r_w) * cfg.kw + s_w;
        },
            [&](int ph, int, int) {
            const int s_val = (4 * ph + gg) - q_val;
            if(s_val < 0 || s_val >= cfg.kw)
                return false;
            // Narrow path: this lane's diagonal channel may run past C in the last
            // (partial) 8-channel group; leave it zero so its output is 0.
            if constexpr(cfg.narrow_c)
                return gch < C;
            return true;
        });
    }
    else
    {
        // Sparse diagonal tap placement.
        //
        // Row m carries channel k = m%8 at output column q = m/8, so its live position
        // within a group of 4 is m%4 and only the groups whose logical K names channel k
        // hold a tap. Logical K is 8*t + k for input position t of the pass, so pass sp
        // spans positions [8*sp,8*sp+8) and the filter column is s = 8*sp + t - q.
        using elem_t = typename smat_a_1of4::base_storage_t;
        static_for<KSP>([&]<int sp>() {
            static_for<cfg.kh>([&]<int r>() {
                smat_a_1of4 w1of4;
                w1of4.fill([](int row, int) { return row % 4; }, [&](int row, int kk) -> elem_t {
                    const int k = row % 8;
                    if(kk % 8 != k)
                        return elem_t(0);
                    const int s_val = (8 * sp + kk / 8) - row / 8;
                    if(s_val < 0 || s_val >= cfg.kw)
                        return elem_t(0);
                    const int global_ch = block_k + wave_group * cfg.group_size + k;
                    if constexpr(cfg.narrow_c)
                    {
                        if(global_ch >= C)
                            return elem_t(0);
                    }
                    const size_t wbase = (size_t)global_ch * cfg.kh * cfg.kw;
                    const int r_w      = (cfg.direction == Direction::Dgrad) ? (cfg.kh - 1 - r) : r;
                    const int s_w =
                        (cfg.direction == Direction::Dgrad) ? (cfg.kw - 1 - s_val) : s_val;
                    return elem_t(wei[wbase + (size_t)r_w * cfg.kw + s_w]);
                });
                arch::matrix_cast(sweights_reg.block(sp, r), w1of4);
            });
        });
    }

    // Rebase each buffer at its image so the per-lane 32-bit offset spans only one
    // image (hi*wi*C); the batch offset (possibly >2GB) lives in the 64-bit base.
    //
    // image_bytes is the OOB sentinel offset that makes pad / upsample-zero loads read 0.
    // Folded tiles carry one buffer per sub-image (block_g_idx + s*n_groups).
    const size_t image_elems = static_cast<size_t>(hi) * wi * C;
    const size_t image_bytes = image_elems * sizeof(element_t);
    using rsrc_t             = decltype(arch::make_buffer(in, static_cast<int64_t>(0)));
    rsrc_t input_rsrc[F];
    static_for<F>([&]<int s>() {
        const int n_s = block_g_idx + s * n_groups;
        input_rsrc[s] =
            arch::make_buffer(in + (size_t)n_s * image_elems, static_cast<int64_t>(image_elems));
    });

    // Per-pass input-load plan: active lanes, LDS destination, and buffer v-offset of
    // their dY column.
    //
    // Pad / upsample-zero columns get the OOB sentinel so the load writes 0. The v-offset
    // is image-local, identical across the F segments; only the LDS destination (segment
    // base) and buffer resource differ.
    struct InputLoadPlan
    {
        bool active[LOAD_PASSES];
        uint4* lds[F][LOAD_PASSES];
        uint32_t voffset[LOAD_PASSES];
    };
    const auto in_load = [&] {
        InputLoadPlan pl{};
        static_for<LOAD_PASSES>([&]<int p>() {
            const int loc = tid + p * cfg.block_size(); // slot within one segment
            pl.active[p]  = (loc < SEG_UINT4);
            static_for<F>([&]<int s>() { pl.lds[s][p] = &input_lds[s * SEG_UINT4 + loc]; });

            const int col        = Sw::x(loc);
            const int c_uint4_p  = Sw::c8(loc);
            const int global_col = (block_q - px) + col; // virtual-grid column
            const bool zero_col  = (UPS > 1) && ((global_col % UPS) != 0);
            const int dY_col     = (UPS > 1) ? (global_col / UPS) : global_col;
            if(pl.active[p] && global_col >= 0 && !zero_col && dY_col < wi)
                pl.voffset[p] =
                    sizeof(uint4) * ((size_t)dY_col * C_UINT4 + block_c_uint4 + c_uint4_p);
            else
                pl.voffset[p] = image_bytes;
        });
        return pl;
    }();

    // Issue an async input load for virtual-grid row y_local into ring buffer TIC_PHASE.
    //
    // Dgrad stride-2 interstitial rows (y_local % UPS != 0) are forced OOB so they read 0.
    // TIC_PHASE is a template arg so the ring base is a compile-time constant the backend
    // can disambiguate from the current-row ds_read (a runtime phase would drain vmcnt to
    // 0 before the LDS read).
    auto issue_input_load = [&]<int TIC_PHASE>(int y_local) {
        const bool row_oob  = (y_local < 0 || y_local >= hi_eff);
        const bool zero_row = !row_oob && (UPS > 1) && ((y_local % UPS) != 0);
        const int dY_row    = (UPS > 1) ? (y_local / UPS) : y_local;
        if constexpr(cfg.narrow_c)
        {
            // Narrow path is synchronous, so OOB rows just skip (no in-flight vmcnt to
            // balance); their ring slot is never computed.
            if(row_oob)
                return;
            // A column's 8-channel uint4 group isn't contiguous in global memory, so
            // stream one channel at a time.
            //
            // Lane owns LDS element e (slot e/8, channel e%8): load into a register (OOB
            // reads 0) then write to LDS. Register-staged because a b16 DMA's fixed fan-out
            // doesn't match this one-channel-per-lane swizzled layout.
            const size_t row_elem = (size_t)dY_row * wi * C;
            auto* lds_e           = reinterpret_cast<element_t*>(input_lds) +
                          TIC_PHASE * INPUT_LDS_BUFFER_UINT4 * CH_PER_UINT4;
            element_t regs[NARROW_LOAD_PASSES];
            static_for<NARROW_LOAD_PASSES>([&]<int p>() {
                const int e = tid + p * cfg.block_size();
                if(e >= NARROW_LOAD_ELEMS)
                    return;
                const int slot       = e / CH_PER_UINT4;
                const int ch         = e % CH_PER_UINT4;
                const int col        = Sw::x(slot);
                const int c_uint4_p  = Sw::c8(slot);
                const int global_col = (block_q - px) + col;
                const bool zero_col  = (UPS > 1) && ((global_col % UPS) != 0);
                const int dY_col     = (UPS > 1) ? (global_col / UPS) : global_col;
                const bool col_ok    = (global_col >= 0 && !zero_col && dY_col < wi);
                const int gch        = (block_c_uint4 + c_uint4_p) * CH_PER_UINT4 + ch;
                const bool ok        = col_ok && !zero_row && gch < C;
                const int v_offset =
                    ok ? (int)(sizeof(element_t) * (row_elem + (size_t)dY_col * C + gch))
                       : (int)image_bytes;
                arch::buffer_load<sizeof(element_t)>::load(input_rsrc[0], &regs[p], v_offset, 0);
            });
            arch::s_wait_vmcnt<0>();
            static_for<NARROW_LOAD_PASSES>([&]<int p>() {
                const int e = tid + p * cfg.block_size();
                if(e >= NARROW_LOAD_ELEMS)
                    return;
                lds_e[e] = regs[p];
            });
            arch::s_wait_lgkmcnt<0>();
            return;
        }
        // Wide path: one buffer_load_lds per active lane every call, even for OOB rows
        // (they read 0).
        //
        // A constant per-row issue count lets s_wait_vmcnt<PREFETCH_AHEAD-1> mean exactly
        // "current row landed, next PREFETCH_AHEAD-1 in flight" everywhere.
        const auto row_offset = (row_oob || zero_row) ? (size_t)image_bytes
                                                      : (size_t)dY_row * wi_stride * sizeof(uint4);
        static_for<F>([&]<int s>() {
            static_for<LOAD_PASSES>([&]<int p>() {
                if(in_load.active[p])
                {
                    arch::buffer_load_lds<16>::load(input_rsrc[s],
                                                    in_load.lds[s][p] +
                                                        TIC_PHASE * INPUT_LDS_BUFFER_UINT4,
                                                    in_load.voffset[p] + row_offset,
                                                    0);
                }
            });
        });
    };

    // Prime the ring: issue the first PREFETCH_AHEAD rows so the steady-state loop
    // always has PREFETCH_AHEAD-1 later rows in flight while computing the current one.
    static_for<PREFETCH_AHEAD>(
        [&]<int j>() { issue_input_load.template operator()<j>(stream_y_lo + j); });

    // Folded tiles: the 16 tile-columns (lane % 16) split into F sub-images of SUBTILES.
    //
    // A lane's B fragment comes from its sub-image segment (seg_off) at local column
    // input_x. F == 1 => the original single-segment mapping.
    const int seg_lane = (lane % 16) / SUBTILES; // sub-image index of this lane
    const int n_local  = (lane % 16) % SUBTILES; // tile-column within the sub-image
    const int seg_off  = seg_lane * SEG_UINT4;   // this lane's segment base in LDS
    // Clamp reads to a segment's last column: columns >= SEG_W are always zero-weight
    // taps, so this keeps B finite (else the spill hits uninitialized LDS -> 0*NaN).
    const auto seg_col = [](int c) { return c < SEG_W ? c : (SEG_W - 1); };
    // B fragment LDS uint4 offsets, precomputed once (recomputing in the load hot
    // loop regresses sparse badly). Dense reads one uint4/pass; sparse two, because
    // the B map splits a lane's K into a low and a high 32-deep half, 4 input
    // columns apart.
    int input_lds_offset_pass[KWP];
    int sinput_lds_offset[KSP][2];
    if constexpr(USE_SPARSE)
    {
        const int sinput_x = 2 * n_local + (lane / 16);
        static_for<KSP>([&]<int sp>() {
            sinput_lds_offset[sp][0] =
                seg_off + Sw::offset_uint4(seg_col(sinput_x + 8 * sp), wave_group * GROUP_UINT4);
            sinput_lds_offset[sp][1] = seg_off + Sw::offset_uint4(seg_col(sinput_x + 8 * sp + 4),
                                                                  wave_group * GROUP_UINT4);
        });
    }
    else
    {
        const int input_x = 2 * n_local + lane / 16;
        for(int ph = 0; ph < KWP; ph++)
            input_lds_offset_pass[ph] =
                seg_off + Sw::offset_uint4(seg_col(input_x + 4 * ph), wave_group * GROUP_UINT4);
    }
    // Map-driven loads over the precomputed offsets: map_fun drives the round layout,
    // and the round's K half selects the uint4 (dense: 1; sparse: 2).
    auto load_dinput = [&](int base, int ph) {
        bunnies::reg_tile<mat_b, 1, 1> tile;
        bunnies::load_tile<arch::ds_load_b128>(
            tile, input_lds + base, [&](int, int, int, int) { return input_lds_offset_pass[ph]; });
        return tile.block(0, 0);
    };
    auto load_sinput = [&](int base, int sp) {
        bunnies::reg_tile<smat_b, 1, 1> tile;
        bunnies::load_tile<arch::ds_load_b128>(tile, input_lds + base, [&](int, int, int k, int) {
            return sinput_lds_offset[sp][k / 32];
        });
        return tile.block(0, 0);
    };

    // Output store plan (LDS staging -> global, one uint2 per lane): global destination
    // (null if off-image), the LDS slot it drains, liveness, and where it stages.
    struct OutStore
    {
        const uint4* load_lds;
        uint4* store_global;
        bool lane_active;
        element_t* store_lds; // this lane's uint2 slot in the output staging buffer
        // Narrow path: the wide uint4 store is replaced by per-channel b16 writes.
        bool store_active; // this lane drains a live output column (q_out < wo)
        int q_out;         // output column
        int ch_base;       // first global channel of this lane's 8-channel group
    };
    const auto ostore = [&] {
        OutStore o{nullptr, nullptr, true, nullptr, false, 0, 0};
        if(tid < STORE_VECS)
        {
            // Folded tiles: the OUTPUT_BLOCK_Q staged columns split into F sub-images of
            // OUT_W_SUB each.
            //
            // col belongs to sub-image s_out (image block_g_idx + s_out*n_groups) at local
            // output q_out. F == 1 => the original mapping.
            constexpr int OUT_W_SUB = W_SUB / cfg.stride;
            const int c_uint4       = tid % BLOCK_C_UINT4;
            const int col           = tid / BLOCK_C_UINT4;
            const int s_out         = col / OUT_W_SUB;
            const int q_local       = col % OUT_W_SUB;
            const int q_out = (cfg.stride == 2) ? (block_q / 2) + q_local : block_q + q_local;
            const int n_out = block_g_idx + s_out * n_groups;
            o.load_lds      = &output_lds[Sw::offset_uint4(col, c_uint4)];
            o.q_out         = q_out;
            o.ch_base       = (block_c_uint4 + c_uint4) * CH_PER_UINT4;
            o.store_active  = (q_out < wo);
            if constexpr(!cfg.narrow_c)
            {
                if(q_out < wo)
                {
                    const int K_UINT4 = C / CH_PER_UINT4;
                    o.store_global    = reinterpret_cast<uint4*>(out) +
                                     (size_t)n_out * ho * wo * K_UINT4 + (size_t)q_out * K_UINT4 +
                                     block_c_uint4 + c_uint4;
                }
            }
        }

        const int result_n        = ResultLayout::outer(lane);
        const int result_row      = ResultLayout::inner(lane, 0);
        const int c4_within_group = result_row / 4 % 2;
        int output_col;
        if constexpr(cfg.stride == 2)
        {
            o.lane_active = (GT::q(result_row) == 0);
            output_col    = result_n;
        }
        else
        {
            o.lane_active = true;
            output_col    = 2 * result_n + GT::q(result_row);
        }
        o.store_lds = reinterpret_cast<element_t*>(output_lds) +
                      Sw::offset_uint2(output_col, wave_group * GROUP_UINT4 * 2 + c4_within_group) *
                          (sizeof(uint2) / sizeof(element_t));
        return o;
    }();

    bunnies::reg_tile<mat_acc, cfg.kh, 1> acc{}; // kh-deep ring, one slot per in-flight row

    // Ring slot of the row being computed; the row prefetched this iteration lands in
    // comp_phase + PREFETCH_AHEAD. Carried across stream + remainder for consistency.
    int comp_phase = 0;

    // When kh is a multiple of the ring size, comp_phase is 0 at every y_base entry, so
    // row Y_LOCAL's slot is the constant Y_LOCAL % NBUF.
    constexpr bool CT_PHASE = (cfg.kh % NUM_INPUT_LDS_BUFFERS == 0);

    // Input-strip sync without collapsing the prefetch pipeline.
    auto sync_input = [] {
        __builtin_amdgcn_s_waitcnt(arch::makecnt(PREFETCH_AHEAD - 1, /*lgkmcnt=*/0));
        __builtin_amdgcn_s_barrier();
    };
    // Output-strip sync: keep the input prefetch (vmcnt) alive, drain only the
    // output_lds ds writes (lgkmcnt) before the barrier.
    auto sync_output = [] {
        arch::s_wait_lgkmcnt<0>();
        __builtin_amdgcn_s_barrier();
    };

    auto flush_p = [&](bunnies::reg_tile_concept auto const& slot, int p_out) {
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

        bunnies::reg_tile<mat_out, 1, 1> out_tile;
        out_tile.block(0, 0).data = packed_convert<element_t>(slot.block(0, 0).data);

        // The swizzle already resolved the destination, so the map ignores the coords.
        auto stage_out = [&] {
            bunnies::store_tile<arch::ds_store_b64>(
                out_tile, ostore.store_lds, [](int, int, int, int) { return 0; });
        };
        if constexpr(cfg.stride == 2)
        {
            if(ostore.lane_active) // only q-phase-0 lanes hold a live column
                stage_out();
        }
        else
        {
            stage_out();
        }

        sync_output();

        if constexpr(cfg.narrow_c)
        {
            // Per-channel drain: write only this group's channels < C (last group is
            // partial), one b16 element at a time.
            if(ostore.store_active)
            {
                const int wo_row = (cfg.stride == 2) ? (p_out / 2) : p_out;
                const auto* src  = reinterpret_cast<const element_t*>(ostore.load_lds);
                const size_t base =
                    ((size_t)block_n * ho * wo + (size_t)wo_row * wo + ostore.q_out) * C +
                    ostore.ch_base;
                for(int ch = 0; ch < CH_PER_UINT4; ch++)
                    if(ostore.ch_base + ch < C)
                        out[base + ch] = src[ch];
            }
        }
        else if(ostore.store_global)
        {
            const int wo_row                        = (cfg.stride == 2) ? (p_out / 2) : p_out;
            ostore.store_global[wo_row * wo_stride] = *ostore.load_lds;
        }
    };

    // Per-row compute, shared by the chunk loop and the last-chunk remainder.
    //
    // Prefetch ahead, run this row's horizontal MFMA passes into the kh-deep acc ring,
    // advance the phase, then flush the completed row. Callers differ only in the row
    // index, prefetch clamp, and flush gate (do_flush drops warm-up rows). CT_PHASE uses
    // a compile-time ring base when kh % NBUF == 0, else the runtime comp_phase.
    auto compute_row = [&]<int Y_LOCAL>(int y, int prefetch_y, bool do_flush) {
        sync_input();
        if constexpr(CT_PHASE)
        {
            constexpr int LP = (Y_LOCAL + PREFETCH_AHEAD) % NUM_INPUT_LDS_BUFFERS;
            constexpr int CP = Y_LOCAL % NUM_INPUT_LDS_BUFFERS;
            issue_input_load.template operator()<LP>(prefetch_y);
            if constexpr(USE_SPARSE)
            {
                static_for<KSP>([&]<int SP>() {
                    smat_b sinput = load_sinput(CP * INPUT_LDS_BUFFER_UINT4, SP);
                    static_for<cfg.kh>([&]<int R>() {
                        constexpr int p_idx = (Y_LOCAL - R + cfg.kh) % cfg.kh;
                        arch::smma::wmma(acc.block(p_idx, 0),
                                         sweights_reg.block(SP, R),
                                         sinput,
                                         acc.block(p_idx, 0));
                    });
                });
            }
            else
            {
                static_for<KWP>([&]<int PH>() {
                    mat_b input_reg = load_dinput(CP * INPUT_LDS_BUFFER_UINT4, PH);
                    static_for<cfg.kh>([&]<int R>() {
                        constexpr int p_idx = (Y_LOCAL - R + cfg.kh) % cfg.kh;
                        arch::mma<>::wmma(acc.block(p_idx, 0),
                                          weights_reg.block(PH, R),
                                          input_reg,
                                          acc.block(p_idx, 0));
                    });
                });
            }
        }
        else
        {
            dispatch<NUM_INPUT_LDS_BUFFERS>(
                (comp_phase + PREFETCH_AHEAD) % NUM_INPUT_LDS_BUFFERS,
                [&]<int LP>() { issue_input_load.template operator()<LP>(prefetch_y); });
            if constexpr(USE_SPARSE)
            {
                static_for<KSP>([&]<int SP>() {
                    smat_b sinput = load_sinput(comp_phase * INPUT_LDS_BUFFER_UINT4, SP);
                    static_for<cfg.kh>([&]<int R>() {
                        constexpr int p_idx = (Y_LOCAL - R + cfg.kh) % cfg.kh;
                        arch::smma::wmma(acc.block(p_idx, 0),
                                         sweights_reg.block(SP, R),
                                         sinput,
                                         acc.block(p_idx, 0));
                    });
                });
            }
            else
            {
                static_for<KWP>([&]<int PH>() {
                    mat_b input_reg = load_dinput(comp_phase * INPUT_LDS_BUFFER_UINT4, PH);
                    static_for<cfg.kh>([&]<int R>() {
                        constexpr int p_idx = (Y_LOCAL - R + cfg.kh) % cfg.kh;
                        arch::mma<>::wmma(acc.block(p_idx, 0),
                                          weights_reg.block(PH, R),
                                          input_reg,
                                          acc.block(p_idx, 0));
                    });
                });
            }
        }

        comp_phase = (comp_phase + 1) % NUM_INPUT_LDS_BUFFERS;

        constexpr int P_FLUSH = (Y_LOCAL + 1) % cfg.kh;
        const int p_out       = y + py - (cfg.kh - 1);
        if(do_flush)
            flush_p(acc.template sub<1, 1>(P_FLUSH, 0), p_out);
        acc.block(P_FLUSH, 0) = mat_acc{};
    };

    for(int y_base = stream_y_lo; y_base + cfg.kh <= main_y_hi; y_base += cfg.kh)
    {
        static_for<cfg.kh>([&]<int Y_LOCAL>() {
            const int y = y_base + Y_LOCAL;
            // Rows past this chunk's span become phantom OOB loads (hi_eff reads 0) to
            // hold the in-flight count constant.
            //
            // Warm-up rows (y < own_y_lo) seed the ring, so their flush is suppressed.
            const int prefetch_y = (y + PREFETCH_AHEAD < end_y) ? (y + PREFETCH_AHEAD) : hi_eff;
            compute_row.template operator()<Y_LOCAL>(y, prefetch_y, y >= own_y_lo);
        });
    }

    if(is_last_chunk)
    {
        const int y_rem_base = (hi_eff / cfg.kh) * cfg.kh;
        static_for<cfg.kh>([&]<int Y_LOCAL>() {
            if(Y_LOCAL >= hi_eff % cfg.kh)
                return;
            const int y          = y_rem_base + Y_LOCAL;
            const int prefetch_y = (y + PREFETCH_AHEAD < hi_eff) ? (y + PREFETCH_AHEAD) : hi_eff;
            compute_row.template operator()<Y_LOCAL>(y, prefetch_y, true);
        });
    }

    if(is_last_chunk)
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
            __syncthreads();
            int p_idx = (p_out - py + cfg.kh) % cfg.kh;
            bunnies::reg_tile<mat_acc, 1, 1> slot;
            dispatch<cfg.kh>(p_idx, [&]<int P>() {
                slot.block(0, 0) = acc.block(P, 0);
                acc.block(P, 0)  = mat_acc{};
            });
            flush_p(slot, p_out);
        }
    }
}

template <Config cfg, DataType DT>
__global__ __launch_bounds__(cfg.block_size()) void conv2d_depthwise_1d_toeplitz_nhwc_cdna4(
    const ToType<DT>* __restrict__ in,
    const ToType<DT>* __restrict__ wei,
    double /*alpha*/,
    double /*beta*/,
    ToType<DT>* __restrict__ out,
    int N,
    int C,
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
    if(__builtin_amdgcn_is_invocable(__builtin_amdgcn_mfma_f32_16x16x32_f16) &&
       __builtin_amdgcn_is_invocable(__builtin_amdgcn_mfma_f32_16x16x32_bf16) &&
       __builtin_amdgcn_is_invocable(__builtin_amdgcn_smfmac_f32_16x16x64_f16) &&
       __builtin_amdgcn_is_invocable(__builtin_amdgcn_smfmac_f32_16x16x64_bf16))
    {
        conv2d_depthwise_1d_toeplitz_nhwc_impl<cfg, DT>(in, wei, out, N, C, hi, wi, ho, wo, py, px);
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
        auto kernel = conv2d_depthwise_1d_toeplitz_nhwc_cdna4<cfg, DT>;
        kernel<<<lp.grid, lp.block_size, lp.dynamic_shared_bytes, stream>>>(
            static_cast<const dtype*>(in),
            static_cast<const dtype*>(wei),
            1.0,
            0.0,
            static_cast<dtype*>(out),
            par.n,
            par.c,
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
    else
        typed_launch.template operator()<DataType::fp16>();
}

// Compute-unit count of the active device (cached), used to size H-tiling.
inline int cu_count()
{
    static const int cu = [] {
        int dev = 0;
        if(hipGetDevice(&dev) != hipSuccess)
            return 256;
        hipDeviceProp_t props{};
        if(hipGetDeviceProperties(&props, dev) != hipSuccess || props.multiProcessorCount <= 0)
            return 256;
        return props.multiProcessorCount;
    }();
    return cu;
}

class Depthwise_1D_Toeplitz_ConvKernel : public DepthwiseConvKernel
{
public:
    constexpr Depthwise_1D_Toeplitz_ConvKernel(const Config& cfg, LaunchFn launch_fn)
        : DepthwiseConvKernel(launch_fn)
        , cfg_(cfg)
    {
    }

    std::string_view name() const override { return "depthwise_1d_toeplitz"; }

    std::string describe_config() const override
    {
        // Compact self-contained run label (abbrev name + every knob):
        //
        //   dw1d=depthwise_1d_toeplitz  k=filter(khxkw)  s=stride  F/D/W=direction
        //   u=dgrad upsample  w=waves/wg  c=block_c  nc=narrow-C  f=batch-fold
        const char* dir = (cfg_.direction == Direction::Fprop)   ? "F"
                          : (cfg_.direction == Direction::Dgrad) ? "D"
                                                                 : "W";
        std::string s   = "dw1d k" + std::to_string(cfg_.kh);
        if(cfg_.kw != cfg_.kh)
            s += "x" + std::to_string(cfg_.kw);
        s += " s" + std::to_string(cfg_.stride);
        s += ' ';
        s += dir;
        if(cfg_.dilation > 1)
            s += " u" + std::to_string(cfg_.dilation);
        s += " w" + std::to_string(cfg_.waves_per_wg);
        s += " c" + std::to_string(cfg_.block_c());
        if(cfg_.narrow_c)
            s += " nc";
        if(cfg_.w_fold > 1)
            s += " f" + std::to_string(cfg_.w_fold);
        if(cfg_.lds_buffers != 3)
            s += " b" + std::to_string(cfg_.lds_buffers);
        return s;
    }

    void get_tolerance(const Conv2dParams& par, float& atol, float& rtol) const override
    {
        get_mixed_precision_tolerance(par, atol, rtol);
        // A small atol floor keeps zero-upsampled Dgrad edges from tripping the check.
        //
        // The model budgets error as rtol * conv(|A|,|B|) with atol == 0. Dgrad edge
        // outputs over a zero-upsampled dY collapse to a single ~1e-6 product, so that
        // budget hits ~0 and fp16 rounding trips it; the floor stays far below any genuine
        // error (~1e-3).
        atol = std::max(atol, 1e-6f);
    }

    bool is_applicable(const Conv2dParams& par) const override
    {
        if(!DepthwiseConvKernel::is_applicable(par))
            return false;
        // Dgrad (backward-data) is the rot180 correlation of dY; stride 1 runs it
        // directly, stride 2 runs it over a 2x-upsampled dY (zeros between samples).
        if(par.direction != Direction::Fprop && par.direction != Direction::Dgrad)
            return false;
        // Both C % 8 == 0 (wide uint4) and C % 8 != 0 (narrow b16) are handled, so no
        // channel-count reject here -- is_valid_config routes each C to its path.
        // Shipped filters {3,5,7,9,11}; any odd kw generalizes (taps masked by kw). 5x5
        // is also served by depthwise_2d_toeplitz; this 1D path wins large-spatial.
        if(par.kh != par.kw ||
           (par.kw != 3 && par.kw != 5 && par.kw != 7 && par.kw != 9 && par.kw != 11))
            return false;
        if(par.stride_h != par.stride_w)
            return false;
        if(par.stride_h != 1 && par.stride_h != 2)
            return false;
        if(par.dilation_h != 1 || par.dilation_w != 1)
            return false;
        if(par.pad_h > par.kh - 1 || par.pad_w > par.kw - 1)
            return false;
        // The buffer is rebased per sample, so only a single image must fit a 32-bit
        // offset; the batch may push the full tensors past 2GB.
        if(par.n > 0)
        {
            Conv2dSize sz(par);
            if(sz.input_bytes() / par.n > INT32_MAX || sz.output_bytes() / par.n > INT32_MAX)
                return false;
        }
        return true;
    }

    bool is_valid_config(const Conv2dParams& par) const override
    {
        if(par.direction != cfg_.direction)
            return false;
        if(par.kh != cfg_.kh || par.kw != cfg_.kw)
            return false;
        if(cfg_.direction == Direction::Dgrad)
        {
            // Dgrad maps the forward stride to the upsample factor (cfg.dilation) and
            // forward dilation to cfg.stride, mirroring SizeView<Dgrad>.
            if(par.stride_h != cfg_.dilation || par.stride_w != cfg_.dilation)
                return false;
            if(par.dilation_h != cfg_.stride || par.dilation_w != cfg_.stride)
                return false;
        }
        else
        {
            if(par.stride_h != cfg_.stride || par.stride_w != cfg_.stride)
                return false;
            if(par.dilation_h != 1 || par.dilation_w != 1)
                return false;
        }
        // Route by channel alignment: narrow owns C % 8 != 0; wide owns C % 8 == 0 and
        // needs block_c to tile C exactly.
        if(cfg_.narrow_c)
        {
            if((par.c % 8) == 0)
                return false;
        }
        else
        {
            if((par.c % cfg_.block_c()) != 0)
                return false;
        }
        // Batch-fold auto-pick + auto-pad.
        //
        // Auto-pick takes the first is_valid_config match, so exactly one fold family is
        // valid per shape (w_fold == preferred_wfold(par)); on a small map the F=1 configs
        // drop out and the folded ladder becomes the front. Auto-pad rounds compute up to
        // a full tile (blocks_w = divup(out_w, w_sub)), padded columns read 0 and are
        // masked on store, so unfoldable widths (e.g. 7) fold for free.
        if(cfg_.w_fold != preferred_wfold(par))
            return false;
        return true;
    }

    // Auto-pick heuristic: the batch-fold factor for `par` (1 = none).
    //
    // Single source of truth for auto-selection. The tile stride is the fprop output
    // subsample; dgrad runs a stride-1 tile carrying the forward stride as input
    // upsampling, so its geometry is stride-independent. A single-image tile makes
    // tile_out = BLOCK_Q/kstride outputs; we auto-fold only small maps (out_w <=
    // tile_out/2, tile <= 50% full), where folding is a proven win -- larger maps keep
    // F=1 (halo re-read outweighs the tail reclaim). Prefer the smallest F (widest
    // sub-tile) that fills the tile with a majority-real tail, since wider sub-tiles
    // re-read less halo (F=2 beat F=4 at W=16).
    static int preferred_wfold(const Conv2dParams& par)
    {
        // Applicable directions (Fprop/Dgrad) and stride_h==stride_w in {1,2} are
        // guaranteed by is_applicable.
        const bool is_dgrad = par.direction == Direction::Dgrad;
        if(par.dilation_h != 1 || par.dilation_w != 1)
            return 1; // forward dilation only
        if((par.c % 8) != 0)
            return 1; // wide path only
        // Tile stride: fprop subsamples the output; dgrad runs a stride-1 tile (the
        // forward stride becomes input upsampling), so its geometry is stride-1.
        const int kstride  = is_dgrad ? 1 : par.stride_h;
        const int out_w    = is_dgrad ? par.w : par.q;
        const int tile_out = BLOCK_Q / kstride; // outputs per single-image tile
        if(out_w <= 0 || (out_w % tile_out) == 0)
            return 1; // tile-aligned: no tail
        if(2 * out_w > tile_out)
            return 1; // >50% full: keep F=1
        const auto fills = [&](int F) {
            const int out_sub = (BLOCK_Q / F) / kstride; // outputs per sub-image
            const int rem     = out_w % out_sub;         // padded tail must be majority-real
            return (par.n % F) == 0 && (rem == 0 || rem * 2 > out_sub);
        };
        if(fills(2))
            return 2;
        if(fills(4))
            return 4;
        return 1;
    }

    LaunchParams get_launch_params(const Conv2dParams& par) const override
    {
        // Folded W tiling: the w-tile is w_sub()-wide and the batch is enumerated in
        // n_groups slots of w_fold images.
        const int output_block_q = cfg_.w_sub() / cfg_.stride;
        // Dgrad produces dX (width par.w); Fprop produces Y (width par.q).
        const int out_w    = (cfg_.direction == Direction::Dgrad) ? par.w : par.q;
        const int blocks_w = divup(out_w, output_block_q);
        const int blocks_c = divup(par.c, cfg_.block_c());
        const int n_groups = divup(par.n, cfg_.w_fold);
        // Cap n_fold at n_groups so a small folded batch spawns no empty x-slots.
        // Must match the device decode.
        const int eff_n_fold    = min(cfg_.n_fold, n_groups);
        const int blocks_w_n    = blocks_w * eff_n_fold;
        const int blocks_n_fold = divup(n_groups, eff_n_fold);

        // H-tiling: the serial row loop means waves don't grow with H, under-subscribing
        // the GPU.
        //
        // Split into num_h_chunks workgroups (gridDim.z) chosen to just fill it, capped so
        // chunks stay large enough to bound the warm-up halo re-read.
        const int view_h        = (cfg_.direction == Direction::Dgrad) ? par.p : par.h;
        const int hi_eff        = (cfg_.dilation > 1) ? ((view_h - 1) * cfg_.dilation + 1) : view_h;
        const int n_full_blocks = hi_eff / cfg_.kh;

        int num_h_chunks = 1;
        if(n_full_blocks > 1)
        {
            const int base_wgs = blocks_w_n * blocks_c * blocks_n_fold;
            // Target ~4 workgroups per CU; only tile when the base grid falls short.
            const int target_wgs = 4 * cu_count();
            // Each chunk re-reads one warm-up kh-block, so cap chunks at n_full_blocks/2
            // (halo <= 50%).
            //
            // Floor of 2 (not 1) lets very short maps (n_full_blocks in {2,3}, e.g. 7x7)
            // still split in two instead of a single serial stream; identical to n/2 for
            // n_full_blocks >= 4.
            const int max_chunks_by_halo = max(2, n_full_blocks / 2);
            if(base_wgs < target_wgs)
            {
                num_h_chunks = divup(target_wgs, base_wgs);
                num_h_chunks = min(num_h_chunks, max_chunks_by_halo);
                num_h_chunks = min(num_h_chunks, n_full_blocks);
                num_h_chunks = max(num_h_chunks, 1);
            }
        }

        LaunchParams launch;
        launch.grid       = dim3(blocks_w_n, blocks_c, blocks_n_fold * num_h_chunks);
        launch.block_size = dim3(cfg_.block_size(), 1, 1);
        return launch;
    }

private:
    const Config& cfg_;
};

} // namespace depthwise_1d_toeplitz
} // namespace hipconv::cdna4
