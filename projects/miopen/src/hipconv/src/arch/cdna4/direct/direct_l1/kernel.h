#pragma once

// direct_l1: a CDNA4 direct conv2d kernel that stages weights through L1 cache.
//
// The double-buffered input tile alone (~153 KiB) nearly fills the 160 KiB LDS,
// so weights load from L1 into registers rather than LDS; repeated loads across
// waves hit L1 (the "l1"), a second staging tier beyond what LDS holds. A separate
// launch formats the weights into MFMA tile order.
//
// The device kernel and its host helpers live here, included by the autoshard,
// so the configs[] table builds in parallel. Everything is template/inline.

#include "balanced_l2_grid.h"
#include "round_invariant_context.h"
#include "config.h"
#include "config_table.h"
#include "config_desc.h"
#include "input_loader.h"
#include "input_loader_lds.h"
#include "output_writer.h"
#include "weights_loader.h"
#include "wave_compute_index.h"
#include "magic_division.h"
#include "transpose_weights.h"
#include "matrix_layout.h"
#include "persistent_grid.h"
#include "swizzle.h"
#include "detail.h"
#include "types.h"
#include "grouped/mfma_dispatch.h"
#include "mathutil.h"
#include "launch_params.h"
#include "conv_kernel.h"
#include "conv_kernel_table.h"
#include "direct_conv_kernel.h"
#include "transpose_lds_layout.h"
#include "memory.h"
#include "packed_ops.h"
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <cstdint>


namespace hipconv::cdna4::direct_l1
{
// Maximum filter height and width supported.
constexpr int MAX_KH = 5;
constexpr int MAX_KW = 5;

// Whether cfg is the one tile that should serve this problem.
//
// Matches direction and filter, then applies the family-partitioning gates
// (unfold / K256-vs-K128 overcompute / K-divisibility) that make exactly one enabled config
// claim each problem so dispatch never ties.
inline bool is_valid_config(const Conv2dParams& par, const Config& cfg)
{
    if(par.direction != cfg.direction)
    {
        return false;
    }

    if(par.kh != cfg.kh || par.kw != cfg.kw)
    {
        return false;
    }

    // Batch-unfold gating on direction-mapped output width.
    //
    // Unfold tiles (unfold_n > 1) pack narrow images and serve out_w <= w_unfold
    // (==8); the standard tile cedes that window so the unfold family wins without a
    // tie. Every square filter 2x2..5x5 has a K32/wave unfold tile.
    constexpr int max_unfold_width = 8;
    const bool is_dgrad_dir        = (cfg.direction == Direction::Dgrad);
    const int out_w                = is_dgrad_dir ? par.w : par.q;
    const bool filter_has_unfold   = (cfg.kw == cfg.kh) && (cfg.kh >= 2 && cfg.kh <= 5);
    if(cfg.unfold_n > 1)
    {
        if(out_w > cfg.w_unfold())
        {
            return false;
        }
    }
    else
    {
        // A small reduction (<= 64) at out_w <= 8 is intentionally unserved.
        //
        // single_c tiles are unfold_n == 1 and cede here, and there is no single_c
        // unfold tile, so such shapes get no config (a clean dispatch failure): a rare
        // corner not worth a dedicated tile.
        if(filter_has_unfold && out_w <= max_unfold_width)
        {
            return false;
        }
    }

    // K(256) versus K(128) selection.
    //
    // Pick the configuration that performs less overcompute.
    // Choose K(128) in a tie because of its greater arithmetic intensity.
    {
        const bool filter_has_k256 =
            ((cfg.kw == cfg.kh) && (cfg.kh == 2 || cfg.kh == 3)) || (cfg.kh == 3 && cfg.kw == 1);

        const int out_chan = is_dgrad_dir ? par.channels_per_group() : par.filters_per_group();

        // Both tiles use 16 rows.
        // Total work is differentiated only by width and output-channel padding.
        const int64_t k256_work =
            static_cast<int64_t>(divup(out_w, 16) * 16) * (divup(out_chan, 256) * 256);
        const int64_t k128_work =
            static_cast<int64_t>(divup(out_w, 32) * 32) * (divup(out_chan, 128) * 128);

        const bool k256_wins = k256_work < k128_work;

        if(cfg.waves_k == 4 && cfg.wave_k16 == 4 && cfg.unfold_n == 1)
        {
            if(!k256_wins)
            {
                return false;
            }
        }
        else if(filter_has_k256 && !cfg.single_c && cfg.waves_k != 4 && !cfg.large_tensor)
        {
            // K128 family cedes whenever K256 wins.
            //
            // large_tensor is exempt: it has no K256 counterpart (the K256 tiles fail
            // the per-image size gate below), so ceding would leave the shape unserved.
            if(k256_wins)
            {
                return false;
            }
        }
    }

    // Per-group reduction/output channel counts (direction-mapped).
    //
    // Each group is an independent conv whose counts drive the peel structure,
    // k-divisibility, and the blocks_k bound.
    const bool is_dgrad      = (cfg.direction == Direction::Dgrad);
    const int reduction_chan = is_dgrad ? par.filters_per_group() : par.channels_per_group();
    const int output_chan    = is_dgrad ? par.channels_per_group() : par.filters_per_group();

    // Reduction channels must be even.
    //
    // The per-pixel base is 2*count bytes and CDNA4 dwordx4 loads are Dword-aligned.
    // (A non-multiple of 8 over-reads harmlessly into zero-filled weights.)
    if((reduction_chan % 2 != 0))
    {
        return false;
    }

    // The weights formatter does not support an odd number of input channels.
    //
    // It loads over the conv C axis at dword (2-fp16) granularity, so an odd C puts the
    // last channel in a dword straddling the buffer's num_records, which the bounds
    // check drops whole (zeroing it). Fprop's C is the reduction (even above); dgrad
    // reduces over K, leaving C otherwise unchecked.
    if((par.channels_per_group() % 2 != 0))
    {
        return false;
    }

    // Reduction iteration count must match the config's peel structure.
    //
    // single_c needs exactly one C(64) iteration, otherwise at least two so the
    // first/middle/last peel has room to run.
    if(cfg.single_c)
    {
        if(divup(reduction_chan, 64) != 1)
        {
            return false;
        }
    }
    else
    {
        if(divup(reduction_chan, 64) < 2)
        {
            return false;
        }
    }

    // Output count above which a padded K128 tile beats the exact K96/K64 tiles.
    //
    // The K128 tile's higher throughput outweighs the padding overcompute (measured
    // on MI355X bf16: ~3-15% for 2x2/3x3's wide-store K128, and confirmed for the
    // 4x4/5x5 narrow K128 too, where padded K128 matches or beats exact K96/K64 at
    // every count except K=320, not worth special-casing). Every filter has a
    // block_k=128 tile, so all cede above the threshold.
    constexpr int wide_k128_min_output = 96;

    if(cfg.k_divisible)
    {
        // Fast tiles: exact multiple of block_k required.
        if(output_chan % cfg.block_k() != 0)
        {
            return false;
        }
        // K96/K64 cede K > 96 to the block_k=128 tile (every filter has one).
        //
        // single_c is exempt (tiny C no K128 config covers).
        if(!cfg.single_c && cfg.block_k() < 128 && output_chan > wide_k128_min_output)
        {
            return false;
        }
        // K128 unfold cedes multiples of 256 to the K256 unfold (3x3 only).
        const bool filter_has_k256_unfold = (cfg.kh == 3 && cfg.kw == 3);
        if(cfg.unfold_n > 1 && cfg.block_k() == 128 && filter_has_k256_unfold &&
           output_chan % 256 == 0)
        {
            return false;
        }
    }
    else if(cfg.waves_k == 4 && cfg.wave_k16 == 4)
    {
        // K256 arbitrary-output fallback.
        //
        // Once the K256 selection above claims this shape, the K256 divisible tile is
        // the only competitor, so cede exactly the multiples of block_k (256).
        if(output_chan % cfg.block_k() == 0)
        {
            return false;
        }
    }
    else if(cfg.single_c)
    {
        // single_c arbitrary-output fallback (block_k=64).
        //
        // Its divisible single_c sibling (also block_k=64) serves output % 64 == 0;
        // this serves the rest. Cede ONLY output % 64 == 0. The generic %96/%128
        // cede below would route to divisible tiles that need reduction >= 128,
        // which single_c shapes never have, reopening the hole this config fills.
        if(output_chan % cfg.block_k() == 0)
        {
            return false;
        }
    }
    else if(cfg.unfold_n > 1)
    {
        // Arbitrary-output batch-unfold fallback (block_k=128).
        //
        // The unfold family has only K128/K256 divisible tiles (no K96/K64), so cede
        // exactly the multiples of block_k (128); %256 is a subset, so the K256
        // divisible unfold (3x3) still gets its counts. The generic %96/%64 cede below
        // would route to nonexistent K96/K64 unfold tiles, leaving those counts as
        // holes (same reasoning as the single_c arm above).
        if(output_chan % cfg.block_k() == 0)
        {
            return false;
        }
    }
    else
    {
        // Arbitrary-K fallback.
        //
        // Any count is servable (padded + writer-guarded), but cede counts a faster
        // divisible tile covers. The padded K128 above threshold does not cede (it
        // beats exact K96/K64), so it keeps K > 96 not divisible by 128.
        bool covered = (output_chan % 128 == 0);
        const bool k128_above_thresh =
            (cfg.block_k() == 128) && (output_chan > wide_k128_min_output);
        if(!k128_above_thresh)
        {
            covered = covered || (output_chan % 96 == 0);
            covered = covered || (output_chan % 64 == 0);
        }
        if(covered)
        {
            return false;
        }
    }

    // The K-partition must fit the topology, not the raw block count.
    //
    // per_xcd_blocks is the innermost decode axis (<= 32 CUs) and slices span kparts
    // XCDs (<= 8); plan_k_partition folds larger blocks_k across the grid. Mirrors
    // get_k_partition.
    const int blocks_k  = divup(output_chan, cfg.block_k());
    const KPartition kp = plan_k_partition(blocks_k,
                                           persistent::NUM_XCD,
                                           /*target_per_xcd=*/maximum(1, 512 / cfg.block_k()),
                                           /*pad_threshold=*/cfg.k_divisible ? blocks_k : 7);
    if(kp.per_xcd_blocks > persistent::NUM_CU_PER_XCD || kp.kparts > persistent::NUM_XCD)
    {
        return false;
    }

    // Buffer-window size gate: partitions large-tensor from baseline configs.
    //
    // Buffer offsets and NUM_RECORDS are 32-bit. A baseline config folds only the
    // tile's image origin into the 64-bit base, so its 32-bit window is one whole
    // image (unfold_n images) and it serves only shapes whose per-image span fits
    // int32. A large_tensor config additionally folds the tile's row origin, shrinking
    // the window to the tile's rows, so it serves shapes whose per-image span does NOT
    // fit but whose per-row span does. The two predicates are complementary, so exactly
    // one family claims each shape (a shape too large for even the row window gets
    // neither, a clean dispatch failure).
    //
    // Both tensors are bounded against the same limit, so which is read vs written
    // (swapped for dgrad) does not matter: (h, w, c) is one tensor, (p, q, k) the
    // other. The row window uses block_size_h rows (block_p + kh - 1) for both; the
    // writer only spans block_p, so bounding it at block_size_h too is conservative by
    // kh-1 rows, which lets the check stay direction-agnostic.
    constexpr size_t max_span = INT32_MAX;
    const size_t in_elem      = sizeof_data_type(par.input_type);
    const size_t out_elem     = sizeof_data_type(par.output_type);
    const size_t hwc_row      = static_cast<size_t>(par.w) * par.c * in_elem;
    const size_t pqk_row      = static_cast<size_t>(par.q) * par.k * out_elem;

    const size_t imgs = cfg.unfold_n;
    const size_t rows = static_cast<size_t>(cfg.block_p() + cfg.kh - 1);

    const bool img_fits =
        (imgs * par.h * hwc_row <= max_span) && (imgs * par.p * pqk_row <= max_span);
    if(cfg.large_tensor)
    {
        // Serve only what baseline cannot (per-image overflow), and only if the row
        // window fits so the row-fold addressing is valid.
        if(img_fits)
        {
            return false;
        }
        if(rows * hwc_row > max_span || rows * pqk_row > max_span)
        {
            return false;
        }
    }
    else if(!img_fits)
    {
        return false;
    }

    return true;
}

// Bytes needed to hold the transposed weights in DirectL1 layout.
//
// K_padded x C_padded x Kh x Kw per group (K rounded to block_k, C to 64),
// packed contiguously across groups. Computed at runtime since Kwg is not
// compile-time here.
inline size_t custom_weights_tensor_size(const Config& cfg, const Conv2dParams& par)
{
    constexpr int elem_size = 2; // sizeof(f16)
    // Per-group, direction-mapped counts (dgrad swaps output/reduction).
    const bool is_dgrad      = (cfg.direction == hipconv::Direction::Dgrad);
    const int output_chan    = is_dgrad ? par.channels_per_group() : par.filters_per_group();
    const int reduction_chan = is_dgrad ? par.filters_per_group() : par.channels_per_group();
    // K rounds to the PADDED block count: K-partition padding adds pure-pad blocks
    // whose zero weights must exist in the buffer for the loader to read in-bounds.
    const int blocks_k  = divup(output_chan, cfg.block_k());
    const KPartition kp = plan_k_partition(blocks_k,
                                           persistent::NUM_XCD,
                                           /*target_per_xcd=*/maximum(1, 512 / cfg.block_k()),
                                           /*pad_threshold=*/cfg.k_divisible ? blocks_k : 7);
    const int K_padded  = kp.kparts * kp.per_xcd_blocks * cfg.block_k();
    const int C_padded  = direct_transpose_weights::make_divisible(reduction_chan, 64);
    return static_cast<size_t>(par.groups) * K_padded * C_padded * par.kh * par.kw * elem_size;
}

// Workspace bytes: just the formatted weights at offset 0.
//
// Self-formatting makes the transpose cost part of the timed launch; hipMalloc's
// >= 256-B alignment covers the dwordx4 loads.
inline size_t get_workspace_size(const Config& cfg, const Conv2dParams& par)
{
    return custom_weights_tensor_size(cfg, par);
}

struct BlockCounts
{
    int k_per_g;
    int p;
    int q;
    int n;
    int g;
};

// Output-tile block counts along each axis (direction-mapped).
inline BlockCounts get_block_counts(const Config& cfg, const Conv2dParams& par)
{
    const bool is_dgrad    = (cfg.direction == Direction::Dgrad);
    const int output_per_g = is_dgrad ? par.channels_per_group() : par.filters_per_group();
    const int out_p        = is_dgrad ? par.h : par.p;
    const int out_q        = is_dgrad ? par.w : par.q;
    // Batch-unfold packs unfold_n images along the columns.
    //
    // n divides down and the Q count collapses to 1. unfold_n == 1 leaves originals.
    const int q_blocks = cfg.unfold_n == 1 ? divup(out_q, cfg.block_q()) : 1;
    return {
        .k_per_g = divup(output_per_g, cfg.block_k()),
        .p       = divup(out_p, cfg.block_p()),
        .q       = q_blocks,
        .n       = divup(par.n, cfg.unfold_n),
        .g       = par.groups,
    };
}

// The K-block partition of the persistent grid for this config + problem.
//
// Padding the block count (to give a prime-ish count a power-of-two factor for even
// XCD division) adds pure-pad K-blocks. Those blocks' zero weights must physically
// exist in the formatted workspace, so both the workspace size and the transpose K
// extent key on kparts*per_xcd_blocks, not the raw block_k count. Padding is enabled
// only for k_divisible=false configs (their writer already guards pad blocks); a
// k_divisible=true config gets padded == raw, so its buffer is unchanged.
inline KPartition get_k_partition(const Config& cfg, const Conv2dParams& par)
{
    const int blocks_k = get_block_counts(cfg, par).k_per_g;
    return plan_k_partition(blocks_k,
                            persistent::NUM_XCD,
                            /*target_per_xcd=*/maximum(1, 512 / cfg.block_k()),
                            /*pad_threshold=*/cfg.k_divisible ? blocks_k : 7);
}

// Padded output-channel count the formatted weights must span (per group).
//
// kparts*per_xcd_blocks blocks of block_k channels each; equals the plain block_k
// rounding when no padding applies.
inline int padded_output_channels(const Config& cfg, const Conv2dParams& par)
{
    const KPartition kp = get_k_partition(cfg, par);
    return kp.kparts * kp.per_xcd_blocks * cfg.block_k();
}

// Launch geometry for the persistent grid.
//
// One workgroup per CU, each iterating its assigned
// BalancedL2Grid work, so the grid shape does not encode the
// problem dimensions.
inline LaunchParams get_launch_params(const Config& cfg)
{
    LaunchParams launch;
    launch.grid                 = dim3(persistent::PERSISTENT_GRID_SIZE, 1, 1);
    launch.block_size           = dim3(cfg.num_threads(), 1, 1);
    launch.dynamic_shared_bytes = 0;
    return launch;
}

template <int Priority>
__device__ void schedule_barrier()
{
    __builtin_amdgcn_s_setprio(Priority);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);
}

template <Config cfg, DataType DT>
__device__ void conv2d_direct_l1_impl(const ToType<DT>* __restrict__ in,
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
                                      int per_xcd,
                                      int kparts,
                                      int xgroups,
                                      int blocks_p,
                                      int blocks_q,
                                      int n_blocks,
                                      MagicDiv magic_per_xcd,
                                      MagicDiv magic_p,
                                      MagicDiv magic_q,
                                      MagicDiv magic_n)
{
    // Waves tiling one K-partition's P x Q plane; the K axis splits across waves_k.
    constexpr int waves_per_kpart = cfg.waves_p * cfg.waves_q;

    using datatypex8_t   = std::conditional_t<DT == DataType::bf16, bf16x8_t, fp16x8_t>;
    using InLoaderGlobal = InputLoader<cfg, ToType<DT>>;
    using InLoaderLds    = InputLoaderLds<cfg, ToType<DT>>;
    using WgtsLoader     = WeightsLoader<cfg, ToType<DT>>;

    // Fires when block_p is too large for this filter; reduce wave_p.
    //
    // True even without the round-overlap pad (see input_lds_layout.h overlap_safe).
    static_assert(InLoaderGlobal::lds_size * static_cast<int>(sizeof(ToType<DT>)) <=
                      LDS_BYTES_PER_CU,
                  "direct_l1 input LDS tile exceeds the 160 KiB per-CU budget "
                  "even without the round-overlap pad. Reduce wave_p (block_p) "
                  "for this filter size.");
    __shared__ ToType<DT> input_lds[InLoaderGlobal::lds_size];

    // Full per-pixel channel count; in_base selects the group's interior slice.
    const int C = groups * c_per_group;
    InputPars input_pars{N, hi, wi, C};

    auto thread = threadIdx.x;

    // readfirstlane pins these wave-uniform indices to SGPRs.
    //
    // Else their live ranges spill under K128's operand pressure.

    // Wave within this K-partition's P x Q plane.
    auto wave = __builtin_amdgcn_readfirstlane((thread / WAVE_SIZE) % waves_per_kpart);

    // K-partition (0 .. waves_k-1): this wave's output-channel slice.
    //
    // Drives the weights sub-tensor and output K base. Equals wave_group except at
    // waves_k==4.
    auto wave_k_idx = __builtin_amdgcn_readfirstlane(thread / (waves_per_kpart * WAVE_SIZE));

    // Ping-pong half (0/1): the two halves run out of phase (one compute, one memory).
    //
    // Only the prologue/loop-exit stagger keys on it.
    auto wave_group = __builtin_amdgcn_readfirstlane(thread / ((cfg.num_waves() / 2) * WAVE_SIZE));

    // Only waves 0..num_w_waves-1 own a W-strip and load input.
    //
    // The rest just compute and load weights. Wave-uniform, so the drain branch is
    // scalar.
    const int global_wave  = wave_k_idx * waves_per_kpart + wave;
    const bool loads_input = global_wave < InLoaderGlobal::num_w_waves;

    // Weights sub-tensor geometry and wave LDS bases are round-invariant.
    //
    // Kq spans the padded K-block count (per_xcd * kparts), which the host also uses
    // to size and zero-fill the formatted weights. When padding grows it past the
    // real K, k_idx reaches pure-pad blocks whose zero weights are dropped by the
    // guarded writer (padding only occurs on k_divisible=false configs).
    constexpr int Kwg_total = cfg.block_k();
    const int Kq            = per_xcd * kparts;
    const int K_share       = Kq * cfg.wave_k();
    const int C_padded      = divup(c_per_group, 64) * 64;
    const int C64           = divup(c_per_group, 64);

    // This workgroup's XCD and XCD-local id.
    //
    // readfirstlane for the same spill hazard as above. See persistent_grid.h for
    // why the mapping covers the work once.
    const persistent::WorkgroupIndex wgi = persistent::workgroup_index();
    const unsigned xcc_id                = __builtin_amdgcn_readfirstlane(wgi.xcc_id);
    const int wg_id                      = __builtin_amdgcn_readfirstlane(wgi.workgroup_id);
    // XCD's cell in the kparts x xgroups grid: K-slice and input segment.
    //
    // kslice picks the contiguous K-block range; xgroup the contiguous X segment.
    const int kslice = __builtin_amdgcn_readfirstlane(static_cast<int>(xcc_id) % kparts);
    const int xgroup = __builtin_amdgcn_readfirstlane(static_cast<int>(xcc_id) / kparts);
    // Every round is valid, so no in-loop valid()/break; tail workgroups get zero.
    const int rounds = BalancedL2Grid::num_rounds_for_workgroup(
        per_xcd, blocks_p, blocks_q, n_blocks, groups, xgroups, xgroup, wg_id);

    // Round-invariant decode inputs, parked in LDS.
    //
    // Held live across the compute phase they would spill K128. See
    // round_invariant_context.h.
    __shared__ int round_ctx_lds[RoundInvariantContext::NUM_SLOTS];
    {
        RoundInvariantContext ctx;
        ctx.x_begin =
            BalancedL2Grid::x_begin(blocks_p, blocks_q, n_blocks, groups, xgroups, xgroup);
        ctx.x_count =
            BalancedL2Grid::x_count(blocks_p, blocks_q, n_blocks, groups, xgroups, xgroup);
        ctx.round_count = BalancedL2Grid::round_count(
            per_xcd, blocks_p, blocks_q, n_blocks, groups, xgroups, xgroup);
        ctx.wg_id    = wg_id;
        ctx.per_xcd  = per_xcd;
        ctx.kslice   = kslice;
        ctx.blocks_p = blocks_p;
        ctx.blocks_q = blocks_q;
        ctx.n_blocks = n_blocks;
        ctx.magic_n  = magic_n;
        ctx.store(round_ctx_lds);
    }

    // Compile-time bool tags: let the lambda take value args, not template args.
    constexpr std::true_type On{};
    constexpr std::false_type Off{};

    // tic/toc carry across rounds.
    //
    // Round r's last iteration prefetches round r+1's first tile into tic, which the
    // end-of-iteration swap turns into the toc that r+1 reads first. Wave-uniform;
    // readfirstlane keeps them scalar.
    int tic = 0;
    int toc = 1;

    for(int round = 0; round < rounds; ++round)
    {
        // Reload the round-invariant context parked in LDS.
        //
        // readfirstlane in load() keeps the decode scalar.
        const RoundInvariantContext ctx = RoundInvariantContext::load(round_ctx_lds);
        BalancedL2Grid sched(ctx.per_xcd,
                             ctx.kslice,
                             ctx.blocks_p,
                             ctx.blocks_q,
                             ctx.n_blocks,
                             ctx.x_begin,
                             ctx.x_count,
                             ctx.round_count,
                             magic_per_xcd,
                             magic_p,
                             magic_q,
                             ctx.magic_n,
                             ctx.wg_id,
                             round);

        const int block_n = sched.n_idx();
        const int block_p = sched.p_idx() * cfg.block_p();
        const int block_q = sched.q_idx() * cfg.block_q();
        const int k_idx   = sched.k_idx();
        const int g_idx   = sched.g_idx(); // outermost X dimension

        const int block_h = block_p - py;
        const int block_w = block_q - px;
        const int block_c = 0;

        const int group_c_base    = g_idx * c_per_group;
        const ToType<DT>* in_base = in + static_cast<size_t>(group_c_base);

        InLoaderGlobal input_loader_global(
            input_pars, block_n, block_h, block_w, block_c, input_lds, in_base, group_c_base);

        InLoaderLds input_load_lds(input_lds);

        // One loader, internally selecting the wave_k_idx partition.
        //
        // Each group's formatted weights are contiguous, so the group offset is a
        // base shift.
        const size_t wei_group_stride =
            static_cast<size_t>(Kq) * Kwg_total * C_padded * cfg.kh * cfg.kw;
        const ToType<DT>* wei_base = wei + static_cast<size_t>(g_idx) * wei_group_stride;
        WgtsLoader weights_loader(K_share, C_padded, k_idx, wei_base, wave_k_idx);

        constexpr int wave_input_reg_h = cfg.wave_p + cfg.kh - 1;
        datatypex8_t input_reg[wave_input_reg_h][cfg.wave_q16];

        datatypex8_t weights_reg[cfg.kh][cfg.wave_k() / 16];

        // Per-wave accumulators, one fp32x4 per (P, Q16, K16) mfma tile.
        //
        // Lane L of acc[p][q16][k16] holds K = k16*16 + (L/16)*4 + a, Q = q16*16 + L%16.
        fp32x4_t acc[cfg.wave_p][cfg.wave_q16][cfg.wave_k16];

        // Carried across rounds; readfirstlane keeps the parity scalar.
        tic = __builtin_amdgcn_readfirstlane(tic);
        toc = __builtin_amdgcn_readfirstlane(toc);

        // Recomputed per round rather than held live across the boundary (K128 spill).
        WaveComputeIndex<cfg> wave_compute_idx(wave);
        const auto wave_h_base = wave_compute_idx.p();            // LDS row
        const auto wave_w_base = wave_compute_idx.lds_col_base(); // LDS col

        // Cold-start the first tile into toc.
        //
        // Rounds > 0 inherit toc from the prior round's cross-round prefetch and
        // skip this, avoiding the ~2500-cyc prologue vmcnt(0) stall. Loads toc (not
        // tic) since the pre-loop swap is gated to round 0; for rounds > 0 the step-0
        // prefetch and writer staging use disjoint tic bytes, so no barrier is needed.
        if(round == 0)
        {
            static_for<InLoaderGlobal::num_load_steps>(
                [&]<int S>() { input_loader_global.template load<S>(toc); });
            wait_vmcnt_all();
            __syncthreads();
        }
        else if constexpr(!InLoaderGlobal::LdsLayout::overlap_safe)
        {
            // Serialize writer staging and step-0 prefetch (they share tic bytes).
            //
            // Round-overlap is unaffordable here (pad overflows LDS). The hazard is a
            // pure LDS WAR, so wait_lgkmcnt_all + s_barrier suffices; __syncthreads
            // would force vmcnt(0) and drain the in-flight cross-round prefetch.
            wait_lgkmcnt_all();
            __builtin_amdgcn_s_barrier();
        }

        // Update the accumulators for one step of the Kw x C32 schedule.
        //
        // Operand order weights=srcA, input=srcB makes each lane's output 4
        // contiguous K, which the OutputWriter's wide path coalesces (see
        // output_writer.h). Numerically identical to the swapped order, free on
        // the load side.
        auto mma = [&]() {
#pragma unroll
            for(int kh = 0; kh < cfg.kh; ++kh)
#pragma unroll
                for(int k16 = 0; k16 < cfg.wave_k16; ++k16)
#pragma unroll
                    for(int p = 0; p < cfg.wave_p; ++p)
#pragma unroll
                        for(int q16 = 0; q16 < cfg.wave_q16; ++q16)
                            acc[p][q16][k16] = mfma_16x16x32(
                                weights_reg[kh][k16], input_reg[p + kh][q16], acc[p][q16][k16]);
        };

        // Same as mma except the first mfma passes SrcC=0 to zero the accumulators.
        auto mma_zeroacc = [&]() {
            constexpr fp32x4_t zero = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for(int k16 = 0; k16 < cfg.wave_k16; ++k16)
#pragma unroll
                for(int p = 0; p < cfg.wave_p; ++p)
#pragma unroll
                    for(int q16 = 0; q16 < cfg.wave_q16; ++q16)
                        acc[p][q16][k16] =
                            mfma_16x16x32(weights_reg[0][k16], input_reg[p][q16], zero);
#pragma unroll
            for(int kh = 1; kh < cfg.kh; ++kh)
#pragma unroll
                for(int k16 = 0; k16 < cfg.wave_k16; ++k16)
#pragma unroll
                    for(int p = 0; p < cfg.wave_p; ++p)
#pragma unroll
                        for(int q16 = 0; q16 < cfg.wave_q16; ++q16)
                            acc[p][q16][k16] = mfma_16x16x32(
                                weights_reg[kh][k16], input_reg[p + kh][q16], acc[p][q16][k16]);
        };

        // No pre-loop swap.
        //
        // The invariant is that toc holds this round's first tile and tic is the dead
        // half; a swap would break the carried parity.

        // Prologue stagger: wavegroup 1 absorbs an extra s_barrier.
        //
        // The two wavegroups then run one step out of phase (one compute, one memory).
        if(wave_group == 1)
            __builtin_amdgcn_s_barrier();

        // next_valid gates the cross-round prefetch (false on the last round).
        bool next_valid = false;

        // Run one step's memory + compute phase.
        //
        // Issues weights, optionally prefetches, reads this step's LDS tile,
        // drains, runs one mma. Compile-time specialized on its index, (Kw, C32Half)
        // coordinates, and these flags so no branch survives in the steady body:
        //   Prefetch     : prefetch this iteration's next-channel tile into tic.
        //   NextPrefetch : (last iteration) prefetch the next round's first tile.
        //   ZeroAcc      : step 0 zero-inits the accumulators via mma_zeroacc.
        auto step =
            [&]<int Step, int Kw, int C32Half, bool Prefetch, bool ZeroAcc, bool NextPrefetch>() {
            weights_loader.load_step(weights_reg);
            // Prefetch and NextPrefetch share the loader; only one is ever set.
            if constexpr(Prefetch)
                input_loader_global.template load<Step>(tic);
            if constexpr(NextPrefetch)
                if(next_valid)
                    input_loader_global.template load<Step>(tic);
            input_load_lds.load_step(toc, wave_h_base, wave_w_base, Kw, C32Half, input_reg);
            // Partial drain leaves this step's prefetch in flight to overlap compute.
            //
            // NextPrefetch included, else the last iteration exposes its full latency.
            // Idle waves have nothing in flight, so drain fully.
            if constexpr(Prefetch || NextPrefetch)
            {
                if(loads_input)
                    wait_vmcnt<InLoaderGlobal::template loads_for_step<Step>()>();
                else
                    wait_vmcnt_all();
            }
            else
            {
                wait_vmcnt_all();
            }
            wait_lgkmcnt_all();
            schedule_barrier<1>();
            if constexpr(ZeroAcc)
                mma_zeroacc();
            else
                mma();
        };

        // Run one channel-iteration: every step plus the trailing barrier.
        auto run_iteration = [&](auto ZeroAcc, auto Prefetch, auto NextPrefetch) {
            if constexpr(Prefetch)
                input_loader_global.step_channels();

            // Emit step S, prefixed by an inter-step barrier for S > 0.
            //
            // num_load_steps = Kw * 2 steps, each filter column in two C(32) halves;
            // step S maps to (Kw=S/2, C32Half=S%2). Barriering before every step but
            // the first gives num_load_steps-1 inter-step barriers.
            constexpr int num_steps = InLoaderGlobal::num_load_steps;
            auto emit_step          = [&]<int S>() {
                if constexpr(S > 0)
                    schedule_barrier<0>();
                step.template operator()<S,
                                                  /*Kw=*/S / 2,
                                                  /*C32Half=*/S % 2,
                                                  decltype(Prefetch)::value,
                                                  /*ZeroAcc=*/(S == 0) && decltype(ZeroAcc)::value,
                                                  decltype(NextPrefetch)::value>();
            };
            static_for<num_steps>(emit_step);

            // Pin the drain at the end of the compute phase.
            //
            // Without this fence the scheduler hoists wait_vmcnt_all() up to the first
            // mfma, collapsing the last step's prefetch into an iteration-boundary bubble.
            __builtin_amdgcn_sched_barrier(0);
            wait_vmcnt_all();
            if constexpr(Prefetch)
            {
                schedule_barrier<0>(); // inter-iteration phase-swap barrier
            }
            else
            {
                // Loop exit: WG1 skips its final barrier.
                //
                // That cancels both the count and phase lag of the prologue stagger, so
                // both halves enter the writer balanced and WG1's last mma overlaps
                // WG0's writing.
                __builtin_amdgcn_s_setprio(0);
                if(wave_group == 0)
                    __builtin_amdgcn_s_barrier();
                __builtin_amdgcn_sched_barrier(0);
            }

            tic ^= 1;
            toc ^= 1;
        };

        next_valid = round < rounds - 1;

        // Aim the dead per-round loader at the next round's first tile.
        //
        // Feeds the last iteration's cross-round prefetch. round+1 is a legal index
        // to decode even on the last round (just not loaded), so no clamp.
        auto retarget_next_round = [&]() {
            // Reload fresh, not the round-top ctx.
            //
            // Reusing it recreates the cross-loop live range the LDS park removes.
            const RoundInvariantContext rctx = RoundInvariantContext::load(round_ctx_lds);
            BalancedL2Grid next_sched(rctx.per_xcd,
                                      rctx.kslice,
                                      rctx.blocks_p,
                                      rctx.blocks_q,
                                      rctx.n_blocks,
                                      rctx.x_begin,
                                      rctx.x_count,
                                      rctx.round_count,
                                      magic_per_xcd,
                                      magic_p,
                                      magic_q,
                                      rctx.magic_n,
                                      rctx.wg_id,
                                      round + 1);
            // Aim at the next round's group input slice.
            //
            // The round-top in_base offset applies only to the round-top loader.
            const int next_group_c_base    = next_sched.g_idx() * c_per_group;
            const ToType<DT>* next_in_base = in + static_cast<size_t>(next_group_c_base);
            input_loader_global.retarget(input_pars,
                                         next_in_base,
                                         next_sched.n_idx(),
                                         next_sched.p_idx() * cfg.block_p() - py,
                                         next_sched.q_idx() * cfg.block_q() - px,
                                         next_group_c_base);
        };

        if constexpr(cfg.single_c)
        {
            // C64 == 1: the lone iteration is both first and last.
            //
            // ZeroAcc + NextPrefetch, no own-prefetch, so retarget before it runs.
            // Small C over-computes to C_padded=64 against zero weights; for
            // completeness, not speed.
            retarget_next_round();
            run_iteration(/*ZeroAcc=*/On, /*Prefetch=*/Off, /*NextPrefetch=*/On);
        }
        else
        {
            // C64 >= 2: peel first (ZeroAcc) and last (NextPrefetch).
            //
            // The steady middle prefetches its own next-channel tile. Retarget only
            // after the first + middle, which use the loader for their own loads.
            run_iteration(/*ZeroAcc=*/On, /*Prefetch=*/On, /*NextPrefetch=*/Off);
            for(int c64 = 1; c64 < C64 - 1; ++c64)
                run_iteration(/*ZeroAcc=*/Off, /*Prefetch=*/On, /*NextPrefetch=*/Off);

            retarget_next_round();

            run_iteration(/*ZeroAcc=*/Off, /*Prefetch=*/Off, /*NextPrefetch=*/On);
        }

        // Write this round's accumulators.
        //
        // OutputWriter dispatches on the config: the wide path and unfold tiles stage
        // each P-row through the dead LDS half for coalesced stores; the K64/K96
        // fallbacks store from registers. The waves_k partitions write disjoint K
        // ranges, so they store concurrently.
        using OutWriter = OutputWriter<cfg, ToType<DT>>;

        OutputPars output_pars{N, ho, wo, groups, k_per_group};
        // Staging reuses the top of the dead (tic) LDS half.
        //
        // The next round's step-0 prefetch (bottom rows) then overlaps the writer
        // drain with no barrier; tic_pad keeps them disjoint. The narrow fallback
        // ignores the pointer.
        constexpr int lds_half       = InLoaderGlobal::lds_size / 2;
        constexpr int stage_lds_fp16 = OutWriter::stage_lds_fp16;
        static_assert(stage_lds_fp16 <= lds_half,
                      "output staging buffer exceeds one input LDS half-tile");
        // This config's stage must fit the MAX-size pad the layout reserves.
        //
        // Smaller stages use the top sub-region.
        static_assert(InLoaderGlobal::LdsLayout::writer_stage_uint4 * 8 >= stage_lds_fp16,
                      "InputLdsLayout::writer_stage_uint4 smaller than "
                      "OutputWriter::stage_lds_fp16 (staging would overflow the pad)");
        OutWriter writer(output_pars,
                         out,
                         input_lds + tic * lds_half + (lds_half - stage_lds_fp16),
                         block_n,
                         block_p,
                         block_q,
                         k_idx,
                         g_idx,
                         wave,
                         wave_k_idx);

        writer.write(acc);

        // No de-stagger barrier.
        //
        // WG1's skipped final loop barrier already realigned phase and count, so both
        // wavegroups reach the next round's top __syncthreads balanced and the
        // prologue stagger re-applies the offset.
    }
}

template <Config cfg, DataType DT>
__global__ __launch_bounds__(cfg.num_threads(),
                             1) void conv2d_direct_l1_cdna4(const ToType<DT>* __restrict__ in,
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
                                                            int per_xcd,
                                                            int kparts,
                                                            int xgroups,
                                                            int blocks_p,
                                                            int blocks_q,
                                                            int n_blocks,
                                                            MagicDiv magic_per_xcd,
                                                            MagicDiv magic_p,
                                                            MagicDiv magic_q,
                                                            MagicDiv magic_n)
{
    if(__builtin_amdgcn_is_invocable(__builtin_amdgcn_mfma_f32_16x16x32_f16) &&
       __builtin_amdgcn_is_invocable(__builtin_amdgcn_mfma_f32_16x16x32_bf16))
    {
        conv2d_direct_l1_impl<cfg, DT>(in,
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
                                       per_xcd,
                                       kparts,
                                       xgroups,
                                       blocks_p,
                                       blocks_q,
                                       n_blocks,
                                       magic_per_xcd,
                                       magic_p,
                                       magic_q,
                                       magic_n);
    }
}

// Per-config launch.
//
// Self-formats the raw NHWC weights into the workspace, then runs the conv
// reading from that region. The caller passes raw weights like every other
// ConvKernel.
template <Config cfg>
void launch_impl(const LaunchParams& lp,
                 const Conv2dParams& par,
                 const void* in,
                 const void* wei,
                 void* out,
                 void* workspace,
                 hipStream_t stream)
{
    // Format the weights and launch the kernel for one element type.
    auto typed_launch = [&]<DataType DT>() {
        using dtype = ToType<DT>;
        auto view   = SizeView<cfg.direction>(par);

        const auto bc = get_block_counts(cfg, par);

        const int blocks_p = bc.p;
        const int blocks_q = bc.q;
        // bc.n (not par.n) already divides the batch by unfold_n.
        //
        // Passing raw par.n would spawn unfold_n x too many tiles. Equal at unfold_n == 1.
        const int n_blocks = bc.n;

        // K-block partition across XCDs.
        //
        // Splits the K-blocks into kparts contiguous slices so each XCD holds only
        // 1/kparts of the weights. Padding the block count (for a prime-ish count like
        // K2376's 19) adds pure-pad blocks the guarded writer drops, so it is enabled
        // only for k_divisible=false configs; the fast writer never sees a pad block.
        const KPartition kp = get_k_partition(cfg, par);
        const int per_xcd   = kp.per_xcd_blocks;
        const int kparts    = kp.kparts;
        const int xgroups   = kp.xgroups;
        // Padded output-channel count (per group) the formatted weights span.
        const int padded_out_chan = kparts * per_xcd * cfg.block_k();

        // magic_n (4th divmod) splits the group from the batch index.
        //
        // Parked in LDS rather than carried live so it does not push K128 over its
        // SGPR budget.
        const MagicDiv magic_per_xcd(per_xcd);
        const MagicDiv magic_p(blocks_p);
        const MagicDiv magic_q(blocks_q);
        const MagicDiv magic_n(n_blocks);

        // Format raw weights into the workspace on the same stream.
        //
        // The transpose finishes before the conv reads them. All groups format via a
        // blockIdx.y axis.
        auto* formatted       = reinterpret_cast<dtype*>(workspace);
        constexpr int Kwg     = cfg.wave_k();
        constexpr int waves_k = cfg.waves_k;
        if constexpr(cfg.direction == Direction::Dgrad)
            direct_transpose_weights::
                launch_transpose_weights_dgrad<cfg.kh, cfg.kw, Kwg, DT, waves_k>(
                    static_cast<const dtype*>(wei),
                    formatted,
                    par.filters_per_group(),
                    par.channels_per_group(),
                    par.groups,
                    stream,
                    padded_out_chan);
        else
            direct_transpose_weights::launch_transpose_weights<cfg.kh, cfg.kw, Kwg, DT, waves_k>(
                static_cast<const dtype*>(wei),
                formatted,
                par.filters_per_group(),
                par.channels_per_group(),
                par.groups,
                stream,
                padded_out_chan);
        const dtype* conv_wei = formatted;

        // c_per_group is the MFMA reduction count, k_per_group the output count.
        //
        // dgrad swaps them; spatial dims and padding are direction-mapped by view.
        constexpr bool is_dgrad    = (cfg.direction == Direction::Dgrad);
        const int kern_c_per_group = is_dgrad ? par.filters_per_group() : par.channels_per_group();
        const int kern_k_per_group = is_dgrad ? par.channels_per_group() : par.filters_per_group();

        conv2d_direct_l1_cdna4<cfg, DT>
            <<<lp.grid, lp.block_size, lp.dynamic_shared_bytes, stream>>>(
                static_cast<const dtype*>(in),
                conv_wei,
                1.0,
                0.0,
                static_cast<dtype*>(out),
                par.n,
                par.groups,
                kern_c_per_group,
                kern_k_per_group,
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
                per_xcd,
                kparts,
                xgroups,
                blocks_p,
                blocks_q,
                n_blocks,
                magic_per_xcd,
                magic_p,
                magic_q,
                magic_n);
    };
    if(par.input_type == DataType::bf16)
        typed_launch.template operator()<DataType::bf16>();
    else
        typed_launch.template operator()<DataType::fp16>();
}

// Leaf kernel, one instance per Config.
class DirectL1_ConvKernel : public DirectConvKernel
{
public:
    constexpr DirectL1_ConvKernel(const Config& cfg, LaunchFn launch_fn)
        : DirectConvKernel(launch_fn)
        , cfg_(cfg)
    {
    }

    std::string_view name() const override { return "direct_l1"; }

    std::string describe_config() const override { return ConfigMatcher(cfg_).describe(); }

    bool matches_descriptor(std::string_view spec, std::string* error) const override
    {
        ConfigMatcher matcher(cfg_);
        if(matcher.match(spec))
            return true;
        if(error)
            *error = matcher.error();
        return false;
    }

    bool is_applicable(const Conv2dParams& par) const override
    {
        if(par.input_type != DataType::fp16 && par.input_type != DataType::bf16)
            return false;
        if(par.input_type != par.weight_type || par.input_type != par.output_type)
            return false;
        if(par.order != TensorOrder::NHWC)
            return false;
        if(par.direction != Direction::Fprop && par.direction != Direction::Dgrad)
            return false;
        if(par.kh > MAX_KH || par.kw > MAX_KW)
            return false;
        if(par.stride_h != 1 || par.stride_w != 1)
            return false;
        if(par.dilation_h != 1 || par.dilation_w != 1)
            return false;
        return true;
    }

    bool is_valid_config(const Conv2dParams& par) const override
    {
        return direct_l1::is_valid_config(par, cfg_);
    }

    LaunchParams get_launch_params(const Conv2dParams&) const override
    {
        return direct_l1::get_launch_params(cfg_);
    }

    size_t get_workspace_size(const Conv2dParams& par) const override
    {
        return direct_l1::get_workspace_size(cfg_, par);
    }

private:
    const Config& cfg_;
};

} // namespace hipconv::cdna4::direct_l1
