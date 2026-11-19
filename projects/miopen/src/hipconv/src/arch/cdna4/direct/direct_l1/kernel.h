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
#include "split_partials.h"
#include "split_plan.h"
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
#include "direct_conv_kernel.h"
#include "transpose_lds_layout.h"
#include "memory.h"
#include "packed_ops.h"
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <type_traits>


namespace hipconv::cdna4::direct_l1
{
// Maximum filter height, width, and depth supported.
constexpr int MAX_KH = 5;
constexpr int MAX_KW = 5;
constexpr int MAX_KD = 3;

// Whether a * b fits an int, which the folded batch counts are written to.
inline bool product_fits_int(int a, int b)
{
    const int64_t product = static_cast<int64_t>(a) * b;
    return product <= static_cast<int64_t>(INT32_MAX);
}

// Output count above which a padded K128 tile beats the exact K96/K64 tiles.
//
// The K128 tile's higher throughput outweighs the padding overcompute (measured
// on MI355X bf16: ~3-15% for 2x2/3x3's wide-store K128, and confirmed for the
// 4x4/5x5 narrow K128 too, where padded K128 matches or beats exact K96/K64 at
// every count except K=320, not worth special-casing). Every filter has a
// block_k=128 tile, so all rank below it above the threshold.
constexpr int wide_k128_min_output = 96;

// Output width at or below which a batch-unfold tile packs the columns.
constexpr int max_unfold_width = 8;

// Whether a 16-bit conv2d filter has a batch-unfold tile: every square one 2x2..5x5.
inline bool filter_has_unfold(const Config& cfg)
{
    return cfg.kd == 1 && cfg.kw == cfg.kh && cfg.kh >= 2 && cfg.kh <= 5;
}

// Discount for a config that loses a benchmarked comparison whose margin was not recorded.
// Places it below the winner and above a config that loses two such comparisons.
constexpr float loses_benchmark = 0.9f;

// Discount for the fill tile that loses on estimated launch time: the ratio of the estimates.
//
// Equal estimates would leave the loser at 1 beside the winner, so it takes loses_benchmark.
inline float fill_tile_discount(float winner_us, float loser_us)
{
    const float ratio = winner_us / loser_us;
    return ratio < 1.0f ? ratio : loses_benchmark;
}

// Discount for the narrow K64 tile past one round, where K128 beats it by 1.15x.
constexpr float narrow_k64_past_one_round = 1.0f / 1.15f;

// Discount for K128 and K256 where the narrow K64 tile or split-K fills the machine.
//
// The narrow tile beats K128 by 1.63x over the VAE geometry and K256 by 1.44-1.68x; K128
// and K256 run such shapes in the same time, and split-K beats both by more.
constexpr float k128_k256_underfilled = 1.0f / 1.63f;

// The K64 narrow tile of each dimensionality, P(16) x Q(16) x K(64).
//
// Named so every config evaluates the narrow tile's round count.
constexpr Config NARROW_K64_2D{.waves_k  = 4,
                               .wave_k16 = 1,
                               .wave_p   = 8,
                               .waves_q  = 1,
                               .kh       = 3,
                               .kw       = 3};
constexpr Config NARROW_K64_3D{.waves_k  = 4,
                               .wave_k16 = 1,
                               .wave_p   = 8,
                               .waves_q  = 1,
                               .kh       = 3,
                               .kw       = 3,
                               .kd       = 3};

// The 3x3x3 K128 split-K tile, named so every 3D config evaluates its split plan.
constexpr Config K128_SPLIT_K_TILE{.wave_k16 = 4,
                                   .wave_p   = 8,
                                   .kh       = 3,
                                   .kw       = 3,
                                   .kd       = 3,
                                   .split_k  = true};

// The 3x3 K128 split-K tile, named so every conv2d config evaluates its split plan.
constexpr Config K128_SPLIT_K_TILE_2D{.wave_k16 = 4,
                                      .wave_p   = 8,
                                      .kh       = 3,
                                      .kw       = 3,
                                      .split_k  = true};

inline bool split_active(const Config& cfg, const ConvParams& par);
inline DepthGeometry depth_geometry(const Config& cfg, const ConvParams& par);
inline bool is_valid_config(const ConvParams& par, const Config& cfg);
inline bool fits_the_view(const ConvParams& par, const Config& cfg);

// Whether a 1×kw layer folds into a kw×1 layer over an N·H × W × 1 × C input.
//
// With kh = 1, pad_h = 0 and stride_h = 1, output row p reads input row p alone, so NHWC
// [N, H, W, C] is the same memory as [N·H, W, 1, C]. See
// docs/algorithms/details/direct/direct-l1-fold.md.
//
// get_block_counts computes divup(N·H, unfold_n) in int arithmetic, so N·H plus the largest
// unfold_n has to fit an int too.
inline bool folds(const ConvParams& par)
{
    int max_unfold_n = 1;
    for(const Config& c : configs)
        max_unfold_n = maximum(max_unfold_n, c.unfold_n);
    return par.dims == 2 && par.kh == 1 && par.kw > 1 && par.pad_h == 0 && par.stride_h == 1 &&
           static_cast<int64_t>(par.n) * par.h + max_unfold_n <= static_cast<int64_t>(INT32_MAX);
}

// The kw×1 layer over an N·H × W × 1 × C input equivalent to a folding 1×kw layer.
inline ConvParams folded(const ConvParams& par)
{
    ConvParams view = par;
    view.n          = par.n * par.h;
    view.h          = par.w;
    view.w          = 1;
    view.kh         = par.kw;
    view.kw         = 1;
    view.pad_h      = par.pad_w;
    view.pad_w      = 0;
    view.stride_h   = par.stride_w;
    view.stride_w   = 1;
    view.dilation_h = par.dilation_w;
    view.dilation_w = 1;
    view.compute_output_size();
    return view;
}

// Whether cfg is a width-1 tile, which runs folded layers.
constexpr bool runs_folded(const Config& cfg)
{
    return cfg.unfold_n > 1 && !cfg.unfold_chunks && cfg.w_unfold() == 1;
}

// The layer cfg computes: the fold of a folding 1×kw layer on a width-1 tile, the layer otherwise.
inline ConvParams layer_view(const ConvParams& par, const Config& cfg)
{
    return folds(par) && runs_folded(cfg) ? folded(par) : par;
}


// Whether the K128 split-K tile is valid on this shape.
//
// Its plan splits a plane in one round only where the unsplit tiles would leave the
// machine underfilled. Measured on MI355X bf16 over the VAE and wan_vae c384 layers, it
// takes 0.072 ms at n1 h66 w114 against 0.104 for the narrow tile, and is not valid where
// S falls to 1 (n4 h66 w114, n1 h122 w210), which K128 runs unsplit faster. See
// docs/algorithms/direct/direct-3d.md.
inline bool k128_split_k_valid(const ConvParams& par)
{
    Config tile    = K128_SPLIT_K_TILE;
    tile.direction = par.direction;
    return is_valid_config(par, tile);
}

inline int grid_rounds(const Config& cfg, const ConvParams& par);
inline float narrow_k64_us(const ConvParams& par);
inline float k128_split_k_us(const ConvParams& par);
// Whether cfg ranks by estimated launch time alone: the chunked unfold tiles and the K64 tiles.
//
// See docs/algorithms/direct/direct-l1-ranking.md.
constexpr bool ranks_by_estimate(const Config& cfg, int output_chan)
{
    // The K64 split-K tile competes only where every K128 tile pads its output channels.
    //
    // At output channels divisible by 128 a K128 tile is usually faster. See
    // docs/algorithms/details/direct/direct-l1-c64-step.md.
    if(cfg.split_k && cfg.c64_step)
        return output_chan % 128 != 0;
    return cfg.unfold_chunks || (cfg.block_k() == 64 && cfg.block_p() == 16 && cfg.unfold_n == 1 &&
                                 !cfg.split_k && cfg.k_divisible && !cfg.single_c);
}

// The tiles that rank by estimate against the other 3x3 conv2d tiles.
struct EstimateRank
{
    // Index into configs of the config of shortest estimate among ranks_by_estimate; -1 if none.
    int best      = -1;
    float best_us = 0.0f;
    // Shortest estimate of a valid config of another tile; 0 if none is estimated.
    float other_us = 0.0f;

    bool best_wins() const { return best >= 0 && other_us > 0.0f && best_us < other_us; }
};

inline EstimateRank rank_by_estimate(const ConvParams& par, int out_w);
inline float conv2d_launch_us(const Config& cfg, const ConvParams& par);

// Whether the narrow K64 tile is valid on this shape and runs it in one grid round.
//
// Past one round its lower arithmetic intensity -- half K128's weight reuse per input
// tile, and four waves rather than eight to hide latency with -- costs more than the
// extra tiles gain. Measured on MI355X bf16: over the 3x3x3 VAE geometry it wins by
// 1.63x at up to 216 tiles (one round) and loses by 1.15x from 252 (two); on 3x3 64x64
// K128 it wins by 1.5x at 256 tiles and loses by 1.17x at 288. See
// docs/algorithms/direct/direct-l1-ranking.md.
inline bool narrow_k64_one_round(const ConvParams& par)
{
    // Only conv3d has a dgrad narrow tile.
    Config narrow = par.dims == 3 ? NARROW_K64_3D : NARROW_K64_2D;
    if(par.dims == 3)
        narrow.direction = par.direction;
    return is_valid_config(par, narrow) && grid_rounds(narrow, par) == 1;
}

// Smallest C/64 (K/64 in dgrad) at which the conv2d K128 split-K ranks first.
//
// Measured on MI355X bf16 over 144 3x3 32x32 layers (n 1-4, reduction 256-2560, output
// 256-1280, fprop and dgrad), all of which it splits in one round: it beats every unsplit
// direct_l1 config on all 90 from C/64 = 8 up (1.31x geomean at 8, 2.79x at 40), on 15 of
// 18 at 6 and 3 of 18 at 4. See docs/algorithms/direct/direct-l1-ranking.md.
constexpr int split_k_min_c64 = 8;

// Discount for every other conv2d config where the K128 split-K ranks first: its geomean
// margin at split_k_min_c64.
constexpr float k128_split_k_2d_fills = 1.0f / 1.31f;

// Whether the conv2d K128 split-K ranks first on this shape.
//
// Where the narrow K64 tile also runs in one round, the shorter estimated launch time
// decides, as in 3D: at C/64 = 8 the narrow tile can win, n1 128x128 c512 k256 running
// 0.051 ms against split-K's 0.053.
inline bool k128_split_k_2d_wins(const ConvParams& par, int out_w, int output_chan)
{
    if(par.dims != 2 || output_chan % 128 != 0 || output_chan <= wide_k128_min_output ||
       out_w <= max_unfold_width)
        return false;
    const int reduction =
        par.direction == Direction::Dgrad ? par.filters_per_group() : par.channels_per_group();
    if(divup(reduction, 64) < split_k_min_c64)
        return false;
    Config tile    = K128_SPLIT_K_TILE_2D;
    tile.direction = par.direction;
    if(!is_valid_config(par, tile))
        return false;
    return !narrow_k64_one_round(par) || k128_split_k_us(par) <= narrow_k64_us(par);
}

// Estimated throughput of cfg relative to the fastest conv2d config valid on this shape.
//
// 1 for the config the corpus benchmarks found fastest, exactly one per shape that any
// config supports; each comparison a config loses discounts it by an estimate of its
// margin. The thresholds were established by benchmark on the 2D corpus; conv3d ranks
// separately in relative_throughput_conv3d and relative_throughput_conv3d_tf32.
inline float
relative_throughput_conv2d(const ConvParams& par, const Config& cfg, int out_w, int output_chan)
{
    // The width-1 tiles, the only ones valid on a fold, rank by estimated launch time.
    if(runs_folded(cfg))
    {
        float best_us = 0.0f;
        for(const Config& c : configs)
            if(runs_folded(c) && fits_the_view(par, c))
            {
                const float us = conv2d_launch_us(c, par);
                if(best_us <= 0.0f || us < best_us)
                    best_us = us;
            }
        const float us = conv2d_launch_us(cfg, par);
        return us <= best_us ? 1.0f : fill_tile_discount(best_us, us);
    }

    // The chunked and K64 tiles rank by estimated launch time.
    //
    // Where neither has the shortest estimate, the K64 tiles take the comparisons below. See
    // docs/algorithms/direct/direct-l1-ranking.md.
    const EstimateRank est = rank_by_estimate(par, out_w);
    if(ranks_by_estimate(cfg, output_chan) && (est.best_wins() || cfg.unfold_chunks))
    {
        const float us = conv2d_launch_us(cfg, par);
        if(est.best_wins())
            return cfg == configs[est.best] ? 1.0f : fill_tile_discount(est.best_us, us);
        return est.other_us > 0.0f ? fill_tile_discount(est.other_us, us) : loses_benchmark;
    }
    const float estimate_discount =
        est.best_wins() ? fill_tile_discount(est.best_us, est.other_us) : 1.0f;

    // K128 split-K takes the shapes it splits in one round from split_k_min_c64 up.
    //
    // Every other config ranks below it there, and it ranks below its unsplit tile elsewhere.
    const bool split_wins = k128_split_k_2d_wins(par, out_w, output_chan);
    if(cfg.split_k)
    {
        if(split_wins && cfg.block_k() == 128)
            return estimate_discount;
        Config unsplit  = cfg;
        unsplit.split_k = false;
        return 0.5f * relative_throughput_conv2d(par, unsplit, out_w, output_chan);
    }

    float throughput = (split_wins ? k128_split_k_2d_fills : 1.0f) * estimate_discount;

    // tf32 has no unfold, K256, K96 or narrow K64 sibling, so it loses no comparison
    // with one of those.
    const bool is_tf32 = (cfg.elem_bytes == 4);

    // Batch-unfold gating on direction-mapped output width.
    //
    // Unfold tiles (unfold_n > 1) pack narrow images and support out_w <= w_unfold
    // (==8); a standard tile there computes block_q columns for each out_w. Every
    // square filter 2x2..5x5 has a K32/wave unfold tile.
    if(!is_tf32 && cfg.unfold_n == 1 && filter_has_unfold(cfg) && out_w <= max_unfold_width)
    {
        throughput *= static_cast<float>(max_unfold_width) / cfg.block_q();
    }

    // K(256) versus K(128) selection.
    //
    // The configuration that performs less overcompute ranks first, K(128) in a tie
    // because of its greater arithmetic intensity. Both tiles use 16 rows, so total
    // work is differentiated only by width and output-channel padding.
    const int64_t k256_work =
        static_cast<int64_t>(divup(out_w, 16) * 16) * (divup(output_chan, 256) * 256);
    const int64_t k128_work =
        static_cast<int64_t>(divup(out_w, 32) * 32) * (divup(output_chan, 128) * 128);
    const bool k256_wins = k256_work < k128_work;
    if(!is_tf32)
    {
        const bool filter_has_k256 =
            ((cfg.kw == cfg.kh) && (cfg.kh == 2 || cfg.kh == 3)) || (cfg.kh == 3 && cfg.kw == 1);

        if(cfg.waves_k == 4 && cfg.wave_k16 == 4 && cfg.unfold_n == 1)
        {
            if(!k256_wins)
            {
                throughput *= loses_benchmark * static_cast<float>(k128_work) / k256_work;
            }
        }
        else if(filter_has_k256 && !cfg.single_c && cfg.waves_k != 4 && !cfg.large_tensor)
        {
            // large_tensor is exempt: it has no K256 counterpart (the K256 tiles fail the
            // per-image size gate).
            if(k256_wins)
            {
                throughput *= static_cast<float>(k256_work) / k128_work;
            }
        }
    }

    // The narrow K64 tile takes the K128 shapes too small to fill the machine.
    //
    // A shape inside the unfold window is not a K128 shape. The tile also ranks above K256
    // there, which runs such shapes in K128's time.
    const bool narrow_k64 = !is_tf32 && cfg.waves_q == 1 && cfg.block_k() == 64;
    const bool k128_shape = (output_chan % 128 == 0) && (output_chan > wide_k128_min_output) &&
                            out_w > max_unfold_width;
    const bool narrow_k64_won = k128_shape && narrow_k64_one_round(par);

    if(narrow_k64_won && cfg.waves_k == 4 && cfg.wave_k16 == 4)
    {
        throughput *= k128_k256_underfilled;
    }

    if(cfg.k_divisible)
    {
        if(narrow_k64)
        {
            if(!narrow_k64_won)
            {
                throughput *= narrow_k64_past_one_round;
            }
        }
        else if(narrow_k64_won && cfg.block_k() == 128 && cfg.unfold_n == 1 && !cfg.single_c)
        {
            throughput *= k128_k256_underfilled;
        }
        // K96/K64 rank below the block_k=128 tile above the threshold (every filter has one).
        //
        // tf32 obeys the same threshold: above it the exact tile needs more blocks, and
        // each one re-reads the input tile the halved block_k amortizes over half the
        // channels -- which costs tf32 double, its input tile being twice the bytes.
        //
        // single_c is exempt (tiny C no K128 config covers).
        else if(!cfg.single_c && cfg.block_k() < 128 && output_chan > wide_k128_min_output)
        {
            throughput *= loses_benchmark;
        }
        // K128 unfold ranks below the K256 unfold on multiples of 256 (3x3 only).
        else if(!is_tf32 && cfg.unfold_n > 1 && cfg.block_k() == 128 && cfg.kh == 3 &&
                cfg.kw == 3 && output_chan % 256 == 0)
        {
            throughput *= loses_benchmark;
        }
    }
    else if(cfg.waves_k == 4 && cfg.wave_k16 == 4)
    {
        // K256 arbitrary-output fallback; its divisible sibling takes the multiples of 256.
        if(output_chan % cfg.block_k() == 0)
        {
            throughput *= loses_benchmark;
        }
    }
    else if(cfg.single_c)
    {
        // single_c arbitrary-output fallback (block_k=64).
        //
        // Its divisible single_c sibling (also block_k=64) takes output % 64 == 0. The
        // generic %96/%128 comparison below would favour divisible tiles that need
        // reduction >= 128, which single_c shapes never have.
        if(output_chan % cfg.block_k() == 0)
        {
            throughput *= loses_benchmark;
        }
    }
    else if(cfg.unfold_n > 1)
    {
        // Arbitrary-output batch-unfold fallback (block_k=128).
        //
        // The unfold family has only K128/K256 divisible tiles (no K96/K64), so its
        // divisible siblings take exactly the multiples of 128.
        if(output_chan % cfg.block_k() == 0)
        {
            throughput *= loses_benchmark;
        }
    }
    else
    {
        // Arbitrary-K fallback.
        //
        // Any count is servable (padded + writer-guarded), but a divisible tile covering
        // the count is faster. The padded K128 above threshold keeps K > 96 not
        // divisible by 128, since it beats exact K96/K64.
        //
        // tf32 has a K64 tile but no K96 one, so an output of 96 stays here.
        bool covered = (output_chan % 128 == 0);
        const bool k128_above_thresh =
            (cfg.block_k() == 128) && (output_chan > wide_k128_min_output);
        if(!k128_above_thresh)
        {
            covered = covered || (!is_tf32 && output_chan % 96 == 0);
            covered = covered || (output_chan % 64 == 0);
        }
        if(covered)
        {
            throughput *= loses_benchmark;
        }
    }

    return throughput;
}

// Estimated throughput of a 16-bit cfg relative to the fastest conv3d config valid on this shape.
//
// A conv3d tile covers one output depth plane, so at E == 1 (D == kd, no depth padding)
// depth adds no tiles while the tap loop multiplies each tile's work, and ranking by
// overcompute alone would put first a tile that leaves most of the machine idle. Kept apart
// from relative_throughput_conv2d so 3D measurements do not move its thresholds; the narrow
// K64 one-round test is the only shared rule. See docs/algorithms/direct/direct-3d.md.
inline float
relative_throughput_conv3d(const ConvParams& par, const Config& cfg, int out_w, int output_chan)
{
    float throughput = 1.0f;

    // No 3D config unfolds the batch, so there is no unfold comparison.
    //
    // An unfold tile packs consecutive images along its columns; under the depth fold
    // these are depth planes, each needing a different tap offset.

    // K128 split-K, the narrow K64 tile and the rest rank the K128 shapes three ways.
    //
    // Where both fill tiles are valid, and the narrow one runs in one round, the one with
    // the shorter estimated time ranks first; where only one is, it does; elsewhere the
    // K256 and K128 comparison decides. On an underfilled shape K256 runs no faster than
    // K128, so both rank below the tile that fills the machine. Each predicate is
    // evaluated identically by every config, so exactly one ranks first on each shape.
    const bool k128_shape   = (output_chan % 128 == 0) && (output_chan > wide_k128_min_output);
    const bool split_valid  = k128_shape && k128_split_k_valid(par);
    const bool narrow_valid = k128_shape && narrow_k64_one_round(par);
    const float split_us    = split_valid ? k128_split_k_us(par) : 0.0f;
    const float narrow_us   = narrow_valid ? narrow_k64_us(par) : 0.0f;
    const bool split_wins   = split_valid && (!narrow_valid || split_us <= narrow_us);
    const bool narrow_wins  = narrow_valid && !split_wins;

    if(split_wins || narrow_wins)
    {
        if(cfg.waves_k == 4 && cfg.wave_k16 == 4)
        {
            throughput *= k128_k256_underfilled;
        }
    }
    // K(256) versus K(128) selection, as in 2D.
    else
    {
        const bool filter_has_k256 =
            ((cfg.kw == cfg.kh) && (cfg.kh == 2 || cfg.kh == 3)) || (cfg.kh == 3 && cfg.kw == 1);

        const int64_t k256_work =
            static_cast<int64_t>(divup(out_w, 16) * 16) * (divup(output_chan, 256) * 256);
        const int64_t k128_work =
            static_cast<int64_t>(divup(out_w, 32) * 32) * (divup(output_chan, 128) * 128);

        const bool k256_wins = k256_work < k128_work;

        if(cfg.waves_k == 4 && cfg.wave_k16 == 4)
        {
            if(!k256_wins)
            {
                throughput *= loses_benchmark * static_cast<float>(k128_work) / k256_work;
            }
        }
        else if(filter_has_k256 && !cfg.single_c && !cfg.large_tensor)
        {
            // Everything the K256 tile can take, it takes.
            //
            // The arm above is the K256 tile itself, so waves_k alone no longer
            // identifies it: the narrow K64 tile splits K four ways too.
            if(k256_wins)
            {
                throughput *= static_cast<float>(k256_work) / k128_work;
            }
        }
    }

    if(cfg.k_divisible)
    {
        const bool narrow_k64 = (cfg.waves_q == 1 && cfg.block_k() == 64);

        if(cfg.split_k)
        {
            // The K64 split-K tile wins one conv3d layer of specs/workloads, too few to rank by.
            if(cfg.block_k() != 128)
            {
                throughput *= loses_benchmark;
            }
            else if(narrow_wins)
            {
                throughput *= fill_tile_discount(narrow_us, split_us);
            }
            else if(!split_wins)
            {
                throughput *= loses_benchmark;
            }
        }
        else if(narrow_k64)
        {
            if(split_wins && narrow_valid)
            {
                throughput *= fill_tile_discount(split_us, narrow_us);
            }
            else if(!narrow_wins)
            {
                throughput *= narrow_k64_past_one_round;
            }
        }
        else if(cfg.block_k() == 128 && (split_wins || narrow_wins))
        {
            throughput *= k128_k256_underfilled;
        }
        else if(!cfg.single_c && cfg.block_k() < 128 && output_chan > wide_k128_min_output)
        {
            throughput *= loses_benchmark;
        }
    }
    else if(cfg.waves_k == 4 && cfg.wave_k16 == 4)
    {
        if(output_chan % cfg.block_k() == 0)
        {
            throughput *= loses_benchmark;
        }
    }
    else if(cfg.single_c)
    {
        if(output_chan % cfg.block_k() == 0)
        {
            throughput *= loses_benchmark;
        }
    }
    else
    {
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
            throughput *= loses_benchmark;
        }
    }

    return throughput;
}

inline float tf32_split_k_us(const Config& split, const ConvParams& par);
inline float tf32_unsplit_us(const Config& tile, const ConvParams& par);

// Estimated time in us of a tf32 3x3x3 launch of cfg, split-K or not.
inline float tf32_conv3d_us(const Config& cfg, const ConvParams& par)
{
    return cfg.split_k ? tf32_split_k_us(cfg, par) : tf32_unsplit_us(cfg, par);
}

// Whether cfg is an arbitrary-output config whose divisible twin is in the table and valid on
// this shape; the twin covers the same channels without the per-channel guard, so it ranks in
// cfg's place and cfg just below it.
inline bool cedes_to_divisible(const ConvParams& par, const Config& cfg)
{
    if(cfg.k_divisible)
        return false;
    Config divisible      = cfg;
    divisible.k_divisible = true;
    for(const Config& c : configs)
        if(c == divisible)
            return is_valid_config(par, divisible);
    return false;
}

// The tf32 conv3d config that ranks first on this shape, as an index into configs, with its
// estimated time: the shortest estimate among the valid tf32 3D configs that do not cede to
// a divisible twin, a tie going to the earlier entry; -1 where none is valid. See
// docs/algorithms/direct/direct-l1-ranking.md.
struct Tf32First
{
    int index = -1;
    float us  = 0.0f;
};

inline Tf32First tf32_first(const ConvParams& par)
{
    Tf32First first;
    for(int i = 0; i < num_configs; ++i)
    {
        const Config& c = configs[i];
        if(c.kd == 1 || c.elem_bytes != 4 || c.direction != par.direction ||
           !is_valid_config(par, c) || cedes_to_divisible(par, c))
            continue;
        const float us = tf32_conv3d_us(c, par);
        if(first.index < 0 || us < first.us)
            first = {i, us};
    }
    return first;
}

// Estimated throughput of a tf32 cfg relative to the fastest tf32 conv3d config valid on
// this shape: 1 for tf32_first's config, the ratio of the estimates for the rest, and
// loses_benchmark times its divisible twin's for a config that cedes to it. Kept apart from
// relative_throughput_conv3d, whose comparisons name tiles tf32 does not have. See
// docs/algorithms/direct/direct-l1-ranking.md.
inline float relative_throughput_conv3d_tf32(const ConvParams& par, const Config& cfg)
{
    if(cedes_to_divisible(par, cfg))
    {
        Config divisible      = cfg;
        divisible.k_divisible = true;
        return loses_benchmark * relative_throughput_conv3d_tf32(par, divisible);
    }
    const Tf32First first = tf32_first(par);
    if(first.index >= 0 && cfg == configs[first.index])
        return 1.0f;
    return fill_tile_discount(first.us, tf32_conv3d_us(cfg, par));
}

// Whether cfg supports this problem.
//
// Applies the hard constraints -- dimensionality, filter, depth geometry, channel
// counts, K-partition topology, buffer window and split plan. Which of the configs
// that pass runs the problem fastest is relative_throughput's estimate.
inline bool is_valid_config(const ConvParams& par, const Config& cfg)
{
    // A folding layer gets only width-1 tiles, checked on its fold; any other layer gets none.
    if(folds(par))
        return runs_folded(cfg) && fits_the_view(folded(par), cfg);
    return !runs_folded(cfg) && fits_the_view(par, cfg);
}

// is_valid_config's checks on the layer cfg computes (the fold, for a width-1 tile).
inline bool fits_the_view(const ConvParams& par, const Config& cfg)
{
    if(par.direction != cfg.direction)
    {
        return false;
    }

    if(par.kh != cfg.kh || par.kw != cfg.kw)
    {
        return false;
    }

    // A config's tile is sized for one element width, so the widths must agree. This
    // alone partitions the tf32 configs from the 16-bit ones.
    if(static_cast<int>(sizeof_data_type(par.input_type)) != cfg.elem_bytes)
    {
        return false;
    }
    const bool is_tf32 = (cfg.elem_bytes == 4);

    // Dimensionality, not just the filter depth.
    //
    // A conv3d can reach here with kd == 1: the batch fold also needs pad_d == 0,
    // unit depth stride and unit depth dilation (conv_params.cpp), so a layer like
    // n1_h8_w8_d4_c64_k64_kh3_kw3_kd1 (pad_d defaults to 1) comes back from
    // unfolded() as dims == 3. Matching on kd alone would hand it to a 2D tile,
    // which would ignore depth and write the wrong output extent.
    if(par.dims != (cfg.kd > 1 ? 3 : 2))
    {
        return false;
    }

    if(par.kd != cfg.kd)
    {
        return false;
    }

    if(cfg.kd > 1)
    {
        // The geometry the kernel runs: dgrad's is the transposed problem over dy.
        const DepthGeometry g = depth_geometry(cfg, par);

        // Dgrad's padding is (kd - 1) * dilation_d - pad_d, which a forward padding past
        // the filter's reach makes negative.
        if(g.pad_d < 0)
        {
            return false;
        }

        // Every output plane must have at least one in-bounds depth tap.
        //
        // The tap loop runs the in-bounds range [tap_lo, tap_hi) and has no empty
        // case. Two conditions together make that range non-empty for every plane:
        // the taps must straddle the image, and they must be spaced closely enough
        // to land in it. A dilation wider than the whole input depth can step over
        // the image entirely -- kd 5, d 2, pad_d 8, dilation_d 3 puts plane 2's
        // taps at -4, -1, 2, 5, 8, none of them inside [0, 2). See
        // docs/algorithms/direct/direct-3d.md.
        if(g.pad_d > g.dilation_d * (par.kd - 1))
        {
            return false;
        }
        if(g.dilation_d > g.D)
        {
            return false;
        }

        // Depth folds into the batch axis, and both folded counts are ints.
        if(!product_fits_int(par.n, g.D) || !product_fits_int(par.n, g.E))
        {
            return false;
        }
    }

    // An unfold tile packs unfold_n images along its columns and cannot address
    // an image wider than one of those slots.
    const bool is_dgrad_dir = (cfg.direction == Direction::Dgrad);
    const int out_w         = is_dgrad_dir ? par.w : par.q;
    if(cfg.unfold_n > 1 && !cfg.unfold_chunks && out_w > cfg.w_unfold())
    {
        return false;
    }

    // slot_chunk needs two or more chunks per row.
    if(cfg.unfold_chunks && out_w <= cfg.w_unfold())
    {
        return false;
    }

    // A small reduction (<= 64) at out_w <= 8 is intentionally unsupported.
    //
    // Only single_c tiles take it, and no square filter has a single_c unfold tile, so the
    // standard one would compute 32 columns for each 8; no measurement places it against another
    // family, so such shapes get no config (a clean dispatch failure).
    if(cfg.elem_bytes == 2 && cfg.single_c && filter_has_unfold(cfg) && out_w <= max_unfold_width)
    {
        return false;
    }

    // Per-group reduction/output channel counts (direction-mapped).
    //
    // Each group is an independent conv whose counts drive the peel structure,
    // k-divisibility, and the blocks_k bound.
    const bool is_dgrad      = (cfg.direction == Direction::Dgrad);
    const int reduction_chan = is_dgrad ? par.filters_per_group() : par.channels_per_group();
    const int output_chan    = is_dgrad ? par.channels_per_group() : par.filters_per_group();

    // Reduction channels must be even at 2 bytes.
    //
    // The per-pixel base is 2*count bytes and CDNA4 dwordx4 loads are Dword-aligned.
    // (A non-multiple of 8 over-reads harmlessly into zero-filled weights.) tf32's
    // 4-byte element is Dword-aligned at every count.
    if(!is_tf32 && (reduction_chan % 2 != 0))
    {
        return false;
    }

    // The weights formatter does not support an odd number of input channels at 2 bytes.
    //
    // It loads over the conv C axis at dword (2-fp16) granularity, so an odd C puts the
    // last channel in a dword straddling the buffer's num_records, which the bounds
    // check drops whole (zeroing it). A tf32 dword is one channel, so nothing straddles.
    // Fprop's C is the reduction (even above); dgrad reduces over K, leaving C
    // otherwise unchecked.
    if(!is_tf32 && (par.channels_per_group() % 2 != 0))
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

    // A k_divisible tile writes without a per-channel guard, so it needs an exact
    // multiple of block_k.
    if(cfg.k_divisible && output_chan % cfg.block_k() != 0)
    {
        return false;
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

    // Offsets into one group's formatted weights are ints counting 2-byte units
    // (transpose_weights_kernel, WeightsLoader).
    const int64_t formatted_weights_per_group =
        static_cast<int64_t>(kp.kparts) * kp.per_xcd_blocks * cfg.block_k() *
        direct_transpose_weights::make_divisible(reduction_chan,
                                                 direct_transpose_weights::C_divisor) *
        cfg.kd * par.kh * par.kw * (cfg.elem_bytes / 2);
    if(formatted_weights_per_group > INT32_MAX)
    {
        return false;
    }

    // Buffer-window size gate: partitions large-tensor from baseline configs.
    //
    // Buffer offsets and NUM_RECORDS are 32-bit. A baseline config folds only the
    // tile's image origin into the 64-bit base, so its 32-bit window is one whole
    // image (unfold_n images) and it supports only shapes whose per-image span fits
    // int32. A large_tensor config additionally folds the tile's row origin, shrinking
    // the window to the tile's rows, so it supports shapes whose per-image span does NOT
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

    // A chunk group spans at most two images.
    const size_t imgs = cfg.unfold_chunks ? 2 : cfg.unfold_n;
    const size_t rows = static_cast<size_t>(cfg.block_p() + cfg.kh - 1);

    const bool img_fits =
        (imgs * par.h * hwc_row <= max_span) && (imgs * par.p * pqk_row <= max_span);
    if(cfg.large_tensor)
    {
        // Accept only what baseline cannot (per-image overflow), and only if the row
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

    // A split of one is the unsplit tile plus a round trip through fp32 partials of the
    // whole output.
    if(cfg.split_k && !split_active(cfg, par))
    {
        return false;
    }

    return true;
}

// Estimated throughput of a valid cfg relative to the fastest config valid on this problem.
inline float relative_throughput(const ConvParams& par, const Config& cfg)
{
    const bool is_dgrad   = (cfg.direction == Direction::Dgrad);
    const int out_w       = is_dgrad ? par.w : par.q;
    const int output_chan = is_dgrad ? par.channels_per_group() : par.filters_per_group();
    if(cfg.kd == 1)
    {
        return relative_throughput_conv2d(par, cfg, out_w, output_chan);
    }
    return cfg.elem_bytes == 4 ? relative_throughput_conv3d_tf32(par, cfg)
                               : relative_throughput_conv3d(par, cfg, out_w, output_chan);
}

// Bytes needed to hold the transposed weights in DirectL1 layout.
//
// K_padded x C_padded x Kd x Kh x Kw per group (K rounded to block_k, C to 64),
// packed contiguously across groups. Computed at runtime since Kwg is not
// compile-time here.
inline size_t custom_weights_tensor_size(const Config& cfg, const ConvParams& par)
{
    // Bytes per logical weight: one 2-byte element, or tf32's two bf16 planes.
    const int elem_size = cfg.elem_bytes;
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
    const int C_padded  = direct_transpose_weights::make_divisible(
        reduction_chan, direct_transpose_weights::C_divisor);
    return static_cast<size_t>(par.groups) * K_padded * C_padded * cfg.kd * par.kh * par.kw *
           elem_size;
}

// Byte offset of a split-K launch's fp32 partials: past the formatted weights, 256-B aligned.
inline size_t split_partials_offset(const Config& cfg, const ConvParams& par)
{
    return make_divisible(custom_weights_tensor_size(cfg, par), size_t{256});
}

struct BlockCounts
{
    int k_per_g;
    int p;
    int q;
    int n;
    int g;
    // Output depth planes; one per tile, so the count is the extent. 1 at kd == 1.
    int e;
};

// Output-tile block counts along each axis (direction-mapped).
inline BlockCounts get_block_counts(const Config& cfg, const ConvParams& par)
{
    const bool is_dgrad    = (cfg.direction == Direction::Dgrad);
    const int output_per_g = is_dgrad ? par.channels_per_group() : par.filters_per_group();
    const int out_p        = is_dgrad ? par.h : par.p;
    const int out_q        = is_dgrad ? par.w : par.q;
    const int out_e        = cfg.kd == 1 ? 1 : (is_dgrad ? par.d : par.e);
    // Batch-unfold packs unfold_n images along the columns.
    //
    // n divides down and the Q count collapses to 1. unfold_n == 1 leaves originals.
    // unfold_chunks: n counts slot groups over N x chunks.
    const int q_blocks = cfg.unfold_n == 1 ? divup(out_q, cfg.block_q()) : 1;
    const int chunks   = cfg.unfold_chunks ? divup(out_q, cfg.w_unfold()) : 1;
    return {
        .k_per_g = divup(output_per_g, cfg.block_k()),
        .p       = divup(out_p, cfg.block_p()),
        .q       = q_blocks,
        .n       = divup(par.n * chunks, cfg.unfold_n),
        .g       = par.groups,
        .e       = out_e,
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
inline KPartition get_k_partition(const Config& cfg, const ConvParams& par)
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
inline int padded_output_channels(const Config& cfg, const ConvParams& par)
{
    const KPartition kp = get_k_partition(cfg, par);
    return kp.kparts * kp.per_xcd_blocks * cfg.block_k();
}

// Output tiles of the launch, counting every padded K-block.
//
// The index space SplitPartials::tile_index spans.
inline size_t tile_count(const Config& cfg, const ConvParams& par)
{
    const BlockCounts bc = get_block_counts(cfg, par);
    const KPartition kp  = get_k_partition(cfg, par);
    return static_cast<size_t>(bc.g) * bc.n * bc.q * bc.p * bc.e * kp.kparts * kp.per_xcd_blocks;
}

// The depth geometry the kernel for cfg sees: dgrad's is the transposed problem over dy.
inline DepthGeometry depth_geometry(const Config& cfg, const ConvParams& par)
{
    // Conv2d: one output plane and Kd = 1, so a split-K plan divides C/64 alone.
    if(cfg.kd == 1)
        return {1, 1, 1, 0, 1};
    if(cfg.direction == Direction::Dgrad)
    {
        const auto view = SizeView<Direction::Dgrad>(par);
        return {view.d(), view.e(), view.stride_d(), view.pad_d(), view.dilation_d()};
    }
    const auto view = SizeView<Direction::Fprop>(par);
    return {view.d(), view.e(), view.stride_d(), view.pad_d(), view.dilation_d()};
}

inline SplitPlan split_plan(const Config& cfg, const ConvParams& par);

// Whether a launch of cfg decodes its grid with RaggedL2Grid instead of BalancedL2Grid.
//
// The chunked tiles' launches need the CUs BalancedL2Grid leaves idle at per_xcd = 5 or 15
// (K1280, C1920 dgrad). See docs/algorithms/direct/direct-l1-chunked-unfold.md.
constexpr bool ragged_grid(const Config& cfg)
{
    return cfg.unfold_chunks;
}

// Uniform bound on the rounds of cfg's grid over the given X axis.
inline int grid_num_rounds(const Config& cfg,
                           int per_xcd,
                           int blocks_p,
                           int blocks_q,
                           int n_blocks,
                           int groups,
                           int xgroups)
{
    return ragged_grid(cfg)
               ? RaggedL2Grid::num_rounds(per_xcd, blocks_p, blocks_q, n_blocks, groups, xgroups)
               : BalancedL2Grid::num_rounds(per_xcd, blocks_p, blocks_q, n_blocks, groups, xgroups);
}

// Rounds of the persistent grid a launch of cfg takes on par.
//
// A split-K launch carries the plane in the group axis instead of P (decode_item).
inline int grid_rounds(const Config& cfg, const ConvParams& par)
{
    const BlockCounts bc = get_block_counts(cfg, par);
    const KPartition kp  = get_k_partition(cfg, par);
    const int blocks_p   = cfg.split_k ? bc.p : bc.p * bc.e;
    const int items      = cfg.split_k ? split_plan(cfg, par).T : 1;
    return grid_num_rounds(cfg, kp.per_xcd_blocks, blocks_p, bc.q, bc.n, bc.g * items, kp.xgroups);
}

// The split-K plan of a launch; one item per plane (T == E) when no split applies.
//
// Each plane splits its own live steps, so the items' lengths, not their counts per
// tile, are what the plan equalizes: it picks an item length, and a plane with fewer
// live taps takes fewer items. Among the lengths whose ranges are valid at every live
// tap count and whose items all run in one round, take the shortest items, and the
// fewest items among those, which writes the fewest partials.
inline SplitPlan split_plan(const Config& cfg, const ConvParams& par)
{
    const DepthGeometry g = depth_geometry(cfg, par);
    const int reduction =
        cfg.direction == Direction::Dgrad ? par.filters_per_group() : par.channels_per_group();
    const int C64  = divup(reduction, 64);
    const int full = cfg.kd * C64;
    SplitPlan best = make_split_plan(g, cfg.kd, C64, full);
    if(!cfg.split_k || cfg.kd > SPLIT_PLAN_MAX_KD ||
       best.e_lo + (g.E - best.e_hi) > SPLIT_PLAN_MAX_EDGE_PLANES)
        return best;

    uint32_t counts = 0;
    for(int e = 0; e < g.E; ++e)
        counts |= 1u << live_taps(g, cfg.kd, e);

    const BlockCounts bc = get_block_counts(cfg, par);
    const KPartition kp  = get_k_partition(cfg, par);
    int best_len         = full;

    // Lengths downward, one per distinct plan.
    //
    // T rises as the length falls, so the first multi-round plan ends the walk; <= keeps
    // the shortest length on a tie.
    for(int len = full - 1, next; len >= 2; len = next)
    {
        const SplitPlan p = make_split_plan(g, cfg.kd, C64, len);
        next              = 0;
        for(int nt = 1; nt <= cfg.kd; ++nt)
            next = maximum(next, divup(nt * C64, p.S_nt[nt]) - 1);

        if(grid_num_rounds(cfg, kp.per_xcd_blocks, bc.p, bc.q, bc.n, bc.g * p.T, kp.xgroups) != 1)
            break;
        bool valid   = true;
        int item_len = 0;
        for(int nt = 1; nt <= cfg.kd; ++nt)
        {
            if(!(counts & (1u << nt)))
                continue;
            valid    = valid && split_ranges_valid(p.S_nt[nt], nt * C64, C64);
            item_len = maximum(item_len, divup(nt * C64, p.S_nt[nt]));
        }
        if(!valid)
            continue;
        if(item_len < best_len || (item_len == best_len && p.T <= best.T))
        {
            best     = p;
            best_len = item_len;
        }
    }
    return best;
}

// Whether a split-K launch of cfg splits any plane.
inline bool split_active(const Config& cfg, const ConvParams& par)
{
    return cfg.split_k && split_plan(cfg, par).T > depth_geometry(cfg, par).E;
}

// Partial slots of a split-K launch: one per (item, K-block).
inline size_t split_slot_count(const Config& cfg, const ConvParams& par)
{
    const BlockCounts bc = get_block_counts(cfg, par);
    const KPartition kp  = get_k_partition(cfg, par);
    return static_cast<size_t>(bc.g) * split_plan(cfg, par).T * bc.n * bc.q * bc.p * kp.kparts *
           kp.per_xcd_blocks;
}

// Reduction channels per group of a direct_l1 problem: K for dgrad, C otherwise.
inline int reduction_channels(Direction direction, const ConvParams& par)
{
    return direction == Direction::Dgrad ? par.filters_per_group() : par.channels_per_group();
}

// Estimated time in us of the narrow K64 tile's launch, given it runs in one round.
//
// Linear in the longest tile's (live Kd, C/64) steps, C/64 alone in 2D; fitted over 578
// MI355X bf16 3x3x3 layers. See docs/algorithms/direct/direct-l1-ranking.md.
inline float narrow_k64_us(const ConvParams& par)
{
    Config narrow = par.dims == 3 ? NARROW_K64_3D : NARROW_K64_2D;
    if(par.dims == 3)
        narrow.direction = par.direction;
    const DepthGeometry g = depth_geometry(narrow, par);
    int taps              = 0;
    for(int e = 0; e < g.E && taps < narrow.kd; ++e)
        taps = maximum(taps, live_taps(g, narrow.kd, e));
    const int steps = taps * divup(reduction_channels(par.direction, par), 64);
    return 4.78f * steps + 10.8f;
}

// (live tap, C(64)) steps of the longest item of a split-K launch of cfg.
inline int split_item_steps(const Config& cfg, const ConvParams& par)
{
    const DepthGeometry g = depth_geometry(cfg, par);
    const SplitPlan plan  = split_plan(cfg, par);
    const int C64         = divup(reduction_channels(cfg.direction, par), 64);
    uint32_t counts       = 0;
    for(int e = 0; e < g.E; ++e)
        counts |= 1u << live_taps(g, cfg.kd, e);
    int steps = 0;
    for(int nt = 1; nt <= cfg.kd; ++nt)
        if(counts & (1u << nt))
            steps = maximum(steps, divup(nt * C64, plan.S_nt[nt]));
    return steps;
}

// Estimated time in us of the K128 split-K launch, partials and reduction included.
//
// Linear in the longest item's (live tap, C(64)) steps, the C(64) blocks per tap and the
// partial slots up to 64, past which the slot count stops adding time; fitted with
// narrow_k64_us.
inline float k128_split_k_us(const ConvParams& par)
{
    Config split    = par.dims == 3 ? K128_SPLIT_K_TILE : K128_SPLIT_K_TILE_2D;
    split.direction = par.direction;
    const int C64   = divup(reduction_channels(par.direction, par), 64);
    const int slots = static_cast<int>(minimum(split_slot_count(split, par), size_t{64}));
    return 7.87f * split_item_steps(split, par) + 0.89f * C64 + 0.22f * slots + 15.1f;
}

// Estimated time in us of a tf32 3x3x3 split-K launch of the K128 or K64 tile, partials and
// reduction included: linear in the longest item's steps and the C(64) blocks per tap,
// fitted per tile over 444 MI355X layers. See docs/algorithms/direct/direct-3d.md.
inline float tf32_split_k_us(const Config& split, const ConvParams& par)
{
    const int steps = split_item_steps(split, par);
    const int C64   = divup(reduction_channels(par.direction, par), 64);
    return split.block_k() == 64 ? 7.47f * steps + 1.09f * C64 + 16.3f
                                 : 9.86f * steps + 2.86f * C64 + 20.7f;
}

// A busy CU's share of its peak with `busy` CUs working; the clock falls as more work.
//
// MI355X bf16, 3x3 conv2d. See docs/algorithms/direct/direct-l1-ranking.md.
inline float cu_rate(int busy)
{
    constexpr int n            = 6;
    constexpr int at[n]        = {0, 128, 176, 208, 240, 256};
    constexpr float rate_at[n] = {0.89f, 0.885f, 0.84f, 0.775f, 0.73f, 0.72f};
    for(int i = 1; i < n; ++i)
        if(busy <= at[i])
            return rate_at[i - 1] + (rate_at[i] - rate_at[i - 1]) *
                                        static_cast<float>(busy - at[i - 1]) /
                                        static_cast<float>(at[i] - at[i - 1]);
    return rate_at[n - 1];
}

// Rounds of a BalancedL2Grid launch, and the sum over them of 1 / cu_rate(busy CUs).
struct GridOccupancy
{
    int rounds;
    float inverse_rate;
};

// The occupancy of a BalancedL2Grid launch over x X items, in closed form.
//
// See docs/algorithms/direct/direct-l1-ranking.md.
inline GridOccupancy balanced_grid_occupancy(int x, int xgroups, int per_xcd, int cus_per_slot)
{
    const int seg         = divup(x, xgroups);
    const int x_xcd       = BalancedL2Grid::x_xcd(per_xcd);
    const int full_segs   = x / seg;
    const int rest        = x % seg;
    const int rounds      = divup(seg, x_xcd);
    const int rest_rounds = divup(rest, x_xcd);
    const int m_full      = seg % rounds;
    const int m_rest      = rest_rounds > 0 ? rest % rest_rounds : 0;
    const int lo          = minimum(minimum(m_full, m_rest), rest_rounds);
    const int hi          = maximum(maximum(m_full, m_rest), rest_rounds);
    const int bounds[]    = {0, lo, m_full + m_rest + rest_rounds - lo - hi, hi, rounds};
    float inverse_rate    = 0.0f;
    for(int i = 0; i < 4; ++i)
    {
        const int r = bounds[i];
        if(bounds[i + 1] == r)
            continue;
        int slots = full_segs * (seg / rounds + (r < m_full ? 1 : 0));
        if(r < rest_rounds)
            slots += rest / rest_rounds + (r < m_rest ? 1 : 0);
        inverse_rate += static_cast<float>(bounds[i + 1] - r) / cu_rate(slots * cus_per_slot);
    }
    return {rounds, inverse_rate};
}

// The occupancy of a RaggedL2Grid launch over `pairs` (X item, K block) pairs of a K-slice,
// each busy CU counted on all kparts slices.
//
// Round r runs, on each XCD, its pairs [32r, 32r + 32), so a segment of c pairs keeps all 32
// CUs busy before round c / 32, c % 32 in that round and none after. The pairs fill
// pairs / seg segments and leave pairs % seg in one more, so the busy count changes at no
// more than four rounds.
inline GridOccupancy ragged_grid_occupancy(int pairs, int xgroups, int kparts)
{
    constexpr int cus   = persistent::NUM_CU_PER_XCD;
    const int seg       = divup(pairs, xgroups);
    const int full_segs = pairs / seg;
    const int rest      = pairs % seg;
    const int rounds    = divup(seg, cus);
    const int bounds[]  = {0,
                           rest / cus,
                           minimum(divup(rest, cus), seg / cus),
                           maximum(divup(rest, cus), seg / cus),
                           rounds};
    float inverse_rate  = 0.0f;
    for(int i = 0; i < 4; ++i)
    {
        const int r = bounds[i];
        if(bounds[i + 1] == r)
            continue;
        const int busy =
            full_segs * RaggedL2Grid::busy_cus(seg, r) + RaggedL2Grid::busy_cus(rest, r);
        inverse_rate += static_cast<float>(bounds[i + 1] - r) / cu_rate(busy * kparts);
    }
    return {rounds, inverse_rate};
}

// A conv2d tile's rate relative to cu_rate, in a one-round launch and past one round.
struct TileRate
{
    float one_round;
    float rounds;
};

// The tiles conv2d_launch_us covers; false for the rest.
inline bool tile_rate(const Config& cfg, TileRate& rate)
{
    if(cfg.unfold_chunks && cfg.split_k)
        rate = {0.857f, 0.857f};
    else if(cfg.unfold_chunks && cfg.block_k() == 256)
        rate = {0.990f, 0.943f};
    else if(cfg.unfold_chunks && cfg.block_k() == 192)
        rate = {0.934f, 0.929f};
    else if(cfg.unfold_chunks)
        rate = {0.738f, 0.74f};
    // Width-1 tiles. A tile with no one-round measurement uses its past-one-round rate for both.
    else if(runs_folded(cfg) && cfg.single_c)
        rate = {0.513f, 0.513f};
    else if(runs_folded(cfg) && cfg.kh == 5)
        rate = {0.723f, 0.723f};
    else if(runs_folded(cfg) && cfg.block_k() == 256 && cfg.unfold_n == 32)
        rate = {1.285f, 0.820f};
    else if(runs_folded(cfg) && cfg.block_k() == 256)
        rate = {1.131f, 0.812f};
    else if(runs_folded(cfg) && cfg.unfold_n == 32)
        rate = {1.175f, 0.660f};
    else if(runs_folded(cfg))
        rate = {1.164f, 0.632f};
    else if(cfg.unfold_n > 1 || cfg.block_p() != 16)
        return false;
    // A c64_step tile has one rate with or without split-K.
    else if(cfg.block_k() == 64 && cfg.block_q() == 32 && cfg.c64_step)
        rate = {0.948f, 0.84f};
    else if(cfg.split_k)
        rate = {1.007f, 1.007f};
    else if(cfg.block_k() == 256 && cfg.block_q() == 16)
        rate = {1.060f, 1.017f};
    else if(cfg.block_k() == 128 && cfg.block_q() == 32 && cfg.wave_k16 == 4)
        rate = {1.067f, 1.025f};
    else if(cfg.block_k() == 64 && cfg.block_q() == 32)
        rate = {0.735f, 0.774f};
    else if(cfg.block_k() == 64 && cfg.block_q() == 16)
        rate = {0.406f, 0.498f};
    else
        return false;
    return true;
}

// The occupancy of cfg's launch on par, with `items` split items per output tile.
//
// K blocks added by padding the block count run the main loop against zero weights, so a
// busy slot keeps kparts * per_xcd CUs busy, not k_per_g; RaggedL2Grid's pairs include them.
inline GridOccupancy launch_occupancy(const Config& cfg, const ConvParams& par, int items)
{
    const BlockCounts bc = get_block_counts(cfg, par);
    const KPartition kp  = get_k_partition(cfg, par);
    const int groups     = bc.g * items;
    if(ragged_grid(cfg))
        return ragged_grid_occupancy(
            RaggedL2Grid::pair_total(kp.per_xcd_blocks, bc.p, bc.q, bc.n, groups),
            kp.xgroups,
            kp.kparts);
    return balanced_grid_occupancy(
        groups * bc.n * bc.q * bc.p, kp.xgroups, kp.per_xcd_blocks, kp.kparts * kp.per_xcd_blocks);
}

// Estimated time in us of an unsplit tf32 3x3x3 launch of the K128 tile, divisible or
// arbitrary output, or of the K64 tile, single_c or not: linear in the busiest grid slot's
// (live tap, C(64)) steps, each over cu_rate at the CUs busy in its round, fitted per tile
// over 1,055 MI355X layers, the single_c tiles over 284. See
// docs/algorithms/direct/direct-3d.md.
inline float tf32_unsplit_us(const Config& tile, const ConvParams& par)
{
    const BlockCounts bc    = get_block_counts(tile, par);
    const KPartition kp     = get_k_partition(tile, par);
    const DepthGeometry g   = depth_geometry(tile, par);
    const GridOccupancy occ = balanced_grid_occupancy(bc.g * bc.n * bc.q * bc.p * bc.e,
                                                      kp.xgroups,
                                                      kp.per_xcd_blocks,
                                                      kp.kparts * kp.per_xcd_blocks);

    // A slot runs occ.rounds consecutive items with the plane innermost: whole cycles of the
    // E planes, then the window of the rest over the most live taps.
    const int rest = occ.rounds % g.E;
    int cycle      = 0;
    for(int e = 0; e < g.E; ++e)
        cycle += live_taps(g, tile.kd, e);
    int window = 0;
    if(rest > 0)
    {
        int taps = 0;
        for(int e = 0; e < rest; ++e)
            taps += live_taps(g, tile.kd, e);
        window = taps;
        for(int s = 1; s < g.E; ++s)
        {
            taps += live_taps(g, tile.kd, (s + rest - 1) % g.E) - live_taps(g, tile.kd, s - 1);
            window = maximum(window, taps);
        }
    }
    const int steps =
        (occ.rounds / g.E * cycle + window) * divup(reduction_channels(par.direction, par), 64);
    const float clocked_steps = steps * occ.inverse_rate / occ.rounds;
    if(tile.single_c)
        return tile.block_k() == 64 ? 5.91f * clocked_steps + 14.0f
                                    : 10.03f * clocked_steps + 14.4f;
    if(tile.block_k() == 64)
        return 6.24f * clocked_steps + 15.1f;
    return tile.k_divisible ? 10.6f * clocked_steps + 11.4f : 9.99f * clocked_steps + 21.0f;
}

// Estimated us of a conv2d launch of cfg; negative for a tile it does not cover.
//
// Fitted on MI355X bf16. See docs/algorithms/direct/direct-l1-ranking.md.
inline float conv2d_launch_us(const Config& cfg, const ConvParams& par)
{
    TileRate rate;
    if(!tile_rate(cfg, rate))
        return -1.0f;
    const int C64  = divup(reduction_channels(par.direction, par), 64);
    int items      = 1;
    int item_steps = C64;
    int splits     = 1;
    if(cfg.split_k)
    {
        const SplitPlan plan = split_plan(cfg, par);
        items                = plan.T;
        splits               = plan.S_nt[1];
        item_steps           = divup(C64, splits);
    }
    const GridOccupancy occ = launch_occupancy(cfg, par, items);

    // FLOPs of one C(64) step of a tile, over a CU's peak of 4096 FLOP per cycle at 2.4 GHz.
    const float step_us = 2.0f * cfg.block_p() * cfg.block_q() * cfg.block_k() * 64 * cfg.kh *
                          cfg.kw / (4096.0f * 2400.0f);
    const float weight_mb =
        static_cast<float>(par.groups) * reduction_channels(par.direction, par) *
        (par.direction == Direction::Dgrad ? par.channels_per_group() : par.filters_per_group()) *
        cfg.kh * cfg.kw * 2 / 1.0e6f;
    const float tile_rate_share = occ.rounds == 1 ? rate.one_round : rate.rounds;
    const float split_us        = cfg.split_k ? 7.72f + 1.40f * splits : 0.0f;
    // A c64_step tile's fixed cost measures about 3 us above the 13.05 us intercept below.
    const float c64_us = cfg.c64_step ? 3.0f : 0.0f;
    return item_steps * step_us * occ.inverse_rate / tile_rate_share + 13.05f + 0.072f * weight_mb +
           split_us + c64_us;
}

inline EstimateRank rank_by_estimate(const ConvParams& par, int out_w)
{
    EstimateRank rank;
    if(par.dims != 2 || par.kh != 3 || par.kw != 3 || sizeof_data_type(par.input_type) != 2 ||
       out_w <= max_unfold_width)
        return rank;
    const int output_chan =
        par.direction == Direction::Dgrad ? par.channels_per_group() : par.filters_per_group();
    for(int i = 0; i < num_configs; ++i)
    {
        const Config& c = configs[i];
        if(c.kh != 3 || c.kw != 3 || c.kd != 1 || c.direction != par.direction ||
           c.elem_bytes != 2 || !is_valid_config(par, c))
            continue;
        // Skip a K64 split-K tile that does not rank by estimate, so it does not set other_us.
        if(c.split_k && c.c64_step && !ranks_by_estimate(c, output_chan))
            continue;
        const float us = conv2d_launch_us(c, par);
        if(us < 0.0f)
            continue;
        if(ranks_by_estimate(c, output_chan))
        {
            if(rank.best < 0 || us < rank.best_us)
            {
                rank.best    = i;
                rank.best_us = us;
            }
        }
        else if(rank.other_us <= 0.0f || us < rank.other_us)
        {
            rank.other_us = us;
        }
    }
    return rank;
}

// Workspace bytes: the formatted weights at offset 0, then a split-K launch's partials.
//
// Self-formatting makes the transpose cost part of the timed launch; hipMalloc's
// >= 256-B alignment covers the dwordx4 loads.
inline size_t get_workspace_size(const Config& cfg, const ConvParams& par)
{
    if(!cfg.split_k)
        return custom_weights_tensor_size(cfg, par);
    return split_partials_offset(cfg, par) + split_partials_bytes(cfg, split_slot_count(cfg, par));
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

// Depth geometry of a conv3d launch; read only under cfg.kd > 1.
//
// Carried in a struct rather than seven more scalars so the 2D entry point can
// default-construct it and let the constant folder delete every use, leaving that
// kernel's kernarg segment (and hence its preload SGPR allocation) untouched.
struct DepthPars
{
    // Input and output depth extents. blocks_e == E: one output plane per tile.
    int D = 1;
    int E = 1;

    int stride_d   = 1;
    int pad_d      = 0;
    int dilation_d = 1;

    // Splits the grid's inflated P axis into (p, e) with e innermost.
    MagicDiv magic_e;

    // Split-K: the plan, the divmods splitting the grid's group axis into (group,
    // item) and an interior plane's items into (plane, split), and the fp32 partials.
    SplitPlan plan;
    MagicDiv magic_items;
    MagicDiv magic_full;
    fp32x4_t* partials = nullptr;
};


// One tile's P-block and output depth plane, decoded from the inflated P axis.
struct PlaneIndex
{
    int p;
    int e;
};

// Split the grid's P axis into its P-block and depth plane, e innermost.
//
// A split-K launch keeps P spatial and carries the plane in the group axis instead
// (decode_item).
//
// readfirstlane because the halves feed the tap bounds, which bound a loop
// containing s_barrier: a VGPR bound would put that barrier in control flow the
// compiler must treat as divergent.
template <int Kd>
__device__ PlaneIndex split_depth(int p_and_e, const DepthPars& dpars)
{
    if constexpr(Kd == 1)
    {
        return {p_and_e, 0};
    }
    else
    {
        uint32_t p = 0, e = 0;
        dpars.magic_e.divmod(static_cast<uint32_t>(p_and_e), static_cast<uint32_t>(dpars.E), p, e);
        return {__builtin_amdgcn_readfirstlane(static_cast<int>(p)),
                __builtin_amdgcn_readfirstlane(static_cast<int>(e))};
    }
}

// A work item: its P-block, depth plane and group, and under split-K split s of S.
struct WorkItem
{
    int p;
    int e;
    int g;
    int s;
    int S;
};

// Decode the grid's P and group axes into a work item.
//
// Split-K numbers a column's items plane-major in the inner half of the group axis
// (SplitPlan).
template <Config cfg>
__device__ WorkItem decode_item(int p_idx, int g_idx, const DepthPars& dpars)
{
    if constexpr(!cfg.split_k)
    {
        const PlaneIndex pe = split_depth<cfg.kd>(p_idx, dpars);
        return {pe.p, pe.e, g_idx, 0, 1};
    }
    else
    {
        uint32_t g = 0, j = 0;
        dpars.magic_items.divmod(
            static_cast<uint32_t>(g_idx), static_cast<uint32_t>(dpars.plan.T), g, j);
        const PlaneItem it = plane_item(dpars.plan, dpars, cfg.kd, static_cast<int>(j), [&](int x) {
            return static_cast<int>(dpars.magic_full.div(static_cast<uint32_t>(x)));
        });
        return {p_idx, it.e, static_cast<int>(g), it.s, it.S};
    }
}

template <int Kd>
__device__ TapRange tap_range(int d0, const DepthPars& dpars)
{
    if constexpr(Kd == 1)
    {
        return {0, 1};
    }
    else
    {
        const TapRange r = tap_range_for(d0, dpars.D, dpars.dilation_d, Kd);
        return {__builtin_amdgcn_readfirstlane(r.lo), __builtin_amdgcn_readfirstlane(r.hi)};
    }
}

template <Config cfg, DataType DT>
__device__ void
conv_direct_l1_impl(const ToType<DT>* __restrict__ in,
                    const direct_transpose_weights::ToSplitType<DT>* __restrict__ wei,
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
                    MagicDiv magic_n,
                    DepthPars dpars)
{
    // Waves tiling one K-partition's P x Q plane; the K axis splits across waves_k.
    constexpr int waves_per_kpart = cfg.waves_p * cfg.waves_q;

    // The depth fold reuses the loaders' image axis, which the unfold also claims.
    static_assert(cfg.kd == 1 || cfg.unfold_n == 1,
                  "batch-unfold packs consecutive images along the columns; under the "
                  "depth fold those are depth planes needing different tap offsets");
    // large_tensor composes with the depth fold but is not exercised; no 3D
    // large_tensor config exists, so refuse one rather than ship it untested.
    static_assert(cfg.kd == 1 || !cfg.large_tensor, "no 3D large_tensor config exists");

    // Split-K takes its plan and partials from DepthPars, which only the conv3d entry
    // point carries; launch_impl sends a conv2d split-K config there too.

    // LDS and the formatted weights are bf16 even for tf32; only `in` and `out` are fp32.
    using LdsType        = direct_transpose_weights::ToSplitType<DT>;
    using InLoaderGlobal = InputLoader<cfg, LdsType>;
    using InLoaderLds    = InputLoaderLds<cfg, LdsType>;
    using WgtsLoader     = WeightsLoader<cfg, LdsType>;
    // One MFMA operand tile row: a plain x8 vector, or the bf16 pair for tf32.
    using datatypex8_t = typename InLoaderLds::datatypex8_t;

    static_assert(std::is_same_v<datatypex8_t, typename WgtsLoader::datatypex8_t>,
                  "the input and weights loaders must agree on the mma operand type");

    // Fires when block_p is too large for this filter; reduce wave_p.
    //
    // True even without the round-overlap pad (see input_lds_layout.h overlap_safe).
    // tf32 counts double here: its buffer carries both bf16 planes.
    static_assert(InLoaderGlobal::lds_size * static_cast<int>(sizeof(LdsType)) <= LDS_BYTES_PER_CU,
                  "direct_l1 input LDS tile exceeds the 160 KiB per-CU budget "
                  "even without the round-overlap pad. Reduce wave_p (block_p) "
                  "for this filter size.");

    // Two wavegroups of four waves, which is what the ping-pong is for.
    //
    // The halves run out of phase so one computes while the other moves memory, and
    // a CU has four SIMDs, so a compute phase needs four waves to fill it. wave_group
    // splits the workgroup in half whatever its size, so a smaller workgroup still
    // runs and still verifies -- it just computes on half the SIMDs.
    static_assert(cfg.num_waves() == 8,
                  "direct_l1 is an 8-wave ping-pong: two wavegroups of four, one per "
                  "SIMD. Set waves_p * waves_q * waves_k == 8.");

    static_assert(!cfg.c64_step || cfg.elem_bytes == 2, "c64_step is built for 16-bit tiles only");

    // The workgroup must have a wave for every W-strip of the input tile.
    //
    // Waves beyond num_w_waves are legal and just compute; waves short of it are
    // not, and fail silently: the strips nobody owns are never loaded and the
    // kernel convolves whatever the LDS held. Implied by the wave count above, and
    // kept because it names the failure that made it necessary.
    static_assert(cfg.num_waves() >= InLoaderGlobal::num_w_waves,
                  "direct_l1 workgroup has fewer waves than the input tile has "
                  "W-strips, so some strip would never be loaded. Raise waves_p, "
                  "waves_q or waves_k, or narrow block_q.");
    __shared__ LdsType input_lds[InLoaderGlobal::lds_size];

    // Full per-pixel channel count; in_base selects the group's interior slice.
    const int C = groups * c_per_group;

    // Depth folds into the batch axis: [N][D][H][W][C] is [N*D][H][W][C] because the
    // tensors are channels-last, so N and D are adjacent. The loaders and the writer
    // then need no depth awareness at all -- a depth plane IS an image to them.
    InputPars input_pars{cfg.kd == 1 ? N : N * dpars.D,
                         hi,
                         wi,
                         C,
                         cfg.unfold_chunks ? divup(wo, cfg.w_unfold()) : 1};

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
    // A split-K launch carries its items as the inner half of the grid's group axis.
    const int grid_groups = cfg.split_k ? groups * dpars.plan.T : groups;

    using Grid = std::conditional_t<ragged_grid(cfg), RaggedL2Grid, BalancedL2Grid>;

    // Every round is valid, so no in-loop valid()/break; tail workgroups get zero.
    const int rounds = Grid::num_rounds_for_workgroup(
        per_xcd, blocks_p, blocks_q, n_blocks, grid_groups, xgroups, xgroup, wg_id);

    // Round-invariant decode inputs, parked in LDS.
    //
    // Held live across the compute phase they would spill K128. See
    // round_invariant_context.h.
    __shared__ int round_ctx_lds[RoundInvariantContext::NUM_SLOTS];
    {
        RoundInvariantContext ctx;
        if constexpr(ragged_grid(cfg))
        {
            ctx.seg_begin = RaggedL2Grid::pair_begin(
                per_xcd, blocks_p, blocks_q, n_blocks, grid_groups, xgroups, xgroup);
            ctx.seg_count = RaggedL2Grid::pair_count(
                per_xcd, blocks_p, blocks_q, n_blocks, grid_groups, xgroups, xgroup);
        }
        else
        {
            ctx.seg_begin =
                BalancedL2Grid::x_begin(blocks_p, blocks_q, n_blocks, grid_groups, xgroups, xgroup);
            ctx.seg_count =
                BalancedL2Grid::x_count(blocks_p, blocks_q, n_blocks, grid_groups, xgroups, xgroup);
        }
        ctx.round_count =
            Grid::round_count(per_xcd, blocks_p, blocks_q, n_blocks, grid_groups, xgroups, xgroup);
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
        Grid sched(ctx.per_xcd,
                   ctx.kslice,
                   ctx.blocks_p,
                   ctx.blocks_q,
                   ctx.n_blocks,
                   ctx.seg_begin,
                   ctx.seg_count,
                   ctx.round_count,
                   magic_per_xcd,
                   magic_p,
                   magic_q,
                   ctx.magic_n,
                   ctx.wg_id,
                   round);

        const int block_n = sched.n_idx();
        // The grid's P axis carries the depth sub-axis, e innermost (see the host's
        // blocks_p inflation), except under split-K. Splitting it here costs one divmod,
        // and only in 3D.
        const WorkItem item = decode_item<cfg>(sched.p_idx(), sched.g_idx(), dpars);
        const int block_e   = item.e;
        const int block_p   = item.p * cfg.block_p();
        const int block_q   = sched.q_idx() * cfg.block_q();
        const int k_idx     = sched.k_idx();
        const int g_idx     = item.g; // outermost X dimension

        const int block_h = block_p - py;
        const int block_w = block_q - px;
        const int block_c = 0;

        // Input depth of tap 0 for this output plane; tap t reads d0 + t*dilation_d.
        const int d0 = cfg.kd == 1 ? 0 : block_e * dpars.stride_d - dpars.pad_d;


        // The in-bounds tap range, monotonic in t so it is a contiguous interval.
        //
        // Running only this range is a correctness requirement, not a saving: an
        // out-of-image depth plane does NOT null the way an out-of-image row or
        // column does. At d == -1 the folded image index n*D - 1 addresses the
        // previous batch item's last plane and at d == D the next item's first --
        // real data in both cases, and the buffer bounds check sees nothing wrong.
        // is_valid_config's pad_d gate is what makes the range non-empty.
        //
        // A split-K item runs its share of those taps' (tap, C(64)) steps: taps
        // [taps.lo, taps.hi), starting at C(64) block c_first in the first and ending
        // after block c_last_end in the last.
        TapRange taps   = tap_range<cfg.kd>(d0, dpars);
        int c_first     = 0;
        int c_last_end  = C64;
        int split_first = 0;
        if constexpr(cfg.split_k)
        {
            const StepRange r = split_live_range(item.s, item.S, taps, C64);
            split_first       = __builtin_amdgcn_readfirstlane(r.lo);
            taps              = {__builtin_amdgcn_readfirstlane(r.lo / C64),
                                 __builtin_amdgcn_readfirstlane((r.hi - 1) / C64 + 1)};
            c_first           = __builtin_amdgcn_readfirstlane(r.lo % C64);
            c_last_end        = __builtin_amdgcn_readfirstlane(r.hi - (taps.hi - 1) * C64);
        }

        const int group_c_base                        = g_idx * c_per_group;
        const typename InLoaderGlobal::src_t* in_base = in + static_cast<size_t>(group_c_base);

        // Folded image index of this tile's first tap.
        const int tile_image =
            cfg.kd == 1 ? block_n : block_n * dpars.D + d0 + taps.lo * dpars.dilation_d;

        // The first tap starts at C(64) block c_first: a channel offset into the pixel,
        // applied to the base the way a group's is. Later taps start at block 0.
        const int first_c_base = group_c_base + c_first * 64;
        InLoaderGlobal input_loader_global(input_pars,
                                           tile_image,
                                           block_h,
                                           block_w,
                                           block_c,
                                           input_lds,
                                           in + static_cast<size_t>(first_c_base),
                                           first_c_base);

        InLoaderLds input_load_lds(input_lds);

        // One loader, internally selecting the wave_k_idx partition.
        //
        // Each group's formatted weights are contiguous, so the group offset is a
        // base shift.
        // Scaled by the loader's plane count: tf32 stores two bf16 tiles per operand.
        const size_t wei_group_stride = static_cast<size_t>(WgtsLoader::planes) * Kq * Kwg_total *
                                        C_padded * cfg.kd * cfg.kh * cfg.kw;
        const LdsType* wei_base = wei + static_cast<size_t>(g_idx) * wei_group_stride;
        WgtsLoader weights_loader(K_share, C_padded, k_idx, wei_base, wave_k_idx);

        // Aim at the first live tap; the rest roll over as the cursor advances,
        // because Kd sits directly inside kq in the formatted layout.
        if constexpr(cfg.split_k)
        {
            weights_loader.seek_step(split_first, C_padded);
        }
        else if constexpr(cfg.kd > 1)
        {
            weights_loader.seek_tap(taps.lo, C_padded);
        }

        constexpr int wave_input_reg_h = cfg.wave_p + cfg.kh - 1;
        datatypex8_t input_reg[cfg.c32_per_step()][wave_input_reg_h][cfg.wave_q16];

        datatypex8_t weights_reg[cfg.c32_per_step()][cfg.kh][cfg.wave_k() / 16];

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

        // Register staging for the tf32 split; empty on the 16-bit DMA path.
        //
        // Step S fetches into stage[S & 1] and splits stage[(S-1) & 1], so a step's
        // loads get a whole mma phase to land. Fully drained at every iteration tail,
        // so nothing stays live into the writer.
        typename InLoaderGlobal::Stage stage[2];

        // Split and store step S's staged fetch into LDS buffer buf (no-op unless tf32).
        // Idle waves fetched nothing and own no columns, so they sit out.
        auto split_step = [&]<int S>(int buf) {
            if constexpr(InLoaderGlobal::is_tf32)
                if(loads_input)
                    input_loader_global.template convert_and_store<S>(buf, stage[S & 1]);
        };

        // Cold-start the first tile into toc.
        //
        // Rounds > 0 inherit toc from the prior round's cross-round prefetch and
        // skip this, avoiding the ~2500-cyc prologue vmcnt(0) stall. Loads toc (not
        // tic) since the pre-loop swap is gated to round 0; for rounds > 0 the step-0
        // prefetch and writer staging use disjoint tic bytes, so no barrier is needed.
        if(round == 0)
        {
            static_for<InLoaderGlobal::num_load_steps>([&]<int S>() {
                input_loader_global.template load<S>(toc, stage[S & 1]);
                // Split one step behind, so the drain below leaves S in flight.
                //
                // tf32-only: the DMA path has nothing to split, and the drain would
                // then serialize its prefetch issue for no one's benefit.
                if constexpr(S > 0 && InLoaderGlobal::is_tf32)
                {
                    if(loads_input)
                        wait_vmcnt<InLoaderGlobal::template loads_for_step<S>()>();
                    split_step.template operator()<S - 1>(toc);
                }
            });
            wait_vmcnt_all();
            split_step.template operator()<InLoaderGlobal::num_load_steps - 1>(toc);
            // __syncthreads drains lgkmcnt, so the splits land before the first read.
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

        // Update the accumulators for one step: one filter column, C(32) or C(64).
        //
        // Operand order weights=srcA, input=srcB makes each lane's output 4
        // contiguous K, which the OutputWriter's wide path coalesces (see
        // output_writer.h). Numerically identical to the swapped order, free on
        // the load side.
        //
        // ZeroAcc: the step's first mfma on each accumulator passes SrcC=0.
        auto mma_step = [&](auto ZeroAcc) {
            constexpr fp32x4_t zero = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for(int c = 0; c < cfg.c32_per_step(); ++c)
#pragma unroll
                for(int kh = 0; kh < cfg.kh; ++kh)
#pragma unroll
                    for(int k16 = 0; k16 < cfg.wave_k16; ++k16)
#pragma unroll
                        for(int p = 0; p < cfg.wave_p; ++p)
#pragma unroll
                            for(int q16 = 0; q16 < cfg.wave_q16; ++q16)
                            {
                                const bool first = decltype(ZeroAcc)::value && c == 0 && kh == 0;
                                acc[p][q16][k16] = mfma_16x16x32(weights_reg[c][kh][k16],
                                                                 input_reg[c][p + kh][q16],
                                                                 first ? zero : acc[p][q16][k16]);
                            }
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
        //   ZeroAcc      : step 0 zero-inits the accumulators.
        // C32Half is the step's first C(32) half; with c64_step the step covers both halves.
        auto step =
            [&]<int Step, int Kw, int C32Half, bool Prefetch, bool ZeroAcc, bool NextPrefetch>() {
            static_for<cfg.c32_per_step()>(
                [&]<int C>() { weights_loader.load_step(weights_reg[C]); });
            // Prefetch and NextPrefetch share the loader; only one is ever set.
            if constexpr(Prefetch)
                input_loader_global.template load<Step>(tic, stage[Step & 1]);
            if constexpr(NextPrefetch)
                if(next_valid)
                    input_loader_global.template load<Step>(tic, stage[Step & 1]);
            static_for<cfg.c32_per_step()>([&]<int C>() {
                input_load_lds.load_step(
                    toc, wave_h_base, wave_w_base, Kw, C32Half + C, input_reg[C]);
            });
            // Partial drain leaves this step's prefetch in flight to overlap compute.
            //
            // NextPrefetch included, else the last iteration exposes its full latency.
            // Idle waves have nothing in flight, so drain fully.
            //
            // On the tf32 path the count is only a scheduling hint (the compiler waits
            // on the register dependencies itself); it stays load-bearing for the DMA
            // path, whose LDS destination it cannot see.
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
            // tf32: split the PREVIOUS step's staged fetch into the two LDS planes.
            //
            // After the operand drain, so these ds_writes do not lengthen the wait the
            // mma is already taking; one step behind the fetch, so the vmcnt drain above
            // has already retired it.
            if constexpr((Prefetch || NextPrefetch) && Step > 0)
                if(Prefetch || next_valid)
                    split_step.template operator()<Step - 1>(tic);
            schedule_barrier<1>();
            mma_step(std::bool_constant<ZeroAcc>{});
        };

        // Run one channel-iteration: every step plus the trailing barrier.
        //
        // LoopExit selects the trailing barrier and is NOT implied by !Prefetch: a
        // depth-tap boundary has the same (Prefetch=Off, NextPrefetch=On) signature
        // as the round's last iteration, but the writer does not run there and the
        // prologue stagger must survive to the next tap. 3D therefore passes
        // LoopExit=Off everywhere and de-staggers once, after the tap loop.
        auto run_iteration = [&](auto ZeroAcc, auto Prefetch, auto NextPrefetch, auto LoopExit) {
            if constexpr(Prefetch)
                input_loader_global.step_channels();

            // Emit step S, prefixed by an inter-step barrier for S > 0.
            //
            // num_load_steps = Kw * 2 steps of one filter column and one C(32) half, or
            // Kw steps of one filter column and C(64) under c64_step. Barriering before
            // every step but the first gives num_load_steps-1 inter-step barriers.
            constexpr int num_steps = InLoaderGlobal::num_load_steps;
            auto emit_step          = [&]<int S>() {
                if constexpr(S > 0)
                    schedule_barrier<0>();
                step.template operator()<S,
                                                  /*Kw=*/S * cfg.c32_per_step() / 2,
                                                  /*C32Half=*/S * cfg.c32_per_step() % 2,
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

            // tf32: the last step's staged fetch has no following step to ride on, so
            // split it here. Its rows are the ragged remainder, so the exposed latency
            // is cheaper than the staging registers a compressed schedule would need.
            if constexpr((decltype(Prefetch)::value || decltype(NextPrefetch)::value) &&
                         InLoaderGlobal::is_tf32)
            {
                if(decltype(Prefetch)::value || next_valid)
                    split_step.template operator()<num_steps - 1>(tic);

                // Land the splits before the buffer swap: s_barrier does not drain LDS,
                // and these are the only LDS writes outliving the per-step wait.
                wait_lgkmcnt_all();
            }

            if constexpr(!LoopExit)
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
            Grid next_sched(rctx.per_xcd,
                            rctx.kslice,
                            rctx.blocks_p,
                            rctx.blocks_q,
                            rctx.n_blocks,
                            rctx.seg_begin,
                            rctx.seg_count,
                            rctx.round_count,
                            magic_per_xcd,
                            magic_p,
                            magic_q,
                            rctx.magic_n,
                            rctx.wg_id,
                            round + 1);
            // The next item, and for split-K its first step on its plane's taps.
            const WorkItem next  = decode_item<cfg>(next_sched.p_idx(), next_sched.g_idx(), dpars);
            int next_split_first = 0;
            if constexpr(cfg.split_k)
            {
                const TapRange live =
                    tap_range<cfg.kd>(next.e * dpars.stride_d - dpars.pad_d, dpars);
                next_split_first = split_live_range(next.s, next.S, live, C64).lo;
            }

            // Aim at the next round's group input slice, from its first C(64) block.
            //
            // The round-top in_base offset applies only to the round-top loader.
            const int next_group_c_base =
                next.g * c_per_group + (cfg.split_k ? next_split_first % C64 * 64 : 0);
            const ToType<DT>* next_in_base = in + static_cast<size_t>(next_group_c_base);
            if constexpr(cfg.kd == 1)
            {
                input_loader_global.retarget(input_pars,
                                             next_in_base,
                                             next_sched.n_idx(),
                                             next_sched.p_idx() * cfg.block_p() - py,
                                             next_sched.q_idx() * cfg.block_q() - px,
                                             next_group_c_base);
            }
            else
            {
                // The next round's P-block, depth plane, and first live tap.
                const int next_d0 = next.e * dpars.stride_d - dpars.pad_d;
                const int next_tap =
                    cfg.split_k ? next_split_first / C64 : tap_range<cfg.kd>(next_d0, dpars).lo;
                const int next_image =
                    next_sched.n_idx() * dpars.D + next_d0 + next_tap * dpars.dilation_d;
                input_loader_global.retarget(input_pars,
                                             next_in_base,
                                             next_image,
                                             next.p * cfg.block_p() - py,
                                             next_sched.q_idx() * cfg.block_q() - px,
                                             next_group_c_base);
            }
        };

        if constexpr(cfg.kd == 1)
        {
            if constexpr(cfg.single_c)
            {
                // C64 == 1: the lone iteration is both first and last.
                //
                // ZeroAcc + NextPrefetch, no own-prefetch, so retarget before it runs.
                // Small C over-computes to C_padded=64 against zero weights; for
                // completeness, not speed.
                retarget_next_round();
                run_iteration(/*ZeroAcc=*/On,
                              /*Prefetch=*/Off,
                              /*NextPrefetch=*/On,
                              /*LoopExit=*/On);
            }
            else
            {
                // C64 >= 2: peel first (ZeroAcc) and last (NextPrefetch).
                //
                // The steady middle prefetches its own next-channel tile. Retarget only
                // after the first + middle, which use the loader for their own loads.
                // A split-K item runs its range [c_first, c_last_end) of C/64, at least two
                // (split_ranges_valid).
                const int c_count = cfg.split_k ? c_last_end - c_first : C64;
                run_iteration(/*ZeroAcc=*/On,
                              /*Prefetch=*/On,
                              /*NextPrefetch=*/Off,
                              /*LoopExit=*/Off);
                for(int c64 = 1; c64 < c_count - 1; ++c64)
                    run_iteration(/*ZeroAcc=*/Off,
                                  /*Prefetch=*/On,
                                  /*NextPrefetch=*/Off,
                                  /*LoopExit=*/Off);

                retarget_next_round();

                run_iteration(/*ZeroAcc=*/Off,
                              /*Prefetch=*/Off,
                              /*NextPrefetch=*/On,
                              /*LoopExit=*/On);
            }
        }
        else
        {
            // Aim the dead per-round loader at this tile's tap `tap`.
            //
            // Same P/Q/group as the round; only the depth plane moves, which under
            // the fold is only the image index.
            auto retarget_tap = [&](int tap) {
                input_loader_global.retarget(input_pars,
                                             in_base,
                                             block_n * dpars.D + d0 + tap * dpars.dilation_d,
                                             block_h,
                                             block_w,
                                             group_c_base);
            };

            // Run c_count C blocks of one depth tap.
            //
            // Mirrors the 2D peel above, with retarget aimed at whatever comes next:
            // the following tap, or the next round's first tap. Every iteration takes
            // the phase-swap barrier (LoopExit=Off), so the prologue stagger survives
            // the tap boundaries; the de-stagger fires once, after the loop. c_count is
            // C64 except at a split-K item's first and last taps, and is never 1 there
            // (split_step_range): a one-iteration branch here spills K128.
            auto run_tap = [&](auto ZeroAcc, int tap, int c_count) {
                const bool last_tap = (tap + 1 >= taps.hi);
                next_valid          = last_tap ? (round < rounds - 1) : true;
                auto retarget       = [&]() {
                    if(last_tap)
                        retarget_next_round();
                    else
                        retarget_tap(tap + 1);
                };

                if constexpr(cfg.single_c)
                {
                    retarget();
                    run_iteration(ZeroAcc,
                                  /*Prefetch=*/Off,
                                  /*NextPrefetch=*/On,
                                  /*LoopExit=*/Off);
                }
                else
                {
                    run_iteration(ZeroAcc,
                                  /*Prefetch=*/On,
                                  /*NextPrefetch=*/Off,
                                  /*LoopExit=*/Off);
                    for(int c64 = 1; c64 < c_count - 1; ++c64)
                        run_iteration(/*ZeroAcc=*/Off,
                                      /*Prefetch=*/On,
                                      /*NextPrefetch=*/Off,
                                      /*LoopExit=*/Off);

                    retarget();

                    run_iteration(/*ZeroAcc=*/Off,
                                  /*Prefetch=*/Off,
                                  /*NextPrefetch=*/On,
                                  /*LoopExit=*/Off);
                }
            };

            // Tap taps.lo zeroes the accumulators, the rest add into them. The weights
            // cursor rolls from tap to tap on its own.
            if constexpr(cfg.split_k)
            {
                run_tap(
                    /*ZeroAcc=*/On, taps.lo, (taps.lo + 1 == taps.hi ? c_last_end : C64) - c_first);
                for(int tap = taps.lo + 1; tap < taps.hi; ++tap)
                    run_tap(/*ZeroAcc=*/Off, tap, tap + 1 == taps.hi ? c_last_end : C64);
            }
            else
            {
                run_tap(/*ZeroAcc=*/On, taps.lo, C64);
                for(int tap = taps.lo + 1; tap < taps.hi; ++tap)
                    run_tap(/*ZeroAcc=*/Off, tap, C64);
            }

            // De-stagger once, cancelling the prologue's extra WG1 barrier so both
            // halves enter the writer balanced (what LoopExit does for the 2D path).
            __builtin_amdgcn_s_setprio(0);
            if(wave_group == 0)
                __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);
        }

        // A split-K item stores its raw accumulators; reduce_splits sums them.
        if constexpr(cfg.split_k)
        {
            using Partials    = SplitPartials<cfg>;
            const size_t slot = Partials::slot_index(sched.g_idx(),
                                                     block_n,
                                                     sched.q_idx(),
                                                     sched.p_idx(),
                                                     k_idx,
                                                     n_blocks,
                                                     blocks_q,
                                                     blocks_p,
                                                     Kq);
            const int lane    = static_cast<int>(
                __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u)));
            fp32x4_t* dst = dpars.partials + Partials::offset_vec4(slot, global_wave) + lane;
#pragma unroll
            for(int p = 0; p < cfg.wave_p; ++p)
#pragma unroll
                for(int q16 = 0; q16 < cfg.wave_q16; ++q16)
#pragma unroll
                    for(int k16 = 0; k16 < cfg.wave_k16; ++k16)
                        dst[((p * cfg.wave_q16 + q16) * cfg.wave_k16 + k16) * WAVE_SIZE] =
                            acc[p][q16][k16];
            continue;
        }

        // Write this round's accumulators.
        //
        // OutputWriter dispatches on the config: the wide path and unfold tiles stage
        // each P-row through the dead LDS half for coalesced stores; the K64/K96
        // fallbacks store from registers. The waves_k partitions write disjoint K
        // ranges, so they store concurrently.
        using OutWriter = OutputWriter<cfg, ToType<DT>>;

        // Output depth folds into the batch axis the same way the input's does:
        // [N][E][Ho][Wo][K] is [N*E][Ho][Wo][K], so the writer stays 2D.
        OutputPars output_pars{cfg.kd == 1 ? N : N * dpars.E, ho, wo, groups, k_per_group};
        const int out_image = cfg.kd == 1 ? block_n : block_n * dpars.E + block_e;
        // Staging reuses the top of the dead (tic) LDS half.
        //
        // The next round's step-0 prefetch (bottom rows) then overlaps the writer
        // drain with no barrier; tic_pad keeps them disjoint. A config that does not
        // stage reserves no bytes, so it gets nullptr rather than a pointer one past
        // the tile end.
        constexpr int lds_half       = InLoaderGlobal::lds_size / 2;
        constexpr int stage_lds_fp16 = OutWriter::stage_lds_fp16;
        static_assert(stage_lds_fp16 <= lds_half,
                      "output staging buffer exceeds one input LDS half-tile");
        // This config's stage must fit the MAX-size pad the layout reserves.
        //
        // Smaller stages use the top sub-region; both sides are zero when the writer
        // takes the narrow register store (every tf32 config).
        static_assert(InLoaderGlobal::LdsLayout::writer_stage_uint4 * 8 >= stage_lds_fp16,
                      "InputLdsLayout::writer_stage_uint4 smaller than "
                      "OutputWriter::stage_lds_fp16 (staging would overflow the pad)");
        OutWriter writer(output_pars,
                         out,
                         OutWriter::stages_output
                             ? reinterpret_cast<ToType<DT>*>(input_lds + tic * lds_half +
                                                             (lds_half - stage_lds_fp16))
                             : nullptr,
                         out_image,
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
__global__ __launch_bounds__(cfg.num_threads(), 1) void conv2d_direct_l1_cdna4(
    const ToType<DT>* __restrict__ in,
    const direct_transpose_weights::ToSplitType<DT>* __restrict__ wei,
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
        conv_direct_l1_impl<cfg, DT>(in,
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
                                     magic_n,
                                     DepthPars{});
    }
}

// Conv3d entry point: the 2D kernargs plus the depth geometry.
//
// A separate entry point rather than seven more arguments on the 2D one. The
// kernarg segment sets the kernarg-preload SGPR allocation and hence the register
// allocator's starting state, so widening the shared signature would perturb the
// generated code of every tuned 2D config even though every use would be dead.
template <Config cfg, DataType DT>
__global__ __launch_bounds__(cfg.num_threads(), 1) void conv3d_direct_l1_cdna4(
    const ToType<DT>* __restrict__ in,
    const direct_transpose_weights::ToSplitType<DT>* __restrict__ wei,
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
    MagicDiv magic_n,
    DepthPars dpars)
{
    if(__builtin_amdgcn_is_invocable(__builtin_amdgcn_mfma_f32_16x16x32_f16) &&
       __builtin_amdgcn_is_invocable(__builtin_amdgcn_mfma_f32_16x16x32_bf16))
    {
        conv_direct_l1_impl<cfg, DT>(in,
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
                                     magic_n,
                                     dpars);
    }
}

// Sum a split-K launch's partials and write the output, one workgroup per (tile, wave).
//
// Each lane rebuilds the accumulators the conv kernel's matching (wave, lane) held,
// so the output goes through the same OutputWriter as an unsplit tile. A single-wave
// workgroup spreads the tiles' waves over the machine; at n1 a tile per workgroup
// would occupy 48 CUs.
template <Config cfg, DataType DT>
__global__ __launch_bounds__(WAVE_SIZE) void reduce_splits(DepthPars dpars,
                                                           ToType<DT>* __restrict__ out,
                                                           int N,
                                                           int groups,
                                                           int k_per_group,
                                                           int ho,
                                                           int wo,
                                                           int Kq,
                                                           int blocks_p,
                                                           int blocks_q,
                                                           int n_blocks)
{
    // OutputWriter's s_waitcnt uses the gfx9 encoding; no builtin names that, so gate on gfx950's.
    if(!(__builtin_amdgcn_is_invocable(__builtin_amdgcn_mfma_f32_16x16x32_f16) &&
         __builtin_amdgcn_is_invocable(__builtin_amdgcn_mfma_f32_16x16x32_bf16)))
        return;

    static_assert(cfg.split_k);
    using Partials                = SplitPartials<cfg>;
    using OutWriter               = OutputWriter<cfg, ToType<DT>>;
    constexpr int waves_per_kpart = cfg.waves_p * cfg.waves_q;
    const int E                   = dpars.E;

    const int global_wave = blockIdx.x % cfg.num_waves();
    const size_t tile     = blockIdx.x / cfg.num_waves();
    const int wave        = global_wave % waves_per_kpart;
    const int wave_k_idx  = global_wave / waves_per_kpart;

    // Tiles run P (with depth inside it) fastest, then Q, N, G, and the K-block minor.
    const int k_idx   = static_cast<int>(tile % Kq);
    size_t x          = tile / Kq;
    const int p_and_e = static_cast<int>(x % (blocks_p * E));
    x /= blocks_p * E;
    const int q_idx = static_cast<int>(x % blocks_q);
    x /= blocks_q;
    const int n_idx = static_cast<int>(x % n_blocks);
    const int g_idx = static_cast<int>(x / n_blocks);

    // The plane's items are consecutive in the group axis, so their slots are a fixed
    // stride apart.
    const int e           = p_and_e % E;
    const int j0          = plane_first_item(dpars.plan, dpars, cfg.kd, e);
    const int S           = dpars.plan.S_nt[live_taps(dpars, cfg.kd, e)];
    const size_t slot0    = Partials::slot_index(g_idx * dpars.plan.T + j0,
                                              n_idx,
                                              q_idx,
                                              p_and_e / E,
                                              k_idx,
                                              n_blocks,
                                              blocks_q,
                                              blocks_p,
                                              Kq);
    const size_t s_stride = static_cast<size_t>(n_blocks) * blocks_q * blocks_p * Kq *
                            cfg.num_waves() * Partials::wave_vec4;
    const fp32x4_t* src = dpars.partials + Partials::offset_vec4(slot0, global_wave) + threadIdx.x;
    fp32x4_t acc[cfg.wave_p][cfg.wave_q16][cfg.wave_k16];
    auto for_each_fragment = [&](auto f) {
#pragma unroll
        for(int p = 0; p < cfg.wave_p; ++p)
#pragma unroll
            for(int q16 = 0; q16 < cfg.wave_q16; ++q16)
#pragma unroll
                for(int k16 = 0; k16 < cfg.wave_k16; ++k16)
                    f(acc[p][q16][k16],
                      ((p * cfg.wave_q16 + q16) * cfg.wave_k16 + k16) * WAVE_SIZE);
    };
    for_each_fragment([&](fp32x4_t& a, int f) { a = src[f]; });
    for(int s = 1; s < S; ++s)
    {
        const fp32x4_t* part = src + s * s_stride;
        for_each_fragment([&](fp32x4_t& a, int f) { a += part[f]; });
    }

    // The writer's staging addresses every wave's slot, so the whole buffer is reserved.
    __shared__ uint16_t stage[OutWriter::stages_output ? OutWriter::stage_lds_fp16 : 1];

    const OutputPars output_pars{N * E, ho, wo, groups, k_per_group};
    OutWriter writer(output_pars,
                     out,
                     OutWriter::stages_output ? reinterpret_cast<ToType<DT>*>(stage) : nullptr,
                     n_idx * E + p_and_e % E,
                     p_and_e / E * cfg.block_p(),
                     q_idx * cfg.block_q(),
                     k_idx,
                     g_idx,
                     wave,
                     wave_k_idx);
    writer.write(acc);
}

// Per-config launch.
//
// Self-formats the raw NHWC weights into the workspace, then runs the conv
// reading from that region. The caller passes raw weights like every other
// ConvKernel.
template <Config cfg>
void launch_impl(const LaunchParams& lp,
                 const Device&,
                 const ConvParams& layer,
                 const void* in,
                 const void* wei,
                 void* out,
                 void* workspace,
                 hipStream_t stream)
{
    const ConvParams par = layer_view(layer, cfg);

    // Format the weights and launch the kernel for one element type.
    auto typed_launch = [&]<DataType DT>() {
        using dtype = ToType<DT>;
        // The workspace holds the formatted weights, which are bf16 for tf32.
        using wtype = direct_transpose_weights::ToSplitType<DT>;
        auto view   = SizeView<cfg.direction>(par);

        const auto bc = get_block_counts(cfg, par);

        // The depth axis rides inside the grid's P axis, e innermost.
        //
        // P is already the fastest X axis, so this puts depth innermost without a
        // fifth divmod in BalancedL2Grid -- which would land on every 2D config's
        // per-round decode. Consecutive rounds on a CU then walk depth, so the input
        // planes they read stay within a narrow depth window in L2 (neighbouring
        // output planes share kd-1 of them at unit depth stride and dilation).
        // bc.e is 1 at cfg.kd == 1, leaving the axis exactly what it was.
        //
        // A split-K launch carries the plane in the group axis instead (decode_item).
        const int blocks_p = cfg.split_k ? bc.p : bc.p * bc.e;
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
        auto* formatted       = reinterpret_cast<wtype*>(workspace);
        constexpr int Kwg     = cfg.wave_k();
        constexpr int waves_k = cfg.waves_k;
        if constexpr(cfg.direction == Direction::Dgrad)
            direct_transpose_weights::
                launch_transpose_weights_dgrad<cfg.kh, cfg.kw, Kwg, DT, waves_k, cfg.kd>(
                    static_cast<const dtype*>(wei),
                    formatted,
                    par.filters_per_group(),
                    par.channels_per_group(),
                    par.groups,
                    stream,
                    padded_out_chan);
        else
            direct_transpose_weights::
                launch_transpose_weights<cfg.kh, cfg.kw, Kwg, DT, waves_k, cfg.kd>(
                    static_cast<const dtype*>(wei),
                    formatted,
                    par.filters_per_group(),
                    par.channels_per_group(),
                    par.groups,
                    stream,
                    padded_out_chan);
        const wtype* conv_wei = formatted;

        // c_per_group is the MFMA reduction count, k_per_group the output count.
        //
        // dgrad swaps them; spatial dims and padding are direction-mapped by view.
        constexpr bool is_dgrad    = (cfg.direction == Direction::Dgrad);
        const int kern_c_per_group = is_dgrad ? par.filters_per_group() : par.channels_per_group();
        const int kern_k_per_group = is_dgrad ? par.channels_per_group() : par.filters_per_group();

        // The two entry points take the same arguments bar the trailing DepthPars.
        //
        // Spelled out twice because <<<>>> needs a literal __global__ callee. Keeping
        // them separate leaves the 2D kernarg segment (and so its kernarg-preload
        // SGPR allocation) exactly what it was. A conv2d split-K config takes the conv3d
        // entry point for its DepthPars, with Kd = 1.
        if constexpr(cfg.kd == 1 && !cfg.split_k)
        {
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
        }
        else
        {
            // Split-K: a column's items as the inner half of the grid's group axis.
            const SplitPlan plan = split_plan(cfg, par);
            fp32x4_t* const partials =
                cfg.split_k ? reinterpret_cast<fp32x4_t*>(static_cast<char*>(workspace) +
                                                          split_partials_offset(cfg, par))
                            : nullptr;
            const DepthGeometry g = depth_geometry(cfg, par);
            const DepthPars dpars{g.D,
                                  g.E,
                                  g.stride_d,
                                  g.pad_d,
                                  g.dilation_d,
                                  MagicDiv(bc.e),
                                  plan,
                                  MagicDiv(plan.T),
                                  MagicDiv(plan.S_nt[cfg.kd]),
                                  partials};
            conv3d_direct_l1_cdna4<cfg, DT>
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
                    magic_n,
                    dpars);

            if constexpr(cfg.split_k)
            {
                const size_t reduce_blocks = tile_count(cfg, par) * cfg.num_waves();
                reduce_splits<cfg, DT>
                    <<<dim3(static_cast<unsigned>(reduce_blocks)), dim3(WAVE_SIZE), 0, stream>>>(
                        dpars,
                        static_cast<dtype*>(out),
                        par.n,
                        par.groups,
                        kern_k_per_group,
                        view.p(),
                        view.q(),
                        kparts * per_xcd,
                        blocks_p,
                        blocks_q,
                        n_blocks);
            }
        }
    };
    // A config is sized for one element width, so the 4-byte one instantiates once.
    if constexpr(cfg.elem_bytes == 4)
        typed_launch.template operator()<DataType::tf32>();
    else if(par.input_type == DataType::bf16)
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

    KVDescriptor config_descriptor() const override { return ConfigMatcher(cfg_); }

    // Family dimensionality gate.
    //
    // Reads dims alone, never cfg_: the dispatcher asks the first kernel in the
    // span and takes the answer for all of them. Which config supports a given
    // dimensionality is is_valid_config's par.dims / par.kd match.
    bool supports_dims(int dims) const override { return dims == 2 || dims == 3; }

    bool is_applicable(const Device& /*dev*/, const ConvParams& par) const override
    {
        // tf32 takes tf32 operands and an fp32 result; the 16-bit families are same-type.
        const bool ok_16bit =
            (par.input_type == DataType::fp16 || par.input_type == DataType::bf16) &&
            par.weight_type == par.input_type && par.output_type == par.input_type;
        const bool ok_tf32 = par.input_type == DataType::tf32 &&
                             par.weight_type == DataType::tf32 && par.output_type == DataType::fp32;
        if(!ok_16bit && !ok_tf32)
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
        // Depth predicates. The base class has none, so the leaf carries them.
        //
        // A depth tap is a whole plane, so fprop's stride_d and dilation_d cost nothing
        // and are left unconstrained -- unlike stride_h/stride_w, which would perturb
        // the loader. Dgrad runs the transposed problem, which is a convolution over dy
        // only at unit depth stride; SizeView also swaps stride and dilation, which holds
        // only while both are 1.
        if(par.dims == 3)
        {
            if(par.kd > MAX_KD)
                return false;
            if(par.direction == Direction::Dgrad && (par.stride_d != 1 || par.dilation_d != 1))
                return false;
        }
        return true;
    }

    bool is_valid_config(const Device& /*dev*/, const ConvParams& par) const override
    {
        return direct_l1::is_valid_config(par, cfg_);
    }

    // The family's confidence, scaled by this config's relative throughput.
    //
    // The first-ranked config keeps the family's index, so ranking direct_l1 against
    // other families is unchanged by how many configs it returns.
    float get_weighted_throughput_index(const Device& dev, const ConvParams& par) const override
    {
        return DirectConvKernel::get_weighted_throughput_index(dev, par) *
               direct_l1::relative_throughput(layer_view(par, cfg_), cfg_);
    }

    LaunchParams get_launch_params(const Device& /*dev*/, const ConvParams&) const override
    {
        return direct_l1::get_launch_params(cfg_);
    }

    size_t get_workspace_size(const Device& /*dev*/, const ConvParams& par) const override
    {
        return direct_l1::get_workspace_size(cfg_, layer_view(par, cfg_));
    }

private:
    const Config& cfg_;
};

} // namespace hipconv::cdna4::direct_l1
