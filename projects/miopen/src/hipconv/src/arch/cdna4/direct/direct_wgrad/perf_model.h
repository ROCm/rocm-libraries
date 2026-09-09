#pragma once

// The performance model estimates the cost (i.e., latency) of a config on a layer.
//
// docs/algorithms/direct/direct-wgrad-estimated-cost.md documents the model: the three terms, the
// coefficients, the sweep that fits them, and the layers it gets wrong.
//
// Free of device code, so the dispatch test can score a layer without running a kernel.

#include "config.h"
#include "config_table.h"
#include "grid.h"
#include "mathutil.h"
#include "persistent_grid.h"
#include "hipconv/conv2d_params.hpp"
#include <cmath>
#include <cstdint>

namespace hipconv::cdna4::direct_wgrad
{

// What a byte costs against a FLOP, fitted over the measured corpus.
//
// The corpus pins the 18.5 ratio between the two tightly and the scale only to within a factor,
// so a refit that moves the scale by less than that is noise.
constexpr double L2_READ_FLOPS_PER_BYTE   = 178.0;
constexpr double DW_ATOMIC_FLOPS_PER_BYTE = 3293.0;

// How much of the idle machine to charge for.
//
// 1 turns the total into a makespan, which over-corrects on the narrow-tile shapes.
constexpr double MACHINE_FILL_EXPONENT = 0.8;

// Output columns the machine would run under one packing.
//
// The packing rule and the model's compute term both read this expression. It takes the packing
// rather than a config because the column block's width follows unfold_n alone, which is what
// lets the packing rule cost the three packings without walking the table.
inline int packed_columns(const Conv2dParams& par, int unfold_n)
{
    const int w_unfold = MFMA_K / unfold_n;
    return divup(par.q, w_unfold) * w_unfold;
}

inline int packed_columns(const Conv2dParams& par, const Config& cfg)
{
    return packed_columns(par, cfg.unfold_n);
}

// Output pixels the machine runs, so the columns a packing leaves empty count as work.
//
// Rows pad by the same factor for every arrangement, so leaving them out moves no ranking.
inline int64_t padded_pixels(const Conv2dParams& par, const Config& cfg)
{
    const int64_t images = int64_t{divup(par.n, cfg.unfold_n)} * cfg.unfold_n;
    return images * par.p * packed_columns(par, cfg);
}

// The grid this config would run under.
//
// Built the way the kernel body builds it, so tiles(), items() and splits() cannot drift from the
// launch's. The launch width is the persistent grid; only a test runs a narrower one.
inline FlatGrid model_grid(const Conv2dParams& par, const Config& cfg)
{
    return FlatGrid{.groups           = par.groups,
                    .c_per_group      = par.channels_per_group(),
                    .k_per_group      = par.filters_per_group(),
                    .images           = par.n,
                    .out_cols         = par.q,
                    .block_c          = cfg.group_c(),
                    .block_k          = cfg.group_k(),
                    .groups_per_block = cfg.waves_g,
                    .block_cols       = cfg.w_unfold(),
                    .unfold_n         = cfg.unfold_n,
                    .workgroups       = persistent::PERSISTENT_GRID_SIZE};
}

// How much longer this config takes over the same work than the filter's widest wave would.
//
// Relative, so an entry on that wave scales by exactly one and the fitted coefficients stay
// comparable across filter sizes.
inline double compute_slowdown(const Config& cfg)
{
    int widest = 0;
    for(const Config& c : configs)
        if(c.kh == cfg.kh && c.kw == cfg.kw && c.wave_c16 * c.wave_k16 > widest)
            widest = c.wave_c16 * c.wave_k16;
    return static_cast<double>(widest) / (cfg.wave_c16 * cfg.wave_k16);
}

// Bytes one operand stream fetches for each byte the tile keeps.
//
// window is the loader's contiguous run in channels, kept the part of it this tile uses, total the
// tensor's whole channel axis; fp16 makes them bytes. The three cases and the floor on the run
// reward are in the short-channel-runs section of direct-wgrad-estimated-cost.md.
inline double line_cost(int window, int kept, int total)
{
    if(window >= total)
        return 1.0;

    constexpr double line_bytes       = 128.0;
    constexpr double most_a_run_earns = 0.5;

    const double run     = 2.0 * window;
    const double fetched = run < line_bytes ? line_bytes : run;

    const double waste = fetched / (2.0 * kept);
    const double reward =
        run <= line_bytes
            ? 1.0
            : (line_bytes / run < most_a_run_earns ? most_a_run_earns : line_bytes / run);
    return waste * reward;
}

// The share of the launch's capacity that has work, averaged over the whole kernel.
//
// The denominator counts workgroup-rounds, so this is a duty cycle. It agrees with the share of
// workgroups that get a cell only inside one round, which is what MACHINE_FILL_EXPONENT was fitted
// on.
inline double machine_fill(const FlatGrid& grid)
{
    const double rounds = divup(grid.cells(), grid.workgroups);
    return grid.cells() / (rounds * grid.workgroups);
}

// What this config would cost the machine on this layer, in FLOP.
//
// Derived in double because the compute term reaches 1e19 on the widest layer the dispatch sweep
// constructs, past a signed 64-bit integer. Returned as float, which is already wider than the
// model is accurate: it is a two-coefficient fit over a measured corpus, so a pair it separates
// only below float resolution it has not separated at all, and the table's order decides those.
inline float estimated_cost(const Conv2dParams& par, const Config& cfg)
{
    const FlatGrid grid = model_grid(par, cfg);

    const double pixels   = static_cast<double>(padded_pixels(par, cfg));
    const double filter   = par.kh * par.kw;
    const double groups   = par.groups;
    const double c_blocks = grid.c_blocks();
    const double k_blocks = grid.k_blocks();
    const double chans    = par.channels_per_group();
    const double filters  = par.filters_per_group();

    // What a line delivers, per operand. S is the k_blocks term and delta the c_blocks one.
    //
    // The corpus decided the gate on a grouped layer: priced ungated it moves 27 of the 128 picks
    // and 11 of those come out wrong.
    const int c_total   = par.groups * par.channels_per_group();
    const int k_total   = par.groups * par.filters_per_group();
    const int c_kept    = minimum(cfg.block_c(), cfg.waves_g * par.channels_per_group());
    const int k_kept    = minimum(cfg.block_k(), cfg.waves_g * par.filters_per_group());
    const double s_line = par.groups > 1 ? line_cost(cfg.block_c(), c_kept, c_total) : 1.0;
    const double d_line = par.groups > 1 ? line_cost(cfg.block_k(), k_kept, k_total) : 1.0;

    const double compute = compute_slowdown(cfg) * 2 * filter * pixels * groups *
                           (c_blocks * cfg.group_c()) * (k_blocks * cfg.group_k());
    const double reads =
        2 * pixels * groups * (k_blocks * chans * s_line + c_blocks * filters * d_line);
    const double atomics = 4 * filter * groups * chans * filters * grid.splits();

    const double total =
        compute + L2_READ_FLOPS_PER_BYTE * reads + DW_ATOMIC_FLOPS_PER_BYTE * atomics;
    return static_cast<float>(total / std::pow(machine_fill(grid), MACHINE_FILL_EXPONENT));
}

// The FLOPs the gradient needs, before any config spends anything on top.
inline double ideal_flops(const Conv2dParams& par)
{
    return 2.0 * par.n * par.p * par.q * par.groups * par.channels_per_group() *
           par.filters_per_group() * par.kh * par.kw;
}

// The share of what a config spends that lands on the gradient, for get_weighted_throughput_index.
//
// estimated_cost is denominated in FLOP, so the layer's own FLOP count over it is a fraction of
// peak in the sense that interface documents. It reaches 1 only for a config that pads nothing,
// refetches nothing, sends no atomics and leaves no workgroup idle, and the compute term alone
// keeps it under 1 everywhere else.
inline float throughput_index(const Conv2dParams& par, const Config& cfg)
{
    return static_cast<float>(ideal_flops(par) / estimated_cost(par, cfg));
}

} // namespace hipconv::cdna4::direct_wgrad
