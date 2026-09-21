#pragma once

// The concrete LDS rings, row loaders, and register ring a Config produces.
//
// docs/algorithms/direct/direct-wgrad.md has the ring geometry.

#include "config.h"
#include "lds_layout.h"
#include "operand_loader.h"
#include "row_loader.h"
#include "types.h"

#include "hipconv/conv2d_params.hpp"

namespace hipconv::cdna4::direct_wgrad
{

// Per-lane width of one load of each row, 4 or 16.
//
// Only the S row can gain from a narrow load, and its 4.5 KiB of recovered padding costs more
// than it saves: three loads per wave where a wide one takes a single load, and three address
// chains live at once in the memory phase. Measured at 0.411 ms against 0.406, and 256
// main-loop VGPRs against 248.
constexpr int S_LANE_BYTES     = 16;
constexpr int DELTA_LANE_BYTES = 16;

// Columns of one load round, a round being one wave's buffer load of the row.
constexpr int col_granularity(int chans, int lane_bytes)
{
    return WAVE_SIZE * lane_bytes / (chans * 2);
}

// Columns one item's row holds: live columns padded to the whole round RowLoader requires.
constexpr int buffer_cols(int live, int chans, int lane_bytes)
{
    const int g = col_granularity(chans, lane_bytes);
    return (live + g - 1) / g * g;
}

constexpr int s_live_cols(const Config& cfg)
{
    return cfg.unfold_n * cfg.s_cols_per_image();
}

// The S row carries a kw-1 halo per packed image; the delta row needs none.
//
// Delta is read at a single column offset, so its live columns are the MFMA's whatever the
// unfold is.
template <Config cfg>
using SRowLayout = RowLayout<buffer_cols(s_live_cols(cfg), cfg.block_c(), S_LANE_BYTES),
                             cfg.block_c(),
                             s_live_cols(cfg),
                             cfg.w_unfold(),
                             cfg.kw - 1>;
template <Config cfg>
using DeltaRowLayout = RowLayout<buffer_cols(MFMA_K, cfg.block_k(), DELTA_LANE_BYTES),
                                 cfg.block_k(),
                                 MFMA_K,
                                 cfg.w_unfold(),
                                 0>;

template <Config cfg, hipconv::DataType DT>
using SRowLoader =
    RowLoader<SRowLayout<cfg>, ToType<DT>, cfg.waves_per_item(), cfg.waves_q, S_LANE_BYTES>;
template <Config cfg, hipconv::DataType DT>
using DeltaRowLoader =
    RowLoader<DeltaRowLayout<cfg>, ToType<DT>, cfg.waves_per_item(), cfg.waves_q, DELTA_LANE_BYTES>;

template <Config cfg, hipconv::DataType DT>
using SRing = RowRing<ToType<DT>, cfg.row_buffers()>;
template <Config cfg, hipconv::DataType DT>
using DeltaRing = RowRing<ToType<DT>, cfg.row_buffers()>;
template <Config cfg, hipconv::DataType DT>
using DeltaScratch = RowRing<ToType<DT>, cfg.scratch_rows()>;

template <Config cfg, hipconv::DataType DT>
constexpr int lds_elems =
    cfg.row_buffers() * SRowLoader<cfg, DT>::buffer_elems +
    (cfg.row_buffers() + cfg.scratch_rows()) * DeltaRowLoader<cfg, DT>::buffer_elems;

// Buffer loads one wave issues per iteration, the same count for every wave.
//
// Each row's rounds are dealt out evenly and the waves with none left take the drain, so one
// s_waitcnt immediate serves the whole workgroup.
template <Config cfg, hipconv::DataType DT>
constexpr int loads_per_iteration =
    DeltaRowLoader<cfg, DT>::loads_per_row + SRowLoader<cfg, DT>::loads_per_row;

// Loads every wave leaves in flight at each drain, in the prologue and in the loop.
//
// One row deeper than the row about to be read: both wavegroups stand at "issued through row
// i - 1 + d, must have retired row i", which leaves rows i + 1 .. i + d - 1 moving.
//
// The two wavegroups share one constant only because the memory phase between their drains
// issues nothing but the row. They guard different rows, ping's row i and pong's row i + 1,
// and each has issued one row more than the last, so the counts coincide. Anything else issued
// there separates them and has to be counted per wavegroup.
template <Config cfg, hipconv::DataType DT>
constexpr int loads_in_flight = (cfg.prefetch_rows - 1) * loads_per_iteration<cfg, DT>;

// Declare the workgroup's LDS and hand `body` the two rings and the prologue scratch.
//
// One __shared__ declaration per ring slot, because the row loop reads one slot while DMA is
// in flight to another. The scratch rows are a single object: nothing reads them while DMA is
// in flight to any of them, and separate objects cost an address register apiece.
template <Config cfg, hipconv::DataType DT, typename Body>
__device__ void with_lds_rings(Body&& body)
{
    using T                   = ToType<DT>;
    constexpr int s_elems     = SRowLoader<cfg, DT>::buffer_elems;
    constexpr int delta_elems = DeltaRowLoader<cfg, DT>::buffer_elems;

    with_rows<T, s_elems, cfg.row_buffers()>([&](auto*... s) {
        const SRing<cfg, DT> s_ring{{s...}};
        with_rows<T, delta_elems, cfg.row_buffers()>([&](auto*... d) {
            const DeltaRing<cfg, DT> delta_ring{{d...}};
            with_row_block<T, delta_elems, cfg.scratch_rows()>(
                [&](const DeltaScratch<cfg, DT>& scratch) { body(s_ring, delta_ring, scratch); });
        });
    });
}

// The delta register ring: the kh rows one compute phase cross-correlates against one S row.
//
// Slots are indexed relative to the iteration (see RowSchedule::reg_slot), so an unroll by kh
// makes every index a compile-time constant and the ring costs no address arithmetic.
template <Config cfg, hipconv::DataType DT>
struct DeltaRegRing
{
    rt_delta<cfg, DT> rows[cfg.kh];

    __device__ rt_delta<cfg, DT>& row(int slot) { return rows[slot]; }
    __device__ const rt_delta<cfg, DT>& row(int slot) const { return rows[slot]; }
};

} // namespace hipconv::cdna4::direct_wgrad
