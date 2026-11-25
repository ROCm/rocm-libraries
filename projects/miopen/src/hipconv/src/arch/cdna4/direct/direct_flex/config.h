#pragma once

// Constants and tiles of direct_flex (docs/algorithms/direct/direct-flex-cdna4.md).

namespace hipconv::cdna4::direct_flex
{
static constexpr int WAVE_SIZE = 64;

// The MFMA the compute phase tiles onto: 16x16x32.
constexpr int MFMA_M = 16;
constexpr int MFMA_N = 16;
constexpr int MFMA_K = 32;

// Elements a 16-byte load moves, which is the granule both LDS tiles are swizzled on.
constexpr int GRANULE = 8;

// Channels one reduction step stages: one 128-byte fibre per pixel or filter row.
constexpr int C_CHUNK = 64;

// Filter rows and columns.
constexpr int MAX_FILTER = 16;

// Filter positions per positions-outer filter tile, one input-table row each.
constexpr int TAB_POSITIONS = 15;
constexpr int TAB_ROWS      = TAB_POSITIONS + 1;

// Filter tiles: at most two across a filter's columns and sixteen down its rows.
constexpr int MAX_FTILES = 32;

// The filter-tile sizes compiled with positions inner, the loop over a tile's positions unrolled.
constexpr int SWEEPS[] = {7, 9};

// Strided dgrad's output phases, sh * sw, and the segments of filter positions their tiles run.
constexpr int MAX_PHASES = 8;
constexpr int MAX_SEGS   = 16;

// Loop-order filter positions plus three, for loads that run into the next X tile.
constexpr int MAX_SEQ = MAX_FILTER * MAX_FILTER + MAX_SEGS;

// A tile: a workgroup's X and K extents and its wave grid.
struct Config
{
    int x_tile16;   // output positions per workgroup / 16
    int k_tile16;   // output channels per workgroup / 16
    int waves_x;    // waves along X within a half-tile pair
    int waves_k;    // waves along K within a half-tile pair
    int csplit = 1; // ways the workgroup splits the reduction, 1, 2 or 4 (csplit_kernel.h)

    constexpr int x_tile() const { return MFMA_M * x_tile16; }
    constexpr int k_tile() const { return MFMA_N * k_tile16; }
    constexpr int waves() const { return waves_x * waves_k * csplit; }
    constexpr int threads() const { return WAVE_SIZE * waves(); }

    // A wave owns wave_x() rows of each X half and wave_k() rows of each K half.
    constexpr int half_x() const { return x_tile() / 2; }
    constexpr int half_k() const { return k_tile() / 2; }
    constexpr int wave_x() const { return half_x() / waves_x; }
    constexpr int wave_k() const { return half_k() / waves_k; }

    static constexpr int granules() { return C_CHUNK / GRANULE; }
    static constexpr int k_steps() { return C_CHUNK / MFMA_K; }

    // A K half stages as many rows as an X half, so every tiling keeps one load schedule.
    constexpr int in_rounds() const { return x_tile() * granules() / threads(); }
    constexpr int wgt_rounds() const { return in_rounds(); }

    // Both tiles are double-buffered, the K tile at the X tile's size.
    constexpr int lds_bytes() const { return 2 * 2 * x_tile() * C_CHUNK * 2; }

    constexpr bool operator==(const Config&) const = default;
};

} // namespace hipconv::cdna4::direct_flex
