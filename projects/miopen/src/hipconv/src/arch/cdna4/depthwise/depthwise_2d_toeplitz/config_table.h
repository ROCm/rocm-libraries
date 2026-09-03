#pragma once

#include "hipconv/conv2d_params.hpp"

// The compiled set of depthwise 2D patch-Toeplitz configs.
//
// Free of device code so hipconv_autoshard can read `num_configs` (and tests can
// name a config) without compiling the kernel; the kernel body lives in kernel.h.

namespace hipconv::cdna4::depthwise_2d_toeplitz
{
using namespace hipconv;

constexpr int WAVE_SIZE = 64;

// One sparse MFMA covers 16 (2x2) output tiles across 4 channels.
//
// How the 16 tiles map to output space is a config knob (see Config::tpr):
// unfolded they are a 1x16 run along W (32 cols); folded they are a TPR-wide x
// FR-tall rectangle (TPR*FR == 16) so small-W shapes fill the MFMA.
constexpr int TILES = 16;

constexpr int next_pow2(int x)
{
    int p = 1;
    while(p < x)
        p <<= 1;
    return p;
}

// Kernel config: per-wave independent loader with a private sliding-window ring.
//
// A workgroup owns one tile group + a CH = 4*waves*subgroups channel block, but
// each wave owns its own uint4-aligned CW = 4*subgroups slice in a private ring
// (async buffer_load_lds). The sliding window fetches each input row from HBM once;
// nothing is shared across waves, so there are no workgroup barriers, only
// wave-scoped s_waitcnt. Requires CH | C and CW a multiple of 8.
struct Config
{
    int waves_per_wg;
    int subgroups       = 2; // sparse sub-groups (4 channels) per wave
    int stride          = 1;
    Direction direction = Direction::Fprop;
    int kh              = 3;
    int kw              = 3;
    // Use the dense F(4x4,5x5) encoding (16x16x32 MFMA, 4x4 tile, no sparsity).
    //
    // For filters whose (out+K-1) patch does not split into whole K=64 sparse
    // passes (e.g. 5x5). See toeplitz-weights.pdf.
    bool dense = false;
    // 2D tile folding: (2x2) tiles laid along W per folded tile-row. Must divide 16.
    //
    // tpr=16 (default) => 1x16 run (32-wide strip). tpr<16 folds FR=16/tpr
    // output-row-slices into the MFMA columns so small-W shapes (Wo <= 2*tpr) fill
    // the tile and shrink the ring LDS -> higher occupancy.
    int tpr = 16;

    // Output tile edge: sparse emits 2x2 tiles, dense F(4x4,5x5) emits 4x4.
    constexpr int out_edge() const { return dense ? 4 : 2; }
    constexpr int tile_cols() const { return out_edge() * tpr; }
    constexpr int block_size() const { return waves_per_wg * WAVE_SIZE; }
    // Channels per workgroup: one wide contiguous run per pixel.
    //
    // 4 channels per sub-group, `subgroups` sub-groups per wave. Wider runs make
    // the async HBM->LDS loads cache-line efficient.
    constexpr int ch_per_wg() const { return 4 * waves_per_wg * subgroups; }
};

// Config list; find_config auto-picks configs[0] absent tuning, else the external
// tuner picks the winning config per shape.
//
// Each wave owns a uint4-aligned slice CW=4*subgroups, so CH = 4*waves*subgroups
// spans 32/16/64/8. W4/CH32 and W8/CH64 are the measured workhorses; the rest
// cover channel counts they can't tile.
constexpr Config configs[] = {
    // subgroups=2 => CW=8 (uint4-aligned).
    {.waves_per_wg = 4, .subgroups = 2}, // CH=32 (workhorse)
    {.waves_per_wg = 2, .subgroups = 2}, // CH=16
    {.waves_per_wg = 8, .subgroups = 2}, // CH=64 (wide c, block_size 512)
    // CH=64 via subgroups=4 (block_size 256): half the waves/WG of w8sg2.
    //
    // Smaller WGs whose concurrent input footprint stays L2-resident when strongly
    // HBM-bound; above the crossover w8sg2 (bs512) overflows L2 and re-fetches ~2x.
    // The second-tier gate below defers w8sg2 to this config when HBM-bound.
    {.waves_per_wg = 4, .subgroups = 4}, // CH=64 (block_size 256, HBM-bound)
    {.waves_per_wg = 1, .subgroups = 2}, // CH=8
    // 2D-folded tiles for small-W shapes: fill the MFMA and shrink the ring LDS.
    {.waves_per_wg = 4, .subgroups = 2, .tpr = 8}, // 8x2 fold, Wo<=16
    {.waves_per_wg = 4, .subgroups = 2, .tpr = 4}, // 4x4 fold, Wo<=8
    {.waves_per_wg = 8, .subgroups = 2, .tpr = 4}, // 4x4 fold, wide c
    {.waves_per_wg = 2, .subgroups = 2, .tpr = 4}, // 4x4 fold, narrow c
    // F(2x2,7x7): 8x8 patch, KP=4 sparse passes, halo 6. Same packing as 3x3.
    {.waves_per_wg = 4, .subgroups = 2, .kh = 7, .kw = 7},           // 7x7 CH=32
    {.waves_per_wg = 2, .subgroups = 2, .kh = 7, .kw = 7},           // 7x7 CH=16
    {.waves_per_wg = 8, .subgroups = 2, .kh = 7, .kw = 7},           // 7x7 CH=64
    {.waves_per_wg = 4, .subgroups = 2, .kh = 7, .kw = 7, .tpr = 4}, // 7x7 4x4 fold
    // Stride-2 (sparse, unfolded) for downsampling layers.
    //
    // The strided receptive field is padded up to a whole-pass patch (3x3 s2 ->
    // 8x6, KP=3; 7x7 s2 -> 16x9, KP=9).
    {.waves_per_wg = 4, .subgroups = 2, .stride = 2},                   // 3x3 s2 CH=32
    {.waves_per_wg = 2, .subgroups = 2, .stride = 2},                   // 3x3 s2 CH=16
    {.waves_per_wg = 8, .subgroups = 2, .stride = 2},                   // 3x3 s2 CH=64
    {.waves_per_wg = 1, .subgroups = 2, .stride = 2},                   // 3x3 s2 CH=8
    {.waves_per_wg = 4, .subgroups = 2, .stride = 2, .kh = 7, .kw = 7}, // 7x7 s2 CH=32
    {.waves_per_wg = 2, .subgroups = 2, .stride = 2, .kh = 7, .kw = 7}, // 7x7 s2 CH=16
    {.waves_per_wg = 8, .subgroups = 2, .stride = 2, .kh = 7, .kw = 7}, // 7x7 s2 CH=64
    // Stride-2 folded (tpr<16): fills the MFMA, shrinks the staged input strip.
    //
    // For narrow-output downsampling (Wo<=2*tpr): big LDS/occupancy win when the
    // unfolded 32-wide tile would stage mostly out-of-image columns (e.g. Wo=14).
    {.waves_per_wg = 4, .subgroups = 2, .stride = 2, .tpr = 8}, // 3x3 s2 8x2 fold
    {.waves_per_wg = 4, .subgroups = 2, .stride = 2, .tpr = 4}, // 3x3 s2 4x4 fold
    {.waves_per_wg = 8, .subgroups = 2, .stride = 2, .tpr = 4}, // 3x3 s2 4x4 wide-c
    {.waves_per_wg = 2, .subgroups = 2, .stride = 2, .tpr = 4}, // 3x3 s2 4x4 narrow-c
    {.waves_per_wg = 4, .subgroups = 2, .stride = 2, .kh = 5, .kw = 5, .tpr = 8}, // 5x5 s2 8x2 fold
    {.waves_per_wg = 4, .subgroups = 2, .stride = 2, .kh = 5, .kw = 5, .tpr = 4}, // 5x5 s2 4x4 fold
    {.waves_per_wg = 4, .subgroups = 2, .stride = 2, .kh = 7, .kw = 7, .tpr = 8}, // 7x7 s2 8x2 fold
    {.waves_per_wg = 4, .subgroups = 2, .stride = 2, .kh = 7, .kw = 7, .tpr = 4}, // 7x7 s2 4x4 fold
    // 5x5 via the sparse encoding (patch width padded to 8).
    //
    // Stride 1 (8x6, KP=3) and stride 2 (8x8, KP=4). Sparse is the only 5x5
    // stride-2 path; the dense F(4x4,5x5) configs below cover stride-1 (tuner picks).
    {.waves_per_wg = 4, .subgroups = 2, .kh = 5, .kw = 5},              // 5x5 s1 CH=32
    {.waves_per_wg = 2, .subgroups = 2, .kh = 5, .kw = 5},              // 5x5 s1 CH=16
    {.waves_per_wg = 8, .subgroups = 2, .kh = 5, .kw = 5},              // 5x5 s1 CH=64
    {.waves_per_wg = 4, .subgroups = 2, .stride = 2, .kh = 5, .kw = 5}, // 5x5 s2 CH=32
    {.waves_per_wg = 2, .subgroups = 2, .stride = 2, .kh = 5, .kw = 5}, // 5x5 s2 CH=16
    {.waves_per_wg = 8, .subgroups = 2, .stride = 2, .kh = 5, .kw = 5}, // 5x5 s2 CH=64
    // F(4x4,5x5): dense 16x16x32 MFMA (no sparsity), 4x4 tile, 8x8 patch, KP=2.
    //
    // Ordered best-first: find_config auto-picks valid_configs[0], so the fastest
    // config per (C, Wo) must come first. The tile fold (tpr<16) fills small-Wo
    // tiles and shrinks the ring LDS (measured 17-35% over unfolded for Wo<=32);
    // is_valid_config rejects a fold when Wo > 4*tpr, falling back to the unfolded
    // workhorse. The C%32/16/64/8 chain covers every applicable channel count.
    {.waves_per_wg = 4, .subgroups = 2, .kh = 5, .kw = 5, .dense = true, .tpr = 8}, // Wo<=32 C%32
    {.waves_per_wg = 2, .subgroups = 2, .kh = 5, .kw = 5, .dense = true, .tpr = 8}, // Wo<=32 C%16
    {.waves_per_wg = 8, .subgroups = 2, .kh = 5, .kw = 5, .dense = true, .tpr = 8}, // Wo<=32 C%64
    {.waves_per_wg = 1, .subgroups = 2, .kh = 5, .kw = 5, .dense = true, .tpr = 8}, // Wo<=32 C%8
    // Unfolded (tpr=16) workhorse: fallback for Wo>32 (folded rejected).
    {.waves_per_wg = 4, .subgroups = 2, .kh = 5, .kw = 5, .dense = true}, // 5x5 CH=32
    {.waves_per_wg = 2, .subgroups = 2, .kh = 5, .kw = 5, .dense = true}, // 5x5 CH=16
    {.waves_per_wg = 8, .subgroups = 2, .kh = 5, .kw = 5, .dense = true}, // 5x5 CH=64
    {.waves_per_wg = 1, .subgroups = 2, .kh = 5, .kw = 5, .dense = true}, // 5x5 CH=8
    // Extra tile folds (tpr=4/2) for explicit --config selection / tuning.
    {.waves_per_wg = 4, .subgroups = 2, .kh = 5, .kw = 5, .dense = true, .tpr = 4}, // Wo<=16
    {.waves_per_wg = 8, .subgroups = 2, .kh = 5, .kw = 5, .dense = true, .tpr = 4}, // Wo<=16 wide-c
    {.waves_per_wg = 4, .subgroups = 2, .kh = 5, .kw = 5, .dense = true, .tpr = 2}, // Wo<=8
    {.waves_per_wg = 8, .subgroups = 2, .kh = 5, .kw = 5, .dense = true, .tpr = 2}, // Wo<=8 wide-c
};

constexpr int num_configs = sizeof(configs) / sizeof(configs[0]);

} // namespace hipconv::cdna4::depthwise_2d_toeplitz
