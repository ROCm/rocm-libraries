#pragma once

#include "hipconv/conv2d_params.hpp"

#include <array>
#include <cstddef>

// The compiled set of depthwise 1D column-Toeplitz configs.
//
// Intentionally free of device code so the autoshard generator can read
// `num_configs` (and tests can name a config) without compiling the kernel; the
// kernel body lives in kernel.h.

namespace hipconv::cdna4::depthwise_1d_toeplitz
{
using namespace hipconv;

constexpr int WAVE_SIZE = 64;
constexpr int BLOCK_Q   = 32; // 16 tiles x 2 outputs per tile

// waves_per_wg == 16 is a 1024-thread workgroup, where a wave is capped at 128 VGPRs
// (4 waves/SIMD x 128 == the 512-VGPR file). The 9x9/11x11 register footprint overflows
// that into scratch, so large filters omit the 16 rung; block_size <= 512 (waves <= 8)
// grants 256 VGPRs and no spill. c % 128 == 0 shapes remain covered by the waves == 8 rung.
constexpr bool wave_ok(int kh, int w)
{
    return !(kh >= 9 && w == 16);
}
constexpr int ladder_len(int kh)
{
    return kh >= 9 ? 5 : 6; // {16,12,8,4,2,1} minus the 16 rung for large filters
}
// Total generated configs: sum of the (per-filter) wave-ladder lengths across the wide
// families (4 variants) and the folded families (4 variants x 2 folds), plus one narrow
// entry per (filter, variant). Kept in lockstep with make_configs's loops.
constexpr std::size_t num_dw_configs()
{
    constexpr auto filters = std::array{3, 5, 7, 9, 11};
    std::size_t wide       = 0;
    for(int kh : filters)
        wide += ladder_len(kh);
    constexpr std::size_t num_variants = 4, num_wfold_variants = 4, num_wfolds = 2;
    return wide * num_variants + filters.size() * num_variants +
           wide * num_wfold_variants * num_wfolds;
}

struct Config
{
    int waves_per_wg;
    int kh         = 3;
    int kw         = 3; // filter width (== kh for the shipped square filters)
    int group_size = 8;
    int n_fold     = 8;
    int stride     = 1;
    // Input-upsample factor: 1 for Fprop; for Dgrad it carries the forward stride
    // (dgrad = stride-1 rot180 correlation over dY upsampled by s).
    int dilation        = 1;
    Direction direction = Direction::Fprop;
    // Narrow-channel path (C % 8 != 0): no coalesced uint4 load/store, so one
    // 8-channel group per workgroup (waves_per_wg == 1), one channel at a time (b16)
    // and masked past C. LDS layout and MFMA match the wide path.
    bool narrow_c = false;
    // Batch-folded W tiling: pack w_fold images into one 32-wide MFMA tile, each
    // contributing BLOCK_Q/w_fold columns of the same w-window. Fills the tile when a
    // single image leaves trailing tile-columns idle. w_fold == 1 is the one-image tile.
    int w_fold = 1;
    // Input LDS ring depth: 3 == triple-buffered (2 rows in flight to hide load
    // latency), 2 == double-buffered (33% less input LDS -> higher occupancy).
    // Must be >= 2.
    int lds_buffers = 3;

    constexpr int block_c() const { return group_size * waves_per_wg; }
    constexpr int block_size() const { return waves_per_wg * WAVE_SIZE; }
    // Outputs per sub-image, and MFMA tile-columns per sub-image (2 outputs each).
    constexpr int w_sub() const { return BLOCK_Q / w_fold; }
    constexpr int subtiles() const { return 16 / w_fold; }
};

// Generated config set: for every filter size and (stride, dilation, direction)
// family, the wide (C % 8 == 0) wave ladder 16..1 plus one narrow entry.
//
// Order contract: auto-pick takes the first valid config. is_valid_config fixes
// kh/stride/direction and gates waves by c % block_c(), so only the matching
// family's descending ladder competes; numeric --config indices are not a stable ABI.
constexpr auto make_configs()
{
    // One (stride, dilation, direction) family. Fprop varies stride at dilation 1;
    // Dgrad stays stride 1 and carries the forward stride in the upsample dilation.
    struct Variant
    {
        int stride;
        int dilation;
        Direction direction;
    };

    constexpr auto filters  = std::array{3, 5, 7, 9, 11};
    constexpr auto waves    = std::array{16, 12, 8, 4, 2, 1};
    constexpr auto variants = std::array<Variant, 4>{{
        {1, 1, Direction::Fprop}, // stride-1 fprop
        {2, 1, Direction::Fprop}, // stride-2 fprop
        {1, 1, Direction::Dgrad}, // stride-1 dgrad
        {1, 2, Direction::Dgrad}, // stride-2 dgrad
    }};

    // Batch-fold factors (wide, same compute path), stamped as extra families. F=2
    // (w_sub=16) reclaims a half-empty tail tile; F=4 (w_sub=8) fills sub-tile feature
    // maps. is_valid_config / preferred_wfold gate and auto-pick them per shape.
    constexpr auto wfolds         = std::array{2, 4};
    constexpr auto wfold_variants = std::array<Variant, 4>{{
        {1, 1, Direction::Fprop}, // stride-1 fprop
        {2, 1, Direction::Fprop}, // stride-2 fprop
        {1, 1, Direction::Dgrad}, // stride-1 dgrad
        {1, 2, Direction::Dgrad}, // stride-2 dgrad (input upsampled 2x)
    }};

    // Wide: capped wave ladder per family. Narrow: one waves==1 entry per family.
    // Folded: the wide stride-1 wave ladder once per (fold variant, w_fold). Large
    // filters shorten the ladder (ladder_len) to skip the spilling waves==16 rung.
    std::array<Config, num_dw_configs()> configs{};
    std::size_t cfg = 0;
    // Wide (C % 8 == 0): coalesced uint4 path, full wave ladder.
    for(int kh : filters)
        for(const auto& v : variants)
            for(int w : waves)
            {
                if(!wave_ok(kh, w))
                    continue;
                configs[cfg++] = Config{.waves_per_wg = w,
                                        .kh           = kh,
                                        .kw           = kh,
                                        .stride       = v.stride,
                                        .dilation     = v.dilation,
                                        .direction    = v.direction};
            }
    // Narrow (C % 8 != 0): one 8-channel group per workgroup, per-channel b16
    // transfers, masked past C.
    for(int kh : filters)
        for(const auto& v : variants)
            configs[cfg++] = Config{.waves_per_wg = 1,
                                    .kh           = kh,
                                    .kw           = kh,
                                    .stride       = v.stride,
                                    .dilation     = v.dilation,
                                    .direction    = v.direction,
                                    .narrow_c     = true,
                                    // Single-wave (64-thread) fallback caps at 256 VGPRs;
                                    // large filters need double- not triple-buffering to
                                    // keep the input ring's in-flight rows out of scratch.
                                    .lds_buffers = (kh >= 5) ? 2 : 3};
    // Batch-folded (F in wfolds) wide stride-1 ladders (fprop + dgrad).
    for(int kh : filters)
        for(const auto& v : wfold_variants)
            for(int f : wfolds)
                for(int w : waves)
                {
                    if(!wave_ok(kh, w))
                        continue;
                    configs[cfg++] = Config{.waves_per_wg = w,
                                            .kh           = kh,
                                            .kw           = kh,
                                            .stride       = v.stride,
                                            .dilation     = v.dilation,
                                            .direction    = v.direction,
                                            .w_fold       = f};
                }
    return configs;
}

constexpr auto configs = make_configs();

constexpr int num_configs = static_cast<int>(configs.size());

} // namespace hipconv::cdna4::depthwise_1d_toeplitz
