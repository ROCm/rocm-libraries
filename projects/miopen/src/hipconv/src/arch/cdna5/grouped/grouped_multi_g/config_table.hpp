#pragma once

#include "config.hpp"
#include <array>

namespace hipconv::cdna5::grouped_multi_g
{

// One stride/dilation/direction family, stamped out for each group size so a
// single kernel TU serves G in {4,8,16,32} (G<=16 via the GPW-packing kernel,
// G=32 via the full-K / two-M-tile kernel; launch_impl dispatches on
// group_size).
//   * Fprop stride=1 dil=1 / Dgrad stride=1 dil=1: prefetch_depth=3 (2 TDM
//     row-loads in flight) is the measured sweet spot.
//   * Fprop stride=2 dil=1 / Dgrad dil=2: the stride-2/dilation-2 pair; dgrad
//     dil=2 uses prefetch_depth=2 since only every other virtual row loads.
// Each group size takes that depth. Two stride-1 dgrad shapes take a measured
// exception: n64_h162_w92_c512_k512_kh3_kw3_g32 runs PF=2, and
// n64_h182_w322_c128_k128_kh3_kw3_g16 runs PF=4. The kernel and its LDS ring
// read cfg.prefetch_depth, so the descriptor names the depth that runs.
//
// Stride-1 fprop and dgrad carry every legal (h_split, waves_per_simd) pair. Splits
// 3 and 6 use 3 waves/SIMD; 2 uses 2, 3, or 4; 4 and 8 use 2 or 4; 1 uses
// 2, 3, or 4; 12 uses 3 and 16 uses 4. Splits 1, 2, and 4 keep a 4-wave
// workgroup. Split 3
// is 3 waves: 3 per SIMD leaves no room for a 4th. Splits 6, 8, 12, and 16
// are the workgroup. Everything else stays at (1, 4). G=32 uses the same
// pairs, one group per wave.
constexpr auto make_configs()
{
    struct Variant
    {
        int stride;
        int dilation;
        int prefetch_depth;
        hipconv::Direction direction;
    };
    struct Split
    {
        int h_split;
        int waves_per_simd;
        int waves_per_wg;
    };

    constexpr auto group_sizes   = std::array{4, 8, 16, 32};
    constexpr auto stride1_fprop = std::array<Split, 14>{{
        {1, 4, 4},
        {1, 3, 4},
        {1, 2, 4},
        {2, 4, 4},
        {2, 3, 4},
        {2, 2, 4},
        {3, 3, 3},
        {4, 4, 4},
        {4, 2, 4},
        {6, 3, 6},
        {8, 4, 8},
        {8, 2, 8},
        {12, 3, 12},
        {16, 4, 16},
    }};
    constexpr auto plain         = std::array<Split, 1>{{{1, 4, 4}}};
    constexpr auto variants      = std::array<Variant, 4>{{
        {1, 1, 3, hipconv::Direction::Fprop}, // stride-1 fprop
        {1, 1, 3, hipconv::Direction::Dgrad}, // stride-1 dgrad
        {2, 1, 3, hipconv::Direction::Fprop}, // stride-2 fprop
        {1, 2, 2, hipconv::Direction::Dgrad}, // dilation-2 dgrad
    }};

    // The two measured dgrad exceptions, appended after the variant stamp.
    constexpr int n_dgrad_pf_overrides = 2;
    // Stride-1 fprop and dgrad, splits 1..4, at the wave counts those splits
    // allow, for prefetch 2 and 4 (3 is the variant above). Split 1 also gets
    // 1 wave/SIMD at prefetch 2, 3, and 4. Two of those cells are the dgrad
    // overrides, so they are not stamped again.
    constexpr int n_sweep_pairs = 9;
    constexpr int n_sweep = group_sizes.size() * 2 * 2 * n_sweep_pairs - n_dgrad_pf_overrides +
                            group_sizes.size() * 2 * 3;
    constexpr int n_configs =
        group_sizes.size() * (2 * stride1_fprop.size() + (variants.size() - 2) * plain.size()) +
        n_dgrad_pf_overrides + n_sweep;
    std::array<Config, n_configs> configs{};
    std::size_t cfg = 0;
    for(int g : group_sizes)
        for(const auto& v : variants)
        {
            const bool stride1  = v.stride == 1 && v.dilation == 1;
            const Split* splits = stride1 ? stride1_fprop.data() : plain.data();
            const int split_count =
                stride1 ? static_cast<int>(stride1_fprop.size()) : static_cast<int>(plain.size());
            for(int i = 0; i < split_count; ++i)
            {
                const Split& s = splits[i];
                configs[cfg++] = Config{.waves_per_wg   = s.waves_per_wg,
                                        .group_size     = g,
                                        .stride         = v.stride,
                                        .dilation       = v.dilation,
                                        .prefetch_depth = v.prefetch_depth,
                                        .direction      = v.direction,
                                        .h_split        = s.h_split,
                                        .waves_per_simd = s.waves_per_simd};
            }
        }
    // h162 g32 dgrad, group size 16, split 1-4. The sweep's best depth is 2.
    configs[cfg++] = Config{.waves_per_wg   = 4,
                            .group_size     = 16,
                            .prefetch_depth = 2,
                            .direction      = hipconv::Direction::Dgrad,
                            .h_split        = 1,
                            .waves_per_simd = 4};
    // w322 g16 dgrad, group size 8, split 1-3. The sweep's best depth is 4.
    configs[cfg++] = Config{.waves_per_wg   = 4,
                            .group_size     = 8,
                            .prefetch_depth = 4,
                            .direction      = hipconv::Direction::Dgrad,
                            .h_split        = 1,
                            .waves_per_simd = 3};
    // Extra prefetch and the split-1 single-wave cells used by the tf32 sweep.
    constexpr auto sweep_splits = std::array<Split, n_sweep_pairs>{{
        {1, 4, 4},
        {1, 3, 4},
        {1, 2, 4},
        {2, 4, 4},
        {2, 3, 4},
        {2, 2, 4},
        {3, 3, 3},
        {4, 4, 4},
        {4, 2, 4},
    }};
    constexpr auto sweep_dirs   = std::array{hipconv::Direction::Fprop, hipconv::Direction::Dgrad};
    for(int g : group_sizes)
        for(auto dir : sweep_dirs)
            for(int pf : {2, 4})
                for(const auto& s : sweep_splits)
                {
                    const bool override_pf2 = dir == hipconv::Direction::Dgrad && g == 16 &&
                                              s.h_split == 1 && s.waves_per_simd == 4 && pf == 2;
                    const bool override_pf4 = dir == hipconv::Direction::Dgrad && g == 8 &&
                                              s.h_split == 1 && s.waves_per_simd == 3 && pf == 4;
                    if(override_pf2 || override_pf4)
                        continue;
                    configs[cfg++] = Config{.waves_per_wg   = s.waves_per_wg,
                                            .group_size     = g,
                                            .stride         = 1,
                                            .dilation       = 1,
                                            .prefetch_depth = pf,
                                            .direction      = dir,
                                            .h_split        = s.h_split,
                                            .waves_per_simd = s.waves_per_simd};
                }
    for(int g : group_sizes)
        for(auto dir : sweep_dirs)
            for(int pf : {2, 3, 4})
                configs[cfg++] = Config{.waves_per_wg   = 4,
                                        .group_size     = g,
                                        .stride         = 1,
                                        .dilation       = 1,
                                        .prefetch_depth = pf,
                                        .direction      = dir,
                                        .h_split        = 1,
                                        .waves_per_simd = 1};
    if(cfg != configs.size())
        throw "grouped config table did not fill every slot";
    return configs;
}

constexpr auto configs    = make_configs();
constexpr int num_configs = configs.size();

} // namespace hipconv::cdna5::grouped_multi_g
