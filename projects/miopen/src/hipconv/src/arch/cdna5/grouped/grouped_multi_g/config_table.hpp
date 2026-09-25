#pragma once

#include "config.hpp"
#include <array>

namespace hipconv::cdna5::grouped_multi_g
{

// One stride/dilation/direction family, replicated across waves_per_wg. The
// table below stamps this family out for each supported group_size so a single
// kernel TU serves G in {4,8,16,32} (G<=16 via the GPW-packing kernel, G=32 via
// the full-K / two-M-tile kernel; launch_impl dispatches on group_size).
//   * Fprop stride=1 dil=1 / Dgrad stride=1 dil=1: prefetch_depth=3 (2 TDM
//     row-loads in flight) is the measured sweet spot.
//   * Fprop stride=2 dil=1 / Dgrad dil=2: the stride-2/dilation-2 pair; dgrad
//     dil=2 uses prefetch_depth=2 since only every other virtual row loads.
constexpr auto make_configs()
{
    // One stride/dilation/direction "family" variant, replicated across every
    // wave count and stamped out for each group size. Mirrors the four blocks of
    // the old HIPCONV_MULTI_G_FAMILY macro, in the same order.
    struct Variant
    {
        int stride;
        int dilation;
        int prefetch_depth;
        hipconv::Direction direction;
    };

    constexpr auto group_sizes = std::array{4, 8, 16, 32};
    constexpr auto waves       = std::array{8, 4, 2, 1};
    constexpr auto variants    = std::array<Variant, 4>{{
        {1, 1, 3, hipconv::Direction::Fprop}, // stride-1 fprop
        {1, 1, 3, hipconv::Direction::Dgrad}, // stride-1 dgrad
        {2, 1, 3, hipconv::Direction::Fprop}, // stride-2 fprop
        {1, 2, 2, hipconv::Direction::Dgrad}, // dilation-2 dgrad
    }};

    std::array<Config, group_sizes.size() * variants.size() * waves.size()> configs{};
    std::size_t cfg = 0;
    for(int g : group_sizes)
        for(const auto& v : variants)
            for(int w : waves)
                configs[cfg++] = Config{.waves_per_wg   = w,
                                        .group_size     = g,
                                        .stride         = v.stride,
                                        .dilation       = v.dilation,
                                        .prefetch_depth = v.prefetch_depth,
                                        .direction      = v.direction};
    return configs;
}

constexpr auto configs    = make_configs();
constexpr int num_configs = configs.size();

} // namespace hipconv::cdna5::grouped_multi_g
