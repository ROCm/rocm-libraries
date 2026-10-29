#pragma once

#include "config.hpp"
#include "hipconv/conv_params.hpp"

#include <array>
#include <cstdint>

namespace hipconv::cdna5::direct
{

constexpr auto tile_configs = std::array{
    Config{.tile_size_h = 16, .tile_size_n = 1, .tile_size_w = 16},
    Config{.tile_size_h = 8, .tile_size_n = 4, .tile_size_w = 8},
    Config{.tile_size_h = 16, .tile_size_n = 1, .tile_size_w = 16, .aligned = false},
    Config{.tile_size_h = 8, .tile_size_n = 4, .tile_size_w = 8, .aligned = false},
    // TF32 stores 4-byte elements, so the 2-byte tiles above do not carry over:
    //   * tile_size_c 128 -> 64: the interleaved double buffer makes the TDM row pad
    //     num_buf * tile_size_c dwords, which at 4 bytes is 132 > the 128 dword field maximum.
    //   * tile_size_k 256 -> 128: same limit on the dgrad weight pad, which follows
    //     tile_size_k / tiles_k. It also halves the accumulator, paying for the (big, small)
    //     bf16 pair that doubles every A/B operand register.
    // Both together land the LDS footprint below the 2-byte tiles'.
    Config{.tile_size_h = 16,
           .tile_size_n = 1,
           .tile_size_w = 16,
           .tile_size_k = 128,
           .tile_size_c = 64,
           .elem_bytes  = 4},
    Config{.tile_size_h = 16,
           .tile_size_n = 1,
           .tile_size_w = 16,
           .tile_size_k = 128,
           .tile_size_c = 64,
           .aligned     = false,
           .elem_bytes  = 4},
    // The batch-folded tile carries over unchanged: only the halved c and k are forced by the pad
    // field, and at 4 bytes this shape needs 253.7 KiB of LDS against the 2-byte tile's 319.7.
    Config{.tile_size_h = 8,
           .tile_size_n = 4,
           .tile_size_w = 8,
           .tile_size_k = 128,
           .tile_size_c = 64,
           .elem_bytes  = 4},
    Config{.tile_size_h = 8,
           .tile_size_n = 4,
           .tile_size_w = 8,
           .tile_size_k = 128,
           .tile_size_c = 64,
           .aligned     = false,
           .elem_bytes  = 4},
};
constexpr auto directions   = std::array{hipconv::Direction::Fprop, hipconv::Direction::Dgrad};
constexpr auto filter_sizes = std::array{1, 2, 3, 4, 5};
constexpr auto make_configs()
{
    // The batch-folded tiles stop at filter size 3, so each contributes that many configs fewer.
    constexpr std::size_t num_folded = [] {
        std::size_t n = 0;
        for(auto& tc : tile_configs)
            n += tc.tile_size_n == 4;
        return n;
    }();
    constexpr std::size_t num_configs =
        tile_configs.size() * directions.size() * filter_sizes.size() -
        num_folded * (filter_sizes.size() - 3) * directions.size();

    std::array<Config, num_configs> configs;
    std::size_t cfg = 0;
    for(auto& tc : tile_configs)
    {
        for(auto& dir : directions)
        {
            for(auto& f : filter_sizes)
            {
                if(tc.tile_size_n == 4 && f >= 4)
                    continue;
                auto& c     = configs[cfg++];
                c           = tc;
                c.direction = dir;
                c.kh        = f;
                c.kw        = f;
                // Need to limit max padding due to LDS shortage
                if(tc.tile_size_n == 4)
                    c.max_px = 4;
            }
        }
    }
    return configs;
}

// Needed for autoshard
constexpr auto configs    = make_configs();
constexpr int num_configs = configs.size();

} // namespace hipconv::cdna5::direct
