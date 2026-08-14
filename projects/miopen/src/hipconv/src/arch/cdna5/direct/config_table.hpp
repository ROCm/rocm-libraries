#pragma once

#include "config.hpp"
#include "hipconv/conv2d_params.hpp"

#include <array>
#include <cstdint>

namespace hipconv::cdna5::direct
{

constexpr auto tile_configs = std::array{
    Config{.tile_size_h = 16, .tile_size_n = 1, .tile_size_w = 16},
    Config{.tile_size_h = 8, .tile_size_n = 4, .tile_size_w = 8},
    Config{.tile_size_h = 16, .tile_size_n = 1, .tile_size_w = 16, .aligned = false},
    Config{.tile_size_h = 8, .tile_size_n = 4, .tile_size_w = 8, .aligned = false},
};
constexpr auto directions   = std::array{hipconv::Direction::Fprop, hipconv::Direction::Dgrad};
constexpr auto filter_sizes = std::array{1, 2, 3, 4, 5};
constexpr auto make_configs()
{
    constexpr std::size_t num_configs =
        tile_configs.size() * directions.size() * filter_sizes.size() -
        2 * (filter_sizes.size() - 3) * directions.size();

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
