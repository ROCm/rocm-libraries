#pragma once

#include "config.hpp"
#include "hipconv/conv2d_params.hpp"

#include <array>
#include <cstdint>

namespace hipconv::cdna5::direct
{

constexpr auto tile_configs = std::array{
    Config{},
    Config{.aligned = false},
};
constexpr auto directions   = std::array{hipconv::Direction::Fprop, hipconv::Direction::Dgrad};
constexpr auto filter_sizes = std::array{1, 2, 3, 4};
constexpr auto make_configs()
{
    constexpr std::size_t num_configs =
        tile_configs.size() * directions.size() * filter_sizes.size();

    std::array<Config, num_configs> configs;
    std::size_t cfg = 0;
    for(auto& tc : tile_configs)
    {
        for(auto& dir : directions)
        {
            for(auto& f : filter_sizes)
            {
                auto& c     = configs[cfg++];
                c           = tc;
                c.direction = dir;
                c.kh        = f;
                c.kw        = f;
            }
        }
    }
    return configs;
}

// Needed for autoshard
constexpr auto configs    = make_configs();
constexpr int num_configs = configs.size();

} // namespace hipconv::cdna5::direct
