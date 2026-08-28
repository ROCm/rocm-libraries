#pragma once

#include "config.hpp"
#include <array>

namespace hipconv::cdna4::direct
{

constexpr auto tile_configs = std::array{
    // Big tiles come first
    Config{.tile_size_k = 256, .tile_size_c = 64},
    Config{.tile_size_k = 256,
           .tile_size_c = 64,
           .tile_size_n = 4,
           .tile_size_h = 8,
           .tile_size_w = 8},
    Config{.tile_size_k = 128, .tile_size_c = 64, .k_parts = 1},
    Config{.tile_size_k = 64, .tile_size_c = 64, .k_parts = 1},
};
constexpr auto types        = std::array{hipconv::DataType::fp16, hipconv::DataType::bf16};
constexpr auto directions   = std::array{hipconv::Direction::Fprop, hipconv::Direction::Dgrad};
constexpr auto filter_sizes = std::array{1, 3, 4, 5, 6, 7};
constexpr auto make_configs()
{
    constexpr std::size_t num_configs =
        tile_configs.size() * types.size() * directions.size() * filter_sizes.size() -
        (filter_sizes.size() - 2) * types.size() * directions.size();

    std::array<Config, num_configs> configs;
    std::size_t cfg = 0;
    for(auto& dir : directions)
    {
        for(auto& type : types)
        {
            for(auto& f : filter_sizes)
            {
                for(auto& tc : tile_configs)
                {
                    // Need to exclude these configs due to out of LDS
                    if(f >= 4 && tc.tile_size_n == 4)
                        continue;
                    auto& c     = configs[cfg++];
                    c           = tc;
                    c.direction = dir;
                    c.type      = type;
                    c.kh        = f;
                    c.kw        = f;
                }
            }
        }
    }
    return configs;
}

// All instantiated configurations.
constexpr auto configs    = make_configs();
constexpr int num_configs = configs.size();

} // namespace hipconv::cdna4::direct
