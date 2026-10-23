#pragma once

#include "config.hpp"
#include "hipconv/conv_params.hpp"

#include <array>
#include <cstdint>

namespace hipconv::cdna5::direct_wgrad
{

constexpr auto configs = std::array{
    Config{.kh = 2, .kw = 2, .tile_size_c = 128, .tile_size_k = 128},
    Config{.kh = 3, .kw = 3, .tile_size_c = 64, .tile_size_k = 128},
    Config{.kh = 4, .kw = 4, .tile_size_c = 32, .tile_size_k = 128},
    Config{.kh = 5, .kw = 5, .tile_size_c = 32, .tile_size_k = 64},
    Config{.kh = 3, .kw = 1, .tile_size_c = 128, .tile_size_k = 128},
    Config{.kh = 1, .kw = 3, .tile_size_c = 128, .tile_size_k = 128},
};
constexpr int num_configs = configs.size();

} // namespace hipconv::cdna5::direct_wgrad
