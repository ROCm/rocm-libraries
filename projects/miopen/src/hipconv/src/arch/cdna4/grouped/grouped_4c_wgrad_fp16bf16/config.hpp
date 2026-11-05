#pragma once

#include "hipconv/conv_params.hpp"

namespace hipconv::cdna4::grouped_4c_wgrad
{
// CDNA4 has 64 threads per wave.
constexpr int WAVE_SIZE = 64;

// There are four channels per channel-group.
constexpr int GROUP_SIZE = 4;

// Every wave processes 16 channel-groups.
constexpr int GROUPS_PER_WAVE = 16;

// Kernel configuration (i.e., tiling).
struct Config
{
    // Number of waves tiled along the channel-groups dimension.
    int waves_g;

    // Number of waves tiled along the q-dimension (i.e., output columns).
    int waves_q;

    // Number of output columns per wave divided by 4.
    int wave_q4 = 8;

    // Convolution kernel height ..
    int kh = 3;

    // .. and width.
    int kw = 3;

    // Stride 2 runs as stride 1 over delta upsampled 2x (zeros between samples).
    int stride = 1;

    Direction direction = Direction::Wgrad;

    constexpr int waves() const { return waves_g * waves_q; }
    constexpr int block_groups() const { return waves_g * GROUPS_PER_WAVE; }
    constexpr int block_c() const { return block_groups() * GROUP_SIZE; }
    constexpr int block_q() const { return waves_q * wave_q4 * 4; }
    constexpr int threads() const { return waves() * WAVE_SIZE; }
};

} // namespace hipconv::cdna4::grouped_4c_wgrad
