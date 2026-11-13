#pragma once

#include "hipconv/conv_params.hpp"

namespace hipconv::cdna4::grouped_8c_wgrad
{

constexpr int WAVE_SIZE = 64;

struct Config
{
    int waves_per_wg;
    int wave_q32 = 1;

    int kh = 3;
    int kw = 3;

    int group_size = 8;

    // Stride 2 runs as stride 1 over delta upsampled 2x (zeros between samples).
    int stride = 1;

    Direction direction = Direction::Wgrad;

    constexpr int block_c() const { return waves_per_wg * group_size; }
    constexpr int block_q() const { return wave_q32 * 32; }
    constexpr int block_size() const { return waves_per_wg * WAVE_SIZE; }
};

} // namespace hipconv::cdna4::grouped_8c_wgrad
