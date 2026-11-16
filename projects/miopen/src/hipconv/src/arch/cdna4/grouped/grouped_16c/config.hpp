#pragma once

#include "hipconv/conv_params.hpp"

namespace hipconv::cdna4::grouped_16c
{

// 64 threads per wave.
constexpr int WAVE_SIZE = 64;

// Block output is 16 columns wide.
constexpr int BLOCK_Q = 16;


// Kernel configuration parameters.
struct Config
{
    int waves_per_wg;

    int kh = 3;
    int kw = 3;
    // Depth taps held per output plane: 1 for conv2d, else the largest kd served.
    int kd = 1;

    int group_size = 16;

    int n_fold = 8;

    int stride = 1;

    int dilation = 1;

    Direction direction = Direction::Fprop;

    // Each input load takes a descriptor spanning its row, for an image past 2 GiB.
    bool per_row = false;

    constexpr int block_c() const { return group_size * waves_per_wg; }

    constexpr int block_size() const { return waves_per_wg * WAVE_SIZE; }
};

} // namespace hipconv::cdna4::grouped_16c
