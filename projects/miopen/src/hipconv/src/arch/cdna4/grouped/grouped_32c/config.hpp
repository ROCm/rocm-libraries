#pragma once

#include "hipconv/conv_params.hpp"

namespace hipconv::cdna4::grouped_32c
{

// 64 threads per wave.
constexpr int WAVE_SIZE = 64;

// Block output is 16 columns wide.
constexpr int BLOCK_Q = 16;

// 32 channels per channel-group.
constexpr int GROUP_SIZE = 32;


// Kernel configuration parameters.
struct Config
{
    int groups_per_wg;

    int kh = 3;
    int kw = 3;
    // Depth taps held per output plane: 1 for conv2d, else the largest kd served.
    int kd = 1;

    int n_fold = 8;

    int stride = 1;

    int dilation = 1;

    Direction direction = Direction::Fprop;

    // Each input load takes a descriptor spanning its row, for an image past 2 GiB.
    bool per_row = false;

    constexpr int waves_per_wg() const { return groups_per_wg * 2; }
    constexpr int block_c() const { return GROUP_SIZE * groups_per_wg; }
    constexpr int block_size() const { return waves_per_wg() * WAVE_SIZE; }
};

} // namespace hipconv::cdna4::grouped_32c
