#pragma once

#include "hipconv/conv_params.hpp"

namespace hipconv::cdna4::grouped_4c
{

// 64 threads per wave.
constexpr int WAVE_SIZE = 64;

// Block output is 16 columns wide.
constexpr int WARP_Q = 4;

// 4 channels per group.
constexpr int GROUP_SIZE = 4;

// Kernel configuration parameters.
struct Config
{
    int waves_c64;

    int waves_q4;

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

    constexpr int num_waves() const { return waves_c64 * waves_q4; }

    constexpr int block_c() const { return waves_c64 * 64; }

    constexpr int block_q() const { return waves_q4 * 4; }

    constexpr int block_groups() const { return waves_c64 * 16; }

    constexpr int block_size() const { return num_waves() * WAVE_SIZE; }
};

} // namespace hipconv::cdna4::grouped_4c
