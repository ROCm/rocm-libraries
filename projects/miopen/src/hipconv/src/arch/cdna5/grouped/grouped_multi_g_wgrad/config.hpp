#pragma once

#include "hipconv/conv2d_params.hpp"

namespace hipconv::cdna5::grouped_multi_g_wgrad
{

constexpr int WAVE_SIZE = 32;

struct Config
{
    int group_size; // 4, 8, 16, or 32
    int waves_per_wg;
    int kh                       = 3;
    int kw                       = 3;
    int prefetch_depth           = 3;
    hipconv::Direction direction = hipconv::Direction::Wgrad;
    // dW reduction strategy when >1 workgroup contributes (q-tiles or batch).
    bool split_k = false;

    constexpr int block_size() const { return waves_per_wg * WAVE_SIZE; }
};

} // namespace hipconv::cdna5::grouped_multi_g_wgrad
