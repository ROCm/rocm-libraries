#pragma once

#include "hipconv/conv2d_params.hpp"

namespace hipconv::cdna5::grouped_multi_g
{

constexpr int WAVE_SIZE = 32;

struct Config
{
    int waves_per_wg;
    int kh = 3;
    int kw = 3;
    // Channels per group. G in {4,8,16} pack GPW=16/G groups per wave into one
    // 16x16x32 WMMA (block-diagonal weight). G=32 uses one group per wave with
    // a full K=32 contraction across two M-tiles. The same templated kernel
    // covers all of them; `group_size` selects the regime at compile time.
    int group_size = 16;
    int stride     = 1;
    int dilation   = 1;
    // Number of LDS input-row slots = max in-flight TDM loads. PF=2 is the
    // classic double buffer (1 load overlaps 1 compute step); PF>2 keeps
    // (PF-1) row loads in flight to hide TDM latency (`s_wait_tensorcnt`).
    int prefetch_depth           = 2;
    hipconv::Direction direction = hipconv::Direction::Fprop;

    constexpr int block_size() const { return waves_per_wg * WAVE_SIZE; }
};

} // namespace hipconv::cdna5::grouped_multi_g
