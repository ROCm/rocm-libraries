#pragma once

#include "config.hpp"

namespace hipconv::cdna4::grouped_16c_wgrad_tf32
{

// Mirrors the fp16/bf16 16c wgrad config ladder. TF32 LDS occupancy is
// ~2x fp16/bf16 wgrad (4 LDS halves at uint2 granularity vs 2 LDS regions
// at uint4 granularity); the largest config (waves_per_wg=4, wave_q32=2)
// still fits under the 65 KB LDS limit on gfx950.
constexpr Config configs[] = {
    {.waves_per_wg = 4, .wave_q32 = 2},
    {.waves_per_wg = 4},
    {.waves_per_wg = 3},
    {.waves_per_wg = 2},
    {.waves_per_wg = 1},
    {.waves_per_wg = 4, .wave_q32 = 2, .stride = 2},
    {.waves_per_wg = 4, .stride = 2},
    {.waves_per_wg = 3, .stride = 2},
    {.waves_per_wg = 2, .stride = 2},
    {.waves_per_wg = 1, .stride = 2},
};

constexpr int num_configs = sizeof(configs) / sizeof(configs[0]);

} // namespace hipconv::cdna4::grouped_16c_wgrad_tf32
