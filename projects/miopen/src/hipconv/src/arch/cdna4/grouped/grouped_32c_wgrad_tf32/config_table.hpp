#pragma once

#include "config.hpp"

namespace hipconv::cdna4::grouped_32c_wgrad_tf32
{

// Mirrors the fp16/bf16 32c wgrad config ladder. TF32 LDS occupancy is ~2x
// fp16/bf16 wgrad (4 LDS halves at uint2 granularity vs 2 LDS regions at
// uint4 granularity); each combination of (groups_per_wg, wave_q32, wave_n)
// still fits within the 65 KB compute / 76 KB staging budget.
constexpr Config configs[] = {
    {.groups_per_wg = 2, .wave_q32 = 2},
    {.groups_per_wg = 2, .wave_q32 = 1, .wave_n = 2},
    {.groups_per_wg = 2, .wave_q32 = 1},
    {.groups_per_wg = 1, .wave_q32 = 2},
    {.groups_per_wg = 1, .wave_q32 = 1, .wave_n = 2},
    {.groups_per_wg = 1, .wave_q32 = 1},
    {.groups_per_wg = 2, .wave_q32 = 2, .stride = 2},
    {.groups_per_wg = 2, .wave_q32 = 1, .wave_n = 2, .stride = 2},
    {.groups_per_wg = 2, .wave_q32 = 1, .stride = 2},
    {.groups_per_wg = 1, .wave_q32 = 2, .stride = 2},
    {.groups_per_wg = 1, .wave_q32 = 1, .wave_n = 2, .stride = 2},
    {.groups_per_wg = 1, .wave_q32 = 1, .stride = 2},
};

constexpr int num_configs = sizeof(configs) / sizeof(configs[0]);

} // namespace hipconv::cdna4::grouped_32c_wgrad_tf32
