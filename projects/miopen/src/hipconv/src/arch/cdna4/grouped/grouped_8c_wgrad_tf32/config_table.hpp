#pragma once

#include "config.hpp"

namespace hipconv::cdna4::grouped_8c_wgrad_tf32
{

// TF32 LDS budget is ~2x BF16/FP16 wgrad (4 LDS halves at uint2 granularity
// vs 2 LDS regions at uint4 granularity). The full waves_per_wg ladder still
// fits under the 65 KB LDS limit on gfx950, so we expose the same range as
// the fp16/bf16 sibling.
constexpr Config configs[] = {
    {.waves_per_wg = 8},
    {.waves_per_wg = 7},
    {.waves_per_wg = 6},
    {.waves_per_wg = 5},
    {.waves_per_wg = 4},
    {.waves_per_wg = 3},
    {.waves_per_wg = 2},
    {.waves_per_wg = 1},
    {.waves_per_wg = 8, .stride = 2},
    {.waves_per_wg = 7, .stride = 2},
    {.waves_per_wg = 6, .stride = 2},
    {.waves_per_wg = 5, .stride = 2},
    {.waves_per_wg = 4, .stride = 2},
    {.waves_per_wg = 3, .stride = 2},
    {.waves_per_wg = 2, .stride = 2},
    {.waves_per_wg = 1, .stride = 2},
};

constexpr int num_configs = sizeof(configs) / sizeof(configs[0]);

} // namespace hipconv::cdna4::grouped_8c_wgrad_tf32
