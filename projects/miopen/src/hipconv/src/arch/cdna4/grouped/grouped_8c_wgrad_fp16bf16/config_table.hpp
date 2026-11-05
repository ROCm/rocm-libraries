#pragma once

#include "config.hpp"

namespace hipconv::cdna4::grouped_8c_wgrad
{

// wave_q32 = 1 is slightly faster than wave_q32 = 2 when waves_per_wg = 8,
// yet further tuning might reveal that wave_q32 = 2 is faster for other
// configurations.
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

} // namespace hipconv::cdna4::grouped_8c_wgrad
