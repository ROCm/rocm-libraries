#pragma once

#include "config.hpp"

namespace hipconv::cdna4::grouped_16c_wgrad
{

constexpr Config configs[] = {
    {.waves_per_wg = 4, .wave_q32 = 2},
    {.waves_per_wg = 4},
    {.waves_per_wg = 3},
    {.waves_per_wg = 2},
    {.waves_per_wg = 1},
    {.waves_per_wg = 4, .stride = 2},
    {.waves_per_wg = 3, .stride = 2},
    {.waves_per_wg = 2, .stride = 2},
    {.waves_per_wg = 1, .stride = 2},
};

constexpr int num_configs = sizeof(configs) / sizeof(configs[0]);

} // namespace hipconv::cdna4::grouped_16c_wgrad
