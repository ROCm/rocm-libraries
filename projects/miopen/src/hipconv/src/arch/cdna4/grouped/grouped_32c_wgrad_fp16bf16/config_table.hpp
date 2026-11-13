#pragma once

#include "config.hpp"

namespace hipconv::cdna4::grouped_32c_wgrad
{

constexpr Config configs[] = {
    {.groups_per_wg = 2, .wave_q32 = 2},
    {.groups_per_wg = 2, .wave_q32 = 1, .wave_n = 2},
    {.groups_per_wg = 2, .wave_q32 = 1},
    {.groups_per_wg = 1, .wave_q32 = 2},
    {.groups_per_wg = 1, .wave_q32 = 1, .wave_n = 2},
    {.groups_per_wg = 1, .wave_q32 = 1},
    {.groups_per_wg = 2, .stride = 2},
    {.groups_per_wg = 1, .stride = 2},
};

constexpr int num_configs = sizeof(configs) / sizeof(configs[0]);

} // namespace hipconv::cdna4::grouped_32c_wgrad
