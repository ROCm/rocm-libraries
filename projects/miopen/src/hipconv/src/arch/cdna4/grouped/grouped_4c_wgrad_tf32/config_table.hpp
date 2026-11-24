#pragma once

#include "config.hpp"

namespace hipconv::cdna4::grouped_4c_wgrad_tf32
{

constexpr Config configs[] = {
    // The smallest configuration is often the fastest.
    // Increasing occupancy is more effective than decreasing global atomicAdd
    // here (the kernel is memory-bound).
    {.waves_g = 1, .waves_q = 4, .wave_q4 = 1},
    {.waves_g = 1, .waves_q = 4, .wave_q4 = 1, .stride = 2},
};

constexpr int num_configs = sizeof(configs) / sizeof(configs[0]);

} // namespace hipconv::cdna4::grouped_4c_wgrad_tf32
