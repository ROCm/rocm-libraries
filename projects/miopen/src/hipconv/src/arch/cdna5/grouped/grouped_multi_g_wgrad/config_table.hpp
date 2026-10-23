#pragma once

#include "config.hpp"

namespace hipconv::cdna5::grouped_multi_g_wgrad
{
constexpr Config configs[] = {
    // G x waves; split_k=false (single-contributor / cascading-atomic) first,
    // then the split-K variants. is_valid_config matches par.channels_per_group
    // to group_size, so only the configs for the shape's G are ever selected.
    {.group_size = 4, .waves_per_wg = 8},
    {.group_size = 4, .waves_per_wg = 4},
    {.group_size = 4, .waves_per_wg = 2},
    {.group_size = 4, .waves_per_wg = 1},
    {.group_size = 8, .waves_per_wg = 8},
    {.group_size = 8, .waves_per_wg = 4},
    {.group_size = 8, .waves_per_wg = 2},
    {.group_size = 8, .waves_per_wg = 1},
    {.group_size = 16, .waves_per_wg = 8},
    {.group_size = 16, .waves_per_wg = 4},
    {.group_size = 16, .waves_per_wg = 2},
    {.group_size = 16, .waves_per_wg = 1},
    {.group_size = 32, .waves_per_wg = 8},
    {.group_size = 32, .waves_per_wg = 4},
    {.group_size = 32, .waves_per_wg = 2},
    {.group_size = 32, .waves_per_wg = 1},
    {.group_size = 4, .waves_per_wg = 8, .split_k = true},
    {.group_size = 4, .waves_per_wg = 4, .split_k = true},
    {.group_size = 8, .waves_per_wg = 8, .split_k = true},
    {.group_size = 8, .waves_per_wg = 4, .split_k = true},
    {.group_size = 16, .waves_per_wg = 8, .split_k = true},
    {.group_size = 16, .waves_per_wg = 4, .split_k = true},
    {.group_size = 32, .waves_per_wg = 8, .split_k = true},
    {.group_size = 32, .waves_per_wg = 4, .split_k = true},
};
constexpr int num_configs = sizeof(configs) / sizeof(Config);

} // namespace hipconv::cdna5::grouped_multi_g_wgrad
