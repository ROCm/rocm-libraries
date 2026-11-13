#pragma once

#include "config.hpp"

namespace hipconv::cdna4::grouped_32c
{

// All instantiated configurations. Shared across fp16 / bf16 / tf32.
// Max groups_per_wg = 8; limited by register allocation and LDS.
// groups_per_wg = 2 performs better in practice.
constexpr Config configs[] = {
    {.groups_per_wg = 2, .direction = Direction::Dgrad},
    {.groups_per_wg = 1, .direction = Direction::Dgrad},
    {.groups_per_wg = 2, .dilation = 2, .direction = Direction::Dgrad},
    {.groups_per_wg = 1, .dilation = 2, .direction = Direction::Dgrad},
    {.groups_per_wg = 2},
    {.groups_per_wg = 1},
    {.groups_per_wg = 2, .stride = 2},
    {.groups_per_wg = 1, .stride = 2},
    // conv3d, kd <= 3.
    {.groups_per_wg = 2, .kd = 3, .direction = Direction::Dgrad},
    {.groups_per_wg = 1, .kd = 3, .direction = Direction::Dgrad},
    {.groups_per_wg = 2, .kd = 3, .dilation = 2, .direction = Direction::Dgrad},
    {.groups_per_wg = 1, .kd = 3, .dilation = 2, .direction = Direction::Dgrad},
    {.groups_per_wg = 2, .kd = 3},
    {.groups_per_wg = 1, .kd = 3},
    {.groups_per_wg = 2, .kd = 3, .stride = 2},
    {.groups_per_wg = 1, .kd = 3, .stride = 2},
    // conv3d, kd <= 5.
    {.groups_per_wg = 2, .kd = 5, .direction = Direction::Dgrad},
    {.groups_per_wg = 1, .kd = 5, .direction = Direction::Dgrad},
    {.groups_per_wg = 2, .kd = 5, .dilation = 2, .direction = Direction::Dgrad},
    {.groups_per_wg = 1, .kd = 5, .dilation = 2, .direction = Direction::Dgrad},
    {.groups_per_wg = 2, .kd = 5},
    {.groups_per_wg = 1, .kd = 5},
    {.groups_per_wg = 2, .kd = 5, .stride = 2},
    {.groups_per_wg = 1, .kd = 5, .stride = 2},
    // conv3d, kd <= 7.
    {.groups_per_wg = 2, .kd = 7, .direction = Direction::Dgrad},
    {.groups_per_wg = 1, .kd = 7, .direction = Direction::Dgrad},
    {.groups_per_wg = 2, .kd = 7, .dilation = 2, .direction = Direction::Dgrad},
    {.groups_per_wg = 1, .kd = 7, .dilation = 2, .direction = Direction::Dgrad},
    {.groups_per_wg = 2, .kd = 7},
    {.groups_per_wg = 1, .kd = 7},
    {.groups_per_wg = 2, .kd = 7, .stride = 2},
    {.groups_per_wg = 1, .kd = 7, .stride = 2},
    // An image past 2 GiB: per class the widest, the widest of 8 waves or fewer, the narrowest.
    {.groups_per_wg = 2, .direction = Direction::Dgrad, .per_row = true},
    {.groups_per_wg = 1, .direction = Direction::Dgrad, .per_row = true},
    {.groups_per_wg = 2, .dilation = 2, .direction = Direction::Dgrad, .per_row = true},
    {.groups_per_wg = 1, .dilation = 2, .direction = Direction::Dgrad, .per_row = true},
    {.groups_per_wg = 2, .per_row = true},
    {.groups_per_wg = 1, .per_row = true},
    {.groups_per_wg = 2, .stride = 2, .per_row = true},
    {.groups_per_wg = 1, .stride = 2, .per_row = true},
    {.groups_per_wg = 2, .kd = 3, .direction = Direction::Dgrad, .per_row = true},
    {.groups_per_wg = 1, .kd = 3, .direction = Direction::Dgrad, .per_row = true},
    {.groups_per_wg = 2, .kd = 3, .dilation = 2, .direction = Direction::Dgrad, .per_row = true},
    {.groups_per_wg = 1, .kd = 3, .dilation = 2, .direction = Direction::Dgrad, .per_row = true},
    {.groups_per_wg = 2, .kd = 3, .per_row = true},
    {.groups_per_wg = 1, .kd = 3, .per_row = true},
    {.groups_per_wg = 2, .kd = 3, .stride = 2, .per_row = true},
    {.groups_per_wg = 1, .kd = 3, .stride = 2, .per_row = true},
    {.groups_per_wg = 2, .kd = 5, .direction = Direction::Dgrad, .per_row = true},
    {.groups_per_wg = 1, .kd = 5, .direction = Direction::Dgrad, .per_row = true},
    {.groups_per_wg = 2, .kd = 5, .dilation = 2, .direction = Direction::Dgrad, .per_row = true},
    {.groups_per_wg = 1, .kd = 5, .dilation = 2, .direction = Direction::Dgrad, .per_row = true},
    {.groups_per_wg = 2, .kd = 5, .per_row = true},
    {.groups_per_wg = 1, .kd = 5, .per_row = true},
    {.groups_per_wg = 2, .kd = 5, .stride = 2, .per_row = true},
    {.groups_per_wg = 1, .kd = 5, .stride = 2, .per_row = true},
    {.groups_per_wg = 2, .kd = 7, .direction = Direction::Dgrad, .per_row = true},
    {.groups_per_wg = 1, .kd = 7, .direction = Direction::Dgrad, .per_row = true},
    {.groups_per_wg = 2, .kd = 7, .dilation = 2, .direction = Direction::Dgrad, .per_row = true},
    {.groups_per_wg = 1, .kd = 7, .dilation = 2, .direction = Direction::Dgrad, .per_row = true},
    {.groups_per_wg = 2, .kd = 7, .per_row = true},
    {.groups_per_wg = 1, .kd = 7, .per_row = true},
    {.groups_per_wg = 2, .kd = 7, .stride = 2, .per_row = true},
    {.groups_per_wg = 1, .kd = 7, .stride = 2, .per_row = true},
};

constexpr int num_configs = sizeof(configs) / sizeof(configs[0]);

} // namespace hipconv::cdna4::grouped_32c
