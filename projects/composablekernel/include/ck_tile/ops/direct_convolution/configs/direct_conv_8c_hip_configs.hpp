// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Configuration set for the 8-channel HIP (intrinsic) grouped direct
// convolution kernel. This header holds ONLY the instantiated configuration
// data (configs_map). The Config struct it parameterizes, and all kernel
// implementation logic, live in kernel/impl/grouped_8c_fp16_hip_conv_impl.hpp,
// which defines Config and then includes this header.

#include "ck_tile/ops/direct_convolution/utils/conv_params.hpp"
#include "ck_tile/ops/direct_convolution/utils/config_map.hpp"

namespace ck_tile::direct_hip_conv::grouped_8c
{

using namespace ck_tile::direct_conv;

// All instantiated configurations. Keys are explicit integers — the mapping from key to
// Config is stable regardless of insertion order, unlike a plain array where position
// implicitly encodes the key.
constexpr auto configs_map = make_config_map<Config>({
    // Dgrad (keys 0–8)
    {0, {.waves_per_wg = 16, .direction = Direction::Dgrad}},
    {1, {.waves_per_wg = 8,  .direction = Direction::Dgrad}},
    {2, {.waves_per_wg = 7,  .direction = Direction::Dgrad}},
    {3, {.waves_per_wg = 6,  .direction = Direction::Dgrad}},
    {4, {.waves_per_wg = 5,  .direction = Direction::Dgrad}},
    {5, {.waves_per_wg = 4,  .direction = Direction::Dgrad}},
    {6, {.waves_per_wg = 3,  .direction = Direction::Dgrad}},
    {7, {.waves_per_wg = 2,  .direction = Direction::Dgrad}},
    {8, {.waves_per_wg = 1,  .direction = Direction::Dgrad}},
    // Fprop (keys 9–17)
    { 9, {.waves_per_wg = 16}},
    {10, {.waves_per_wg = 8}},
    {11, {.waves_per_wg = 7}},
    {12, {.waves_per_wg = 6}},
    {13, {.waves_per_wg = 5}},
    {14, {.waves_per_wg = 4}},
    {15, {.waves_per_wg = 3}},
    {16, {.waves_per_wg = 2}},
    {17, {.waves_per_wg = 1}},
});
static_assert(configs_map.is_valid(), "Duplicate or negative config key in grouped_8c configs_map");

constexpr int NUM_CONFIGS = configs_map.size;

} // namespace ck_tile::direct_hip_conv::grouped_8c
