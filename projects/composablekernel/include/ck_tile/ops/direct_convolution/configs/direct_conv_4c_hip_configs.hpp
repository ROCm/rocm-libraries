// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Configuration set for the 4-channel HIP (intrinsic) grouped direct
// convolution kernel. This header holds ONLY the instantiated configuration
// data (configs_map). The Config struct it parameterizes, and all kernel
// implementation logic, live in kernel/impl/grouped_4c_fp16_hip_conv_impl.hpp,
// which defines Config and then includes this header.

#include "ck_tile/ops/direct_convolution/utils/conv_params.hpp"
#include "ck_tile/ops/direct_convolution/utils/config_map.hpp"

namespace ck_tile::direct_hip_conv::grouped_4c
{

using namespace ck_tile::direct_conv;

// All instantiated configurations. The first valid config is expected to be the fastest.
// Keys are explicit integers — the mapping from key to Config is stable regardless of
// insertion order, unlike a plain array where position implicitly encodes the key.
constexpr auto configs_map = make_config_map<Config>({
    // Dgrad (keys 0–4)
    {0, {.waves_c64 = 2, .waves_q4 = 8, .direction = Direction::Dgrad}},
    {1, {.waves_c64 = 2, .waves_q4 = 4, .direction = Direction::Dgrad}},
    {2, {.waves_c64 = 2, .waves_q4 = 2, .direction = Direction::Dgrad}},
    {3, {.waves_c64 = 2, .waves_q4 = 1, .direction = Direction::Dgrad}},
    {4, {.waves_c64 = 1, .waves_q4 = 1, .direction = Direction::Dgrad}},
    // Fprop (keys 5–9)
    {5, {.waves_c64 = 2, .waves_q4 = 8}},
    {6, {.waves_c64 = 2, .waves_q4 = 4}},
    {7, {.waves_c64 = 2, .waves_q4 = 2}},
    {8, {.waves_c64 = 2, .waves_q4 = 1}},
    {9, {.waves_c64 = 1, .waves_q4 = 1}},
});
static_assert(configs_map.is_valid(), "Duplicate or negative config key in grouped_4c configs_map");

constexpr int NUM_CONFIGS = configs_map.size;

} // namespace ck_tile::direct_hip_conv::grouped_4c
