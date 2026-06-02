// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Configuration set for the 32-channel HIP (intrinsic) grouped direct
// convolution kernel. This header holds ONLY the instantiated configuration
// data (configs_map). The Config struct it parameterizes, and all kernel
// implementation logic, live in kernel/impl/grouped_32c_fp16_hip_conv_impl.hpp,
// which defines Config and then includes this header.

#include "ck_tile/ops/direct_convolution/utils/conv_params.hpp"
#include "ck_tile/ops/direct_convolution/utils/config_map.hpp"

namespace ck_tile::direct_hip_conv::grouped_32c
{

using namespace ck_tile::direct_conv;

// All instantiated configurations. Keys are explicit integers — the mapping from key to
// Config is stable regardless of insertion order, unlike a plain array where position
// implicitly encodes the key.
// waves_per_wg = groups_per_wg * 2 (since 32c uses 2 waves per group)
// groups_per_wg=2 -> waves_per_wg=4, groups_per_wg=1 -> waves_per_wg=2
constexpr auto configs_map = make_config_map<Config>({
    // Dgrad (keys 0–1)
    {0, {.waves_per_wg = 4, .direction = Direction::Dgrad}},
    {1, {.waves_per_wg = 2, .direction = Direction::Dgrad}},
    // Fprop (keys 2–3)
    {2, {.waves_per_wg = 4}},
    {3, {.waves_per_wg = 2}},
});
static_assert(configs_map.is_valid(), "Duplicate or negative config key in grouped_32c configs_map");

constexpr int NUM_CONFIGS = configs_map.size;

} // namespace ck_tile::direct_hip_conv::grouped_32c
