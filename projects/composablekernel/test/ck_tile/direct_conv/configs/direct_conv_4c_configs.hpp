// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Configuration set for the 4-channel grouped direct convolution kernel (v3).
// This header holds ONLY the instantiated configuration data (configs_map /
// KernelConfigurations). The Config struct it parameterizes, and all kernel
// implementation logic, live in kernel/impl/grouped_4c_tile_conv_impl_v3.hpp,
// which defines Config and then includes this header.

#include "ck_tile/ops/direct_convolution/utils/common.hpp"
#include "ck_tile/ops/direct_convolution/utils/conv_params.hpp"
#include "ck_tile/ops/direct_convolution/utils/config_map.hpp"

namespace ck_tile::direct_conv::grouped_4c_tile::v3
{

// All instantiated configurations. The first valid config is expected to be the fastest.
//
// Layout: 4 variant groups × 10 configs each = 40 configs total.
// Each group has 5 Dgrad (indices 0-4) + 5 Fprop (indices 5-9) configs:
//   waves_c64=2,waves_q4=8 / waves_c64=2,waves_q4=4 / waves_c64=2,waves_q4=2 /
//   waves_c64=2,waves_q4=1 / waves_c64=1,waves_q4=1
//
// Group 0 (indices  0- 9): Cyclic-shift swizzle, direct DRAM epilogue
// Group 1 (indices 10-19): Cyclic-shift swizzle, LDS-staged epilogue
// Group 2 (indices 20-29): XOR swizzle, direct DRAM epilogue
// Group 3 (indices 30-39): XOR swizzle, LDS-staged epilogue
template <DataType DT = DataType::fp16>
struct KernelConfigurations
{
static constexpr auto configs_map = make_config_map<Config<DT>>({
    // ---- Group 0: No swizzle, direct DRAM epilogue ----
    // Dgrad (keys 0-4)
    { 0, {.waves_c64 = 2, .waves_q4 = 8, .direction = Direction::Dgrad}},
    { 1, {.waves_c64 = 2, .waves_q4 = 4, .direction = Direction::Dgrad}},
    { 2, {.waves_c64 = 2, .waves_q4 = 2, .direction = Direction::Dgrad}},
    { 3, {.waves_c64 = 2, .waves_q4 = 1, .direction = Direction::Dgrad}},
    { 4, {.waves_c64 = 1, .waves_q4 = 1, .direction = Direction::Dgrad}},
    // Fprop (keys 5-9)
    { 5, {.waves_c64 = 2, .waves_q4 = 8}},
    { 6, {.waves_c64 = 2, .waves_q4 = 4}},
    { 7, {.waves_c64 = 2, .waves_q4 = 2}},
    { 8, {.waves_c64 = 2, .waves_q4 = 1}},
    { 9, {.waves_c64 = 1, .waves_q4 = 1}},
    // ---- Group 1: No swizzle, LDS-staged epilogue ----
    // Dgrad (keys 10-14)
    {10, {.waves_c64 = 2, .waves_q4 = 8, .direction = Direction::Dgrad,
          .epilogue = EpilogueType::RegistersToLdsToGlobalMemory}},
    {11, {.waves_c64 = 2, .waves_q4 = 4, .direction = Direction::Dgrad,
          .epilogue = EpilogueType::RegistersToLdsToGlobalMemory}},
    {12, {.waves_c64 = 2, .waves_q4 = 2, .direction = Direction::Dgrad,
          .epilogue = EpilogueType::RegistersToLdsToGlobalMemory}},
    {13, {.waves_c64 = 2, .waves_q4 = 1, .direction = Direction::Dgrad,
          .epilogue = EpilogueType::RegistersToLdsToGlobalMemory}},
    {14, {.waves_c64 = 1, .waves_q4 = 1, .direction = Direction::Dgrad,
          .epilogue = EpilogueType::RegistersToLdsToGlobalMemory}},
    // Fprop (keys 15-19)
    {15, {.waves_c64 = 2, .waves_q4 = 8,
          .epilogue = EpilogueType::RegistersToLdsToGlobalMemory}},
    {16, {.waves_c64 = 2, .waves_q4 = 4,
          .epilogue = EpilogueType::RegistersToLdsToGlobalMemory}},
    {17, {.waves_c64 = 2, .waves_q4 = 2,
          .epilogue = EpilogueType::RegistersToLdsToGlobalMemory}},
    {18, {.waves_c64 = 2, .waves_q4 = 1,
          .epilogue = EpilogueType::RegistersToLdsToGlobalMemory}},
    {19, {.waves_c64 = 1, .waves_q4 = 1,
          .epilogue = EpilogueType::RegistersToLdsToGlobalMemory}},
    // ---- Group 2: XOR swizzle, direct DRAM epilogue ----
    // Dgrad (keys 20-24)
    {20, {.waves_c64 = 2, .waves_q4 = 8, .direction = Direction::Dgrad,
          .swizzle_type = SwizzleType::XOR}},
    {21, {.waves_c64 = 2, .waves_q4 = 4, .direction = Direction::Dgrad,
          .swizzle_type = SwizzleType::XOR}},
    {22, {.waves_c64 = 2, .waves_q4 = 2, .direction = Direction::Dgrad,
          .swizzle_type = SwizzleType::XOR}},
    {23, {.waves_c64 = 2, .waves_q4 = 1, .direction = Direction::Dgrad,
          .swizzle_type = SwizzleType::XOR}},
    {24, {.waves_c64 = 1, .waves_q4 = 1, .direction = Direction::Dgrad,
          .swizzle_type = SwizzleType::XOR}},
    // Fprop (keys 25-29)
    {25, {.waves_c64 = 2, .waves_q4 = 8, .swizzle_type = SwizzleType::XOR}},
    {26, {.waves_c64 = 2, .waves_q4 = 4, .swizzle_type = SwizzleType::XOR}},
    {27, {.waves_c64 = 2, .waves_q4 = 2, .swizzle_type = SwizzleType::XOR}},
    {28, {.waves_c64 = 2, .waves_q4 = 1, .swizzle_type = SwizzleType::XOR}},
    {29, {.waves_c64 = 1, .waves_q4 = 1, .swizzle_type = SwizzleType::XOR}},
    // ---- Group 3: XOR swizzle, LDS-staged epilogue ----
    // Dgrad (keys 30-34)
    {30, {.waves_c64 = 2, .waves_q4 = 8, .direction = Direction::Dgrad,
          .swizzle_type = SwizzleType::XOR,
          .epilogue = EpilogueType::RegistersToLdsToGlobalMemory}},
    {31, {.waves_c64 = 2, .waves_q4 = 4, .direction = Direction::Dgrad,
          .swizzle_type = SwizzleType::XOR,
          .epilogue = EpilogueType::RegistersToLdsToGlobalMemory}},
    {32, {.waves_c64 = 2, .waves_q4 = 2, .direction = Direction::Dgrad,
          .swizzle_type = SwizzleType::XOR,
          .epilogue = EpilogueType::RegistersToLdsToGlobalMemory}},
    {33, {.waves_c64 = 2, .waves_q4 = 1, .direction = Direction::Dgrad,
          .swizzle_type = SwizzleType::XOR,
          .epilogue = EpilogueType::RegistersToLdsToGlobalMemory}},
    {34, {.waves_c64 = 1, .waves_q4 = 1, .direction = Direction::Dgrad,
          .swizzle_type = SwizzleType::XOR,
          .epilogue = EpilogueType::RegistersToLdsToGlobalMemory}},
    // Fprop (keys 35-39)
    {35, {.waves_c64 = 2, .waves_q4 = 8,
          .swizzle_type = SwizzleType::XOR,
          .epilogue = EpilogueType::RegistersToLdsToGlobalMemory}},
    {36, {.waves_c64 = 2, .waves_q4 = 4,
          .swizzle_type = SwizzleType::XOR,
          .epilogue = EpilogueType::RegistersToLdsToGlobalMemory}},
    {37, {.waves_c64 = 2, .waves_q4 = 2,
          .swizzle_type = SwizzleType::XOR,
          .epilogue = EpilogueType::RegistersToLdsToGlobalMemory}},
    {38, {.waves_c64 = 2, .waves_q4 = 1,
          .swizzle_type = SwizzleType::XOR,
          .epilogue = EpilogueType::RegistersToLdsToGlobalMemory}},
    {39, {.waves_c64 = 1, .waves_q4 = 1,
          .swizzle_type = SwizzleType::XOR,
          .epilogue = EpilogueType::RegistersToLdsToGlobalMemory}},
    // Cyclic-shift instances (keys 40-43)
    {40, {.waves_c64 = 2, .waves_q4 = 8,
          .swizzle_type = SwizzleType::CyclicShift,
          .epilogue = EpilogueType::RegistersToLdsToGlobalMemory}},
    {41, {.waves_c64 = 2, .waves_q4 = 8,
          .swizzle_type = SwizzleType::CyclicShift,
          .epilogue = EpilogueType::RegistersToGlobalMemory}},
    {42, {.waves_c64 = 2, .waves_q4 = 8, .direction = Direction::Dgrad,
          .swizzle_type = SwizzleType::CyclicShift,
          .epilogue = EpilogueType::RegistersToLdsToGlobalMemory}},
    {43, {.waves_c64 = 2, .waves_q4 = 8, .direction = Direction::Dgrad,
          .swizzle_type = SwizzleType::CyclicShift,
          .epilogue = EpilogueType::RegistersToGlobalMemory}},
    // ---- Group 4: CyclicShift, small vector sizes for padding ----
    // Dgrad (keys 44-47), Fprop (keys 48-51)
    {44, {.waves_c64 = 2, .waves_q4 = 8, .direction = Direction::Dgrad,
          .swizzle_type = SwizzleType::CyclicShift, .vector_size = 2}},
    {45, {.waves_c64 = 2, .waves_q4 = 8, .direction = Direction::Dgrad,
          .swizzle_type = SwizzleType::CyclicShift, .vector_size = 1}},
    {46, {.waves_c64 = 2, .waves_q4 = 8, .direction = Direction::Fprop,
          .swizzle_type = SwizzleType::CyclicShift, .vector_size = 2}},
    {47, {.waves_c64 = 2, .waves_q4 = 8, .direction = Direction::Fprop,
          .swizzle_type = SwizzleType::CyclicShift, .vector_size = 1}},
    {48, {.waves_c64 = 2, .waves_q4 = 4, .direction = Direction::Dgrad,
          .swizzle_type = SwizzleType::CyclicShift, .vector_size = 2}},
    {49, {.waves_c64 = 2, .waves_q4 = 4, .direction = Direction::Dgrad,
          .swizzle_type = SwizzleType::CyclicShift, .vector_size = 1}},
    {50, {.waves_c64 = 2, .waves_q4 = 4, .direction = Direction::Fprop,
          .swizzle_type = SwizzleType::CyclicShift, .vector_size = 2}},
    {51, {.waves_c64 = 2, .waves_q4 = 4, .direction = Direction::Fprop,
          .swizzle_type = SwizzleType::CyclicShift, .vector_size = 1}},
    // TODO: These configurations produce wrong results.
    //  {52, {.waves_c64=2,.waves_q4=2,.direction=Direction::Dgrad,.swizzle_type=SwizzleType::CyclicShift,.vector_size=2}},
    //  {53, {.waves_c64=2,.waves_q4=2,.direction=Direction::Dgrad,.swizzle_type=SwizzleType::CyclicShift,.vector_size=1}},
    //  {54, {.waves_c64=2,.waves_q4=2,.direction=Direction::Fprop,.swizzle_type=SwizzleType::CyclicShift,.vector_size=2}},
    //  {55, {.waves_c64=2,.waves_q4=2,.direction=Direction::Fprop,.swizzle_type=SwizzleType::CyclicShift,.vector_size=1}},
});
static_assert(configs_map.is_valid(), "Duplicate or negative config key in grouped_4c_tile configs_map");
static constexpr int NUM_CONFIGS = configs_map.size;
}; // KernelConfigurations


} // namespace ck_tile::direct_conv::grouped_4c_tile::v3
