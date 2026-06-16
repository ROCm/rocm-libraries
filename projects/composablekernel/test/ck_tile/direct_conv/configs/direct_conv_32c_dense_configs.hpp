// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Configuration set for the non-grouped (dense) 32-channel direct convolution
// kernel with cross-wave LDS reduction (v3). This header holds ONLY the
// instantiated configuration data (configs_map / KernelConfigurations). The
// MfmaShape enum and Config struct it parameterizes, and all kernel
// implementation logic, live in kernel/impl/conv_32c_tile_impl_v3.hpp, which
// defines Config and then includes this header.

#include "ck_tile/ops/direct_convolution/utils/common.hpp"
#include "ck_tile/ops/direct_convolution/utils/conv_params.hpp"
#include "ck_tile/ops/direct_convolution/utils/config_map.hpp"

namespace ck_tile::direct_conv::conv_32c_tile::v3 {

// ===================================================================
// KernelConfigurations — v3 configs.
// Direct DRAM epilogue only (no LDS epilogue needed since cross-wave
// reduction uses LDS for the reduction itself).
// ===================================================================
template <DataType DT = DataType::fp16>
struct KernelConfigurations
{
    static constexpr auto configs_map = make_config_map<Config<DT>>({
        // --- SwizzleType::None (keys 0-3) ---
        {0, {.waves_per_wg = 4, .direction = Direction::Dgrad}},
        {1, {.waves_per_wg = 2, .direction = Direction::Dgrad}},
        {2, {.waves_per_wg = 4}},
        {3, {.waves_per_wg = 2}},
        // --- SwizzleType::CyclicShift (keys 4-7) ---
        {4,
         {.waves_per_wg = 4,
          .direction    = Direction::Dgrad,
          .swizzle_type = SwizzleType::CyclicShift}},
        {5,
         {.waves_per_wg = 2,
          .direction    = Direction::Dgrad,
          .swizzle_type = SwizzleType::CyclicShift}},
        {6, {.waves_per_wg = 4, .swizzle_type = SwizzleType::CyclicShift}},
        {7, {.waves_per_wg = 2, .swizzle_type = SwizzleType::CyclicShift}},
        // --- SwizzleType::XOR (keys 8-11) ---
        {8, {.waves_per_wg = 4, .direction = Direction::Dgrad, .swizzle_type = SwizzleType::XOR}},
        {9, {.waves_per_wg = 2, .direction = Direction::Dgrad, .swizzle_type = SwizzleType::XOR}},
        {10, {.waves_per_wg = 4, .swizzle_type = SwizzleType::XOR}},
        {11, {.waves_per_wg = 2, .swizzle_type = SwizzleType::XOR}},
        // --- CyclicShift + LDS epilogue (keys 12-15) ---
        {12,
         {.waves_per_wg = 4,
          .direction    = Direction::Dgrad,
          .swizzle_type = SwizzleType::CyclicShift,
          .epilogue     = EpilogueType::RegistersToLdsToGlobalMemory}},
        {13,
         {.waves_per_wg = 2,
          .direction    = Direction::Dgrad,
          .swizzle_type = SwizzleType::CyclicShift,
          .epilogue     = EpilogueType::RegistersToLdsToGlobalMemory}},
        {14,
         {.waves_per_wg = 4,
          .swizzle_type = SwizzleType::CyclicShift,
          .epilogue     = EpilogueType::RegistersToLdsToGlobalMemory}},
        {15,
         {.waves_per_wg = 2,
          .swizzle_type = SwizzleType::CyclicShift,
          .epilogue     = EpilogueType::RegistersToLdsToGlobalMemory}},
        // --- XOR + LDS epilogue (keys 16-19) ---
        {16,
         {.waves_per_wg = 4,
          .direction    = Direction::Dgrad,
          .swizzle_type = SwizzleType::XOR,
          .epilogue     = EpilogueType::RegistersToLdsToGlobalMemory}},
        {17,
         {.waves_per_wg = 2,
          .direction    = Direction::Dgrad,
          .swizzle_type = SwizzleType::XOR,
          .epilogue     = EpilogueType::RegistersToLdsToGlobalMemory}},
        {18,
         {.waves_per_wg = 4,
          .swizzle_type = SwizzleType::XOR,
          .epilogue     = EpilogueType::RegistersToLdsToGlobalMemory}},
        {19,
         {.waves_per_wg = 2,
          .swizzle_type = SwizzleType::XOR,
          .epilogue     = EpilogueType::RegistersToLdsToGlobalMemory}},
        // --- CyclicShift 8-wave (keys 20-23) ---
        {20,
         {.waves_per_wg = 8,
          .direction    = Direction::Dgrad,
          .swizzle_type = SwizzleType::CyclicShift}},
        {21, {.waves_per_wg = 8, .swizzle_type = SwizzleType::CyclicShift}},
        {22,
         {.waves_per_wg = 8,
          .direction    = Direction::Dgrad,
          .swizzle_type = SwizzleType::CyclicShift,
          .epilogue     = EpilogueType::RegistersToLdsToGlobalMemory}},
        {23,
         {.waves_per_wg = 8,
          .swizzle_type = SwizzleType::CyclicShift,
          .epilogue     = EpilogueType::RegistersToLdsToGlobalMemory}},
        // --- CyclicShift waves_per_wg=3 (keys 24-27) ---
        {24,
         {.waves_per_wg = 3,
          .direction    = Direction::Dgrad,
          .swizzle_type = SwizzleType::CyclicShift}},
        {25, {.waves_per_wg = 3, .swizzle_type = SwizzleType::CyclicShift}},
        {26,
         {.waves_per_wg = 3,
          .direction    = Direction::Dgrad,
          .swizzle_type = SwizzleType::CyclicShift,
          .epilogue     = EpilogueType::RegistersToLdsToGlobalMemory}},
        {27,
         {.waves_per_wg = 3,
          .swizzle_type = SwizzleType::CyclicShift,
          .epilogue     = EpilogueType::RegistersToLdsToGlobalMemory}},
        // --- CyclicShift waves_per_wg=5 (keys 28-31) ---
        {28,
         {.waves_per_wg = 5,
          .direction    = Direction::Dgrad,
          .swizzle_type = SwizzleType::CyclicShift}},
        {29, {.waves_per_wg = 5, .swizzle_type = SwizzleType::CyclicShift}},
        {30,
         {.waves_per_wg = 5,
          .direction    = Direction::Dgrad,
          .swizzle_type = SwizzleType::CyclicShift,
          .epilogue     = EpilogueType::RegistersToLdsToGlobalMemory}},
        {31,
         {.waves_per_wg = 5,
          .swizzle_type = SwizzleType::CyclicShift,
          .epilogue     = EpilogueType::RegistersToLdsToGlobalMemory}},
        // --- CyclicShift waves_per_wg=6 (keys 32-35) ---
        {32,
         {.waves_per_wg = 6,
          .direction    = Direction::Dgrad,
          .swizzle_type = SwizzleType::CyclicShift}},
        {33, {.waves_per_wg = 6, .swizzle_type = SwizzleType::CyclicShift}},
        {34,
         {.waves_per_wg = 6,
          .direction    = Direction::Dgrad,
          .swizzle_type = SwizzleType::CyclicShift,
          .epilogue     = EpilogueType::RegistersToLdsToGlobalMemory}},
        {35,
         {.waves_per_wg = 6,
          .swizzle_type = SwizzleType::CyclicShift,
          .epilogue     = EpilogueType::RegistersToLdsToGlobalMemory}},
        // --- CyclicShift waves_per_wg=7 (keys 36-39) ---
        {36,
         {.waves_per_wg = 7,
          .direction    = Direction::Dgrad,
          .swizzle_type = SwizzleType::CyclicShift}},
        {37, {.waves_per_wg = 7, .swizzle_type = SwizzleType::CyclicShift}},
        {38,
         {.waves_per_wg = 7,
          .direction    = Direction::Dgrad,
          .swizzle_type = SwizzleType::CyclicShift,
          .epilogue     = EpilogueType::RegistersToLdsToGlobalMemory}},
        {39,
         {.waves_per_wg = 7,
          .swizzle_type = SwizzleType::CyclicShift,
          .epilogue     = EpilogueType::RegistersToLdsToGlobalMemory}},
        // --- XOR waves_per_wg=8 (keys 40-43) ---
        {40, {.waves_per_wg = 8, .direction = Direction::Dgrad, .swizzle_type = SwizzleType::XOR}},
        {41, {.waves_per_wg = 8, .swizzle_type = SwizzleType::XOR}},
        {42,
         {.waves_per_wg = 8,
          .direction    = Direction::Dgrad,
          .swizzle_type = SwizzleType::XOR,
          .epilogue     = EpilogueType::RegistersToLdsToGlobalMemory}},
        {43,
         {.waves_per_wg = 8,
          .swizzle_type = SwizzleType::XOR,
          .epilogue     = EpilogueType::RegistersToLdsToGlobalMemory}},
        // --- c_slices_per_wave > 1, waves_per_wg=2 (keys 44-51) ---
        {44, {.waves_per_wg = 2, .c_slices_per_wave = 2}},
        {45, {.waves_per_wg = 2, .c_slices_per_wave = 2, .direction = Direction::Dgrad}},
        {46, {.waves_per_wg = 2, .c_slices_per_wave = 4}},
        {47, {.waves_per_wg = 2, .c_slices_per_wave = 4, .direction = Direction::Dgrad}},
        {48, {.waves_per_wg = 2, .c_slices_per_wave = 2, .swizzle_type = SwizzleType::CyclicShift}},
        {49,
         {.waves_per_wg      = 2,
          .c_slices_per_wave = 2,
          .direction         = Direction::Dgrad,
          .swizzle_type      = SwizzleType::CyclicShift}},
        {50, {.waves_per_wg = 2, .c_slices_per_wave = 2, .swizzle_type = SwizzleType::XOR}},
        {51,
         {.waves_per_wg      = 2,
          .c_slices_per_wave = 2,
          .direction         = Direction::Dgrad,
          .swizzle_type      = SwizzleType::XOR}},
        // --- c_slices_per_wave > 1, waves_per_wg=4 (keys 52-59) ---
        {52, {.waves_per_wg = 4, .c_slices_per_wave = 2}},
        {53, {.waves_per_wg = 4, .c_slices_per_wave = 2, .direction = Direction::Dgrad}},
        {54, {.waves_per_wg = 4, .c_slices_per_wave = 4}},
        {55, {.waves_per_wg = 4, .c_slices_per_wave = 4, .direction = Direction::Dgrad}},
        {56, {.waves_per_wg = 4, .c_slices_per_wave = 2, .swizzle_type = SwizzleType::CyclicShift}},
        {57,
         {.waves_per_wg      = 4,
          .c_slices_per_wave = 2,
          .direction         = Direction::Dgrad,
          .swizzle_type      = SwizzleType::CyclicShift}},
        {58, {.waves_per_wg = 4, .c_slices_per_wave = 2, .swizzle_type = SwizzleType::XOR}},
        {59,
         {.waves_per_wg      = 4,
          .c_slices_per_wave = 2,
          .direction         = Direction::Dgrad,
          .swizzle_type      = SwizzleType::XOR}},
        // --- reduction <= 32 (waves_per_wg=1, keys 60-63) ---
        {60, {.waves_per_wg = 1, .direction = Direction::Dgrad}},
        {61, {.waves_per_wg = 1}},
        {62, {.waves_per_wg = 1, .direction = Direction::Dgrad, .swizzle_type = SwizzleType::CyclicShift}},
        {63, {.waves_per_wg = 1, .swizzle_type = SwizzleType::CyclicShift}},
        // --- covering-window gap fill (keys 64-67) ---
        // W=9: waves_per_wg=3, c_slices_per_wave=3 (CyclicShift; non-pow2 waves)
        {64,
         {.waves_per_wg      = 3,
          .c_slices_per_wave = 3,
          .direction         = Direction::Dgrad,
          .swizzle_type      = SwizzleType::CyclicShift}},
        {65, {.waves_per_wg = 3, .c_slices_per_wave = 3, .swizzle_type = SwizzleType::CyclicShift}},
        // W=10: waves_per_wg=5, c_slices_per_wave=2 (CyclicShift; non-pow2 waves)
        {66,
         {.waves_per_wg      = 5,
          .c_slices_per_wave = 2,
          .direction         = Direction::Dgrad,
          .swizzle_type      = SwizzleType::CyclicShift}},
        {67, {.waves_per_wg = 5, .c_slices_per_wave = 2, .swizzle_type = SwizzleType::CyclicShift}},
        // --- waves_per_wg=1, c_slices_per_wave > 1 (keys 68-83) ---
        // Single-wave configs that stream the C-reduction as N=cspw chunks of 32
        // channels through one wavefront, with NO cross-wave LDS reduction.
        // total_block_c = cspw * 32, covering reduction in (32*(cspw-1), 32*cspw]:
        //   cspw=2 -> (32,64]   cspw=3 -> (64,96]
        //   cspw=4 -> (96,128]  cspw=6 -> (160,192]
        // CyclicShift swizzle (matches the waves=1 / odd-wave covering-gap style).
        // Both DRAM and LDS-staged epilogues; Fprop and Dgrad.
        // cspw=2 (total_block_c=64):
        {68, {.waves_per_wg = 1, .c_slices_per_wave = 2, .swizzle_type = SwizzleType::CyclicShift}},
        {69,
         {.waves_per_wg      = 1,
          .c_slices_per_wave = 2,
          .direction         = Direction::Dgrad,
          .swizzle_type      = SwizzleType::CyclicShift}},
        {70,
         {.waves_per_wg      = 1,
          .c_slices_per_wave = 2,
          .swizzle_type      = SwizzleType::CyclicShift,
          .epilogue          = EpilogueType::RegistersToLdsToGlobalMemory}},
        {71,
         {.waves_per_wg      = 1,
          .c_slices_per_wave = 2,
          .direction         = Direction::Dgrad,
          .swizzle_type      = SwizzleType::CyclicShift,
          .epilogue          = EpilogueType::RegistersToLdsToGlobalMemory}},
        // cspw=3 (total_block_c=96):
        {72, {.waves_per_wg = 1, .c_slices_per_wave = 3, .swizzle_type = SwizzleType::CyclicShift}},
        {73,
         {.waves_per_wg      = 1,
          .c_slices_per_wave = 3,
          .direction         = Direction::Dgrad,
          .swizzle_type      = SwizzleType::CyclicShift}},
        {74,
         {.waves_per_wg      = 1,
          .c_slices_per_wave = 3,
          .swizzle_type      = SwizzleType::CyclicShift,
          .epilogue          = EpilogueType::RegistersToLdsToGlobalMemory}},
        {75,
         {.waves_per_wg      = 1,
          .c_slices_per_wave = 3,
          .direction         = Direction::Dgrad,
          .swizzle_type      = SwizzleType::CyclicShift,
          .epilogue          = EpilogueType::RegistersToLdsToGlobalMemory}},
        // cspw=4 (total_block_c=128):
        {76, {.waves_per_wg = 1, .c_slices_per_wave = 4, .swizzle_type = SwizzleType::CyclicShift}},
        {77,
         {.waves_per_wg      = 1,
          .c_slices_per_wave = 4,
          .direction         = Direction::Dgrad,
          .swizzle_type      = SwizzleType::CyclicShift}},
        {78,
         {.waves_per_wg      = 1,
          .c_slices_per_wave = 4,
          .swizzle_type      = SwizzleType::CyclicShift,
          .epilogue          = EpilogueType::RegistersToLdsToGlobalMemory}},
        {79,
         {.waves_per_wg      = 1,
          .c_slices_per_wave = 4,
          .direction         = Direction::Dgrad,
          .swizzle_type      = SwizzleType::CyclicShift,
          .epilogue          = EpilogueType::RegistersToLdsToGlobalMemory}},
        // cspw=6 (total_block_c=192):
        {80, {.waves_per_wg = 1, .c_slices_per_wave = 6, .swizzle_type = SwizzleType::CyclicShift}},
        {81,
         {.waves_per_wg      = 1,
          .c_slices_per_wave = 6,
          .direction         = Direction::Dgrad,
          .swizzle_type      = SwizzleType::CyclicShift}},
        {82,
         {.waves_per_wg      = 1,
          .c_slices_per_wave = 6,
          .swizzle_type      = SwizzleType::CyclicShift,
          .epilogue          = EpilogueType::RegistersToLdsToGlobalMemory}},
        {83,
         {.waves_per_wg      = 1,
          .c_slices_per_wave = 6,
          .direction         = Direction::Dgrad,
          .swizzle_type      = SwizzleType::CyclicShift,
          .epilogue          = EpilogueType::RegistersToLdsToGlobalMemory}},
    });
    static_assert(configs_map.is_valid(),
                  "Duplicate or negative config key in conv_32c_tile_v3 configs_map");
    static constexpr int NUM_CONFIGS = configs_map.size;
};

} // namespace ck_tile::direct_conv::conv_32c_tile::v3
