// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/host/device_prop.hpp"
#include "ck_tile/ops/gemm.hpp"

// The tile of example/ck_tile/03_gemm/gemm_permute.hpp.
struct EPermuteTestConfig
{
    static constexpr int M_Tile = 128, N_Tile = 256, K_Tile = 32;
    static constexpr int M_Warp = 2, N_Warp = 2;
    static constexpr int M_Warp_Tile = 32, N_Warp_Tile = 32, K_Warp_Tile = 16;
    static constexpr bool TransposeC = true, DoubleSmemBuffer = false, PadN = false;
    template <typename Problem>
    using Pipeline = ck_tile::GemmPipelineAgBgCrCompV3<Problem>;
    static bool DeviceSupported() { return ck_tile::is_gfx95_supported(); }
};

#include "test_gemm_e_permute_cases.inc"
