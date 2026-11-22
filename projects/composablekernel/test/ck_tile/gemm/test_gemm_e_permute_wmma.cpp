// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/host/device_prop.hpp"
#include "ck_tile/ops/gemm.hpp"

// The bf16 tile of GemmConfigComputeV3_WMMA on gfx1250.
struct EPermuteTestConfig
{
    static constexpr int M_Tile = 128, N_Tile = 128, K_Tile = 32;
    static constexpr int M_Warp = 4, N_Warp = 2;
    static constexpr int M_Warp_Tile = 16, N_Warp_Tile = 16, K_Warp_Tile = 32;
    static constexpr bool TransposeC = false, DoubleSmemBuffer = false, PadN = false;
    template <typename Problem>
    using Pipeline = ck_tile::GemmPipelineAgBgCrCompV3<Problem>;
    static bool DeviceSupported() { return ck_tile::is_gfx125_supported(); }
};

#include "test_gemm_e_permute_cases.inc"
