// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/host/device_prop.hpp"
#include "ck_tile/ops/gemm.hpp"

// The gfx1250 TDM pipelines of GemmEPermuteKernel; CMakeLists.txt builds one binary per config.
// The TDM loads clip the M tail and the E view masks it, so neither pads M.
struct EPermuteTestConfig
{
#ifdef E_PERMUTE_TDM_V3
    // comp_tdm_v3 with kPadN: the TDM pipeline whose CShuffle store also takes a ragged N.
    static constexpr int M_Tile = 160, N_Tile = 256, K_Tile = 64;
    static constexpr bool PadN = true;
    template <typename Problem>
    using Pipeline = ck_tile::GemmPipelineAgBgCrCompTDMV3<Problem>;
#else
    // The run-ahead ring with two slots, the fastest TDM tile for the gemm_permute example shape.
    static constexpr int M_Tile = 192, N_Tile = 256, K_Tile = 64;
    static constexpr bool PadN = false;
    template <typename Problem>
    using Pipeline = ck_tile::GemmPipelineAgBgCrCompTDMRunAhead<
        Problem,
        ck_tile::GemmPipelineAgBgCrCompTDMRunAheadPolicy<2, 1>>;
#endif
    static constexpr int M_Warp = 2, N_Warp = 8;
    static constexpr int M_Warp_Tile = 16, N_Warp_Tile = 16, K_Warp_Tile = 32;
    static constexpr bool TransposeC = true, DoubleSmemBuffer = true;
    static bool DeviceSupported() { return ck_tile::is_gfx125_supported(); }
};

#include "test_gemm_e_permute_cases.inc"
