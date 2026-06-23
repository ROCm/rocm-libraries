// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdio>

#include "gemm_utils.hpp"
#include "ck_tile/host/device_prop.hpp"

// Arch-dispatched launcher for AB-quant grouped GEMM.
// - Instantiates the GFX950-optimized (8-wave) kernel only when GFX950 is a build target.
// - Instantiates the generic kernel only when non-GFX950 targets are present.
// - At runtime, selects the correct kernel based on the actual device.
//
// NOTE: This header must be included AFTER run_gemm_quant_example.inc (which defines
// run_gemm_example_prec_type).

template <typename T,
          bool TransposeC,
          typename TypeConfig,
          typename AQuantGroupSize,
          typename BQuantGroupSize,
          ck_tile::QuantType QT>
int run_gemm_abquant_quantgrouped(const ck_tile::ArgParser& arg_parser)
{
    using namespace ck_tile::core::arch;
    if constexpr(getCMakeTargetsContain<amdgcn_target_id::GFX950>() &&
                 (ck_tile::core::amdgcn_compiler_target_state::CK_TILE_HOST_COMPILE ||
                  get_compiler_target().TARGET_ID == amdgcn_target_id::GFX950))
    {
        if(ck_tile::is_gfx95_supported())
        {
            return run_gemm_example_prec_type<GemmConfigEightWaves<T, TransposeC>,
                                              TypeConfig,
                                              AQuantGroupSize,
                                              BQuantGroupSize,
                                              QT>(arg_parser);
        }
    }
    if constexpr(getCMakeTargetsContainOtherThan<amdgcn_target_id::GFX950>() &&
                 (ck_tile::core::amdgcn_compiler_target_state::CK_TILE_HOST_COMPILE ||
                  get_compiler_target().TARGET_ID != amdgcn_target_id::GFX950))
    {
        return run_gemm_example_prec_type<GemmConfigABQuantPrefill<T, TransposeC>,
                                          TypeConfig,
                                          AQuantGroupSize,
                                          BQuantGroupSize,
                                          QT>(arg_parser);
    }
    std::fprintf(stderr, "No AB-quant grouped GEMM kernel was compiled for the current device.\n");
    return -1;
}

template <typename T,
          bool TransposeC,
          typename TypeConfig,
          typename AQuantGroupSize,
          typename BQuantGroupSize,
          ck_tile::QuantType QT>
int run_gemm_abquant_quantgrouped_preshuffleb(const ck_tile::ArgParser& arg_parser)
{
    using namespace ck_tile::core::arch;
    if constexpr(getCMakeTargetsContain<amdgcn_target_id::GFX950>() &&
                 (ck_tile::core::amdgcn_compiler_target_state::CK_TILE_HOST_COMPILE ||
                  get_compiler_target().TARGET_ID == amdgcn_target_id::GFX950))
    {
        if(ck_tile::is_gfx95_supported())
        {
            return run_gemm_example_prec_type<GemmConfigPreshuffleBEightWaves<T, TransposeC>,
                                              TypeConfig,
                                              AQuantGroupSize,
                                              BQuantGroupSize,
                                              QT>(arg_parser);
        }
    }
    if constexpr(getCMakeTargetsContainOtherThan<amdgcn_target_id::GFX950>() &&
                 (ck_tile::core::amdgcn_compiler_target_state::CK_TILE_HOST_COMPILE ||
                  get_compiler_target().TARGET_ID != amdgcn_target_id::GFX950))
    {
        return run_gemm_example_prec_type<GemmConfigPreshuffleB_ABQuant_Prefill<T, TransposeC>,
                                          TypeConfig,
                                          AQuantGroupSize,
                                          BQuantGroupSize,
                                          QT>(arg_parser);
    }
    std::fprintf(stderr,
                 "No preshuffle-B AB-quant grouped GEMM kernel was compiled for the current "
                 "device.\n");
    return -1;
}
