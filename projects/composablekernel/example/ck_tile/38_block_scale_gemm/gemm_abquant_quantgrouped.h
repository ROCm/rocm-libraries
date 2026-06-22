// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

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
    if constexpr(getCMakeTargetsContain<amdgcn_target_id::GFX950>())
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
    if constexpr(getCMakeTargetsContainOtherThan<amdgcn_target_id::GFX950>())
    {
        return run_gemm_example_prec_type<GemmConfigABQuantPrefill<T, TransposeC>,
                                          TypeConfig,
                                          AQuantGroupSize,
                                          BQuantGroupSize,
                                          QT>(arg_parser);
    }
    __builtin_unreachable();
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
    if constexpr(getCMakeTargetsContain<amdgcn_target_id::GFX950>())
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
    if constexpr(getCMakeTargetsContainOtherThan<amdgcn_target_id::GFX950>())
    {
        return run_gemm_example_prec_type<GemmConfigPreshuffleB_ABQuant_Prefill<T, TransposeC>,
                                          TypeConfig,
                                          AQuantGroupSize,
                                          BQuantGroupSize,
                                          QT>(arg_parser);
    }
    __builtin_unreachable();
}
