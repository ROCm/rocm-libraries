// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdio>

#include "ck_tile/host/device_prop.hpp"

template <typename WmmaConfig, typename GenericConfig, typename TypeConfig, typename QuantGroupSize>
int run_gemm_bquant_quantgrouped_preshuffleb(const ck_tile::ArgParser& arg_parser)
{
    using namespace ck_tile::core::arch;
    constexpr bool is_host_compile =
        ck_tile::core::amdgcn_compiler_target_state::CK_TILE_HOST_COMPILE;
    constexpr bool is_mixed_gfx950_build =
        getCMakeTargetsContain<amdgcn_target_id::GFX950>() &&
        getCMakeTargetsContainOtherThan<amdgcn_target_id::GFX950>();
    constexpr bool build_needs_generic =
        !CK_TILE_USE_WMMA || getCMakeTargetsContain<amdgcn_target_id::GFX950>();
    constexpr bool compile_generic =
        build_needs_generic && (is_host_compile || !CK_TILE_USE_WMMA || !is_mixed_gfx950_build ||
                                get_compiler_target().TARGET_ID == amdgcn_target_id::GFX950);

#if CK_TILE_USE_WMMA
    if constexpr(getCMakeTargetsContainOtherThan<amdgcn_target_id::GFX950>() &&
                 (is_host_compile || get_compiler_target().TARGET_ID != amdgcn_target_id::GFX950))
    {
        if(!ck_tile::is_gfx95_supported())
        {
            return run_gemm_example_prec_type<WmmaConfig,
                                              TypeConfig,
                                              QuantGroupSize,
                                              ck_tile::QuantType::BQuantGrouped>(arg_parser);
        }
    }
#endif

    if constexpr(compile_generic)
    {
#if CK_TILE_USE_WMMA
        if(ck_tile::is_gfx95_supported())
#endif
        {
            return run_gemm_example_prec_type<GenericConfig,
                                              TypeConfig,
                                              QuantGroupSize,
                                              ck_tile::QuantType::BQuantGrouped>(arg_parser);
        }
    }

    std::fprintf(stderr,
                 "No preshuffle-B B-quant grouped GEMM kernel was compiled for the current "
                 "device.\n");
    return -1;
}
