// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/mfma/mfma.hpp"
#include "ck_tile/core/arch/mma/wmma/wmma.hpp"
#include "ck_tile/core/arch/mma/utility/tile_distribution_encoding_register_mapper.hpp"
#include "ck_tile/core/arch/mma/utility/tile_distribution_encoding_calculator.hpp"

using namespace ck_tile;
using namespace ck_tile::core::arch;
using namespace mma;
using F16       = fp16_t;
using F32       = fp32_t;
using Target90a = decltype(make_amdgcn_gfx9_target<amdgcn_target_id::GFX90A>());
using Target11  = decltype(make_amdgcn_gfx11_target<amdgcn_target_id::GFX1100>());
using Target12  = decltype(make_amdgcn_gfx12_target<amdgcn_target_id::GFX1201>());

template <typename MmaOp>
void print_builtin_tile_distr_enc()
{
    TileDistrEncRegMap<typename TileDistrEncCalc<MmaOp>::AWarpDstrEncoding>::print();
    TileDistrEncRegMap<typename TileDistrEncCalc<MmaOp>::BWarpDstrEncoding>::print();
    TileDistrEncRegMap<typename TileDistrEncCalc<MmaOp>::CWarpDstrEncoding>::print();
}

// List of builtins to inspect.
// clang-format off
using mfma_f32_16x16x16f16            = amdgcn_mma<F16, F16, F32, 16u, 16u, 16u, DefaultMfmaCtrlFlags,                Target90a, MmaOpFamily::DENSE>;
using wmma_f32_16x16x16_f16_w32       = amdgcn_mma<F16, F16, F32, 16u, 16u, 16u, DefaultWmmaCtrlFlags<F16, F16, F32>, Target11,  MmaOpFamily::DENSE>;
using wmma_f32_16x16x16_f16_w32_gfx12 = amdgcn_mma<F16, F16, F32, 16u, 16u, 16u, DefaultWmmaCtrlFlags<F16, F16, F32>, Target12,  MmaOpFamily::DENSE>;
// clang-format on

int main()
{
    print_builtin_tile_distr_enc<mfma_f32_16x16x16f16>();
    print_builtin_tile_distr_enc<wmma_f32_16x16x16_f16_w32>();
    print_builtin_tile_distr_enc<wmma_f32_16x16x16_f16_w32_gfx12>();
    return 0;
}
