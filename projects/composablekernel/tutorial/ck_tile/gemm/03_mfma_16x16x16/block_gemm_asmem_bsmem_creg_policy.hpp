// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm.hpp"
#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/amdgcn_mma.hpp"
#include "ck_tile/core/arch/mma/mfma/mfma.hpp"

namespace ck_tile {

using namespace ck_tile::core::arch;
using namespace mma;

using F16       = fp16_t;
using F32       = fp32_t;
using Target908 = decltype(make_amdgcn_gfx9_target<amdgcn_target_id::GFX908>());

// TODO: The CompilerTarget template param has a default value of decltype(get_compiler_target()).
// The get_compiler_target() function only returns a real device arch during the device compiler
// pass. In more complex code, this is a problem because it is hard to make sure that the host pass
// never sees definitions involving WaveWiseMmaPipeline, otherwise the default value will be locked
// to HOST forever and the pipeline will not work at all!
using MyWarpGemmAsMmaPipeline = WaveWiseMmaPipeline<F16, // ADatatType
                                                    F16, // BDataType
                                                    F32, // CDataType
                                                    16,  // M
                                                    16,  // N
                                                    16,  // K
                                                    MmaOpFamily::DENSE,
                                                    MmaAccumPolicy::ROW_MAJOR,
                                                    true,       // CTranspose
                                                    Target908>; // Workaround!

using MyWarpGemmAsMmaPipelineKIter = WaveWiseMmaPipeline<F16, // ADatatType
                                                         F16, // BDataType
                                                         F32, // CDataType
                                                         16,  // M
                                                         16,  // N
                                                         32,  // K
                                                         MmaOpFamily::DENSE,
                                                         MmaAccumPolicy::ROW_MAJOR,
                                                         true,       // CTranspose
                                                         Target908>; // Workaround!

// Policy for BlockGemmASmemBSmemCReg with MFMA_16x16x16 instruction
struct BlockGemmASmemBSmemCRegPolicy
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemmMWarpNWarp()
    {
        // KERNEL_B uses 4x1 warp configuration
        constexpr index_t kMWarp = 4;
        constexpr index_t kNWarp = 1;

        // KERNEL_B uses mfma m16 n16 k16
        if constexpr(std::is_same_v<typename Problem::ADataType, half_t> &&
                     std::is_same_v<typename Problem::BDataType, half_t> &&
                     std::is_same_v<typename Problem::CDataType, float>)
        {
            // return make_tuple(
            //     WarpGemmMfmaF16F16F32M16N16K16TransposedCDistribution{}, kMWarp, kNWarp);

            // return make_tuple(
            //     WarpGemmMfmaF16F16F32M16N16K32TransposedCDistribution{}, kMWarp, kNWarp);

            // return make_tuple(MyWarpGemmAsMmaPipeline{}, kMWarp, kNWarp);

            return make_tuple(MyWarpGemmAsMmaPipelineKIter{}, kMWarp, kNWarp);
        }
        // else if constexpr(std::is_same_v<typename Problem::ADataType, bf16_t> &&
        //                   std::is_same_v<typename Problem::BDataType, bf16_t> &&
        //                   std::is_same_v<typename Problem::CDataType, float>)
        // {
        //     return make_tuple(
        //         WarpGemmMfmaBf16Bf16F32M16N16K16TransposedCDistribution{}, kMWarp, kNWarp);
        // }
        else
        {
            static_assert(false, "Unsupported data type configuration for GEMM warp execution.");
        }
    }
};

} // namespace ck_tile
