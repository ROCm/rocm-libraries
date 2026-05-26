// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/amdgcn_mma.hpp"
#include "ck_tile/core/arch/mma/mma_selector.hpp"
#include "ck_tile/core/arch/mma/mma_traits.hpp"
#include "ck_tile/core/arch/mma/scale/scale_traits.hpp"
#include "ck_tile/core/arch/mma/scale/wmma/scale_gfx12.hpp"

#include <cstdint>
#include <type_traits>

namespace ck_tile::core::arch::mma {

/**
 * @class ScaleWmmaDefaultSelector
 * @brief Implements a default scale WMMA selector strategy. The SelectedOp can be unsupported.
 * @tparam ADataType      Data type of matrix A
 * @tparam BDataType      Data type of matrix B
 * @tparam CDataType      Data type of the accumulator
 * @tparam WaveTileM      Size of the M dimension
 * @tparam WaveTileN      Size of the N dimension
 * @tparam WaveTileKTest  Size of the K dimension
 * @tparam CompilerTarget The compiler target
 */
template <typename ADataType,
          typename BDataType,
          typename CDataType,
          std::uint32_t WaveTileM,
          std::uint32_t WaveTileN,
          std::uint32_t WaveTileKTest,
          typename CompilerTarget>
// TODO: c++20 amdgcn_target_arch_id CompilerTarget>
// TODO: c++20 requires(is_target_arch_rdna(CompilerTarget) &&
// is_power_of_two_integer(WaveTileKTest))
struct ScaleWmmaDefaultSelector
{
    private:
    // Define our candidate WMMA implementation for the current parameters
    using CandidateOp = amdgcn_mma<ADataType,
                                   BDataType,
                                   CDataType,
                                   WaveTileM,
                                   WaveTileN,
                                   WaveTileKTest,
                                   DefaultScaleWmmaCtrlFlags,
                                   CompilerTarget,
                                   MmaOpFamily::SCALE>;

    public:
    // If the candidate is supported (e.g., a backend implementation exists), then select it.
    // Otherwise, fall back to the unsupported pass-through implementation.
    using SelectedOp = std::conditional_t<MmaOpTraits<CandidateOp>::IsSupported,
                                          CandidateOp,
                                          amdgcn_mma<ADataType,
                                                     BDataType,
                                                     CDataType,
                                                     WaveTileM,
                                                     WaveTileN,
                                                     WaveTileKTest,
                                                     void,
                                                     amdgcn_target<>,
                                                     MmaOpFamily::UNDEFINED>>;
};

/**
 * @struct MmaDefaultSelector
 * @brief Implements the RDNA default MMA selector strategy for scale WMMA on GFX1250.
 * If no supported instruction is found, falls back to an unsupported pass-through implementation.
 * @tparam ADataType      Data type of matrix A
 * @tparam BDataType      Data type of matrix B
 * @tparam CDataType      Data type of the accumulator
 * @tparam WaveTileM      Size of the M dimension of the WaveTile to decompose
 * @tparam WaveTileN      Size of the N dimension of the WaveTile to decompose
 * @tparam WaveTileK      Size of the K dimension of the WaveTile to decompose
 * @tparam CompilerTarget The compiler target
 * @tparam OpFamily       The MMA operation family
 */
template <typename ADataType,
          typename BDataType,
          typename CDataType,
          std::uint32_t WaveTileM,
          std::uint32_t WaveTileN,
          std::uint32_t WaveTileK,
          typename CompilerTarget,
          MmaOpFamily OpFamily>
// TODO: c++20 amdgcn_target_arch_id CompilerTarget>
// TODO: c++20 requires
struct MmaDefaultSelector<ADataType,
                          BDataType,
                          CDataType,
                          WaveTileM,
                          WaveTileN,
                          WaveTileK,
                          CompilerTarget,
                          OpFamily,
                          enable_if_all<enable_if_target_family_gfx1250_t<CompilerTarget>,
                                        std::enable_if_t<OpFamily == MmaOpFamily::SCALE>>>
{
    private:
    // Provide the default depth-K search strategy for scale WMMA shapes.
    using CandidateOp16x16 = typename ScaleWmmaDefaultSelector<ADataType,
                                                               BDataType,
                                                               CDataType,
                                                               16u,
                                                               16u,
                                                               128u,
                                                               CompilerTarget>::SelectedOp;

    // Default operation triggers pass-through
    using DefaultOp = typename ScaleWmmaDefaultSelector<ADataType,
                                                        BDataType,
                                                        CDataType,
                                                        1u,
                                                        1u,
                                                        1u,
                                                        CompilerTarget>::SelectedOp;

    // Check if each candidate is supported for the given WaveTile sizes
    static constexpr bool IsSupported16x16 =
        MmaOpTraits<CandidateOp16x16>::IsSupported && (WaveTileM % CandidateOp16x16::kM == 0u) &&
        (WaveTileN % CandidateOp16x16::kN == 0u) && (WaveTileK % CandidateOp16x16::kK == 0u);

    public:
    // Select the largest supported WMMA operation for the given WaveTile shape
    using SelectedOp = std::conditional_t<IsSupported16x16, CandidateOp16x16, DefaultOp>;
};

} // namespace ck_tile::core::arch::mma
