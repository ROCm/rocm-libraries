/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#ifndef ROCWMMA_GEMM_TEST_DEVICE_PREDICATES
#define ROCWMMA_GEMM_TEST_DEVICE_PREDICATES

#include "gemm_predicates_base.hpp"

namespace rocwmma
{
    template <uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename InputT,
              typename OutputT,
              typename ComputeT,
              typename LayoutA,
              typename LayoutB,
              typename LayoutC,
              typename LayoutD,
              typename LayoutLds,
              typename GemmConfig,
              uint32_t BlocksX,
              uint32_t BlocksY,
              uint32_t TBlockX,
              uint32_t TBlockY,
              uint32_t WaveSize,
              uint32_t ArchId>
    struct gemm_PGR1_LB2_MP0_MB_CP_guard : public GemmPredicatesBase<BlockM,
                                                                     BlockN,
                                                                     BlockK,
                                                                     InputT,
                                                                     OutputT,
                                                                     ComputeT,
                                                                     BlocksX,
                                                                     BlocksY,
                                                                     TBlockX,
                                                                     TBlockY,
                                                                     WaveSize,
                                                                     ArchId>
    {
        using Base = GemmPredicatesBase<BlockM,
                                        BlockN,
                                        BlockK,
                                        InputT,
                                        OutputT,
                                        ComputeT,
                                        BlocksX,
                                        BlocksY,
                                        TBlockX,
                                        TBlockY,
                                        WaveSize,
                                        ArchId>;

    private:
        enum struct GlobalPredicates : bool
        {
            // Quirk for LdsRF is that it requires matching waves in X and Y directions
            // for correctness.
            // Second part is that the ldsRF crosses threshold from 16/32 block sizes to 64, which has different considerations
            // for the MaxVW. This unfortunately limits applicability in cooperative environment.
            LdsRFTest = !(std::is_same_v<GemmConfig, typename CooperativeGemm::BlockLevel::LdsRF>)
                        || ((BlockM * BlockK / WaveSize > 8u) && (BlockN * BlockK / WaveSize > 8u)),

            Enable = (LdsRFTest)
        };

        using TestTraits = typename Base::TestTraits;

        enum struct Gfx9Predicates : bool
        {
            // Valid for gfx9 only
            ArchTest = (bool)TestTraits::Arch::IsGfx9,

            // Quirk for LdsRF is that it requires matching waves in X and Y directions
            // for correctness.
            // Second part is that the ldsRF layout supports only one wave due to MaxVW considerations.
            // This unfortunately limits applicability in cooperative environment.
            LdsRFTest = !(std::is_same_v<GemmConfig, typename CooperativeGemm::BlockLevel::LdsRF>)
                        || (((TBlockX / WaveSize) * TBlockY) == 1),

            CostABTest
            = ((2u * ((uint32_t)TestTraits::Cost::TileA + (uint32_t)TestTraits::Cost::TileB))
               <= 256u),
            CostAccTest  = ((uint32_t)TestTraits::Cost::TileC <= 256u),
            CostTailTest = (((uint32_t)TestTraits::Cost::TileA + (uint32_t)TestTraits::Cost::TileB
                             + 2u * (uint32_t)TestTraits::Cost::TileD)
                            <= 256u),

            Enable = (ArchTest && LdsRFTest && CostABTest && CostAccTest && CostTailTest)
        };

#if !NDEBUG
        static constexpr void debugGfx9Predicates()
        {
            std::cout << "Gfx9 Predicates:\n";
            std::cout << "ArchTest: " << (bool)Gfx9Predicates::ArchTest << std::endl;
            std::cout << "LdsRFTest: " << (bool)Gfx9Predicates::LdsRFTest << std::endl;
            std::cout << "CostABTest: " << (bool)Gfx9Predicates::CostABTest << std::endl;
            std::cout << "CostAccTest: " << (bool)Gfx9Predicates::CostAccTest << std::endl;
            std::cout << "CostTailTest: " << (bool)Gfx9Predicates::CostTailTest << std::endl;
            std::cout << "Enable: " << (bool)Gfx9Predicates::Enable << std::endl;
        }
#endif // !NDEBUG

        enum struct Gfx11Predicates : bool
        {
            // Valid for gfx11 only
            ArchTest = (bool)TestTraits::Arch::IsGfx11 || (bool)TestTraits::Arch::IsGfx12,

            // AB inputs are duplicated, double buffered
            // Acc tiles are unpacked.
            // Tail requires A, B, C & D tiles + FMA
            CostABTest
            = ((4u * ((uint32_t)TestTraits::Cost::TileA + (uint32_t)TestTraits::Cost::TileB))
               <= 256u),
            CostAccTest  = ((2u * (uint32_t)TestTraits::Cost::TileC) <= 256u),
            CostTailTest = (((uint32_t)TestTraits::Cost::TileA + (uint32_t)TestTraits::Cost::TileB
                             + 2u * (uint32_t)TestTraits::Cost::TileD)
                            <= 256u),

            Enable = (ArchTest && CostABTest && CostAccTest && CostTailTest)
        };

#if !NDEBUG
        static constexpr void debugGfx11Predicates()
        {
            std::cout << "Gfx11 Predicates:\n";
            std::cout << "ArchTest: " << (bool)Gfx11Predicates::ArchTest << std::endl;
            std::cout << "CostABTest: " << (bool)Gfx11Predicates::CostABTest << std::endl;
            std::cout << "CostAccTest: " << (bool)Gfx11Predicates::CostAccTest << std::endl;
            std::cout << "CostTailTest: " << (bool)Gfx11Predicates::CostTailTest << std::endl;
            std::cout << "Enable: " << (bool)Gfx11Predicates::Enable << std::endl;
        }
#endif // !NDEBUG

    public:
        constexpr static bool enableBuild()
        {
            return Base::enableBuild() && (bool)GlobalPredicates::Enable
                   && ((bool)Gfx9Predicates::Enable || (bool)Gfx11Predicates::Enable);
        }

        constexpr static bool enableRun()
        {
            return Base::enableRun() && (bool)GlobalPredicates::Enable
                   && ((bool)Gfx9Predicates::Enable || (bool)Gfx11Predicates::Enable);
        }

#if !NDEBUG
        constexpr static void debugPredicates()
        {
            std::cout << "Base predicates:\n";
            Base::debugPredicates();
            std::cout << "\nDerived Predicates:\n";
            debugGfx9Predicates();
            debugGfx11Predicates();

            std::cout << "Overall enable build: " << enableBuild() << std::endl;
            std::cout << "Overall enable run: " << enableRun() << std::endl;
        }
#endif // !NDEBUG
    };
} // namespace rocwmma

#endif // ROCWMMA_GEMM_TEST_DEVICE_PREDICATES
