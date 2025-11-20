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
              typename GemmConfig,
              uint32_t BlocksX,
              uint32_t BlocksY,
              uint32_t TBlockX,
              uint32_t TBlockY,
              uint32_t WaveSize,
              typename Enabler = void>
    struct CooperativePredicates
    {
        static constexpr bool IsBlockLevel
            = is_same_v<
                  GemmConfig,
                  typename CooperativeGemm::BlockLevel::
                      LdsRF> || is_same_v<GemmConfig, typename CooperativeGemm::BlockLevel::LdsNT> || is_same_v<GemmConfig, typename CooperativeGemm::BlockLevel::LdsTN>;

        static constexpr bool IsWaveLevel
            = is_same_v<
                  GemmConfig,
                  typename CooperativeGemm::WaveLevel::
                      LdsNT> || is_same_v<GemmConfig, typename CooperativeGemm::WaveLevel::LdsTN>;

        static constexpr bool IsWgLevel
            = is_same_v<
                  GemmConfig,
                  typename CooperativeGemm::WorkgroupLevel::
                      LdsNT> || is_same_v<GemmConfig, typename CooperativeGemm::WorkgroupLevel::LdsTN>;

        static constexpr uint32_t WavesX     = TBlockX / WaveSize;
        static constexpr uint32_t WavesY     = TBlockY;
        static constexpr uint32_t TotalWaves = WavesX * WavesY;

        // Minimum vector elements to get full registers
        static constexpr uint32_t MinElementsPerReg
            = max(1u, (uint32_t)(sizeof(float32_t) / sizeof(InputT)));

        // In block-wise cooperation:
        // - GR/LW FragA/B are mfma tile size
        // - GR/LW FragA will be cooperative for waves in same ROW
        // - GR/LW FragB will be cooperative for waves in same COL
        // Ensure filled registers are divisible by the participating waves
        static constexpr uint32_t CoopElementsA_blk = BlockM * BlockK;
        static constexpr uint32_t CoopElementsB_blk = BlockN * BlockK;
        static constexpr uint32_t CoopRegsA_blk = CoopElementsA_blk / WaveSize / MinElementsPerReg;
        static constexpr uint32_t CoopRegsB_blk = CoopElementsB_blk / WaveSize / MinElementsPerReg;
        static constexpr bool     CoopCheckA_blk
            = (CoopRegsA_blk >= WavesY) && (CoopRegsA_blk % WavesY == 0u);
        static constexpr bool CoopCheckB_blk
            = (CoopRegsB_blk >= WavesX) && (CoopRegsB_blk % WavesX == 0u);
        static constexpr bool CoopCheck_blk = CoopCheckA_blk && CoopCheckB_blk;

        // In wave-wise cooperation:
        // - GR/LW FragA/B are wave tile size
        // - GR/LW FragA will be cooperative for waves in same ROW
        // - GR/LW FragB will be cooperative for waves in same COL
        // Ensure filled registers are divisible by the participating waves
        static constexpr uint32_t CoopElementsA_wv = CoopElementsA_blk * BlocksX;
        static constexpr uint32_t CoopElementsB_wv = CoopElementsB_blk * BlocksY;
        static constexpr uint32_t CoopRegsA_wv = CoopElementsA_wv / WaveSize / MinElementsPerReg;
        static constexpr uint32_t CoopRegsB_wv = CoopElementsB_wv / WaveSize / MinElementsPerReg;
        static constexpr bool     CoopCheckA_wv
            = (CoopRegsA_wv >= WavesY) && (CoopRegsA_wv % WavesY == 0u);
        static constexpr bool CoopCheckB_wv
            = (CoopRegsB_wv >= WavesX) && (CoopRegsB_wv % WavesX == 0u);
        static constexpr bool CoopCheck_wv = CoopCheckA_wv && CoopCheckB_wv;

        // In wg-wise cooperation:
        // - GR/LW FragA/B are macro tile size
        // - GR/LW FragA will be cooperative for all waves
        // - GR/LW FragB will be cooperative for all waves
        // Ensure filled registers are divisible by the participating waves
        static constexpr uint32_t CoopElementsA_wg = CoopElementsA_wv * WavesX;
        static constexpr uint32_t CoopElementsB_wg = CoopElementsB_wv * WavesY;
        static constexpr uint32_t CoopRegsA_wg = CoopElementsA_wg / WaveSize / MinElementsPerReg;
        static constexpr uint32_t CoopRegsB_wg = CoopElementsB_wg / WaveSize / MinElementsPerReg;
        static constexpr bool     CoopCheckA_wg
            = (CoopRegsA_wg >= TotalWaves) && (CoopRegsA_wg % TotalWaves == 0u);
        static constexpr bool CoopCheckB_wg
            = (CoopRegsB_wg >= TotalWaves) && (CoopRegsB_wg % TotalWaves == 0u);
        static constexpr bool CoopCheck_wg = CoopCheckA_wg && CoopCheckB_wg;

        static constexpr bool CoopCheck = (IsBlockLevel && CoopCheck_blk)
                                          || (IsWaveLevel && CoopCheck_wv)
                                          || (IsWgLevel && CoopCheck_wg);
    };

    template <uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename InputT,
              typename GemmConfig,
              uint32_t BlocksX,
              uint32_t BlocksY,
              uint32_t TBlockX,
              uint32_t TBlockY,
              uint32_t WaveSize>
    struct CooperativePredicates<BlockM,
                                 BlockN,
                                 BlockK,
                                 InputT,
                                 GemmConfig,
                                 BlocksX,
                                 BlocksY,
                                 TBlockX,
                                 TBlockY,
                                 WaveSize,
                                 enable_if_t<TBlockX % WaveSize != 0u>>
    {
        // TBlockX must be a multiple of the WaveSize
        // E.g., Don't attempt TBlockX = 32 with WaveSize = 64
        static constexpr bool CoopCheck = false;
    };

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
        using CooperativePredicates = CooperativePredicates<BlockM,
                                                            BlockN,
                                                            BlockK,
                                                            InputT,
                                                            GemmConfig,
                                                            BlocksX,
                                                            BlocksY,
                                                            TBlockX,
                                                            TBlockY,
                                                            WaveSize>;

        enum struct GlobalPredicates : bool
        {

            // Quirk for LdsRF is that it requires matching waves in X and Y directions
            // for correctness.
            // Second part is that the ldsRF crosses threshold from 16/32 block sizes to 64, which has different considerations
            // for the MaxVW. This unfortunately limits applicability in cooperative environment.
            LdsRFTest = !(std::is_same_v<GemmConfig, typename CooperativeGemm::BlockLevel::LdsRF>)
                        || ((BlockM * BlockK / WaveSize > 8u) && (BlockN * BlockK / WaveSize > 8u)),

            // The register file mapping is only valid when A and B have the same layout.
            LdsRFLayoutTest = !(std::is_same_v<GemmConfig, typename CooperativeGemm::BlockLevel::LdsRF>)
                              || (std::is_same_v<LayoutA, LayoutB>),

            // Need to filter out some coop tests
            Enable = (LdsRFTest && LdsRFLayoutTest) && CooperativePredicates::CoopCheck
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
