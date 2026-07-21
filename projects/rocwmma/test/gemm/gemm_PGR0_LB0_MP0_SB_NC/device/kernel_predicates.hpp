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
              uint32_t TBlockX,
              uint32_t TBlockY,
              uint32_t WaveSize,
              uint32_t ArchId>
    struct gemm_PGR0_LB0_MP0_SB_NC_guard : public GemmPredicatesBase<BlockM,
                                                                     BlockN,
                                                                     BlockK,
                                                                     InputT,
                                                                     OutputT,
                                                                     ComputeT,
                                                                     1u,
                                                                     1u,
                                                                     TBlockX,
                                                                     TBlockY,
                                                                     WaveSize,
                                                                     ArchId>
    {
        using Base       = GemmPredicatesBase<BlockM,
                                        BlockN,
                                        BlockK,
                                        InputT,
                                        OutputT,
                                        ComputeT,
                                        1u,
                                        1u,
                                        TBlockX,
                                        TBlockY,
                                        WaveSize,
                                        ArchId>;
        using TestTraits = typename Base::TestTraits;

    private:
        enum struct Gfx9Predicates : bool
        {
            // Valid for gfx9 only
            ArchTest = (bool)TestTraits::Arch::IsGfx9,

            // Must skip int8 tests on gfx9 for now
            CostABTest
            = (((uint32_t)TestTraits::Cost::TileA + (uint32_t)TestTraits::Cost::TileB) <= 256u),
            CostCTest = ((uint32_t)TestTraits::Cost::TileC <= 256u),
            CostDTest = ((uint32_t)TestTraits::Cost::TileD <= 256u),

            Enable = (ArchTest && CostABTest && CostCTest && CostDTest)
        };

#if !NDEBUG
        static constexpr void debugGfx9Predicates()
        {
            std::cout << "Gfx9 Predicates:\n";
            std::cout << "ArchTest: " << (bool)Gfx9Predicates::ArchTest << std::endl;
            std::cout << "CostABTest: " << (bool)Gfx9Predicates::CostABTest << std::endl;
            std::cout << "CostCTest: " << (bool)Gfx9Predicates::CostCTest << std::endl;
            std::cout << "CostDTest: " << (bool)Gfx9Predicates::CostDTest << std::endl;
            std::cout << "Enable: " << (bool)Gfx9Predicates::Enable << std::endl;
        }
#endif // !NDEBUG

        enum struct Gfx11Predicates : bool
        {
            // Valid for gfx11 only
            ArchTest = (bool)TestTraits::Arch::IsGfx11 || (bool)TestTraits::Arch::IsGfx12,

            // AB inputs are duplicated, single buffered
            // C tiles are unpacked.
            CostABTest
            = ((2u * ((uint32_t)TestTraits::Cost::TileA + (uint32_t)TestTraits::Cost::TileB))
               <= 256u),
            CostCTest = ((2u * (uint32_t)TestTraits::Cost::TileC) <= 256u),
            CostDTest = ((uint32_t)TestTraits::Cost::TileD <= 256u),

            Enable = (ArchTest && CostABTest && CostCTest && CostDTest)
        };

#if !NDEBUG
        static constexpr void debugGfx11Predicates()
        {
            std::cout << "Gfx11 Predicates:\n";
            std::cout << "ArchTest: " << (bool)Gfx11Predicates::ArchTest << std::endl;
            std::cout << "CostABTest: " << (bool)Gfx11Predicates::CostABTest << std::endl;
            std::cout << "CostCTest: " << (bool)Gfx11Predicates::CostCTest << std::endl;
            std::cout << "CostDTest: " << (bool)Gfx11Predicates::CostDTest << std::endl;
            std::cout << "Enable: " << (bool)Gfx11Predicates::Enable << std::endl;
        }
#endif // !NDEBUG

    public:
        constexpr static bool enableBuild()
        {
            return Base::enableBuild()
                   && ((bool)Gfx9Predicates::Enable || (bool)Gfx11Predicates::Enable);
        }

        constexpr static bool enableRun()
        {
            return Base::enableRun()
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
