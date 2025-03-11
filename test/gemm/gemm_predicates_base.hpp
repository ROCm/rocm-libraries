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
#ifndef ROCWMMA_TEST_GEMM_PREDICATES_BASE_HPP
#define ROCWMMA_TEST_GEMM_PREDICATES_BASE_HPP

#include "gemm_test_traits.hpp"

namespace rocwmma
{

    template <uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename InputT,
              typename OutputT,
              typename ComputeT,
              uint32_t BlocksX,
              uint32_t BlocksY,
              uint32_t TBlockX,
              uint32_t TBlockY,
              uint32_t WaveSize,
              uint32_t ArchId>
    struct GemmPredicatesBase
    {
        using TestTraits = GemmTestTraits<BlockM,
                                          BlockN,
                                          BlockK,
                                          InputT,
                                          OutputT,
                                          ComputeT,
                                          BlocksX,
                                          BlocksY,
                                          WaveSize,
                                          ArchId>;

    protected:
        enum struct GlobalPredicates : bool
        {
            // ThreadblockX must be a multiple of the wave size
            TBlockXTest = (TBlockX % WaveSize == 0u),

            // Ensure that we have at least 1 wave
            MinTBlockTest = (TBlockX >= WaveSize && TBlockY >= 1),

            // Ensure that we only build for the current compiler target, which
            // will also exclude the host.
            CurrentArchTest = (ArchId == Constants::AMDGCN_CURRENT_ARCH_ID),

            // Ensure that we only build for the current wave size
            CurrentWaveSizeTest = (WaveSize == Constants::AMDGCN_WAVE_SIZE),

            // Only supported hardware allowed
            ArchTest = (bool)TestTraits::Arch::IsGfx9 || (bool)TestTraits::Arch::IsGfx11
                       || (bool)TestTraits::Arch::IsGfx12,

            // During the build phase, we have information about current target arch.
            // This means only the current arch and wave size are valid.
            EnableBuild
            = (TBlockXTest && MinTBlockTest && CurrentArchTest && CurrentWaveSizeTest && ArchTest),

            // During run phase on the host, we don't have compile time info about current arch or wave size.
            // We have to trust that the runtime params obtained through HipDevice will dispatch correctly for
            // the current arch and wave size.
            EnableRun = (TBlockXTest && MinTBlockTest && ArchTest),
        };

#if !NDEBUG
        static constexpr void debugGlobalPredicates()
        {
            std::cout << "Global Predicates:\n";
            std::cout << "TBlockXTest: " << (bool)GlobalPredicates::TBlockXTest << std::endl;
            std::cout << "MinTBlockTest: " << (bool)GlobalPredicates::MinTBlockTest << std::endl;
            std::cout << "CurrentArchTest: " << (bool)GlobalPredicates::CurrentArchTest
                      << std::endl;
            std::cout << "CurrentWaveSizeTest: " << (bool)GlobalPredicates::CurrentWaveSizeTest
                      << std::endl;
            std::cout << "ArchTest: " << (bool)GlobalPredicates::ArchTest << std::endl;
            std::cout << "EnableBuild: " << (bool)GlobalPredicates::EnableBuild << std::endl;
            std::cout << "EnableRun: " << (bool)GlobalPredicates::EnableRun << std::endl;
        }
#endif // !NDEBUG

        enum struct Gfx9Predicates : bool
        {
            ArchTest = (bool)TestTraits::Arch::IsGfx9,

            WaveSizeTest = (bool)TestTraits::Arch::IsWave64,

            TBlockTest
            = (TBlockX * TBlockY >= Constants::AMDGCN_WAVE_SIZE_64) && (TBlockX * TBlockY <= 256u),

            InputTypesTest
            = (bool)TestTraits::InputType::IsFloat8 || (bool)TestTraits::InputType::IsBFloat8
              || (bool)TestTraits::InputType::IsFloat8Fnuz
              || (bool)TestTraits::InputType::IsBFloat8Fnuz || (bool)TestTraits::InputType::IsInt8
              || (bool)TestTraits::InputType::IsFloat16 || (bool)TestTraits::InputType::IsBFloat16
              || (bool)TestTraits::InputType::IsFloat32 || (bool)TestTraits::InputType::IsXFloat32
              || (bool)TestTraits::InputType::IsFloat64,

            // Gfx940/1/2 arch req'd for float8_fnuz_t, bfloat8_fnuz_t and xfloat32_t
            F8FnuzXF32ArchTest = !((bool)TestTraits::InputType::IsFloat8Fnuz
                                   || (bool)TestTraits::InputType::IsBFloat8Fnuz
                                   || (bool)TestTraits::InputType::IsXFloat32)
                                 || (bool)TestTraits::Arch::IsGfx940
                                 || (bool)TestTraits::Arch::IsGfx941
                                 || (bool)TestTraits::Arch::IsGfx942,

            // Gfx950 arch req'd for float8_t
            F8ArchTest
            = !((bool)TestTraits::InputType::IsFloat8 || (bool)TestTraits::InputType::IsBFloat8)
              || (bool)TestTraits::Arch::IsGfx950,

            // All archs except gfx908 can run float64_t
            F64ArchTest
            = !(bool)TestTraits::InputType::IsFloat64 || !(bool)TestTraits::Arch::IsGfx908,

            // General int8_t block size
            // BlockM/N = 16; Block K >= 16
            // BlockM/N = 32; Block K >= 8
            I8BlockSizeTest = !((bool)TestTraits::InputType::IsInt8)
                              || ((bool)TestTraits::BlockSizes::isBlockMN16 && (BlockK >= 16u)
                                  && (BlockK % 16u == 0u))
                              || ((bool)TestTraits::BlockSizes::isBlockMN32 && (BlockK >= 16u)
                                  && (BlockK % 16u == 0u)),

            // Follow-on to gfx940/1/2/950 int8_t.
            // BlockM/N = 16; Block K >= 32
            // BlockM/N = 32; Block K >= 16
            ExtI8BlockSizeTest
            = !((bool)TestTraits::InputType::IsInt8
                && ((bool)TestTraits::Arch::IsGfx940 || (bool)TestTraits::Arch::IsGfx941
                    || (bool)TestTraits::Arch::IsGfx942 || (bool)TestTraits::Arch::IsGfx950))
              || ((bool)TestTraits::BlockSizes::isBlockMN16 && (BlockK >= 32u)
                  && (BlockK % 32u == 0u))
              || ((bool)TestTraits::BlockSizes::isBlockMN32 && (BlockK >= 16u)
                  && (BlockK % 16u == 0u)),

            // General float8_t / bfloat8_t block size
            // BlockM/N = 16; Block K >= 32
            // BlockM/N = 32; Block K >= 16
            F8BlockSizeTest
            = !((bool)TestTraits::InputType::IsFloat8 || (bool)TestTraits::InputType::IsBFloat8
                || (bool)TestTraits::InputType::IsFloat8Fnuz
                || (bool)TestTraits::InputType::IsBFloat8Fnuz)
              || ((bool)TestTraits::BlockSizes::isBlockMN16 && (BlockK >= 32u)
                  && (BlockK % 32u == 0u))
              || ((bool)TestTraits::BlockSizes::isBlockMN32 && (BlockK >= 16u)
                  && (BlockK % 16u == 0u)),

            // General float16_t / hfloat16_t / bfloat16_t block size
            // BlockM/N = 16; Block K >= 16
            // BlockM/N = 32; Block K >= 8
            F16BlockSizeTest
            = !((bool)TestTraits::InputType::IsFloat16 || (bool)TestTraits::InputType::IsBFloat16)
              || ((bool)TestTraits::BlockSizes::isBlockMN16 && (BlockK >= 16u)
                  && (BlockK % 16u == 0u))
              || ((bool)TestTraits::BlockSizes::isBlockMN32 && (BlockK >= 8u)
                  && (BlockK % 8u == 0u)),

            // Older gfx908 arch has half BlockK on bfloat16_t
            // BlockM/N = 16; Block K >= 8
            // BlockM/N = 32; Block K >= 4
            Gfx908BF16BlockSizeTest
            = !((bool)TestTraits::InputType::IsBFloat16 && (bool)TestTraits::Arch::IsGfx908)
              || ((bool)TestTraits::BlockSizes::isBlockMN16 && (BlockK >= 8u)
                  && (BlockK % 8u == 0u))
              || ((bool)TestTraits::BlockSizes::isBlockMN32 && (BlockK >= 4u)
                  && (BlockK % 4u == 0u)),

            // General float32_t block size
            // BlockM/N = 16; Block K >= 4
            // BlockM/N = 32; Block K >= 2
            F32BlockSizeTest = !((bool)TestTraits::InputType::IsFloat32)
                               || ((bool)TestTraits::BlockSizes::isBlockMN16 && (BlockK >= 4u)
                                   && (BlockK % 4u == 0u))
                               || ((bool)TestTraits::BlockSizes::isBlockMN32 && (BlockK >= 4u)
                                   && (BlockK % 4u == 0u)),

            // General xfloat32_t block size
            // BlockM/N = 16; Block K >= 8
            // BlockM/N = 32; Block K >= 4
            XF32BlockSizeTest = !((bool)TestTraits::InputType::IsXFloat32)
                                || ((bool)TestTraits::BlockSizes::isBlockMN16 && (BlockK >= 8u)
                                    && (BlockK % 8u == 0u))
                                || ((bool)TestTraits::BlockSizes::isBlockMN32 && (BlockK >= 4u)
                                    && (BlockK % 4u == 0u)),

            // General float64_t block size
            // BlockM/N = 16; Block K >= 4
            F64BlockSizeTest = !((bool)TestTraits::InputType::IsFloat64)
                               || ((bool)TestTraits::BlockSizes::isBlockMN16 && (BlockK >= 4u)
                                   && (BlockK % 4u == 0u)),

            Enable = (ArchTest && WaveSizeTest && TBlockTest && InputTypesTest && F8FnuzXF32ArchTest
                      && F8ArchTest && F64ArchTest && I8BlockSizeTest && ExtI8BlockSizeTest
                      && F8BlockSizeTest && F16BlockSizeTest && Gfx908BF16BlockSizeTest
                      && F32BlockSizeTest && XF32BlockSizeTest && F64BlockSizeTest)
        };

#if !NDEBUG
        static constexpr void debugGfx9Predicates()
        {
            std::cout << "Gfx9 Predicates:\n";
            std::cout << "ArchTest: " << (bool)Gfx9Predicates::ArchTest << std::endl;
            std::cout << "WaveSizeTest: " << (bool)Gfx9Predicates::WaveSizeTest << std::endl;
            std::cout << "TBlockTest: " << (bool)Gfx9Predicates::TBlockTest << std::endl;
            std::cout << "InputTypesTest: " << (bool)Gfx9Predicates::InputTypesTest << std::endl;
            std::cout << "F8XF32ArchTest: " << (bool)Gfx9Predicates::F8XF32ArchTest << std::endl;
            std::cout << "F64ArchTest: " << (bool)Gfx9Predicates::F64ArchTest << std::endl;
            std::cout << "I8BlockSizeTest: " << (bool)Gfx9Predicates::I8BlockSizeTest << std::endl;
            std::cout << "Gfx940I8BlockSizeTest: " << (bool)Gfx9Predicates::Gfx940I8BlockSizeTest
                      << std::endl;
            std::cout << "F8BlockSizeTest: " << (bool)Gfx9Predicates::F8BlockSizeTest << std::endl;
            std::cout << "F16BlockSizeTest: " << (bool)Gfx9Predicates::F16BlockSizeTest
                      << std::endl;
            std::cout << "Gfx908BF16BlockSizeTest: "
                      << (bool)Gfx9Predicates::Gfx908BF16BlockSizeTest << std::endl;
            std::cout << "F32BlockSizeTest: " << (bool)Gfx9Predicates::F32BlockSizeTest
                      << std::endl;
            std::cout << "XF32BlockSizeTest: " << (bool)Gfx9Predicates::XF32BlockSizeTest
                      << std::endl;
            std::cout << "F64BlockSizeTest: " << (bool)Gfx9Predicates::F64BlockSizeTest
                      << std::endl;
            std::cout << "Enable: " << (bool)Gfx9Predicates::Enable << std::endl;
        }
#endif // !NDEBUG

        enum struct Gfx11Predicates : bool
        {
            // Valid for gfx11 only
            ArchTest = (bool)TestTraits::Arch::IsGfx11,

            // Wave size on gfx11 is 32
            WaveSizeTest = (bool)TestTraits::Arch::IsWave32,

            // Max recommended TBlock size is 256
            TBlockTest
            = (TBlockX * TBlockY >= Constants::AMDGCN_WAVE_SIZE_32) && (TBlockX * TBlockY <= 256u),

            // Input types supported
            InputTypesTest = (bool)TestTraits::InputType::IsInt8
                             || (bool)TestTraits::InputType::IsFloat16
                             || (bool)TestTraits::InputType::IsBFloat16,

            // General int8_t block size
            // BlockM/N = 16; Block K >= 16
            I8BlockSizeTest = !((bool)TestTraits::InputType::IsInt8)
                              || ((bool)TestTraits::BlockSizes::isBlockMN16 && (BlockK >= 16u)
                                  && (BlockK % 16u == 0u)),

            // General float16_t / hfloat16_t / bfloat16_t block size
            // BlockM/N = 16; Block K >= 16
            F16BlockSizeTest
            = !((bool)TestTraits::InputType::IsFloat16 || (bool)TestTraits::InputType::IsBFloat16)
              || ((bool)TestTraits::BlockSizes::isBlockMN16 && (BlockK >= 16u)
                  && (BlockK % 16u == 0u)),

            Enable = (ArchTest && WaveSizeTest && TBlockTest && InputTypesTest && I8BlockSizeTest
                      && F16BlockSizeTest)
        };

#if !NDEBUG
        static constexpr void debugGfx11Predicates()
        {
            std::cout << "Gfx11 Predicates:\n";
            std::cout << "ArchTest: " << (bool)Gfx11Predicates::ArchTest << std::endl;
            std::cout << "WaveSizeTest: " << (bool)Gfx11Predicates::WaveSizeTest << std::endl;
            std::cout << "TBlockTest: " << (bool)Gfx11Predicates::TBlockTest << std::endl;
            std::cout << "InputTypesTest: " << (bool)Gfx11Predicates::InputTypesTest << std::endl;
            std::cout << "I8BlockSizeTest: " << (bool)Gfx11Predicates::I8BlockSizeTest << std::endl;
            std::cout << "F16BlockSizeTest: " << (bool)Gfx11Predicates::F16BlockSizeTest
                      << std::endl;
            std::cout << "Enable: " << (bool)Gfx11Predicates::Enable << std::endl;
        }
#endif // !NDEBUG

        enum struct Gfx12Predicates : bool
        {
            // Valid for gfx11 only
            ArchTest = (bool)TestTraits::Arch::IsGfx12,

            // Wave size on gfx11 is 32
            WaveSizeTest = (bool)TestTraits::Arch::IsWave32,

            // Max recommended TBlock size is 256
            TBlockTest
            = (TBlockX * TBlockY >= Constants::AMDGCN_WAVE_SIZE_32) && (TBlockX * TBlockY <= 256u),

            // Input types supported
            InputTypesTest
            = (bool)TestTraits::InputType::IsInt8 || (bool)TestTraits::InputType::IsFloat16
              || (bool)TestTraits::InputType::IsBFloat16 || (bool)TestTraits::InputType::IsFloat8
              || (bool)TestTraits::InputType::IsBFloat8,

            // General int8_t block size
            // BlockM/N = 16; Block K >= 16
            I8BlockSizeTest
            = !((bool)TestTraits::InputType::IsInt8 || (bool)TestTraits::InputType::IsFloat8
                || (bool)TestTraits::InputType::IsBFloat8)
              || ((bool)TestTraits::BlockSizes::isBlockMN16 && (BlockK >= 16u)
                  && (BlockK % 16u == 0u)),

            // General float16_t / hfloat16_t / bfloat16_t block size
            // BlockM/N = 16; Block K >= 16
            F16BlockSizeTest
            = !((bool)TestTraits::InputType::IsFloat16 || (bool)TestTraits::InputType::IsBFloat16)
              || ((bool)TestTraits::BlockSizes::isBlockMN16 && (BlockK >= 16u)
                  && (BlockK % 16u == 0u)),

            Enable = (ArchTest && WaveSizeTest && TBlockTest && InputTypesTest && I8BlockSizeTest
                      && F16BlockSizeTest)
        };

#if !NDEBUG
        static constexpr void debugGfx12Predicates()
        {
            std::cout << "Gfx12 Predicates:\n";
            std::cout << "ArchTest: " << (bool)Gfx12Predicates::ArchTest << std::endl;
            std::cout << "WaveSizeTest: " << (bool)Gfx12Predicates::WaveSizeTest << std::endl;
            std::cout << "TBlockTest: " << (bool)Gfx12Predicates::TBlockTest << std::endl;
            std::cout << "InputTypesTest: " << (bool)Gfx12Predicates::InputTypesTest << std::endl;
            std::cout << "I8BlockSizeTest: " << (bool)Gfx12Predicates::I8BlockSizeTest << std::endl;
            std::cout << "F16BlockSizeTest: " << (bool)Gfx12Predicates::F16BlockSizeTest
                      << std::endl;
            std::cout << "Enable: " << (bool)Gfx12Predicates::Enable << std::endl;
        }
#endif // !NDEBUG

    public:
        constexpr static bool enableBuild()
        {
            return ((bool)GlobalPredicates::EnableBuild
                    && ((bool)Gfx9Predicates::Enable || (bool)Gfx11Predicates::Enable
                        || (bool)Gfx12Predicates::Enable));
        }

        constexpr static bool enableRun()
        {
            return ((bool)GlobalPredicates::EnableRun
                    && ((bool)Gfx9Predicates::Enable || (bool)Gfx11Predicates::Enable
                        || (bool)Gfx12Predicates::Enable));
        }

#if !NDEBUG
        constexpr static void debugPredicates()
        {
            debugGlobalPredicates();
            debugGfx9Predicates();
            debugGfx11Predicates();
            debugGfx12Predicates();

            std::cout << "Overall enable build: " << enableBuild() << std::endl;
            std::cout << "Overall enable run: " << enableRun() << std::endl;
        }
#endif // !NDEBUG
    };

} // namespace rocwmma

#endif // ROCWMMA_TEST_GEMM_PREDICATES_BASE_HPP
