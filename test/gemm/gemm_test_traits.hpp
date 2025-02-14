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

#ifndef ROCWMMA_GEMM_TEST_TRAITS_HPP
#define ROCWMMA_GEMM_TEST_TRAITS_HPP

// Silence warnings for calls on unsupported architectures.
// Unsupported architectures will generate no-ops and test
// will be avoided at runtime anyway.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <rocwmma/rocwmma.hpp>
#pragma GCC diagnostic pop

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
              uint32_t WaveSize,
              uint32_t ArchId>
    struct GemmTestTraits
    {
        // Size properties of gemm tiles
        enum struct TileSizes : uint32_t
        {
            A_X    = BlocksX * BlockM,
            A_Y    = BlockK,
            A_Size = A_X * A_Y,

            B_X    = BlockK,
            B_Y    = BlocksY * BlockN,
            B_Size = B_X * B_Y,

            C_X    = BlocksX * BlockM,
            C_Y    = BlocksY * BlockN,
            C_Size = C_X * C_Y,

            D_X    = BlocksX * BlockM,
            D_Y    = BlocksY * BlockN,
            D_Size = D_X * D_Y,
        };

        // Tile costs
        enum struct Cost : uint32_t
        {
            Granularity = 16u,
            DWord       = 4u,

            TileA
            = ceilDiv((uint32_t)TileSizes::A_Size * sizeof(InputT), Granularity* WaveSize* DWord)
              * Granularity,
            TileB
            = ceilDiv((uint32_t)TileSizes::B_Size * sizeof(InputT), Granularity* WaveSize* DWord)
              * Granularity,
            TileC
            = ceilDiv((uint32_t)TileSizes::C_Size * sizeof(ComputeT), Granularity* WaveSize* DWord)
              * Granularity,
            TileD
            = ceilDiv((uint32_t)TileSizes::D_Size * sizeof(OutputT), Granularity* WaveSize* DWord)
              * Granularity,
        };

        // Architecture we are testing
        enum struct Arch : bool
        {
            IsWave32 = (WaveSize == Constants::AMDGCN_WAVE_SIZE_32),
            IsWave64 = (WaveSize == Constants::AMDGCN_WAVE_SIZE_64),

            IsGfx908 = (ArchId == Constants::AMDGCN_ARCH_ID_GFX908),
            IsGfx90A = (ArchId == Constants::AMDGCN_ARCH_ID_GFX90A),
            IsGfx940 = (ArchId == Constants::AMDGCN_ARCH_ID_GFX940),
            IsGfx941 = (ArchId == Constants::AMDGCN_ARCH_ID_GFX941),
            IsGfx942 = (ArchId == Constants::AMDGCN_ARCH_ID_GFX942),
            IsGfx950 = (ArchId == Constants::AMDGCN_ARCH_ID_GFX950),

            IsGfx1100 = (ArchId == Constants::AMDGCN_ARCH_ID_GFX1100),
            IsGfx1101 = (ArchId == Constants::AMDGCN_ARCH_ID_GFX1101),
            IsGfx1102 = (ArchId == Constants::AMDGCN_ARCH_ID_GFX1102),

            IsGfx1200 = (ArchId == Constants::AMDGCN_ARCH_ID_GFX1200),
            IsGfx1201 = (ArchId == Constants::AMDGCN_ARCH_ID_GFX1201),

            IsGfx9  = IsGfx908 || IsGfx90A || IsGfx940 || IsGfx941 || IsGfx942 || IsGfx950,
            IsGfx11 = IsGfx1100 || IsGfx1101 || IsGfx1102,
            IsGfx12 = IsGfx1200 || IsGfx1201,
        };

        enum struct InputType : bool
        {
            IsInt8 = std::is_same_v<InputT, int8_t>,

            // Make sure to include fnuz f8 types
            IsFloat8      = std::is_same_v<InputT, float8_t>,
            IsFloat8Fnuz  = std::is_same_v<InputT, float8_fnuz_t>,
            IsBFloat8     = std::is_same_v<InputT, bfloat8_t>,
            IsBFloat8Fnuz = std::is_same_v<InputT, bfloat8_fnuz_t>,

#if !ROCWMMA_TESTS_NO_HALF
            IsHFloat16 = std::is_same_v<InputT, hfloat16_t>,
#else
            IsHFloat16 = false,
#endif // !ROCWMMA_TESTS_NO_HALF
            IsFloat16  = std::is_same_v<InputT, float16_t> || IsHFloat16,
            IsBFloat16 = std::is_same_v<InputT, bfloat16_t>,

            IsFloat32  = std::is_same_v<InputT, float32_t>,
            IsXFloat32 = std::is_same_v<InputT, xfloat32_t>,

            IsFloat64 = std::is_same_v<InputT, float64_t>,
        };

        enum struct OutputType : bool
        {
            IsInt8        = std::is_same_v<OutputT, int8_t>,
            IsFloat8      = std::is_same_v<InputT, float8_t>,
            IsFloat8Fnuz  = std::is_same_v<InputT, float8_fnuz_t>,
            IsBFloat8     = std::is_same_v<InputT, bfloat8_t>,
            IsBFloat8Fnuz = std::is_same_v<InputT, bfloat8_fnuz_t>,

#if !ROCWMMA_TESTS_NO_HALF
            IsHFloat16 = std::is_same_v<OutputT, hfloat16_t>,
#else
            IsHFloat16 = false,
#endif // !ROCWMMA_TESTS_NO_HALF

            IsFloat16  = std::is_same_v<OutputT, float16_t> || IsHFloat16,
            IsBFloat16 = std::is_same_v<OutputT, bfloat16_t>,

            IsFloat32  = std::is_same_v<OutputT, float32_t>,
            IsXFloat32 = std::is_same_v<OutputT, xfloat32_t>,

            IsFloat64 = std::is_same_v<OutputT, float64_t>,
        };

        enum struct ComputeType : bool
        {
            IsInt8        = std::is_same_v<ComputeT, int8_t>,
            IsFloat8      = std::is_same_v<InputT, float8_t>,
            IsFloat8Fnuz  = std::is_same_v<InputT, float8_fnuz_t>,
            IsBFloat8     = std::is_same_v<InputT, bfloat8_t>,
            IsBFloat8Fnuz = std::is_same_v<InputT, bfloat8_fnuz_t>,

#if !ROCWMMA_TESTS_NO_HALF
            IsHFloat16 = std::is_same_v<ComputeT, hfloat16_t>,
#else
            IsHFloat16 = false,
#endif // !ROCWMMA_TESTS_NO_HALF

            IsFloat16  = std::is_same_v<ComputeT, float16_t> || IsHFloat16,
            IsBFloat16 = std::is_same_v<ComputeT, bfloat16_t>,

            IsFloat32  = std::is_same_v<ComputeT, float32_t>,
            IsXFloat32 = std::is_same_v<ComputeT, xfloat32_t>,

            IsFloat64 = std::is_same_v<ComputeT, float64_t>,
        };

        enum struct BlockSizes : bool
        {
            isBlockMN16 = (BlockM == 16u) && (BlockN == 16u),
            isBlockMN32 = (BlockM == 32u) && (BlockN == 32u)
        };
    };

} // namespace rocWMMA

#endif // ROCWMMA_GEMM_TEST_TRAITS_HPP
