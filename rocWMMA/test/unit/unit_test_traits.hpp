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

#ifndef ROCWMMA_UNIT_TEST_TRAITS_HPP
#define ROCWMMA_UNIT_TEST_TRAITS_HPP

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
              typename DataT,
              typename Layout,
              uint32_t WaveSize,
              uint32_t ArchId>
    struct UnitTestTraits
    {
        // Size properties of gemm tiles
        enum struct Sizes : uint32_t
        {
            TileX    = BlockM,
            TileY    = BlockN,
            TileSize = TileX * TileY * sizeof(DataT),
        };

        // Tile costs
        enum struct Cost : uint32_t
        {
            Granularity = 16u,
            DWord       = 4u,

            PackedTile
            = ceilDiv((uint32_t)Sizes::TileSize, Granularity* WaveSize* DWord) * Granularity,
            UnpackedTile = PackedTile * PackTraits<DataT>::PackRatio,
        };

        // Architecture we are testing
        enum Arch : bool
        {
            IsWave32 = (WaveSize == Constants::AMDGCN_WAVE_SIZE_32),
            IsWave64 = (WaveSize == Constants::AMDGCN_WAVE_SIZE_64),

            IsGfx908  = (ArchId == Constants::AMDGCN_ARCH_ID_GFX908),
            IsGfx90A  = (ArchId == Constants::AMDGCN_ARCH_ID_GFX90A),
            IsGfx942  = (ArchId == Constants::AMDGCN_ARCH_ID_GFX942),
            IsGfx950  = (ArchId == Constants::AMDGCN_ARCH_ID_GFX950),
            IsGfx1100 = (ArchId == Constants::AMDGCN_ARCH_ID_GFX1100),
            IsGfx1101 = (ArchId == Constants::AMDGCN_ARCH_ID_GFX1101),
            IsGfx1102 = (ArchId == Constants::AMDGCN_ARCH_ID_GFX1102),
            IsGfx1200 = (ArchId == Constants::AMDGCN_ARCH_ID_GFX1200),
            IsGfx1201 = (ArchId == Constants::AMDGCN_ARCH_ID_GFX1201),

            IsGfx9  = IsGfx908 || IsGfx90A || IsGfx942 || IsGfx950,
            IsGfx11 = IsGfx1100 || IsGfx1101 || IsGfx1102,
            IsGfx12 = IsGfx1200 || IsGfx1201,
        };
    };

    template <uint32_t BlockM,
              uint32_t BlockN,
              typename DataT,
              typename Layout,
              uint32_t WaveSize,
              uint32_t ArchId>
    struct FragSize_guard
    {
        using TestTraits = UnitTestTraits<BlockM, BlockN, DataT, Layout, WaveSize, ArchId>;

    private:
        enum struct Gfx9Predicates : bool
        {
            // Cost for a full tile, unpacked data
            CostTest = ((uint32_t)TestTraits::Cost::UnpackedTile <= 256u),
            Enable   = ((bool)TestTraits::IsGfx9 && (bool)TestTraits::IsWave64 && CostTest)
        };

        enum struct Gfx11Predicates : bool
        {
            // Cost for a full tile, unpacked data
            CostTest = ((uint32_t)TestTraits::Cost::UnpackedTile <= 256u),
            Enable   = ((bool)TestTraits::IsGfx11 && (bool)TestTraits::IsWave32 && CostTest)
        };

        enum struct Gfx12Predicates : bool
        {
            // Cost for a full tile, unpacked data
            CostTest = ((uint32_t)TestTraits::Cost::UnpackedTile <= 256u),
            Enable   = ((bool)TestTraits::IsGfx12 && (bool)TestTraits::IsWave32 && CostTest)
        };

    public:
        constexpr static bool enable()
        {
            return ((bool)Gfx9Predicates::Enable || (bool)Gfx11Predicates::Enable
                    || (bool)Gfx12Predicates::Enable);
        }
    };

} // namespace rocWMMA

#endif // ROCWMMA_UNIT_TEST_TRAITS_HPP
