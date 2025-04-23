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
#ifndef GEMM_LOCAL_MAPPING_IMPL_HPP
#define GEMM_LOCAL_MAPPING_IMPL_HPP

#include "gemm_global_mapping.hpp"
#include "gemm_local_mapping.hpp"

namespace rocwmma
{
    namespace LocalMapping
    {

#define LdsMappingT typename GlobalMapping, typename LayoutLds

#define LdsMappingT_impl GlobalMapping, LayoutLds

        template <LdsMappingT>
        __device__ constexpr inline auto LdsMappingTN<LdsMappingT_impl>::waveOffsetA()
        {
            return swap(GlobalMapping::waveOffsetA());
        }

        template <LdsMappingT>
        __device__ constexpr inline auto LdsMappingTN<LdsMappingT_impl>::waveOffsetB()
        {
            return GlobalMapping::waveOffsetB();
        }

        template <LdsMappingT>
        __device__ constexpr inline auto LdsMappingTN<LdsMappingT_impl>::blockOffsetA()
        {
            return swap(GlobalMapping::blockOffsetA());
        }

        template <LdsMappingT>
        __device__ constexpr inline auto LdsMappingTN<LdsMappingT_impl>::blockOffsetB()
        {
            return GlobalMapping::blockOffsetB();
        }

        template <LdsMappingT>
        __device__ constexpr inline auto LdsMappingTN<LdsMappingT_impl>::writeCoordA()
        {
            // Base lds coordA = (0, 0).
            // For local write, must add wave offset if global read tile is a wave tile
            auto baseCoordA = make_coord2d(0u, 0u);
            return GlobalMapping::readABWaveTile() ? baseCoordA + waveOffsetA() : baseCoordA;
        }

        template <LdsMappingT>
        __device__ constexpr inline auto LdsMappingTN<LdsMappingT_impl>::writeCoordB()
        {
            // B data will start right after A data
            // For local write, must add wave offset if global read tile is a wave tile
            auto baseCoordB = swap(GlobalMapping::projCoordA(GlobalMapping::macroTileSizeC()));
            return GlobalMapping::readABWaveTile() ? baseCoordB + waveOffsetB() : baseCoordB;
        }

        template <LdsMappingT>
        __device__ constexpr inline auto LdsMappingTN<LdsMappingT_impl>::readCoordA()
        {
            // Base lds coordA = (0, 0).
            // For local read, will be in MFMA format, so we need the wave offset
            auto baseCoordA = make_coord2d(0u, 0u);
            return baseCoordA + waveOffsetA();
        }

        template <LdsMappingT>
        __device__ constexpr inline auto LdsMappingTN<LdsMappingT_impl>::readCoordB()
        {
            // B data will start right after A data
            // For local read, will be in MFMA format, so we need the wave offset
            auto baseCoordB = swap(GlobalMapping::projCoordA(GlobalMapping::macroTileSizeC()));
            return baseCoordB + waveOffsetB();
        }

        template <LdsMappingT>
        __device__ constexpr inline auto LdsMappingTN<LdsMappingT_impl>::sizeLds()
        {
            auto macroTileC = GlobalMapping::macroTileSizeC();
            return make_coord2d(LdsHeight, get<0>(macroTileC) + get<1>(macroTileC));
        }

        template <LdsMappingT>
        __device__ constexpr inline auto LdsMappingTN<LdsMappingT_impl>::ldLds()
        {
            return DataLayout::leadingDim(sizeLds());
        }

#undef LdsMappingT
#undef LdsMappingT_impl

#define LdsMappingT typename GlobalMapping, typename LayoutLds

#define LdsMappingT_impl GlobalMapping, LayoutLds

        template <LdsMappingT>
        __device__ constexpr inline auto LdsMappingNT<LdsMappingT_impl>::waveOffsetA()
        {
            return GlobalMapping::waveOffsetA();
        }

        template <LdsMappingT>
        __device__ constexpr inline auto LdsMappingNT<LdsMappingT_impl>::waveOffsetB()
        {
            return swap(GlobalMapping::waveOffsetB());
        }

        template <LdsMappingT>
        __device__ constexpr inline auto LdsMappingNT<LdsMappingT_impl>::blockOffsetA()
        {
            return GlobalMapping::blockOffsetA();
        }

        template <LdsMappingT>
        __device__ constexpr inline auto LdsMappingNT<LdsMappingT_impl>::blockOffsetB()
        {
            return swap(GlobalMapping::blockOffsetB());
        }

        template <LdsMappingT>
        __device__ constexpr inline auto LdsMappingNT<LdsMappingT_impl>::writeCoordA()
        {
            // Base lds coordA = (0, 0).
            // For local write, must add wave offset if global read tile is a wave tile
            auto baseCoordA = make_coord2d(0u, 0u);
            return GlobalMapping::readABWaveTile() ? baseCoordA + waveOffsetA() : baseCoordA;
        }

        template <LdsMappingT>
        __device__ constexpr inline auto LdsMappingNT<LdsMappingT_impl>::writeCoordB()
        {
            // B data will start right after A data
            // For local write, must add wave offset if global read tile is a wave tile
            auto baseCoordB = GlobalMapping::projCoordA(GlobalMapping::macroTileSizeC());
            return GlobalMapping::readABWaveTile() ? baseCoordB + waveOffsetB() : baseCoordB;
        }

        template <LdsMappingT>
        __device__ constexpr inline auto LdsMappingNT<LdsMappingT_impl>::readCoordA()
        {
            // Base lds coordA = (0, 0).
            // For local write, must add wave offset if global read tile is a wave tile
            auto baseCoordA = make_coord2d(0u, 0u);
            return baseCoordA + waveOffsetA();
        }

        template <LdsMappingT>
        __device__ constexpr inline auto LdsMappingNT<LdsMappingT_impl>::readCoordB()
        {
            // B data will start right after A data
            // For local write, must add wave offset if global read tile is a wave tile
            auto baseCoordB = GlobalMapping::projCoordA(GlobalMapping::macroTileSizeC());
            return baseCoordB + waveOffsetB();
        }

        template <LdsMappingT>
        __device__ constexpr inline auto LdsMappingNT<LdsMappingT_impl>::sizeLds()
        {
            auto macroTileC = GlobalMapping::macroTileSizeC();
            return make_coord2d(get<0>(macroTileC) + get<1>(macroTileC), LdsWidth);
        }

        template <LdsMappingT>
        __device__ constexpr inline auto LdsMappingNT<LdsMappingT_impl>::ldLds()
        {
            return DataLayout::leadingDim(sizeLds());
        }

#undef LdsMappingT
#undef LdsMappingT_impl

#undef LdsMappingT
#undef LdsMappingT_impl

#define LdsMappingT typename GlobalMapping, typename LayoutLds

#define LdsMappingT_impl GlobalMapping, LayoutLds

        template <LdsMappingT>
        __device__ constexpr inline auto
            LdsMappingRF<LdsMappingT_impl>::projCoordA(Coord2d const& coordA)
        {
            // Scale the A coordinate to register file height
            return make_coord2d(get<0>(coordA) * GlobalMapping::kDim() / LdsWidth, 0u);
        }

        template <LdsMappingT>
        __device__ constexpr inline auto
            LdsMappingRF<LdsMappingT_impl>::projCoordB(Coord2d const& coordB)
        {
            // Scale the B coordinate to register file height
            return make_coord2d(get<1>(coordB) * GlobalMapping::kDim() / LdsWidth, 0u);
        }

        template <LdsMappingT>
        __device__ constexpr inline auto LdsMappingRF<LdsMappingT_impl>::waveOffsetA()
        {
            return projCoordA(GlobalMapping::waveOffsetA());
        }

        template <LdsMappingT>
        __device__ constexpr inline auto LdsMappingRF<LdsMappingT_impl>::waveOffsetB()
        {
            return projCoordB(GlobalMapping::waveOffsetB());
        }

        template <LdsMappingT>
        __device__ constexpr inline auto LdsMappingRF<LdsMappingT_impl>::blockOffsetA()
        {
            return projCoordA(GlobalMapping::blockOffsetA());
        }

        template <LdsMappingT>
        __device__ constexpr inline auto LdsMappingRF<LdsMappingT_impl>::blockOffsetB()
        {
            return projCoordB(GlobalMapping::blockOffsetB());
        }

        template <LdsMappingT>
        __device__ constexpr inline auto LdsMappingRF<LdsMappingT_impl>::writeCoordA()
        {
            // Assuming base coord of (0, 0) for LDS
            return waveOffsetA();
        }

        template <LdsMappingT>
        __device__ constexpr inline auto LdsMappingRF<LdsMappingT_impl>::writeCoordB()
        {
            return projCoordA(GlobalMapping::macroTileSizeC()) + waveOffsetB();
        }

        template <LdsMappingT>
        __device__ constexpr inline auto LdsMappingRF<LdsMappingT_impl>::readCoordA()
        {
            // Assuming base coord of (0, 0) for LDS
            return waveOffsetA();
        }

        template <LdsMappingT>
        __device__ constexpr inline auto LdsMappingRF<LdsMappingT_impl>::readCoordB()
        {
            return projCoordA(GlobalMapping::macroTileSizeC()) + waveOffsetB();
        }

        template <LdsMappingT>
        __device__ constexpr inline auto LdsMappingRF<LdsMappingT_impl>::sizeLds()
        {
            auto macroTileC = GlobalMapping::macroTileSizeC();
            return make_coord2d((get<0>(macroTileC) + get<1>(macroTileC)) * GlobalMapping::kDim()
                                    / LdsWidth,
                                LdsWidth);
        }

        template <LdsMappingT>
        __device__ constexpr inline auto LdsMappingRF<LdsMappingT_impl>::ldLds()
        {
            return DataLayout::leadingDim(sizeLds());
        }

#undef LdsMappingT
#undef LdsMappingT_impl

    } // namespace LocalMapping

} // namespace rocwmma

#endif // GEMM_LOCAL_MAPPING_IMPL_HPP
