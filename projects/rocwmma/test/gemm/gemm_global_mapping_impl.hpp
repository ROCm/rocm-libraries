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
#ifndef GEMM_GLOBAL_MAPPING_IMPL_HPP
#define GEMM_GLOBAL_MAPPING_IMPL_HPP

#include "gemm_global_mapping.hpp"

namespace rocwmma
{
    namespace GlobalMapping
    {

        namespace detail
        {

#define MappingBaseT                                                                               \
    uint32_t BlockM, uint32_t BlockN, uint32_t BlockK, typename InputT, typename OutputT,          \
        typename ComputeT, typename LayoutA, typename LayoutB, typename LayoutC, typename LayoutD, \
        uint32_t BlocksX, uint32_t BlocksY, uint32_t TBlockX, uint32_t TBlockY

#define MappingBaseT_impl                                                                  \
    BlockM, BlockN, BlockK, InputT, OutputT, ComputeT, LayoutA, LayoutB, LayoutC, LayoutD, \
        BlocksX, BlocksY, TBlockX, TBlockY

            template <MappingBaseT>
            template <typename CoordC>
            __device__ constexpr inline auto
                MappingBase<MappingBaseT_impl>::projCoordA(CoordC const& coordC)
            {
                return make_coord2d(get<0>(coordC), 0u);
            }

            template <MappingBaseT>
            template <typename CoordC>
            __device__ constexpr inline auto
                MappingBase<MappingBaseT_impl>::projCoordB(CoordC const& coordC)
            {
                return make_coord2d(0u, get<1>(coordC));
            }

            ///
            /// Dimensions
            ///

            template <MappingBaseT>
            __device__ constexpr inline auto MappingBase<MappingBaseT_impl>::macroTileSizeC()
            {
                return WaveSpace::workgroupDim() * waveTileSizeC();
            }

            template <MappingBaseT>
            __device__ constexpr inline auto MappingBase<MappingBaseT_impl>::waveTileSizeC()
            {
                return blockSizeC() * make_coord2d(BlocksX, BlocksY);
            }

            template <MappingBaseT>
            __device__ constexpr inline auto MappingBase<MappingBaseT_impl>::blockSizeC()
            {
                return make_coord2d((uint32_t)GetIOShape_t<MfmaFragC>::BlockHeight,
                                    (uint32_t)GetIOShape_t<MfmaFragC>::BlockWidth);
            }

            template <MappingBaseT>
            __device__ constexpr inline auto MappingBase<MappingBaseT_impl>::kDim()
            {
                return BlockK;
            }

            ///
            /// Offsets
            ///

            template <MappingBaseT>
            __device__ constexpr inline auto MappingBase<MappingBaseT_impl>::waveOffsetA()
            {
                return projCoordA(waveOffsetC());
            }

            template <MappingBaseT>
            __device__ constexpr inline auto MappingBase<MappingBaseT_impl>::waveOffsetB()
            {
                return projCoordB(waveOffsetC());
            }

            template <MappingBaseT>
            __device__ constexpr inline auto MappingBase<MappingBaseT_impl>::waveOffsetC()
            {
                return WaveSpace::localWaveCoord() * waveTileSizeC();
            }

            template <MappingBaseT>
            __device__ constexpr inline auto MappingBase<MappingBaseT_impl>::blockOffsetA()
            {
                return projCoordA(blockOffsetC());
            }

            template <MappingBaseT>
            __device__ constexpr inline auto MappingBase<MappingBaseT_impl>::blockOffsetB()
            {
                return projCoordB(blockOffsetC());
            }

            template <MappingBaseT>
            __device__ constexpr inline auto MappingBase<MappingBaseT_impl>::blockOffsetC()
            {
                return blockSizeC();
            }

            template <MappingBaseT>
            __device__ constexpr inline auto MappingBase<MappingBaseT_impl>::kStepOffsetA()
            {
                return make_coord2d(0u, BlockK);
            }

            template <MappingBaseT>
            __device__ constexpr inline auto MappingBase<MappingBaseT_impl>::kStepOffsetB()
            {
                return make_coord2d(BlockK, 0u);
            }

            ///
            /// Global matrix coords
            ///

            template <MappingBaseT>
            __device__ constexpr inline auto MappingBase<MappingBaseT_impl>::macroTileCoordC()
            {
                return WaveSpace::workgroupCoord() * macroTileSizeC();
            }

            template <MappingBaseT>
            __device__ constexpr inline auto MappingBase<MappingBaseT_impl>::waveTileCoordC()
            {
                return macroTileCoordC() + waveOffsetC();
            }
        }

#undef MappingBaseT
#undef MappingBaseT_impl

    } // namespace CooperativeGemm

} // namespace rocwmma

#endif // GEMM_GLOBAL_MAPPING_IMPL_HPP
