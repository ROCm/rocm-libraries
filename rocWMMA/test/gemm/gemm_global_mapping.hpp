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
#ifndef GEMM_GLOBAL_MAPPING_HPP
#define GEMM_GLOBAL_MAPPING_HPP

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <rocwmma/rocwmma.hpp>
#include <rocwmma/rocwmma_coop.hpp>
#include <rocwmma/rocwmma_transforms.hpp>
#pragma GCC diagnostic pop

namespace rocwmma
{
    namespace GlobalMapping
    {
        namespace detail
        {
            template <uint32_t BlockM, // MFMA BlockM
                      uint32_t BlockN, // MFMA BlockN
                      uint32_t BlockK, // MFMA BlockK
                      typename InputT, // Type of A / B data
                      typename OutputT, // Type of C / D data
                      typename ComputeT, // Type of accumulator data
                      typename LayoutA, // Data Layout of A (row / col major)
                      typename LayoutB, // Data Layout of B (row / col major)
                      typename LayoutC, // Data Layout of C (row / col major)
                      typename LayoutD, // Data Layout of D (row / col major)
                      uint32_t BlocksX, // MFMA blocks per wave in X direction
                      uint32_t BlocksY, // MFMA blocks per wave in Y direction
                      uint32_t TBlockX = 0, // Thread block X dimension
                      uint32_t TBlockY = 0> // Thread block Y dimension
            struct MappingBase
            {
                /*
                * This global mapping aligns each wave with a gemm C/D tile
                * that the wave will compute independently. All calculations
                * are in matrix coordinates (Coord2d).
                *
                * Coordinates are scaled into three different levels:
                * - Macro level: context of work for entire workgroup
                * - Wave level: context of work for each wave
                * - Block level: context of work for each mfma block
                *
                * Waves computing tiles on the same row (matrix A) and waves
                * computing tiles on the same col (matrix B) have opportunities
                * for data sharing. As a result, waves collaboration is limited
                * to tiles in same locality.
                *
                * Block size:      (BlockM x BlockN)
                * Wave tile size:  (BlocksX * BlockSize.x) x (BlocksY * BlockSize.y)
                * Macro Tile size: (WgDim.x * WaveTileSize.x) x (WgDim.y * WaveTileSize.y)
                *
                * Wave data share A: same row
                * Wave data share B: same col
                *
                * Global C layout & workgroup assignment for WGDim = (2x2):
                *
                * W (X, Y) = wave row X, col Y
                *
                *
                *                                     |--------- Macro Tile Y-------------|
                *                                     |-- Wave Tile Y --|
                *                                     |-BlockN-|
                *
                *                                      BlockN x BlocksY   BlockN x BlocksY
                *                                     |<--------------->|<--------------->|
                *      _ _   _ _      _ _          ___  ________ ________ ________ ________
                *       |     |        |            ^  |        |        |        |        |
                *       | Wave| BlockM |   BlockM   |  |        W        |        W        |
                *       | Tile|       _|_     x     |  |__   (0, 0)    __|__   (0, 1)    __|
                *       |  X  |            BlocksX  |  |                 |                 |
                * Macro |     |                     |  |                 |                 |
                *  Tile |    _|_                   _v_ |________|________|________|________|
                *   X   |                           ^  |        |        |        |        |
                *       |                  BlockM   |  |        W        |        W        |
                *       |                     x     |  |__   (1, 0)    __|__   (1, 1)    __|
                *       |                  BlocksX  |  |                 |                 |
                *       |                           |  |                 |                 |
                *      _|_                         _v_ |________|________|________|________|
                *
                *
                * TLDR: Global mapping aligns global offsets for A / B / C / D  MFMA blocks.
                * Configures fragment and buffer types for per-wave responsibility of
                * BlocksX * BlocksY MFMA blocks.
                */

                // Block fragments that are used as inputs to MFMA operations.
                using MfmaFragA   = fragment<matrix_a, BlockM, BlockN, BlockK, InputT, LayoutA>;
                using MfmaFragB   = fragment<matrix_b, BlockM, BlockN, BlockK, InputT, LayoutB>;
                using MfmaFragC   = fragment<accumulator, BlockM, BlockN, BlockK, OutputT, LayoutC>;
                using MfmaFragD   = fragment<accumulator, BlockM, BlockN, BlockK, OutputT, LayoutD>;
                using MfmaFragAcc = fragment<accumulator, BlockM, BlockN, BlockK, ComputeT>;

                // Mfma fragment buffers required for the gemm driver
                using MfmaBuffA   = MfmaFragA[BlocksX];
                using MfmaBuffB   = MfmaFragB[BlocksY];
                using MfmaBuffC   = MfmaFragC[BlocksX][BlocksY];
                using MfmaBuffD   = MfmaFragD[BlocksX][BlocksY];
                using MfmaBuffAcc = MfmaFragAcc[BlocksX][BlocksY];

                using WaveSpace = typename rocwmma::detail::WaveSpace<TBlockX, TBlockY>;

                // Projection of C coordinate in direction of A
                template <typename CoordC>
                __device__ constexpr static inline auto projCoordA(CoordC const& coordC);

                // Projection of C coordinate in direction of B
                template <typename CoordC>
                __device__ constexpr static inline auto projCoordB(CoordC const& coordC);

                ///
                /// Dimensions
                ///

                // Macro tile = tile processed by entire workgroup
                // Wave tile = tile processed by current wave
                // Block size = mfma block size
                __device__ constexpr static inline auto macroTileSizeC();
                __device__ constexpr static inline auto waveTileSizeC();
                __device__ constexpr static inline auto blockSizeC();
                __device__ constexpr static inline auto kDim();

                ///
                /// Offsets
                ///

                // The offset from macro tile to the mfma tiles for current wave
                __device__ constexpr static inline auto waveOffsetA();
                __device__ constexpr static inline auto waveOffsetB();
                __device__ constexpr static inline auto waveOffsetC();

                // The local offsets between mfma blocks
                __device__ constexpr static inline auto blockOffsetA();
                __device__ constexpr static inline auto blockOffsetB();
                __device__ constexpr static inline auto blockOffsetC();

                // The matrix offset to the next step in the k dimension
                __device__ constexpr static inline auto kStepOffsetA();
                __device__ constexpr static inline auto kStepOffsetB();

                ///
                /// Global matrix coords
                ///

                // Global matrix coordinate of macro tile for the current workgroup
                __device__ constexpr static inline auto macroTileCoordC();

                // Global matrix coordinate of wave tile for the current wave
                __device__ constexpr static inline auto waveTileCoordC();
            };

        } // namespace detail

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
                  uint32_t BlocksX,
                  uint32_t BlocksY,
                  uint32_t TBlockX = 0,
                  uint32_t TBlockY = 0>
        struct BlockLevelMapping : public detail::MappingBase<BlockM,
                                                              BlockN,
                                                              BlockK,
                                                              InputT,
                                                              OutputT,
                                                              ComputeT,
                                                              LayoutA,
                                                              LayoutB,
                                                              LayoutC,
                                                              LayoutD,
                                                              BlocksX,
                                                              BlocksY,
                                                              TBlockX,
                                                              TBlockY>
        {
            /*
            * This flavour of Global Mapping targets A/B/C/D wave tiles iteratively
            * in MFMA fragment chunks:
            * BlocksX x (BlockM x BlockK) for A
            * BlocksY x (BlockN x BlockK) for B
            * BlocksX x BlocksY x (BlockM x BlockN) for C / D
            *
            * This leads to favourable MFMA layouts directly from Global Reads,
            * but may not be the most efficient in some layouts.
            */

            using Base = detail::MappingBase<BlockM,
                                             BlockN,
                                             BlockK,
                                             InputT,
                                             OutputT,
                                             ComputeT,
                                             LayoutA,
                                             LayoutB,
                                             LayoutC,
                                             LayoutD,
                                             BlocksX,
                                             BlocksY,
                                             TBlockX,
                                             TBlockY>;

            // Global wave tile R/W be in sections of MFMA sized fragments
            using GRFragA = typename Base::MfmaFragA;
            using GRFragB = typename Base::MfmaFragB;
            using GRFragC = typename Base::MfmaFragC;
            using GWFragD = typename Base::MfmaFragD;

            // Global wave tile R/W will use MFMA sized fragment buffers
            using GRBuffA = typename Base::MfmaBuffA;
            using GRBuffB = typename Base::MfmaBuffB;
            using GRBuffC = typename Base::MfmaBuffC;
            using GWBuffD = typename Base::MfmaBuffD;

            // The base global matrix coordinate of the current wave tile.
            __device__ constexpr static inline auto readCoordA()
            {
                return Base::projCoordA(readCoordC());
            }
            __device__ constexpr static inline auto readCoordB()
            {
                return Base::projCoordB(readCoordC());
            }
            __device__ constexpr static inline auto readCoordC()
            {
                return Base::waveTileCoordC();
            }
            __device__ constexpr static inline auto writeCoordD()
            {
                return Base::waveTileCoordC();
            }

            // Indicate if global read for A and B are wave tiles
            __device__ constexpr static inline auto readABWaveTile()
            {
                return true;
            }
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
                  uint32_t BlocksX,
                  uint32_t BlocksY,
                  uint32_t TBlockX = 0,
                  uint32_t TBlockY = 0>
        struct WaveLevelMapping : public detail::MappingBase<BlockM,
                                                             BlockN,
                                                             BlockK,
                                                             InputT,
                                                             OutputT,
                                                             ComputeT,
                                                             LayoutA,
                                                             LayoutB,
                                                             LayoutC,
                                                             LayoutD,
                                                             BlocksX,
                                                             BlocksY,
                                                             TBlockX,
                                                             TBlockY>
        {
            /*
            * This flavour of Global Mapping targets A/B as a single wave tile sized fragment.
            * C/D wave tiles are targeted iteratively in MFMA fragment chunks:
            * (BlocksX * BlockM) x BlockK for A
            * (BlocksY * BlockN) x BlockK for B
            * BlocksX x BlocksY x (BlockM x BlockN) for C / D
            *
            * This global read A/B fragments are not MFMA friendly, however when written to LDS
            * smaller MFMA friendly fragment may be read directly from the same LDS layout.
            * Larger GR may be more efficient in certain layouts for data pipelining.
            *
            * This layout can accommodate either compile time or runtime TBlockX/Y params.
            */
            using Base = detail::MappingBase<BlockM,
                                             BlockN,
                                             BlockK,
                                             InputT,
                                             OutputT,
                                             ComputeT,
                                             LayoutA,
                                             LayoutB,
                                             LayoutC,
                                             LayoutD,
                                             BlocksX,
                                             BlocksY,
                                             TBlockX,
                                             TBlockY>;

            // Global reads for A/B are single fragment of wave tile size
            // Global R/W for C/D are MFMA sized fragments
            using GRFragA = fragment<matrix_a, BlockM * BlocksX, BlockN, BlockK, InputT, LayoutA>;
            using GRFragB = fragment<matrix_b, BlockM, BlockN * BlocksY, BlockK, InputT, LayoutB>;
            using GRFragC = typename Base::MfmaFragC;
            using GWFragD = typename Base::MfmaFragD;

            // Global reads for A/B will have one fragment buffer
            // Global R/W for C/D will use MFMA sized multiple fragment buffer
            using GRBuffA = GRFragA;
            using GRBuffB = GRFragB;
            using GRBuffC = typename Base::MfmaBuffC;
            using GWBuffD = typename Base::MfmaBuffD;

            // A/B global reads will collaborate on wave tile
            __device__ constexpr static inline auto readCoordA()
            {
                return Base::projCoordA(readCoordC());
            }
            __device__ constexpr static inline auto readCoordB()
            {
                return Base::projCoordB(readCoordC());
            }

            // C/D global R/W on wave tile
            __device__ constexpr static inline auto readCoordC()
            {
                return Base::waveTileCoordC();
            }

            __device__ constexpr static inline auto writeCoordD()
            {
                return Base::waveTileCoordC();
            }

            // Indicate if global read for A and B are wave tiles
            __device__ constexpr static inline auto readABWaveTile()
            {
                return true;
            }
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
                  uint32_t BlocksX,
                  uint32_t BlocksY,
                  uint32_t TBlockX,
                  uint32_t TBlockY>
        struct WorkgroupLevelMapping : public detail::MappingBase<BlockM,
                                                                  BlockN,
                                                                  BlockK,
                                                                  InputT,
                                                                  OutputT,
                                                                  ComputeT,
                                                                  LayoutA,
                                                                  LayoutB,
                                                                  LayoutC,
                                                                  LayoutD,
                                                                  BlocksX,
                                                                  BlocksY,
                                                                  TBlockX,
                                                                  TBlockY>
        {

            // Must provide valid TBlockX/Y params at compile time.
            static_assert((TBlockX > 0) && (TBlockX % Constants::AMDGCN_WAVE_SIZE == 0),
                          "Invalid TBlockX dimension");
            static_assert(TBlockY > 0, "Invalid TBlockY dimension");

            /*
            * This flavour of Global Mapping targets A/B as a single macro tile sized fragment.
            * C/D wave tiles are targeted iteratively in MFMA fragment chunks:
            * (WgX * BlocksX * BlockM) x BlockK for A
            * (WgY * BlocksY * BlockN) x BlockK for B
            * BlocksX x BlocksY x (BlockM x BlockN) for C / D
            *
            * This global read A/B fragments are not MFMA friendly, however when written to LDS
            * smaller MFMA friendly fragment may be read directly from the same LDS layout.
            * Larger GR may be more efficient in certain layouts for data pipelining.
            */
            using Base = detail::MappingBase<BlockM,
                                             BlockN,
                                             BlockK,
                                             InputT,
                                             OutputT,
                                             ComputeT,
                                             LayoutA,
                                             LayoutB,
                                             LayoutC,
                                             LayoutD,
                                             BlocksX,
                                             BlocksY,
                                             TBlockX,
                                             TBlockY>;


            static constexpr uint32_t WaveTileM = BlocksX * BlockM;
            static constexpr uint32_t WaveTileN = BlocksY * BlockN;
            static constexpr uint32_t WaveTileK = BlockK;

            static constexpr uint32_t MacroTileM = TBlockX / Constants::AMDGCN_WAVE_SIZE * WaveTileM;
            static constexpr uint32_t MacroTileN = TBlockY * WaveTileN;
            static constexpr uint32_t MacroTileK = BlockK;

            // Wave-tiles
            using WaveTileA   = fragment<matrix_a, WaveTileM, WaveTileN, WaveTileK, InputT, LayoutA>;
            using WaveTileB   = fragment<matrix_b, WaveTileM, WaveTileN, WaveTileK, InputT, LayoutB>;
            using WaveTileC   = fragment<accumulator, WaveTileM, WaveTileN, WaveTileK, OutputT, LayoutC>;
            using WaveTileD   = fragment<accumulator, WaveTileM, WaveTileN, WaveTileK, OutputT, LayoutD>;
            using WaveTileAcc = fragment<accumulator, WaveTileM, WaveTileN, WaveTileK, ComputeT>;

            // We can guess layouts are interleaved by testing A's MatrixLayout.
            static constexpr bool IsInterleaved = layout_traits<typename GetIOConfig_t<WaveTileA>::IOLayout::MatrixLayout>::is_interleaved;

            // Macro-tiles
            using MacroTileA   = fragment<matrix_a, MacroTileM, MacroTileN, MacroTileK, InputT, LayoutA>;
            using MacroTileB   = fragment<matrix_b, MacroTileM, MacroTileN, MacroTileK, InputT, LayoutB>;

            // Mma operations
            // Interleaved layouts use wave tile mma inputs
            // Non-interleaved layouts use block tile mma inputs
            using MfmaFragA   = conditional_t<IsInterleaved, WaveTileA, typename Base::MfmaFragA>;
            using MfmaFragB   = conditional_t<IsInterleaved, WaveTileB, typename Base::MfmaFragB>;
            using MfmaFragC   = conditional_t<IsInterleaved, WaveTileC, typename Base::MfmaFragC>;
            using MfmaFragD   = conditional_t<IsInterleaved, WaveTileD, typename Base::MfmaFragD>;
            using MfmaFragAcc = conditional_t<IsInterleaved, WaveTileAcc, typename Base::MfmaFragAcc>;

            // Mfma fragment buffers required for the gemm driver.
            // Interleaved layouts mma inputs are buffered in single wave tile
            // Non-interleaved layouts mma inputs are buffered in array of block tiles
            using MfmaBuffA   = conditional_t<IsInterleaved, MfmaFragA, typename Base::MfmaBuffA>;
            using MfmaBuffB   = conditional_t<IsInterleaved, MfmaFragB, typename Base::MfmaBuffB>;
            using MfmaBuffC   = conditional_t<IsInterleaved, MfmaFragC, typename Base::MfmaBuffC>;
            using MfmaBuffD   = conditional_t<IsInterleaved, MfmaFragD, typename Base::MfmaBuffD>;
            using MfmaBuffAcc = conditional_t<IsInterleaved, MfmaFragAcc, typename Base::MfmaBuffAcc>;

            // Global reads for A/B are macro tiles
            // Global C/D:
            // Interleaved layouts wave tile
            // Non-interleaved layouts block tile
            using GRFragA = fragment<matrix_a, MacroTileM, MacroTileN, MacroTileK, InputT, LayoutA>;
            using GRFragB = fragment<matrix_b, MacroTileM, MacroTileN, MacroTileK, InputT, LayoutB>;
            using GRFragC = conditional_t<IsInterleaved, WaveTileC, typename Base::MfmaFragC>;
            using GWFragD = conditional_t<IsInterleaved, WaveTileD, typename Base::MfmaFragD>;

            // Global reads for A/B will have one macro tile buffer
            // Global R/W:
            // Interleaved layouts buffer will be wave-tile
            // Non-interleaved layouts buffer will be array of block tiles
            using GRBuffA = GRFragA;
            using GRBuffB = GRFragB;
            using GRBuffC = conditional_t<IsInterleaved, GRFragC, typename Base::MfmaBuffC>;
            using GWBuffD = conditional_t<IsInterleaved, GWFragD, typename Base::MfmaBuffD>;

            // A/B global reads will collaborate on macro tile
            __device__ constexpr static inline auto readCoordA()
            {
                return Base::projCoordA(Base::macroTileCoordC());
            }
            __device__ constexpr static inline auto readCoordB()
            {
                return Base::projCoordB(Base::macroTileCoordC());
            }

            // C/D global R/W on wave tile
            __device__ constexpr static inline auto readCoordC()
            {
                return Base::waveTileCoordC();
            }

            __device__ constexpr static inline auto writeCoordD()
            {
                return Base::waveTileCoordC();
            }

            // Indicate if global read for A and B are wave tiles
            __device__ constexpr static inline auto readABWaveTile()
            {
                return false;
            }
        };

    } // namespace GlobalMapping

} // namespace rocwmma

#include "gemm_global_mapping_impl.hpp"

#endif // GEMM_GLOBAL_MAPPING_HPP
