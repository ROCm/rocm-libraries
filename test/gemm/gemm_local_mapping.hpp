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
#ifndef GEMM_LOCAL_MAPPING_HPP
#define GEMM_LOCAL_MAPPING_HPP

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <rocwmma/rocwmma.hpp>
#include <rocwmma/rocwmma_coop.hpp>
#include <rocwmma/rocwmma_transforms.hpp>
#pragma GCC diagnostic pop

namespace rocwmma
{
    namespace LocalMapping
    {
        template <typename GlobalMapping, typename LayoutLds>
        struct LdsMappingTN
        {
            /*
            * This is a helper class to transform global A and B fragments such that they can
            * fit together in a common LDS space without the use of extra padding.
            * We use the fragments defined in the GlobalMapping argument.
            *
            * LdsMappingTN (Block Height = LDS Height = BlockK)
            * Matrix geometry for A and B have a common dimension in BlockK.
            * We can fix the height of the LDS storage to BlockK, and transform
            * fragments of A and B to fit without extra padding.
            *
            * Fragments of A must be transposed to fit this geometry,
            * and both fragments from A and B must accomodate LDS data layout.
            *
            * Local Matrix Layout (LDS):
            *
            * Transposed A fragments [A0 (T) ... AX-1 (T)] are placed first and occupy a total width of MacroTileX.
            * B fragments [B0 ... BY-1] follow A fragments and occupy a total width of MacroTileY
            *
            *
            *  kDim                        A0 (T)             AX-1 (T)            B0                BY-1
            *  |      (0,0)  -->  ______________  ...  ___               ______________  ...  ___
            *  |             |   |__________C0__  ...  ___|     ...     |__________R0__  ...  ___|  ...
            *  |             |   |__________C1__  ...  ___|     ...     |__________R1__  ...  ___|  ...
            *  |      BlockK |   |__________C2__  ...  ___|     ...     |__________R2__  ...  ___|  ...
            *  |             |   |          ...   ...     |     ...     |          ...   ...     |  ...
            *  |             |   |__________Ck__  ...  ___|     ...     |__________Rk__  ...  ___|  ...
            *  v             -->
            *                    ^------- BlockM ---------^             ^-------- BlockN --------^
            *
            *                    ^-------- MacroTileX ----------^ ^----------- MacroTileY ------------^ (BlockK-1, MacroTileX + MacroTileY - 1)
            *
            * TLDR: Take the Global Read fragments, transpose A and write the resulting frags into LDS
            * stacked beside each other using BlockK as common height.
            */

            using DataLayout = DataLayout::Array1d<LayoutLds>;

            /// LOCAL WRITE -> GR frags
            // K = BlockHeight
            // GRFragA Transposed
            // GRFragB unchanged
            // Ensure LW frags are LayoutLds friendly.
            using LWFragA
                = ApplyDataLayout_t<ApplyTranspose_t<typename GlobalMapping::GRFragA>, LayoutLds>;
            using LWFragB = ApplyDataLayout_t<typename GlobalMapping::GRFragB, LayoutLds>;

            /// LOCAL READ -> Mfma frags
            // MfmaFragA transposed
            // MfmaFragB unchanged
            // Ensure LR frags are LayoutLds friendly.
            using LRFragA
                = ApplyDataLayout_t<ApplyTranspose_t<typename GlobalMapping::MfmaFragA>, LayoutLds>;
            using LRFragB = ApplyDataLayout_t<typename GlobalMapping::MfmaFragB, LayoutLds>;

            /// Sanity checks:
            // All local R/W tiles should have same height
            static_assert(GetIOShape_t<LWFragA>::BlockHeight == GetIOShape_t<LWFragB>::BlockHeight,
                          "LW frag heights do not match");
            static_assert(GetIOShape_t<LRFragA>::BlockHeight == GetIOShape_t<LRFragB>::BlockHeight,
                          "LR frag heights do not match");
            static_assert(GetIOShape_t<LWFragA>::BlockHeight == GetIOShape_t<LRFragA>::BlockHeight,
                          "LW and LR frag heights do not match");

            // Heights should equal KDim
            static_assert(GetIOShape_t<LWFragA>::KDim == GetIOShape_t<LWFragB>::KDim,
                          "LW frag K dims do not match");
            static_assert(GetIOShape_t<LRFragA>::KDim == GetIOShape_t<LRFragB>::KDim,
                          "LR frag K dims do not match");
            static_assert(GetIOShape_t<LWFragA>::KDim == GetIOShape_t<LRFragA>::KDim,
                          "LW and LR frag K dims do not match");

            // Finally, height should equal KDim
            static_assert(GetIOShape_t<LWFragA>::BlockHeight == GetIOShape_t<LWFragB>::KDim,
                          "Frag height does not equal K dim");

        private:
            constexpr static uint32_t LdsHeight = GetIOShape_t<LWFragA>::BlockHeight;

        public: // Implicit interface for local mapping object
            // Offset of the current wave in the LDS macro tile
            __device__ constexpr static inline auto waveOffsetA();
            __device__ constexpr static inline auto waveOffsetB();

            // Block offset between local mfma fragments
            __device__ constexpr static inline auto blockOffsetA();
            __device__ constexpr static inline auto blockOffsetB();

            // The base lds write / read coordinates
            __device__ constexpr static inline auto writeCoordA();
            __device__ constexpr static inline auto writeCoordB();

            __device__ constexpr static inline auto readCoordA();
            __device__ constexpr static inline auto readCoordB();

            // Dimensions of shared memory usage
            __device__ constexpr static inline auto sizeLds();

            // Leading dimension of lds matrix
            __device__ constexpr static inline auto ldLds();

            template <uint32_t WaveCount = 1>
            __device__ constexpr static inline auto
                formatLWFragA(typename GlobalMapping::GRFragA const& grFragA)
            {
                return rocwmma::template applyDataLayout<LayoutLds, WaveCount>(
                    applyTranspose(grFragA));
            }

            template <uint32_t WaveCount = 1>
            __device__ constexpr static inline auto
                formatLWFragB(typename GlobalMapping::GRFragB const& grFragB)
            {
                return rocwmma::template applyDataLayout<LayoutLds, WaveCount>(grFragB);
            }

            __device__ constexpr static inline auto
                formatLRFragA(LRFragA const& lrFragA)
            {
                using MfmaLayout = typename GetDataLayout_t<typename GlobalMapping::MfmaFragA>::Orientation;
                return rocwmma::template applyDataLayout<MfmaLayout, 1u>(applyTranspose(lrFragA));
            }

            __device__ constexpr static inline auto
                formatLRFragB(LRFragB const& lrFragB)
            {
                using MfmaLayout = typename GetDataLayout_t<typename GlobalMapping::MfmaFragB>::Orientation;
                return rocwmma::template applyDataLayout<MfmaLayout, 1u>(lrFragB);
            }
        };

        template <typename GlobalMapping, typename LayoutLds>
        struct LdsMappingNT
        {
            /* LdsMappingNT (Block Width = LDS Width = BlockK)
            * Matrix geometry for A and B have a common dimension (BlockK).
            * We can fix one of the LDS dimensions to BlockK (in this case the width),
            * and insert blocks of different heights (BlockM, BlockN) to use the space
            * without the need of extra padding.
            *
            * Fragments of B must be transposed to fit this geometry,
            * and both fragments from A and B must accomodate LDS data layout.
            *
            *
            * Local Layout (LDS):
            *
            * Non - transposed A fragments [A0 ... AX-1] are placed first and occupy a total height
            * of Macro Tile X, where X = number of A blocks and Ck is the kth column of the A block.
            *
            * B fragments [B0 (T) ... BY-1 (T)] follow A fragments and occupy a total height of
            * Macro Tile Y, where Y = number of B blocks, and Rk is the kth row of the B block.
            *
            *
            *                        _____________BlockK_____________
            *                       |                                |
            *                       v                                v
            *                  (0,0) ----------------------------------->
            *          -->       -->  ______________    ...        ______
            *          |         |   |    |    |                  |      |
            *          |         |   |    |    |                  |      |
            *  Macro   |  BlockM |   | C0 | C1 | C2               | Ck-1 |   A0
            *  Tile X  |         |   |    |    |                  |      |
            *          |         --> |___ |___ |____    ...       |______|
            *          |
            *          |                    ...  ...  ...  ...          AX-1
            *          -->
            *          -->       -->  ______________    ...        ______
            *          |         |   |    |    |                  |      |
            *          |         |   |    |    |                  |      |
            *  Macro   |  BlockN |   | R0 | R1 | R2               | Rk-1 |   B0 (T)
            *  Tile Y  |         |   |    |    |                  |      |
            *          |         --> |___ |___ |____    ...       |______|
            *          |
            *          |                    ...  ...  ...  ...        BY-1 (T)
            *          -->                                           (MacroTileX + MacroTileY - 1, BlockK -1)
            *
            * TLDR: Take the Global Read fragments, transpose B and write the resulting frags into LDS
            * stacked on top of each other using BlockK as common width.
            */

            using DataLayout = DataLayout::Array1d<LayoutLds>;

            /// LOCAL WRITE -> GR frags
            // K = BlockWidth
            // GRFragA unchanged
            // GRFragB transposed
            // Ensure LW frags are LayoutLds friendly.
            using LWFragA = ApplyDataLayout_t<typename GlobalMapping::GRFragA, LayoutLds>;
            using LWFragB
                = ApplyDataLayout_t<ApplyTranspose_t<typename GlobalMapping::GRFragB>, LayoutLds>;

            /// LOCAL READ -> Mfma frags
            // MfmaFragA unchanged
            // MfmaFragB transposed
            // Ensure LR frags are LayoutLds friendly.
            using LRFragA = ApplyDataLayout_t<typename GlobalMapping::MfmaFragA, LayoutLds>;
            using LRFragB
                = ApplyDataLayout_t<ApplyTranspose_t<typename GlobalMapping::MfmaFragB>, LayoutLds>;

            // Sanity check:
            // All local R/W tiles should have same width
            static_assert(GetIOShape_t<LWFragA>::BlockWidth == GetIOShape_t<LWFragB>::BlockWidth,
                          "LW frag widths do not match");
            static_assert(GetIOShape_t<LRFragA>::BlockWidth == GetIOShape_t<LRFragB>::BlockWidth,
                          "LR frag widths do not match");
            static_assert(GetIOShape_t<LWFragA>::BlockWidth == GetIOShape_t<LRFragA>::BlockWidth,
                          "LW and LR frag widths do not match");

            // Heights should equal KDim
            static_assert(GetIOShape_t<LWFragA>::KDim == GetIOShape_t<LWFragB>::KDim,
                          "LW frag K dims do not match");
            static_assert(GetIOShape_t<LRFragA>::KDim == GetIOShape_t<LRFragB>::KDim,
                          "LR frag K dims do not match");
            static_assert(GetIOShape_t<LWFragA>::KDim == GetIOShape_t<LRFragA>::KDim,
                          "LW and LR frag K dims do not match");

            // Finally, height should equal KDim
            static_assert(GetIOShape_t<LWFragA>::BlockWidth == GetIOShape_t<LWFragB>::KDim,
                          "Frag width does not equal K dim");

        private:
            constexpr static uint32_t LdsWidth = GetIOShape_t<LWFragA>::BlockWidth;

        public: // Implicit interface for local mapping object
            // Offset of the current wave in the LDS macro tile
            __device__ constexpr static inline auto waveOffsetA();
            __device__ constexpr static inline auto waveOffsetB();

            // Block offset between local mfma fragments
            __device__ constexpr static inline auto blockOffsetA();
            __device__ constexpr static inline auto blockOffsetB();

            // The base lds write / read coordinates
            __device__ constexpr static inline auto writeCoordA();
            __device__ constexpr static inline auto writeCoordB();

            __device__ constexpr static inline auto readCoordA();
            __device__ constexpr static inline auto readCoordB();

            // Dimensions of shared memory usage
            __device__ constexpr static inline auto sizeLds();

            // Leading dimension of shared memory usage
            __device__ constexpr static inline auto ldLds();

            template <uint32_t WaveCount = 1>
            __device__ constexpr static inline auto
                formatLWFragA(typename GlobalMapping::GRFragA const& grFragA)
            {
                return applyDataLayout<LayoutLds, WaveCount>(grFragA);
            }

            template <uint32_t WaveCount = 1>
            __device__ constexpr static inline auto
                formatLWFragB(typename GlobalMapping::GRFragB const& grFragB)
            {
                return applyDataLayout<LayoutLds, WaveCount>(applyTranspose(grFragB));
            }

            __device__ constexpr static inline auto
                formatLRFragA(LRFragA const& lrFragA)
            {
                using MfmaLayout = typename GetDataLayout_t<typename GlobalMapping::MfmaFragA>::Orientation;
                return rocwmma::template applyDataLayout<MfmaLayout, 1u>(lrFragA);
            }

            __device__ constexpr static inline auto
                formatLRFragB(LRFragB const& lrFragB)
            {
                using MfmaLayout = typename GetDataLayout_t<typename GlobalMapping::MfmaFragB>::Orientation;
                return rocwmma::template applyDataLayout<MfmaLayout, 1u>(applyTranspose(lrFragB));
            }
        };

        template <typename GlobalMapping, typename LayoutLds>
        struct LdsMappingRF
        {
            /*
            * This is a helper class to transform global A and B fragments such that they can
            * fit together in a common LDS space without the use of extra padding.
            * We use the fragments defined in the GlobalMapping argument.
            *
            * LdsMappingRF (Block Width = LDS Width = AMDGCN_WAVE_SIZE = 64)
            *
            * We can fix the width of the LDS storage to AMDGCN_WAVE_SIZE, and transform
            * fragments of A and B to fit without extra padding.
            *
            * Fragments of A must be transformed to fit this geometry, essentially
            * treating all fragments directly as a register file of AMDGCN_WAVE_SIZE elements wide.
            *
            * Local Matrix Layout (LDS):
            *
            * A fragments [A0 ... AX-1]  are placed first and occupy a total width
            * of RegCntA * X, where:
            *   RegCntA = BlockM * BlockK / AMDGCN_WAVE_SIZE
            *   X = number of A blocks,
            *   Rk = the kth register of the A block.
            *
            * B fragments [B0 ... BY-1] follow A fragments and occupy a total width of
            * of RegCntB * Y, where:
            *   RegCntB = BlockY * BlockK / AMDGCN_WAVE_SIZE
            *   Y = number of B blocks,
            *   Rk = the kth register of the B block.
            *
            * NOTE: This format is only MFMA friendly for block-wise global read formats.
            * E.g. Global read must read block tiles and not combined wave or macro tiles.
            *
            *
            *                          |--- AMDGCN_WAVE_SIZE ---|
            *
            *         -->   (0,0)  -->  ______________  ...  ___
            *         |            |   |__________R0__  ...  ___|
            *  RegCntA|            |   |__________R1__  ...  ___|
            *     *   |    RegCntA |   |__________R2__  ...  ___|    A0
            *     X   |            |   |          ...   ...     |
            *         |            |   |__________Rk-1  ...  ___|
            *         |            -->
            *         |
            *         |               ...       ...        ...     AX-1
            *         -->
            *         -->          -->  ______________  ...  ___
            *         |            |   |__________R0__  ...  ___|
            * RegCntB |            |   |__________R1__  ...  ___|
            *    *    |    RegCntB |   |__________R2__  ...  ___|    B0
            *    Y    |            |   |          ...   ...     |
            *         |            |   |__________Rk-1  ...  ___|
            *         |            -->
            *         |                 ...       ...        ...     BY-1
            *         -->
            *
            * TLDR: Take the Global Read fragments, write their register file contents to LDS
            * and read them back. Only works with MFMA friendly fragments.
            */

            using DataLayout = DataLayout::Array1d<LayoutLds>;

            /// LOCAL WRITE -> GR frags (MFMA blocks)
            // LDS width = AMDGCN_WAVE_SIZE
            // GRFragA transformed to register file
            // GRFragB transformed to register file
            // Ensure LW frags are LayoutLds friendly.
            using LWFragA = ApplyDataLayout_t<ApplyRegisterFile_t<typename GlobalMapping::GRFragA>,
                                              LayoutLds>;
            using LWFragB = ApplyDataLayout_t<ApplyRegisterFile_t<typename GlobalMapping::GRFragB>,
                                              LayoutLds>;

            /// LOCAL READ -> Mfma frags
            // Same as LWFrags
            using LRFragA = LWFragA;
            using LRFragB = LWFragB;

            /// Sanity checks:
            // All local R/W tiles should have same width
            static_assert(GetIOShape_t<LWFragA>::BlockWidth == GetIOShape_t<LWFragB>::BlockWidth,
                          "LW frag widths do not match");
            static_assert(GetIOShape_t<LRFragA>::BlockWidth == GetIOShape_t<LRFragB>::BlockWidth,
                          "LR frag widths do not match");
            static_assert(GetIOShape_t<LWFragA>::BlockWidth == Constants::AMDGCN_WAVE_SIZE,
                          "LW and LR frag widths do not match register element count");

            // This layout is only valid if global reads are MFMA friendly, due to register file transform.
            // In-register layout for MFMA blocks is not the same as wave tile or macro tile.
            static_assert(std::is_same<typename GlobalMapping::GRFragA,
                                       typename GlobalMapping::MfmaFragA>::value,
                          "GR A block must be MFMA size");
            static_assert(std::is_same<typename GlobalMapping::GRFragB,
                                       typename GlobalMapping::MfmaFragB>::value,
                          "GR B block must be MFMA size");

        private: // Coordinate projection helpers
            constexpr static uint32_t LdsWidth = Constants::AMDGCN_WAVE_SIZE;

            // Project coordinates into stacked register file space
            __device__ constexpr static inline auto projCoordA(Coord2d const& coordA);
            __device__ constexpr static inline auto projCoordB(Coord2d const& coordB);

        public: // Implicit interface for local mapping object
            // Offset of the current wave in the LDS macro tile
            __device__ constexpr static inline auto waveOffsetA();
            __device__ constexpr static inline auto waveOffsetB();

            // Block offset between local mfma fragments
            __device__ constexpr static inline auto blockOffsetA();
            __device__ constexpr static inline auto blockOffsetB();

            // The base lds write / read coordinates
            __device__ constexpr static inline auto writeCoordA();
            __device__ constexpr static inline auto writeCoordB();

            __device__ constexpr static inline auto readCoordA();
            __device__ constexpr static inline auto readCoordB();

            // Dimensions of shared memory usage
            __device__ constexpr static inline auto sizeLds();

            // Leading dimension of lds matrix
            __device__ constexpr static inline auto ldLds();

            template <uint32_t WaveCount = 1>
            __device__ constexpr static inline auto
                formatLWFragA(typename GlobalMapping::GRFragA const& grFragA)
            {
                // When interpreting as a register block (e.g. BlockDim 64 on CDNA), avoid transforming
                // direct register layouts because we want the register contents as they currently are.
                return reinterpret_cast<
                    ApplyDataLayout_t<ApplyRegisterFile_t<typename GlobalMapping::GRFragA>,
                                      LayoutLds> const&>(grFragA);
            }

            template <uint32_t WaveCount = 1>
            __device__ constexpr static inline auto
                formatLWFragB(typename GlobalMapping::GRFragB const& grFragB)
            {
                // When interpreting as a register block (e.g. BlockDim 64 on CDNA), avoid transforming
                // direct register layouts because we want the register contents as they currently are.
                return reinterpret_cast<
                    ApplyDataLayout_t<ApplyRegisterFile_t<typename GlobalMapping::GRFragB>,
                                      LayoutLds> const&>(grFragB);
            }

            __device__ constexpr static inline auto
                formatLRFragA(LRFragA const& lrFragA)
            {
                using MfmaLayout = typename GetDataLayout_t<typename GlobalMapping::MfmaFragA>::Orientation;
                return rocwmma::template applyDataLayout<MfmaLayout, 1u>(lrFragA);
            }

            __device__ constexpr static inline auto
                formatLRFragB(LRFragB const& lrFragB)
            {
                using MfmaLayout = typename GetDataLayout_t<typename GlobalMapping::MfmaFragB>::Orientation;
                return rocwmma::template applyDataLayout<MfmaLayout, 1u>(lrFragB);
            }
        };

    } // namespace LocalMapping

} // namespace rocwmma

#include "gemm_local_mapping_impl.hpp"

#endif // GEMM_LOCAL_MAPPING_HPP
