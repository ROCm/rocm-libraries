/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
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

#pragma once

#include <vector>

#include <rocRoller/TensorDescriptor.hpp>

#include <client/GEMMParameters.hpp>

namespace rocRoller::Client
{
    /**
     * @brief Pre-swizzle and optionally pre-tile the input.
     *
     * This assumes that the incoming TensorDescriptor `desc` has been
     * created with `withNormalizedDimensions()`.  That is, the
     * left-most dimension (the 0 dimension) is the fastest dimension
     * (has the smallest stride).
     */
    template <typename T>
    inline std::vector<T> preSwizzle(std::vector<T> const&      input,
                                     TensorDescriptor const&    desc,
                                     std::vector<size_t> const& preSwizzleSize,
                                     std::vector<size_t> const& preTileSize)
    {
        if(not preSwizzleSize.empty())
        {
            AssertFatal(preSwizzleSize.size() == 3,
                        ShowValue(preSwizzleSize.size()),
                        ShowValue(preSwizzleSize));
        }
        AssertFatal(desc.dimensions() == 2,
                    "Batch dimension not yet supported.",
                    ShowValue(desc.dimensions()),
                    ShowValue(desc));
        AssertFatal(desc.totalAllocatedElements() == input.size(),
                    ShowValue(desc),
                    ShowValue(input.size()));

        std::vector<size_t> srcSizes, dimOrder;

        if((not preSwizzleSize.empty()) && (preTileSize.empty()))
        {
            auto tileMN   = preSwizzleSize[0];
            auto tileK    = preSwizzleSize[1];
            auto subTileK = preSwizzleSize[2];

            AssertFatal(tileMN == 64 || tileMN == 32, ShowValue(tileMN));
            AssertFatal(tileK % 4 == 0, ShowValue(tileK));

            size_t nLanesPerSIMD   = 16;
            size_t nSIMDsPerWave   = 4;
            size_t nSIMDIndex      = tileMN / nLanesPerSIMD;
            size_t nSIMDBlock      = nSIMDsPerWave / nSIMDIndex;
            size_t nVGPRIndex      = std::min(nSIMDIndex, subTileK);
            size_t nVGPRBlock      = tileK / nSIMDBlock / nVGPRIndex;
            size_t nSIMDIndexBlock = nVGPRIndex;
            size_t nSIMDIndexIndex = nSIMDIndex / nSIMDIndexBlock;

            AssertFatal(nVGPRIndex * nVGPRBlock * nSIMDBlock == tileK);
            AssertFatal(nLanesPerSIMD * nSIMDIndexIndex * nSIMDIndexBlock == tileMN);

            srcSizes = {nVGPRIndex,
                        nVGPRBlock,
                        nSIMDBlock,
                        desc.size(0) / (tileK),
                        nLanesPerSIMD,
                        nSIMDIndexIndex,
                        nSIMDIndexBlock,
                        desc.size(1) / (tileMN)};

            if(tileMN == 64)
            {
                // Pre swizzle: swap nSIMDIndexBlock (6) and nVGPRIndex (0)
                dimOrder = {6, 1, 2, 3, 4, 5, 0, 7};
            }
            else if(tileMN == 32 && subTileK == 4)
            {
                // Pre swizzle: swap nSIMDIndexBlock (6) and nVGPRIndex (0)
                //              swap nSIMDBlock (2) and nVGPRBlock (1)
                dimOrder = {6, 2, 1, 3, 4, 5, 0, 7};
            }
            else if(tileMN == 32 && subTileK == 2)
            {
                // Pre swizzle: rotate nVGPRIndex (0), nVGPRBlock (1), nSIMDBlock (2)
                dimOrder = {1, 2, 0, 3, 4, 5, 6, 7};
            }
        }
        else if((preSwizzleSize.empty()) && (not preTileSize.empty()))
        {
            srcSizes = {preTileSize[0],
                        desc.size(0) / preTileSize[0],
                        preTileSize[1],
                        desc.size(1) / preTileSize[1]};

            // Pre-tiling: 1 and 3 are pushed to the back (they become the slowest)
            dimOrder = {0, 2, 1, 3};
        }
        else
        {
            auto tileMN   = preSwizzleSize[0];
            auto tileK    = preSwizzleSize[1];
            auto subTileK = preSwizzleSize[2];

            AssertFatal(tileMN == 64 || tileMN == 32, ShowValue(tileMN));
            AssertFatal(tileK % 4 == 0, ShowValue(tileK));

            size_t wgTileSizeK     = preTileSize[0];
            size_t wgTileSizeMN    = preTileSize[1];
            size_t nLanesPerSIMD   = 16;
            size_t nSIMDsPerWave   = 4;
            size_t nSIMDIndex      = tileMN / nLanesPerSIMD;
            size_t nSIMDBlock      = nSIMDsPerWave / nSIMDIndex;
            size_t nVGPRIndex      = std::min(nSIMDIndex, subTileK);
            size_t nVGPRBlock      = tileK / nSIMDBlock / nVGPRIndex;
            size_t nSIMDIndexBlock = nVGPRIndex;
            size_t nSIMDIndexIndex = nSIMDIndex / nSIMDIndexBlock;

            AssertFatal(nVGPRIndex * nVGPRBlock * nSIMDBlock == tileK);
            AssertFatal(nLanesPerSIMD * nSIMDIndexIndex * nSIMDIndexBlock == tileMN);

            srcSizes = {nVGPRIndex,
                        nVGPRBlock,
                        nSIMDBlock,
                        wgTileSizeK / tileK,
                        desc.size(0) / wgTileSizeK,
                        nLanesPerSIMD,
                        nSIMDIndexIndex,
                        nSIMDIndexBlock,
                        wgTileSizeMN / tileMN,
                        desc.size(1) / wgTileSizeMN};

            if(tileMN == 64)
            {
                // Pre swizzle: swap nSIMDIndexBlock (7) and nVGPRIndex (0)
                // Pre tile: push workgroup tiles (4 and 9) to the end
                dimOrder = {7, 1, 2, 3, 5, 6, 0, 8, 4, 9};
            }
            else if(tileMN == 32 && subTileK == 4)
            {
                // Pre swizzle: swap nSIMDIndexBlock (7) and nVGPRIndex (0)
                //              swap nSIMDBlock (2) and nVGPRBlock (1)
                // Pre tile: push workgroup tiles (4 and 9) to the end
                dimOrder = {7, 2, 1, 3, 5, 6, 0, 8, 4, 9};
            }
            else if(tileMN == 32 && subTileK == 2)
            {
                // Pre swizzle: rotate nVGPRIndex (0), nVGPRBlock (1), nSIMDBlock (2)
                // Pre tile: push workgroup tiles (4 and 9) to the end
                dimOrder = {1, 2, 0, 3, 5, 6, 7, 8, 4, 9};
            }
        }

        AssertFatal(product(srcSizes) == product(desc.sizes()), "PreSwizzle size mismatch.");

        AssertFatal(not srcSizes.empty(), "PreSwizzle source size not populated.");
        AssertFatal(not dimOrder.empty(), "PreSwizzle permutation order not populated.");

        TensorDescriptor src(desc.dataType(), srcSizes);

        AssertFatal(src.totalAllocatedElements() == desc.totalAllocatedElements(),
                    ShowValue(src.totalAllocatedElements()),
                    ShowValue(desc.totalAllocatedElements()),
                    ShowValue(src.totalAllocatedElements() / desc.totalAllocatedElements()),
                    ShowValue(src),
                    ShowValue(desc));

        auto dst = TensorDescriptor::ShuffledNoPadding(desc.dataType(), srcSizes, dimOrder);

        AssertFatal(src.totalAllocatedElements() == dst.totalAllocatedElements(),
                    ShowValue(src.totalAllocatedElements()),
                    ShowValue(dst.totalAllocatedElements()),
                    ShowValue(src),
                    ShowValue(dst));

        return shuffleDims(input, dst, src);
    }

}
