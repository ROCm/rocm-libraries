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

    template <typename T>
    inline std::vector<T> preSwizzle(std::vector<T> const&      input,
                                     TensorDescriptor const&    desc,
                                     std::vector<size_t> const& tile)
    {
        AssertFatal(tile.size() == 3, ShowValue(tile.size()), ShowValue(tile));
        AssertFatal(desc.dimensions() == 2,
                    "Batch dimension not yet supported.",
                    ShowValue(desc.dimensions()),
                    ShowValue(desc));
        AssertFatal(desc.totalAllocatedElements() == input.size(),
                    ShowValue(desc),
                    ShowValue(input.size()));

        auto tileMN   = tile[0];
        auto tileK    = tile[1];
        auto subTileK = tile[2];

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

        std::vector<size_t> srcSizes = {nVGPRIndex,
                                        nVGPRBlock,
                                        nSIMDBlock,
                                        desc.size(0) / (tileK),
                                        nLanesPerSIMD,
                                        nSIMDIndexIndex,
                                        nSIMDIndexBlock,
                                        desc.size(1) / (tileMN)};

        TensorDescriptor src(desc.dataType(), srcSizes), dst;

        AssertFatal(src.totalAllocatedElements() == desc.totalAllocatedElements(),
                    ShowValue(src.totalAllocatedElements()),
                    ShowValue(desc.totalAllocatedElements()),
                    ShowValue(src.totalAllocatedElements() / desc.totalAllocatedElements()),
                    ShowValue(src),
                    ShowValue(desc));

        std::vector<size_t> dimOrder;

        if(tileMN == 64)
        {
            dimOrder = {6, 1, 2, 3, 4, 5, 0, 7};
        }
        else if(tileMN == 32 && subTileK == 4)
        {
            dimOrder = {6, 2, 1, 3, 4, 5, 0, 7};
        }
        else if(tileMN == 32 && subTileK == 2)
        {
            dimOrder = {1, 2, 0, 3, 4, 5, 6, 7};
        }

        AssertFatal(!dimOrder.empty(), "pre-swizzle permutation order not populated");

        dst = TensorDescriptor::ShuffledNoPadding(desc.dataType(), srcSizes, dimOrder);

        AssertFatal(src.totalAllocatedElements() == dst.totalAllocatedElements(),
                    ShowValue(src.totalAllocatedElements()),
                    ShowValue(dst.totalAllocatedElements()),
                    ShowValue(src),
                    ShowValue(dst));

        return shuffleDims(input, dst, src);
    }

}
