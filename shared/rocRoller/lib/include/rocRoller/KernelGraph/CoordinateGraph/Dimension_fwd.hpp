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

#include <variant>

namespace rocRoller
{
    namespace KernelGraph::CoordinateGraph
    {
        /*
         * Nodes (Dimensions)
         */

        struct ForLoop;
        struct Adhoc;
        struct ElementNumber;
        struct Lane;
        struct Linear;
        struct LDS;
        struct MacroTile;
        struct MacroTileIndex;
        struct MacroTileNumber;
        struct SubDimension;
        struct ThreadTile;
        struct ThreadTileIndex;
        struct ThreadTileNumber;
        struct Unroll;
        struct User;
        struct VGPR;
        struct VGPRBlockNumber;
        struct VGPRBlockIndex;
        struct WaveTile;
        struct WaveTileIndex;
        struct WaveTileNumber;
        struct JammedWaveTileNumber;
        struct Wavefront;
        struct Workgroup;
        struct Workitem;

        using Dimension = std::variant<ForLoop,
                                       Adhoc,
                                       ElementNumber,
                                       Lane,
                                       LDS,
                                       Linear,
                                       MacroTile,
                                       MacroTileIndex,
                                       MacroTileNumber,
                                       SubDimension,
                                       ThreadTile,
                                       ThreadTileIndex,
                                       ThreadTileNumber,
                                       Unroll,
                                       User,
                                       VGPR,
                                       VGPRBlockNumber,
                                       VGPRBlockIndex,
                                       WaveTile,
                                       WaveTileIndex,
                                       WaveTileNumber,
                                       JammedWaveTileNumber,
                                       Wavefront,
                                       Workgroup,
                                       Workitem>;

        template <typename T>
        concept CDimension = std::constructible_from<Dimension, T>;

        template <typename T>
        concept CConcreteDimension = (CDimension<T> && !std::same_as<Dimension, T>);
    }
}
