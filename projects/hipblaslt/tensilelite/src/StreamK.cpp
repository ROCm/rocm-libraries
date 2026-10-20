/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <Tensile/StreamK.hpp>

#include <Tensile/MagicDivision.hpp>
#include <Tensile/Utils.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>

namespace TensileLite
{
    StreamK::Static::Split StreamK::Static::computeSplit(size_t tiles,
                                                         size_t itersPerTile,
                                                         size_t skGrid,
                                                         int    skFullTiles,
                                                         bool   forceDPOnly)
    {
        Split split;
        if(skGrid == 0)
            return split;

        // Two-tile algorithm: each workgroup runs an even number of Stream-K
        // iterations followed by an even number of data-parallel tiles. When
        // the grid divides the tile count no Stream-K tiles are needed at all
        // and every tile is data-parallel. Force-DP-only is a persistent
        // DP-only use of StreamK=3: skTiles of zero keeps every output tile in
        // the DP region.
        const bool bigEnough = tiles > skGrid;
        uint32_t   skTiles   = forceDPOnly ? 0u : static_cast<uint32_t>(skGrid);
        if(!forceDPOnly && tiles % skGrid != 0)
        {
            skTiles = bigEnough ? static_cast<uint32_t>(skGrid * skFullTiles + tiles % skGrid)
                                : static_cast<uint32_t>(tiles);
            // Cap Stream-K tiles at total number of tiles in case of large multiplier
            skTiles = std::min(skTiles, static_cast<uint32_t>(tiles));
        }

        split.args.skTiles    = skTiles;
        split.args.itersPerWG = static_cast<uint32_t>(skTiles * itersPerTile / skGrid);
        // Global leftover under the historical first-E mapping. The device may
        // redistribute these extras within each tile when
        // InternalArgsSupport::perTileExtraIters is set and skGrid % skTiles == 0;
        // the packed value itself is unchanged.
        split.extraIters = static_cast<uint32_t>(
            static_cast<size_t>(skTiles) * itersPerTile
            - static_cast<size_t>(split.args.itersPerWG) * skGrid);
        return split;
    }

    StreamK::Static::KernelArguments
        StreamK::Static::calculateKernelArgs(size_t               numTiles,
                                             uint32_t             itersPerTile,
                                             size_t               grid,
                                             origami::reduction_t reduction,
                                             int                  skFullTiles,
                                             bool                 uniformSummationOrderActive)
    {
        KernelArguments streamk3;
        if(reduction == origami::reduction_t::parallel)
        {
            const uint32_t skSplit = static_cast<uint32_t>(grid / numTiles);
            streamk3.skTiles       = skSplit;
            streamk3.itersPerWG    = itersPerTile / std::max(skSplit, uint32_t(1));
        }
        else
        {
            const Split split = computeSplit(numTiles, itersPerTile, grid, skFullTiles, false);
            streamk3.skTiles    = split.args.skTiles;
            streamk3.itersPerWG = split.args.itersPerWG;
        }

        uint32_t magicShift  = 0;
        streamk3.magicNumber = magicNumber(2, itersPerTile, &magicShift);
        assert((magicShift & Slot2Bits::ShiftReservedMask) == 0u);

        if(uniformSummationOrderActive)
            magicShift |= Slot2Bits::UniformSummationOrderBit;
        streamk3.magicShift = magicShift;
        streamk3.grid       = static_cast<uint32_t>(grid);
        return streamk3;
    }

    bool StreamK::Static::isRowUniform(Split const& split,
                                       size_t       tiles,
                                       size_t       itersPerTile,
                                       size_t       skGrid,
                                       bool         perTileCapable,
                                       bool         uniformSummationOrder)
    {
        // The kernel branches at runtime on the uniform-summation-order bit the
        // host packs into MagicShiftItersPerTile: with the bit clear it runs the
        // historical global first-E mapping even when it supports per-tile
        // extras. Model both terms, or host and device disagree about which tile
        // a workgroup owns.
        const bool perTileActive = perTileCapable && uniformSummationOrder;
        // A tile's fold signature is the ordered list of chunk lengths whose
        // partials are summed to produce it, and two tiles are bitwise equal
        // for identical inputs exactly when their signatures match. Every tile
        // in the launch must therefore share one signature. Ways that happens:
        //
        //  skTiles == 0        force-DP-only, so every tile is whole.
        //  skTiles == tiles && I % skItersPerWG == 0 && extraIters == 0
        //                      no data-parallel region. Includes GridEqualsTiles
        //                      (skItersPerWG == I) and all-partial equal chunks
        //                      (tiles | grid and (grid/tiles) | I).
        //                      Mixed GridDividesTiles (skTiles == grid < tiles,
        //                      skItersPerWG == I) is two-tile DP-first then SK.
        //                      gfx950 skips tree-partials workspace when
        //                      tiles % grid == 0, and the SK half of D is never
        //                      stored (uso-row-sweep C1: 992/1024 rows stay
        //                      poison). Refuse that split until the device path
        //                      writes every tile.
        //  per-tile extras: skTiles == tiles && grid % tiles == 0 with a kernel
        //                      that redistributes extras within each tile.
        //                      extraIters may be nonzero; fold signatures still
        //                      match because each tile gets the same intra-tile
        //                      remainder pattern.
        //
        // Without perTileActive, extraIters != 0 means chunks come in two
        // lengths under the global first-E mapping, so the chunk lattice has no
        // single period. skItersPerWG != 0 is not implied by the rest: tiles == 0
        // is reachable from the grouped-GEMM callers and would otherwise leave
        // I % skItersPerWG undefined.
        if(split.args.skTiles == 0)
            return true;

        if(split.args.itersPerWG != 0 && split.extraIters == 0 && split.args.skTiles == tiles
           && itersPerTile % split.args.itersPerWG == 0)
            return true;

        if(perTileActive && split.args.skTiles == tiles && tiles != 0 && skGrid % tiles == 0
           && split.args.itersPerWG != 0)
            return true;

        return false;
    }

    StreamK::Static::IterRange StreamK::Static::workgroupIterRange(size_t w,
                                                                   size_t tiles,
                                                                   size_t itersPerTile,
                                                                   size_t skGrid,
                                                                   bool   perTileCapable,
                                                                   bool   uniformSummationOrder)
    {
        IterRange range;
        if(skGrid == 0 || tiles == 0)
            return range;

        const size_t totalIters = tiles * itersPerTile;
        const size_t W          = totalIters / skGrid;
        const size_t E          = totalIters - W * skGrid;

        // The device takes the per-tile branch only when the kernel supports it
        // AND the packed uniform-summation-order bit is set. Mirror both terms,
        // or the host attributes iterations to the wrong workgroup and the fixup
        // reads the wrong partials.
        const bool perTileActive = perTileCapable && uniformSummationOrder;

        if(perTileActive && skGrid % tiles == 0)
        {
            const size_t F    = skGrid / tiles;
            const size_t q    = w / F;
            const size_t s    = w % F;
            const size_t remI = itersPerTile % F;
            const size_t base = itersPerTile / F;
            range.start       = q * itersPerTile + s * base + std::min(s, remI);
            range.end = range.start + base + (s < remI ? size_t{1} : size_t{0});
            return range;
        }

        // Historical global first-E mapping.
        if(w < E)
        {
            range.start = w * (W + 1);
            range.end   = range.start + (W + 1);
        }
        else
        {
            range.start = E * (W + 1) + (w - E) * W;
            range.end   = range.start + W;
        }
        return range;
    }

    StreamK::Dynamic::KernelArguments
        StreamK::Dynamic::calculateKernelArgs(size_t   numTiles,
                                              uint32_t itersPerTile,
                                              size_t   grid,
                                              int      skSplitOverride,
                                              int      skTilesOverride)
    {
        constexpr uint32_t defaultSplit = 2;

        uint32_t split
            = skSplitOverride > -1 ? static_cast<uint32_t>(skSplitOverride) : defaultSplit;
        uint32_t tiles = skTilesOverride > -1 ? static_cast<uint32_t>(skTilesOverride) : 0;

        const uint32_t itersPerWI = CeilDivide(itersPerTile, split);
        split                     = CeilDivide(itersPerTile, itersPerWI);
        KernelArguments streamk4;
        streamk4.totalItems = static_cast<uint32_t>(numTiles - tiles) + tiles * split;
        streamk4.skTiles    = tiles;
        streamk4.skSplit    = split;
        streamk4.itersPerWI = itersPerWI;
        streamk4.grid       = static_cast<uint32_t>(grid);
        return streamk4;
    }

    bool StreamK::isParallelReductionRowUniform(origami::reduction_t reduction,
                                                size_t               grid,
                                                int                  streamKAtomic,
                                                bool                 staticTwoTilePacking,
                                                size_t               tiles)
    {
        // Parallel Stream-K under static two-tile packing maps each workgroup
        // to TileIdx = StreamKIdx // F and PartialIdx = StreamKIdx % F, with
        // F = grid/tiles. Every tile sees the same PartialIdx set and the same
        // per-partial K ranges (extras go by PartialIdx), so identical A rows
        // produce identical D rows when F >= 2 and grid is an exact multiple
        // of tiles. Atomic fixup and non-static ABIs are out of scope.
        if(reduction != origami::reduction_t::parallel)
            return false;
        if(streamKAtomic != 0)
            return false;
        if(!staticTwoTilePacking)
            return false;
        if(grid == 0 || tiles == 0)
            return false;
        if(grid % tiles != 0)
            return false;
        if((grid / tiles) < 2)
            return false;
        return true;
    }
}
