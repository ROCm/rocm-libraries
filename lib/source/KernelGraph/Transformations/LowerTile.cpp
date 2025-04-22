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

#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/CodeGen/Utils.hpp>
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/CommandSolution_fwd.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/Transforms/LowerTile.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Operations/Operations.hpp>
#include <rocRoller/Utilities/Logging.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        namespace CT = rocRoller::KernelGraph::CoordinateGraph;
        using namespace ControlGraph;
        using namespace CoordinateGraph;
        using namespace Expression;
        using namespace Register;

        ConstraintStatus NoConstructDestructMT(const KernelGraph& k)
        {
            ConstraintStatus retval;
            for(auto element : k.coordinates.getEdges<CoordinateTransformEdge>())
            {
                auto dmt = k.coordinates.get<DestructMacroTile>(element);
                if(dmt)
                {
                    retval.combine(
                        false,
                        concatenate(
                            "DestructMacroTile Coordinate Edge Still Exists: ", element, "."));
                }
                else
                {
                    auto cmt = k.coordinates.get<ConstructMacroTile>(element);
                    if(cmt)
                    {
                        retval.combine(
                            false,
                            concatenate(
                                "ConstructMacroTile Coordinate Edge Still Exists: ", element, "."));
                    }
                }
            }
            return retval;
        }

        void setIsTransposedLoad(int tileTag, KernelGraph& graph)
        {
            auto conns = graph.mapper.getCoordinateConnections(tileTag);
            AssertFatal(conns.size() == 1,
                        "Macrotile(",
                        tileTag,
                        ") is connected to more than 1 operation in control graph!");
            auto opTag = conns[0].control;
            auto e     = graph.control.getElement(opTag);
            std::visit(rocRoller::overloaded{[&](LoadTiled& op) {
                                                 auto newLoadTiled(op);
                                                 newLoadTiled.isTransposedTile = true;
                                                 graph.control.setElement(opTag, newLoadTiled);
                                             },
                                             [&](LoadLDSTile& op) {
                                                 auto newLoadLDSTile(op);
                                                 newLoadLDSTile.isTransposedTile = true;
                                                 graph.control.setElement(opTag, newLoadLDSTile);
                                             },
                                             [&](auto& op) {
                                                 Throw<FatalError>("Unexpected control node ",
                                                                   op.name(),
                                                                   "(",
                                                                   opTag,
                                                                   ") connected to MacroTile(",
                                                                   tileTag,
                                                                   ")");
                                             }},
                       std::get<Operation>(e));
        }

        /**
         * Note that load/store generator uses ElementNumber
         * coordinates like this:
         *
         *     m = getSize(getDimension<ElementNumber>(0))
         *     n = getSize(getDimension<ElementNumber>(1))
         *
         *     for i in range(m):
         *         for j in range(n):
         *             VGPR[i * n + j] = buffer[coordinateTransform(i, j)]
         *
         * Through the code below, the naming convention is:
         *
         *    ElementNumberX: getDimension<ElementNumber>(0)
         *    ElementNumberY: getDimension<ElementNumber>(1)
         *
         * Therefore, ElementNumberY is fast-moving and contiguous in
         * VGPR indexes.
         */

        /**
         * @brief Add coordinate-transforms for tiling two
         * SubDimension coordinates into macro number/index
         * coordinates.
         *
         * The geometry of the tiling is taken from the MacroTile
         * associated with `macTileTag`.
         *
         * Required (deferred) connections are appended to
         * `connections`.
         *
         * @return Tuple of: row MacroTileNumber, row MacroTileIndex,
         * column MacroTileNumber, column MacroTileIndex.
         */
        std::tuple<int, int, int, int>
            addLoadMacroTileCT(KernelGraph&                     graph,
                               std::vector<DeferredConnection>& connections,
                               int                              macTileTag,
                               std::vector<int> const&          sdim)
        {
            auto macTile   = graph.coordinates.getNode<MacroTile>(macTileTag);
            auto sdimX     = sdim[0];
            auto sdimY     = sdim[1];
            auto numTilesX = tileCeilDivide(graph.coordinates.get<SubDimension>(sdimX)->size,
                                            macTile.sizes[0]);

            auto numTilesY = tileCeilDivide(graph.coordinates.get<SubDimension>(sdimY)->size,
                                            macTile.sizes[1]);

            connections.push_back(DC<SubDimension>(sdimX, 0));
            connections.push_back(DC<SubDimension>(sdimY, 1));

            auto nMacX = graph.coordinates.addElement(macTile.tileNumber(0, numTilesX));
            auto nMacY = graph.coordinates.addElement(macTile.tileNumber(1, numTilesY));
            auto iMacX = graph.coordinates.addElement(macTile.tileIndex(0));
            auto iMacY = graph.coordinates.addElement(macTile.tileIndex(1));

            connections.push_back(DC<MacroTileNumber>(nMacX, 0));
            connections.push_back(DC<MacroTileNumber>(nMacY, 1));

            graph.coordinates.addElement(Tile(), {sdimX}, {nMacX, iMacX});
            graph.coordinates.addElement(Tile(), {sdimY}, {nMacY, iMacY});

            return {nMacX, iMacX, nMacY, iMacY};
        }

        std::tuple<int, int, int> addLoad1DMacroTileCT(KernelGraph&                     graph,
                                                       std::vector<DeferredConnection>& connections,
                                                       int                              macTileTag,
                                                       std::vector<int> const&          sdim)
        {
            auto macTile   = graph.coordinates.getNode<MacroTile>(macTileTag);
            auto sdimX     = sdim[0];
            auto sdimY     = sdim[1];
            auto numTilesX = tileCeilDivide(graph.coordinates.get<SubDimension>(sdimX)->size,
                                            macTile.sizes[0]);

            connections.push_back(DC<SubDimension>(sdimX, 0));
            connections.push_back(DC<SubDimension>(sdimY, 1));

            auto nMacX = graph.coordinates.addElement(macTile.tileNumber(0, numTilesX));
            auto iMacX = graph.coordinates.addElement(macTile.tileIndex(0));
            auto iMacY = graph.coordinates.addElement(macTile.tileIndex(1));

            connections.push_back(DC<MacroTileNumber>(nMacX, 0));

            graph.coordinates.addElement(Tile(), {sdimX}, {nMacX, iMacX});
            graph.coordinates.addElement(PassThrough(), {sdimY}, {iMacY});

            return {nMacX, iMacX, iMacY};
        }

        /**
         * @brief Add coordinate-transforms for loading a WaveTile
         * from row/column coordinates `iWaveX` and `iWaveY` for the
         * v_mfma_*_f8f6f4 instruction.
         *
         * The `lane` and `element` parameters are existing
         * coordinates corresponding to a Lane coordiante and VGPR
         * coordinate (which should be thought of as which
         * element/item is being addressed).  Each lane loads 32
         * elements.
         */
        void addLoadWaveTileCTF8F6F4(KernelGraph& graph,
                                     int          iWaveX,
                                     int          iWaveY,
                                     int          lane,
                                     int          element,
                                     uint         K,
                                     uint         bitsPerElement,
                                     int          wavefrontSize)

        {
            AssertFatal((K == 128 || K == 64) && (wavefrontSize == 64 || wavefrontSize == 32));

            uint M = 16 * 128 / K;

            uint const lanesPerSIMD = 16;
            uint const simdsPerWave = wavefrontSize / lanesPerSIMD;

            uint const simdsPerSGroup = M / lanesPerSIMD;
            uint const numVBlocks     = wavefrontSize == 64 ? 2 : (bitsPerElement == 8 ? 4 : 2);

            uint const elementsPerVGPRBlock = ((M * K) / wavefrontSize) / numVBlocks;

            auto SIMD = graph.coordinates.addElement(Adhoc("SIMD", literal(simdsPerWave), nullptr));
            auto laneInSIMD = graph.coordinates.addElement(Lane(literal(lanesPerSIMD), nullptr));

            auto simdBlockNumber = graph.coordinates.addElement(
                Adhoc("simdBlockNumber", literal(simdsPerWave / simdsPerSGroup), nullptr));
            auto simdBlockIndex = graph.coordinates.addElement(
                Adhoc("simdBlockIndex", literal(simdsPerSGroup), nullptr));

            auto elementBlockNumber
                = graph.coordinates.addElement(VGPRBlockNumber(literal(numVBlocks), nullptr));
            auto elementBlockIndex = graph.coordinates.addElement(
                VGPRBlockIndex(literal(elementsPerVGPRBlock), nullptr));

            graph.coordinates.addElement(Tile(), {iWaveX}, {simdBlockIndex, laneInSIMD});

            if(bitsPerElement == 8)
            {
                graph.coordinates.addElement(
                    Tile(), {iWaveY}, {elementBlockNumber, simdBlockNumber, elementBlockIndex});
            }
            else
            {
                graph.coordinates.addElement(
                    Tile(), {iWaveY}, {simdBlockNumber, elementBlockNumber, elementBlockIndex});
            }

            graph.coordinates.addElement(Flatten(), {simdBlockNumber, simdBlockIndex}, {SIMD});
            graph.coordinates.addElement(Flatten(), {SIMD, laneInSIMD}, {lane});
            graph.coordinates.addElement(
                Flatten(), {elementBlockNumber, elementBlockIndex}, {element});
        }

        /** @brief Add coordinate-transforms for transpose-loading a WaveTile
         * from row/column coordinates `iWaveX` and `iWaveY`.
         *
         * The `lane` and `element` parameters are existing coordinates
         * corresponding to a Lane coordiante and VGPR coordinate (which should
         * be thought of as which element/item is being addressed). Each lane
         * loads 32 elements.
         */
        void addTransposeLoadWaveTileCT(KernelGraph& graph,
                                        int          iWaveX,
                                        int          iWaveY,
                                        int          lane,
                                        int          element,
                                        uint         M,
                                        uint         K,
                                        uint         bitsPerElement,
                                        int          wavefrontSize)

        {
            const auto simdsInWave    = 4;
            const auto lanesInSIMD    = 16;
            const auto simdsPerSGroup = M / lanesInSIMD;

            const auto bitsPerTrLoad           = bitsPerTransposeLoad(bitsPerElement);
            const auto elementsTrLoadedPerLane = bitsPerTrLoad / bitsPerElement;
            const auto numTrLoadsPerWave       = 2;
            const auto numTrLoads              = (M * K) / wavefrontSize / elementsTrLoadedPerLane;

            auto simdsPerWave = graph.coordinates.addElement(
                Adhoc("simdsPerWave", literal(simdsInWave), nullptr));
            auto lanesPerSIMD = graph.coordinates.addElement(Lane(literal(lanesInSIMD), nullptr));

            auto simdBlockNumber = graph.coordinates.addElement(
                Adhoc("simdBlockNumber", literal(simdsInWave / simdsPerSGroup), nullptr));
            auto simdBlockIndex = graph.coordinates.addElement(
                Adhoc("simdBlockIndex", literal(simdsPerSGroup), nullptr));

            auto lanesPerSIMDInM = graph.coordinates.addElement(
                Lane(literal(lanesInSIMD / elementsTrLoadedPerLane), nullptr));
            auto lanesPerSIMDInK
                = graph.coordinates.addElement(Lane(literal(elementsTrLoadedPerLane), nullptr));

            auto trLoadBlockNumber = graph.coordinates.addElement(
                Adhoc("trLoadBlockNumber", literal(numTrLoads / numTrLoadsPerWave), nullptr));
            auto trLoadBlockIndex = graph.coordinates.addElement(
                Adhoc("trLoadBlockIndex", literal(numTrLoadsPerWave), nullptr));

            auto elementBlockNumber
                = graph.coordinates.addElement(VGPRBlockNumber(literal(numTrLoads), nullptr));
            auto elementBlockIndex = graph.coordinates.addElement(
                VGPRBlockIndex(literal(elementsTrLoadedPerLane), nullptr));

            graph.coordinates.addElement(
                Flatten(), {trLoadBlockNumber, trLoadBlockIndex}, {elementBlockNumber});

            graph.coordinates.addElement(
                Tile(), {iWaveX}, {simdBlockIndex, lanesPerSIMDInM, elementBlockIndex});

            graph.coordinates.addElement(
                Tile(),
                {iWaveY},
                {trLoadBlockNumber, simdBlockNumber, trLoadBlockIndex, lanesPerSIMDInK});

            graph.coordinates.addElement(
                Flatten(), {lanesPerSIMDInK, lanesPerSIMDInM}, {lanesPerSIMD});
            graph.coordinates.addElement(
                Flatten(), {simdBlockNumber, simdBlockIndex}, {simdsPerWave});
            graph.coordinates.addElement(Flatten(), {simdsPerWave, lanesPerSIMD}, {lane});
            graph.coordinates.addElement(
                Flatten(), {elementBlockNumber, elementBlockIndex}, {element});
        }

        void addLoadSwizzleTileCT(KernelGraph&                     graph,
                                  std::vector<DeferredConnection>& connections,
                                  int                              macTileTag,
                                  int                              iMacX,
                                  int                              iMacY,
                                  VariableType const&              varType,
                                  int                              wavefrontSize,
                                  std::vector<unsigned int> const& jammedTiles,
                                  CommandParametersPtr             params)
        {
            auto macTile = graph.coordinates.getNode<MacroTile>(macTileTag);

            AssertFatal(macTile.subTileSizes.size() == 4, "Invalid tile specification.");

            auto waveTile    = WaveTile(macTile);
            auto waveTileTag = graph.coordinates.addElement(waveTile);

            connections.push_back(DC<WaveTile>(waveTileTag));

            auto nWaveX = graph.coordinates.addElement(waveTile.tileNumber(0));
            auto nWaveY = graph.coordinates.addElement(waveTile.tileNumber(1));
            auto iWaveX = graph.coordinates.addElement(waveTile.tileIndex(0));
            auto iWaveY = graph.coordinates.addElement(waveTile.tileIndex(1));

            graph.coordinates.addElement(Tile(), {iMacX}, {nWaveX, iWaveX});
            graph.coordinates.addElement(Tile(), {iMacY}, {nWaveY, iWaveY});

            graph.coordinates.addElement(Tile(), {waveTileTag}, {iWaveX, iWaveY});

            connections.push_back(DC<WaveTileNumber>(nWaveX, 0));
            connections.push_back(DC<WaveTileNumber>(nWaveY, 1));

            uint const nLaneInSIMD = 16;
            uint const nSIMDBlock  = macTile.miTileSizes[2];
            uint const nSIMDIndex  = 4 / nSIMDBlock;

            auto SIMDBlock
                = graph.coordinates.addElement(Adhoc("SIMDBlock", literal(nSIMDBlock), nullptr));
            auto SIMDIndex
                = graph.coordinates.addElement(Adhoc("SIMDIndex", literal(nSIMDIndex), nullptr));
            auto laneInSIMD = graph.coordinates.addElement(Lane(literal(nLaneInSIMD), nullptr));

            graph.coordinates.addElement(Tile(), {iWaveX}, {SIMDBlock, SIMDIndex, laneInSIMD});

            uint numElements = waveTile.elements();
            uint wfs         = static_cast<uint>(wavefrontSize);
            uint numVgpr     = numElements / wfs;

            uint const nVgprIndex = macTile.miTileSizes[2];
            uint const nVgprBlock = numVgpr / nVgprIndex;
            auto       vgprBlock
                = graph.coordinates.addElement(VGPRBlockNumber(literal(nVgprBlock), literal(1u)));
            auto vgprIndex
                = graph.coordinates.addElement(VGPRBlockIndex(literal(nVgprIndex), literal(1u)));
            auto vgpr = graph.coordinates.addElement(VGPR(literal(numVgpr), literal(1u)));
            graph.coordinates.addElement(Flatten(), {vgprBlock, vgprIndex}, {vgpr});
            connections.push_back(DC<VGPRBlockNumber>(vgprBlock));
            connections.push_back(DC<VGPRBlockIndex>(vgprIndex));
            connections.push_back(DC<VGPR>(vgpr));

            graph.coordinates.addElement(PassThrough(), {iWaveY}, {vgpr});

            auto wavefrontSizeLiteral = literal(wfs);

            auto wave  = graph.coordinates.addElement(Wavefront(-1));
            auto waveX = graph.coordinates.addElement(Wavefront(0));
            auto waveY = graph.coordinates.addElement(Wavefront(1));
            graph.coordinates.addElement(Flatten(), {waveX, waveY}, {wave});

            auto workitem = graph.coordinates.addElement(Workitem(0));
            auto lane     = graph.coordinates.addElement(Lane(wavefrontSizeLiteral, literal(1u)));
            graph.coordinates.addElement(Flatten(), {wave, lane}, {workitem});
            graph.coordinates.addElement(Flatten(), {SIMDBlock, SIMDIndex, laneInSIMD}, {lane});

            auto jammedWavetileX = graph.coordinates.addElement(
                JammedWaveTileNumber(0, literal(jammedTiles[0]), literal(1)));
            graph.coordinates.addElement(Tile(), {nWaveX}, {jammedWavetileX, waveX});
            connections.push_back(DC<JammedWaveTileNumber>(jammedWavetileX, 0));

            auto jammedWavetileY = graph.coordinates.addElement(
                JammedWaveTileNumber(1, literal(jammedTiles[1]), literal(1)));
            graph.coordinates.addElement(Tile(), {nWaveY}, {jammedWavetileY, waveY});
            connections.push_back(DC<JammedWaveTileNumber>(jammedWavetileY, 1));

            graph.coordinates.addElement(DataFlow(), {macTileTag}, {waveTileTag});
        }

        /**
         * @brief Add coordinate-transforms for loading a WaveTile
         * from row/column coordinates iMacX and iMacY.
         *
         * The geometry and layout of the WaveTile is taken from the
         * MacroTile associated with `macTileTag`.
         *
         * Required (deferred) connections are appended to
         * `connections`.
         */
        void addLoadWaveTileCT(KernelGraph&                     graph,
                               std::vector<DeferredConnection>& connections,
                               int                              macTileTag,
                               int                              iMacX,
                               int                              iMacY,
                               VariableType const&              varType,
                               int                              wavefrontSize,
                               bool                             isFromLDS,
                               std::vector<unsigned int> const& jammedTiles,
                               CommandParametersPtr             params,
                               ContextPtr                       context)
        {
            auto macTile = graph.coordinates.getNode<MacroTile>(macTileTag);

            AssertFatal(macTile.subTileSizes.size() == 4, "Invalid tile specification.");

            auto workitem    = graph.coordinates.addElement(Workitem(0));
            auto waveTile    = WaveTile(macTile);
            auto waveTileTag = graph.coordinates.addElement(waveTile);

            graph.coordinates.addElement(PassThrough(), {waveTileTag}, {macTileTag});

            connections.push_back(DC<WaveTile>(waveTileTag));

            auto nWaveX = graph.coordinates.addElement(waveTile.tileNumber(0));
            auto nWaveY = graph.coordinates.addElement(waveTile.tileNumber(1));
            auto iWaveX = graph.coordinates.addElement(waveTile.tileIndex(0));
            auto iWaveY = graph.coordinates.addElement(waveTile.tileIndex(1));

            graph.coordinates.addElement(Tile(), {iMacX}, {nWaveX, iWaveX});
            graph.coordinates.addElement(Tile(), {iMacY}, {nWaveY, iWaveY});

            connections.push_back(DC<WaveTileNumber>(nWaveX, 0));
            connections.push_back(DC<WaveTileNumber>(nWaveY, 1));

            auto waveX = graph.coordinates.addElement(Wavefront(0));
            auto waveY = graph.coordinates.addElement(Wavefront(1));
            auto wave  = graph.coordinates.addElement(Wavefront(-1));

            uint numElements = waveTile.elements();
            uint wfs         = static_cast<uint>(wavefrontSize);
            uint numGPR      = numElements / wfs;

            uint M   = macTile.subTileSizes[0];
            uint N   = macTile.subTileSizes[1];
            uint K   = macTile.subTileSizes[2];
            uint K_L = K / (wfs / M);

            auto wavefrontSizeLiteral = literal(wfs);

            auto lane = graph.coordinates.addElement(Lane(wavefrontSizeLiteral, nullptr));
            auto vgpr = graph.coordinates.addElement(VGPR(literal(numGPR), nullptr));

            graph.coordinates.addElement(Flatten(), {waveX, waveY}, {wave});
            graph.coordinates.addElement(Flatten(), {wave, lane}, {workitem});

            connections.push_back(DC<VGPR>(vgpr));

            auto bitsPerElement = DataTypeInfo::Get(varType).elementBits;

            auto isF8F6F4 = (bitsPerElement <= 8)
                            && (((M == 16) && (N == 16) && (K == 128))
                                || ((M == 32) && (N == 32) && (K == 64)));

            auto isTransposableType = [isF8F6F4, bitsPerElement, context, M, N, K](DataType type) {
                auto isF16TransposableTileLayout = (bitsPerElement == 16)
                                                   && (((M == 16) && (N == 16) && (K == 32))
                                                       || ((M == 32) && (N == 32) && (K == 16)));

                auto hasTransposeInstructionForType = [context](DataType type) {
                    auto const& arch = context->targetArchitecture();
                    switch(type)
                    {
                    case DataType::Half:
                    case DataType::BFloat16:
                        return arch.HasCapability(GPUCapability::HasDSReadTransposeB16);
                    case DataType::FP8:
                    case DataType::BF8:
                        return arch.HasCapability(GPUCapability::HasDSReadTransposeB8);
                    case DataType::FP6:
                    case DataType::BF6:
                        return arch.HasCapability(GPUCapability::HasDSReadTransposeB6);
                    case DataType::FP4:
                        return arch.HasCapability(GPUCapability::HasDSReadTransposeB4);
                    default:
                        return false;
                    };
                };
                return (isF8F6F4 || isF16TransposableTileLayout)
                       && hasTransposeInstructionForType(type);
            };

            bool isTransposeLayout = params->transposeMemoryAccess[waveTile.layout];

            switch(waveTile.layout)
            {
            case LayoutType::MATRIX_A:
            {
                auto jammedWavetileX = graph.coordinates.addElement(
                    JammedWaveTileNumber(0, literal(jammedTiles[0]), literal(1)));
                connections.push_back(DC<JammedWaveTileNumber>(jammedWavetileX, 0));

                if(isFromLDS && isTransposableType(varType.dataType) && !isTransposeLayout)
                {
                    Log::debug("Adding transpose-load CT for A macTileTag {}", macTileTag);
                    addTransposeLoadWaveTileCT(
                        graph, iWaveX, iWaveY, lane, vgpr, M, K, bitsPerElement, wavefrontSize);
                    setIsTransposedLoad(macTileTag, graph);
                }
                else if(!isF8F6F4)
                {
                    auto blockNumber = graph.coordinates.addElement(
                        Adhoc("BlockNumber", literal(static_cast<uint>(K / K_L)), nullptr));
                    auto blockIndex = graph.coordinates.addElement(
                        Adhoc("BlockIndex", literal(static_cast<uint>(K_L)), nullptr));

                    graph.coordinates.addElement(Tile(), {iWaveY}, {blockNumber, blockIndex});

                    graph.coordinates.addElement(Flatten(), {blockNumber, iWaveX}, {lane});
                    graph.coordinates.addElement(PassThrough(), {blockIndex}, {vgpr});
                }
                else
                {
                    addLoadWaveTileCTF8F6F4(
                        graph, iWaveX, iWaveY, lane, vgpr, K, bitsPerElement, wavefrontSize);
                }

                if(params->swizzleScale)
                    graph.coordinates.addElement(Tile(), {nWaveX}, {waveX, jammedWavetileX});
                else
                    graph.coordinates.addElement(Tile(), {nWaveX}, {jammedWavetileX, waveX});
            }
            break;

            case LayoutType::MATRIX_B:
            {
                auto jammedWavetileY = graph.coordinates.addElement(
                    JammedWaveTileNumber(1, literal(jammedTiles[1]), literal(1)));
                connections.push_back(DC<JammedWaveTileNumber>(jammedWavetileY, 1));

                if(isFromLDS && isTransposableType(varType.dataType) && isTransposeLayout)
                {
                    Log::debug("Adding transpose-load CT for B macTileTag {}", macTileTag);
                    addTransposeLoadWaveTileCT(
                        graph, iWaveY, iWaveX, lane, vgpr, M, K, bitsPerElement, wavefrontSize);
                    setIsTransposedLoad(macTileTag, graph);
                }
                else if(!isF8F6F4)
                {
                    auto blockNumber = graph.coordinates.addElement(
                        Adhoc("BlockNumber", literal(static_cast<uint>(K / K_L)), nullptr));
                    auto blockIndex = graph.coordinates.addElement(
                        Adhoc("BlockIndex", literal(static_cast<uint>(K_L)), nullptr));

                    graph.coordinates.addElement(Tile(), {iWaveX}, {blockNumber, blockIndex});

                    graph.coordinates.addElement(Flatten(), {blockNumber, iWaveY}, {lane});
                    graph.coordinates.addElement(PassThrough(), {blockIndex}, {vgpr});
                }
                else
                {
                    addLoadWaveTileCTF8F6F4(
                        graph, iWaveY, iWaveX, lane, vgpr, K, bitsPerElement, wavefrontSize);
                }

                if(params->swizzleScale)
                    graph.coordinates.addElement(Tile(), {nWaveY}, {waveY, jammedWavetileY});
                else
                    graph.coordinates.addElement(Tile(), {nWaveY}, {jammedWavetileY, waveY});
            }
            break;

            case LayoutType::MATRIX_ACCUMULATOR:
            {
                auto numGPRsPerLaneInWave = wfs == 64 ? 4 : 8; // A wave can have 256 GPR in total
                uint simdsPerWave         = wavefrontSize / 16u; // Each SIMD consists of 16 lanes
                auto simdsPerWaveLiteral  = literal(simdsPerWave);
                auto unitStride           = literal(1u);

                auto nRowBlocks = literal(waveTile.sizes[0] / simdsPerWave);
                auto nColBlocks = literal(waveTile.sizes[1] / simdsPerWave);

                auto jammedWavetileX = graph.coordinates.addElement(
                    JammedWaveTileNumber(0, literal(jammedTiles[0]), literal(1)));
                connections.push_back(DC<JammedWaveTileNumber>(jammedWavetileX, 0));

                auto jammedWavetileY = graph.coordinates.addElement(
                    JammedWaveTileNumber(1, literal(jammedTiles[1]), literal(1)));
                connections.push_back(DC<JammedWaveTileNumber>(jammedWavetileY, 1));

                auto nVblk = graph.coordinates.addElement(
                    VGPRBlockNumber(literal(numGPR / numGPRsPerLaneInWave), unitStride));
                auto iVblk = graph.coordinates.addElement(
                    VGPRBlockIndex(literal(numGPRsPerLaneInWave), unitStride));
                auto nLblk = graph.coordinates.addElement(
                    Adhoc("LANEBlockNumber", literal(wfs / simdsPerWave), unitStride));
                auto iLblk = graph.coordinates.addElement(
                    Adhoc("LANEBlockIndex", simdsPerWaveLiteral, unitStride));
                auto block = graph.coordinates.addElement(
                    Adhoc("LinearBlock", literal(numElements / 16u), unitStride));
                auto rowBlock
                    = graph.coordinates.addElement(Adhoc("RowBlock", nRowBlocks, unitStride));
                auto colBlock
                    = graph.coordinates.addElement(Adhoc("ColBlock", nColBlocks, unitStride));

                connections.push_back(DC<VGPRBlockNumber>(nVblk));
                connections.push_back(DC<VGPRBlockIndex>(iVblk));

                graph.coordinates.addElement(Tile(), {iWaveX}, {rowBlock, iVblk});
                graph.coordinates.addElement(Tile(), {iWaveY}, {colBlock, iLblk});

                graph.coordinates.addElement(Flatten(), {rowBlock, colBlock}, {block});
                graph.coordinates.addElement(Tile(), {block}, {nVblk, nLblk});

                graph.coordinates.addElement(Flatten(), {nVblk, iVblk}, {vgpr});
                graph.coordinates.addElement(Flatten(), {nLblk, iLblk}, {lane});

                if(params->swizzleScale)
                {
                    graph.coordinates.addElement(Tile(), {nWaveX}, {waveX, jammedWavetileX});
                    graph.coordinates.addElement(Tile(), {nWaveY}, {waveY, jammedWavetileY});
                }
                else
                {
                    graph.coordinates.addElement(Tile(), {nWaveX}, {jammedWavetileX, waveX});
                    graph.coordinates.addElement(Tile(), {nWaveY}, {jammedWavetileY, waveY});
                }
            }
            break;

            default:
                Throw<FatalError>("addLoadWaveTileCT waveTile.layout not implemented yet.");
            }
        }

        /**
         * @brief Add coordinate-transforms for loading a ThreadTile
         * from row/column coordinates iMacX and iMacY.
         *
         * The geometry of the ThreadTile is taken from the MacroTile
         * associated with `macTileTag`.
         *
         * By default:
         *
         *   - For A/B matrix layouts, the Y thread tile number is
         *     fast wrt the workitem/lane index and the X thread tile
         *     number is slow.  For other layous, the X/Y thread tile
         *     numbers are taken from the X/Y workitem index.
         *
         *   - The row index of a thread tile is fast wrt the VGPR
         *     index.
         *
         * When `useSwappedAccess` is true, both of these orders are
         * reversed.
         *
         * Required (deferred) connections are appended to
         * `connections`.
         */
        void addLoadThreadTileCT(KernelGraph&                       graph,
                                 std::vector<DeferredConnection>&   connections,
                                 int                                macTileTag,
                                 int                                iMacX,
                                 int                                iMacY,
                                 std::array<unsigned int, 3> const& workgroupSizes,
                                 std::vector<unsigned int> const&   jammedTiles,
                                 bool                               useSwappedAccess,
                                 bool                               isDirect2LDS)
        {
            auto macTile = graph.coordinates.getNode<MacroTile>(macTileTag);
            auto thrTile = ThreadTile(macTile);

            auto nThrX
                = graph.coordinates.addElement(ThreadTileNumber(0, literal(thrTile.wsizes.at(0))));
            auto nThrY
                = graph.coordinates.addElement(ThreadTileNumber(1, literal(thrTile.wsizes.at(1))));
            auto iThrX
                = graph.coordinates.addElement(ThreadTileIndex(0, literal(thrTile.sizes.at(0))));
            auto iThrY
                = graph.coordinates.addElement(ThreadTileIndex(1, literal(thrTile.sizes.at(1))));

            int elementNumberX, elementNumberY;

            auto workitemX
                = graph.coordinates.addElement(Workitem(0, literal(workgroupSizes.at(0))));

            if(useSwappedAccess)
            {
                elementNumberX
                    = graph.coordinates.addElement(ElementNumber(0, literal(thrTile.sizes.at(0))));
                elementNumberY
                    = graph.coordinates.addElement(ElementNumber(1, literal(thrTile.sizes.at(1))));

                graph.coordinates.addElement(PassThrough(), {iThrX}, {elementNumberX});
                graph.coordinates.addElement(PassThrough(), {iThrY}, {elementNumberY});

                if(macTile.layoutType == LayoutType::MATRIX_A
                   || macTile.layoutType == LayoutType::MATRIX_B
                   || macTile.layoutType == LayoutType::MATRIX_ACCUMULATOR
                   || macTile.layoutType == LayoutType::SCRATCH)
                {
                    graph.coordinates.addElement(Flatten(), {nThrX, nThrY}, {workitemX});
                }
                else
                {
                    graph.coordinates.addElement(PassThrough(), {nThrX}, {workitemX});

                    auto workitemY
                        = graph.coordinates.addElement(Workitem(1, literal(workgroupSizes.at(1))));
                    graph.coordinates.addElement(PassThrough(), {nThrY}, {workitemY});
                }
            }
            else
            {
                elementNumberX
                    = graph.coordinates.addElement(ElementNumber(0, literal(thrTile.sizes.at(1))));
                elementNumberY
                    = graph.coordinates.addElement(ElementNumber(1, literal(thrTile.sizes.at(0))));

                graph.coordinates.addElement(PassThrough(), {iThrX}, {elementNumberY});
                graph.coordinates.addElement(PassThrough(), {iThrY}, {elementNumberX});

                if(macTile.layoutType == LayoutType::MATRIX_A
                   || macTile.layoutType == LayoutType::MATRIX_B
                   || macTile.layoutType == LayoutType::MATRIX_ACCUMULATOR
                   || macTile.layoutType == LayoutType::SCRATCH)
                {
                    graph.coordinates.addElement(Flatten(), {nThrY, nThrX}, {workitemX});
                }
                else
                {
                    graph.coordinates.addElement(PassThrough(), {nThrX}, {workitemX});

                    auto workitemY
                        = graph.coordinates.addElement(Workitem(1, literal(workgroupSizes.at(1))));
                    graph.coordinates.addElement(PassThrough(), {nThrY}, {workitemY});
                }
            }

            connections.push_back(DC<ElementNumber>(elementNumberX, 0));
            connections.push_back(DC<ElementNumber>(elementNumberY, 1));

            if(jammedTiles.size() > 0 && jammedTiles[0] > 1)
            {
                auto jammedWavetileX = graph.coordinates.addElement(
                    JammedWaveTileNumber(0, literal(jammedTiles[0]), literal(1)));
                connections.push_back(DC<JammedWaveTileNumber>(jammedWavetileX, 0));
                if(useSwappedAccess && isDirect2LDS)
                    graph.coordinates.addElement(Tile(), {iMacX}, {jammedWavetileX, iThrX, nThrX});
                else
                    graph.coordinates.addElement(Tile(), {iMacX}, {jammedWavetileX, nThrX, iThrX});
            }
            else
            {
                if(useSwappedAccess && isDirect2LDS)
                    graph.coordinates.addElement(Tile(), {iMacX}, {iThrX, nThrX});
                else
                    graph.coordinates.addElement(Tile(), {iMacX}, {nThrX, iThrX});
            }

            if(jammedTiles.size() > 1 && jammedTiles[1] > 1)
            {
                auto jammedWavetileY = graph.coordinates.addElement(
                    JammedWaveTileNumber(1, literal(jammedTiles[1]), literal(1)));
                connections.push_back(DC<JammedWaveTileNumber>(jammedWavetileY, 1));
                if(isDirect2LDS)
                {
                    if(useSwappedAccess)
                        graph.coordinates.addElement(
                            Tile(), {iMacY}, {jammedWavetileY, nThrY, iThrY});
                    else
                        graph.coordinates.addElement(
                            Tile(), {iMacY}, {jammedWavetileY, iThrY, nThrY});
                }
                else
                    graph.coordinates.addElement(Tile(), {iMacY}, {jammedWavetileY, nThrY, iThrY});
            }
            else
            {
                if(isDirect2LDS)
                {
                    if(useSwappedAccess)
                        graph.coordinates.addElement(Tile(), {iMacY}, {nThrY, iThrY});
                    else
                        graph.coordinates.addElement(Tile(), {iMacY}, {iThrY, nThrY});
                }
                else
                    graph.coordinates.addElement(Tile(), {iMacY}, {nThrY, iThrY});
            }
        }

        /**
         * @brief Add coordinate-transforms for loading a
         * MATRIX_ACCUMULATOR tile from row/column coordinates iMacX
         * and iMacY.
         *
         * Required (deferred) connections are appended to
         * `connections`.
         */
        void addLoadAccumulatorTileCT(KernelGraph&                       graph,
                                      std::vector<DeferredConnection>&   connections,
                                      int                                macTileTag,
                                      int                                iMacX,
                                      int                                iMacY,
                                      int                                wavefrontSize,
                                      std::array<unsigned int, 3> const& workgroupSizes)
        {
            auto macTile = graph.coordinates.getNode<MacroTile>(macTileTag);
            auto thrTile = ThreadTile(macTile);

            auto nThrX
                = graph.coordinates.addElement(ThreadTileNumber(0, literal(thrTile.wsizes.at(0))));
            auto nThrY
                = graph.coordinates.addElement(ThreadTileNumber(1, literal(thrTile.wsizes.at(1))));
            auto iThrX
                = graph.coordinates.addElement(ThreadTileIndex(0, literal(thrTile.sizes.at(0))));
            auto iThrY
                = graph.coordinates.addElement(ThreadTileIndex(1, literal(thrTile.sizes.at(1))));

            auto workitemX
                = graph.coordinates.addElement(Workitem(0, literal(workgroupSizes.at(0))));

            auto elementNumberX
                = graph.coordinates.addElement(ElementNumber(0, literal(thrTile.sizes.at(1))));
            auto elementNumberY
                = graph.coordinates.addElement(ElementNumber(1, literal(thrTile.sizes.at(0))));

            // fast dim : iThrX, slow dim : iThrY
            graph.coordinates.addElement(PassThrough(), {iThrX}, {elementNumberY});
            graph.coordinates.addElement(PassThrough(), {iThrY}, {elementNumberX});
            // fast dim : nThrX, slow dim : nThrY
            graph.coordinates.addElement(Flatten(), {nThrY, nThrX}, {workitemX});

            connections.push_back(DC<ElementNumber>(elementNumberX, 0));
            connections.push_back(DC<ElementNumber>(elementNumberY, 1));

            graph.coordinates.addElement(Tile(), {iMacX}, {nThrX, iThrX});

            auto numXWorkitems        = thrTile.wsizes.at(0);
            auto numYWorkitemsPerWave = wavefrontSize / numXWorkitems;
            auto numYWorkitems        = thrTile.wsizes.at(1);
            if(numYWorkitemsPerWave > 1 && numYWorkitems > numYWorkitemsPerWave)
            {
                auto numYElements = thrTile.sizes.at(1);
                auto waveNumber   = graph.coordinates.addElement(
                    Adhoc("waveNumber",
                          literal(static_cast<uint>(numYWorkitems / numYWorkitemsPerWave)),
                          nullptr));
                auto workitemYPerWave = graph.coordinates.addElement(Adhoc(
                    "workitemYPerWave", literal(static_cast<uint>(numYWorkitemsPerWave)), nullptr));
                auto waveBlock        = graph.coordinates.addElement(
                    Adhoc("waveBlock",
                          literal(static_cast<uint>(numYWorkitemsPerWave * numYElements)),
                          nullptr));

                graph.coordinates.addElement(Tile(), {iMacY}, {waveNumber, waveBlock});
                graph.coordinates.addElement(Tile(), {waveBlock}, {iThrY, workitemYPerWave});
                graph.coordinates.addElement(Flatten(), {waveNumber, workitemYPerWave}, {nThrY});
            }
            else
            {
                graph.coordinates.addElement(Tile(), {iMacY}, {nThrY, iThrY});
            }
        }

        /**
         * @brief Store version of addLoadAccumulatorTileCT.
         */
        void addStoreAccumulatorTileCT(KernelGraph&                       graph,
                                       std::vector<DeferredConnection>&   connections,
                                       int                                macTileTag,
                                       int                                iMacX,
                                       int                                iMacY,
                                       int                                wavefrontSize,
                                       std::array<unsigned int, 3> const& workgroupSizes)
        {
            auto macTile = graph.coordinates.getNode<MacroTile>(macTileTag);

            auto thrTile = ThreadTile(macTile);

            auto nThrX
                = graph.coordinates.addElement(ThreadTileNumber(0, literal(thrTile.wsizes.at(0))));
            auto nThrY
                = graph.coordinates.addElement(ThreadTileNumber(1, literal(thrTile.wsizes.at(1))));
            auto iThrX
                = graph.coordinates.addElement(ThreadTileIndex(0, literal(thrTile.sizes.at(0))));
            auto iThrY
                = graph.coordinates.addElement(ThreadTileIndex(1, literal(thrTile.sizes.at(1))));

            auto workitemX
                = graph.coordinates.addElement(Workitem(0, literal(workgroupSizes.at(0))));

            auto elementNumberX
                = graph.coordinates.addElement(ElementNumber(0, literal(thrTile.sizes.at(1))));
            auto elementNumberY
                = graph.coordinates.addElement(ElementNumber(1, literal(thrTile.sizes.at(0))));

            connections.push_back(DC<ElementNumber>(elementNumberX, 0));
            connections.push_back(DC<ElementNumber>(elementNumberY, 1));

            graph.coordinates.addElement(Tile(), {workitemX}, {nThrY, nThrX});
            graph.coordinates.addElement(PassThrough(), {elementNumberX}, {iThrY});
            graph.coordinates.addElement(PassThrough(), {elementNumberY}, {iThrX});

            graph.coordinates.addElement(Flatten(), {nThrX, iThrX}, {iMacX});

            auto numXWorkitems        = thrTile.wsizes.at(0);
            auto numYWorkitemsPerWave = wavefrontSize / numXWorkitems;
            auto numYWorkitems        = thrTile.wsizes.at(1);
            if(numYWorkitemsPerWave > 1 && numYWorkitems > numYWorkitemsPerWave)
            {
                auto numYElements = thrTile.sizes.at(1);
                auto waveNumber   = graph.coordinates.addElement(
                    Adhoc("waveNumber",
                          literal(static_cast<uint>(numYWorkitems / numYWorkitemsPerWave)),
                          nullptr));
                auto workitemYPerWave = graph.coordinates.addElement(Adhoc(
                    "workitemYPerWave", literal(static_cast<uint>(numYWorkitemsPerWave)), nullptr));
                auto waveBlock        = graph.coordinates.addElement(
                    Adhoc("waveBlock",
                          literal(static_cast<uint>(numYWorkitemsPerWave * numYElements)),
                          nullptr));

                graph.coordinates.addElement(Tile(), {nThrY}, {waveNumber, workitemYPerWave});
                graph.coordinates.addElement(Flatten(), {iThrY, workitemYPerWave}, {waveBlock});
                graph.coordinates.addElement(Flatten(), {waveNumber, waveBlock}, {iMacY});
            }
            else
            {
                graph.coordinates.addElement(Flatten(), {nThrY, iThrY}, {iMacY});
            }
        }

        /**
         * @brief Store version of addLoadMacroTileCT.
         */
        std::tuple<int, int, int, int>
            addStoreMacroTileCT(KernelGraph&                     graph,
                                std::vector<DeferredConnection>& connections,
                                int                              macTileTag,
                                std::vector<int> const&          sdim,
                                std::vector<unsigned int> const& jammedTiles)
        {
            auto macTile = graph.coordinates.getNode<MacroTile>(macTileTag);
            auto sdimX   = sdim[0];
            auto sdimY   = sdim[1];

            connections.push_back(DC<SubDimension>(sdimX, 0));
            connections.push_back(DC<SubDimension>(sdimY, 1));

            auto nMacX = graph.coordinates.addElement(macTile.tileNumber(0, nullptr));
            auto nMacY = graph.coordinates.addElement(macTile.tileNumber(1, nullptr));
            auto iMacX = graph.coordinates.addElement(macTile.tileIndex(0, jammedTiles[0]));
            auto iMacY = graph.coordinates.addElement(macTile.tileIndex(1, jammedTiles[1]));

            connections.push_back(DC<MacroTileNumber>(nMacX, 0));
            connections.push_back(DC<MacroTileNumber>(nMacY, 1));

            graph.coordinates.addElement(Flatten(), {nMacX, iMacX}, {sdim[0]});
            graph.coordinates.addElement(Flatten(), {nMacY, iMacY}, {sdim[1]});

            return {nMacX, iMacX, nMacY, iMacY};
        }

        std::tuple<int, int, int>
            addStore1DMacroTileCT(KernelGraph&                     graph,
                                  std::vector<DeferredConnection>& connections,
                                  int                              macTileTag,
                                  std::vector<int> const&          sdim,
                                  std::vector<unsigned int> const& jammedTiles)
        {
            auto macTile = graph.coordinates.getNode<MacroTile>(macTileTag);
            auto sdimX   = sdim[0];
            auto sdimY   = sdim[1];

            connections.push_back(DC<SubDimension>(sdimX, 0));
            connections.push_back(DC<SubDimension>(sdimY, 1));

            auto nMacX = graph.coordinates.addElement(macTile.tileNumber(0, nullptr));
            auto iMacX = graph.coordinates.addElement(macTile.tileIndex(0, jammedTiles[0]));
            auto iMacY = graph.coordinates.addElement(macTile.tileIndex(1, jammedTiles[1]));

            connections.push_back(DC<MacroTileNumber>(nMacX, 0));

            graph.coordinates.addElement(Flatten(), {nMacX, iMacX}, {sdimX});
            graph.coordinates.addElement(PassThrough(), {iMacY}, {sdimY});

            return {nMacX, iMacX, iMacY};
        }

        /**
         * @brief Store version of addLoadWaveTileCT.
         */
        void addStoreWaveTileCT(KernelGraph&                     graph,
                                std::vector<DeferredConnection>& connections,
                                int                              macTileTag,
                                int                              iMacX,
                                int                              iMacY,
                                int                              wavefrontSize,
                                std::vector<unsigned int> const& jammedTiles,
                                CommandParametersPtr             params)
        {
            auto macTile = graph.coordinates.getNode<MacroTile>(macTileTag);

            AssertFatal(macTile.layoutType == LayoutType::MATRIX_ACCUMULATOR,
                        "Store must be from accumulator.");

            auto workitem    = graph.coordinates.addElement(Workitem(0));
            auto waveTile    = WaveTile(macTile);
            auto waveTileTag = graph.coordinates.addElement(waveTile);

            uint numElements = waveTile.sizes[0] * waveTile.sizes[1];
            uint wfs         = static_cast<uint>(wavefrontSize);
            uint numGPR      = numElements / wfs;

            auto numGPRsPerLaneInWave = wfs == 64 ? 4 : 8;
            uint simdsPerWave         = wavefrontSize / 16u;
            auto simdsPerWaveLiteral  = literal(simdsPerWave);

            auto nRowBlocks = literal(waveTile.sizes[0] / simdsPerWave);
            auto nColBlocks = literal(waveTile.sizes[1] / simdsPerWave);

            auto nWaveX = graph.coordinates.addElement(waveTile.tileNumber(0));
            auto nWaveY = graph.coordinates.addElement(waveTile.tileNumber(1));
            auto iWaveX = graph.coordinates.addElement(waveTile.tileIndex(0));
            auto iWaveY = graph.coordinates.addElement(waveTile.tileIndex(1));

            auto wavefrontSizeLiteral = literal(static_cast<uint>(wavefrontSize));
            auto unitStride           = literal(1u);

            auto nVblk = graph.coordinates.addElement(
                VGPRBlockNumber(literal(numGPR / numGPRsPerLaneInWave), unitStride));
            auto iVblk = graph.coordinates.addElement(
                VGPRBlockIndex(literal(numGPRsPerLaneInWave), unitStride));
            auto nLblk = graph.coordinates.addElement(
                Adhoc("LANEBlockNumber", literal(wfs / simdsPerWave), unitStride));
            auto iLblk = graph.coordinates.addElement(
                Adhoc("LANEBlockIndex", simdsPerWaveLiteral, unitStride));
            auto block = graph.coordinates.addElement(
                Adhoc("LinearBlock", literal(numElements / 16u), unitStride));
            auto rowBlock = graph.coordinates.addElement(Adhoc("RowBlock", nRowBlocks, unitStride));
            auto colBlock = graph.coordinates.addElement(Adhoc("ColBlock", nColBlocks, unitStride));

            graph.coordinates.addElement(Flatten(), {nWaveX, iWaveX}, {iMacX});
            graph.coordinates.addElement(Flatten(), {nWaveY, iWaveY}, {iMacY});

            auto waveX = graph.coordinates.addElement(Wavefront(0));
            auto waveY = graph.coordinates.addElement(Wavefront(1));
            auto wave  = graph.coordinates.addElement(Wavefront(-1));

            graph.coordinates.addElement(Tile(), {wave}, {waveX, waveY});

            auto lane = graph.coordinates.addElement(Lane(wavefrontSizeLiteral, unitStride));
            auto vgpr = graph.coordinates.addElement(VGPR(literal(numGPR), unitStride));

            connections.push_back(DC<WaveTile>(waveTileTag));
            connections.push_back(DC<VGPRBlockNumber>(nVblk));
            connections.push_back(DC<VGPRBlockIndex>(iVblk));
            connections.push_back(DC<VGPR>(vgpr));

            graph.coordinates.addElement(Tile(), {vgpr}, {nVblk, iVblk});
            graph.coordinates.addElement(Tile(), {lane}, {nLblk, iLblk});
            graph.coordinates.addElement(Flatten(), {nVblk, nLblk}, {block});
            graph.coordinates.addElement(Tile(), {block}, {rowBlock, colBlock});

            if(jammedTiles[0] > 1)
            {
                auto jammedWavetileX = graph.coordinates.addElement(
                    JammedWaveTileNumber(0, literal(jammedTiles[0]), literal(1)));
                connections.push_back(DC<JammedWaveTileNumber>(jammedWavetileX, 0));
                if(params->swizzleScale)
                    graph.coordinates.addElement(Flatten(), {waveX, jammedWavetileX}, {nWaveX});
                else
                    graph.coordinates.addElement(Flatten(), {jammedWavetileX, waveX}, {nWaveX});
            }
            else
            {
                graph.coordinates.addElement(PassThrough(), {waveX}, {nWaveX});
            }

            if(jammedTiles[1] > 1)
            {
                auto jammedWavetileY = graph.coordinates.addElement(
                    JammedWaveTileNumber(1, literal(jammedTiles[1]), literal(1)));
                connections.push_back(DC<JammedWaveTileNumber>(jammedWavetileY, 1));

                if(params->swizzleScale)
                    graph.coordinates.addElement(Flatten(), {waveY, jammedWavetileY}, {nWaveY});
                else
                    graph.coordinates.addElement(Flatten(), {jammedWavetileY, waveY}, {nWaveY});
            }
            else
            {
                graph.coordinates.addElement(PassThrough(), {waveY}, {nWaveY});
            }

            graph.coordinates.addElement(Flatten(), {rowBlock, iVblk}, {iWaveX});
            graph.coordinates.addElement(Flatten(), {colBlock, iLblk}, {iWaveY});

            graph.coordinates.addElement(Tile(), {workitem}, {wave, lane});
        }

        /**
         * @brief Store version of addLoadThreadTileCT.
         */
        void addStoreThreadTileCT(KernelGraph&                       graph,
                                  std::vector<DeferredConnection>&   connections,
                                  int                                macTileTag,
                                  int                                iMacX,
                                  int                                iMacY,
                                  std::array<unsigned int, 3> const& workgroupSizes,
                                  std::vector<unsigned int> const&   jammedTiles,
                                  bool                               useSwappedAccess,
                                  bool                               isDirect2LDS)
        {
            auto macTile = graph.coordinates.getNode<MacroTile>(macTileTag);

            auto thrTile = ThreadTile(macTile);

            auto nThrX
                = graph.coordinates.addElement(ThreadTileNumber(0, literal(thrTile.wsizes.at(0))));
            auto nThrY
                = graph.coordinates.addElement(ThreadTileNumber(1, literal(thrTile.wsizes.at(1))));
            auto iThrX
                = graph.coordinates.addElement(ThreadTileIndex(0, literal(thrTile.sizes.at(0))));
            auto iThrY
                = graph.coordinates.addElement(ThreadTileIndex(1, literal(thrTile.sizes.at(1))));

            auto workitemX
                = graph.coordinates.addElement(Workitem(0, literal(workgroupSizes.at(0))));

            int elementNumberX, elementNumberY;

            if(useSwappedAccess)
            {
                elementNumberX
                    = graph.coordinates.addElement(ElementNumber(0, literal(thrTile.sizes.at(0))));
                elementNumberY
                    = graph.coordinates.addElement(ElementNumber(1, literal(thrTile.sizes.at(1))));

                connections.push_back(DC<ElementNumber>(elementNumberX, 0));
                connections.push_back(DC<ElementNumber>(elementNumberY, 1));

                graph.coordinates.addElement(PassThrough(), {elementNumberX}, {iThrX});
                graph.coordinates.addElement(PassThrough(), {elementNumberY}, {iThrY});

                if(macTile.layoutType == LayoutType::MATRIX_A
                   || macTile.layoutType == LayoutType::MATRIX_B
                   || macTile.layoutType == LayoutType::MATRIX_ACCUMULATOR
                   || macTile.layoutType == LayoutType::SCRATCH)
                {
                    graph.coordinates.addElement(Tile(), {workitemX}, {nThrX, nThrY});
                }
                else
                {
                    auto workitemY
                        = graph.coordinates.addElement(Workitem(1, literal(workgroupSizes.at(1))));

                    graph.coordinates.addElement(PassThrough(), {workitemX}, {nThrX});
                    graph.coordinates.addElement(PassThrough(), {workitemY}, {nThrY});
                }
            }
            else
            {
                elementNumberX
                    = graph.coordinates.addElement(ElementNumber(0, literal(thrTile.sizes.at(1))));
                elementNumberY
                    = graph.coordinates.addElement(ElementNumber(1, literal(thrTile.sizes.at(0))));

                graph.coordinates.addElement(PassThrough(), {elementNumberY}, {iThrX});
                graph.coordinates.addElement(PassThrough(), {elementNumberX}, {iThrY});

                connections.push_back(DC<ElementNumber>(elementNumberX, 0));
                connections.push_back(DC<ElementNumber>(elementNumberY, 1));

                if(macTile.layoutType == LayoutType::MATRIX_A
                   || macTile.layoutType == LayoutType::MATRIX_B
                   || macTile.layoutType == LayoutType::MATRIX_ACCUMULATOR
                   || macTile.layoutType == LayoutType::SCRATCH)
                {
                    graph.coordinates.addElement(Tile(), {workitemX}, {nThrY, nThrX});
                }
                else
                {
                    auto workitemY
                        = graph.coordinates.addElement(Workitem(1, literal(workgroupSizes.at(1))));

                    graph.coordinates.addElement(PassThrough(), {workitemX}, {nThrX});
                    graph.coordinates.addElement(PassThrough(), {workitemY}, {nThrY});
                }
            }

            if(jammedTiles.size() > 0 && jammedTiles[0] > 1)
            {
                auto jammedWavetileX = graph.coordinates.addElement(
                    JammedWaveTileNumber(0, literal(jammedTiles[0]), literal(1)));
                connections.push_back(DC<JammedWaveTileNumber>(jammedWavetileX, 0));
                if(useSwappedAccess && isDirect2LDS)
                    graph.coordinates.addElement(
                        Flatten(), {jammedWavetileX, iThrX, nThrX}, {iMacX});
                else
                    graph.coordinates.addElement(
                        Flatten(), {jammedWavetileX, nThrX, iThrX}, {iMacX});
            }
            else
            {
                if(useSwappedAccess && isDirect2LDS)
                    graph.coordinates.addElement(Flatten(), {iThrX, nThrX}, {iMacX});
                else
                    graph.coordinates.addElement(Flatten(), {nThrX, iThrX}, {iMacX});
            }

            if(jammedTiles.size() > 1 && jammedTiles[1] > 1)
            {
                auto jammedWavetileY = graph.coordinates.addElement(
                    JammedWaveTileNumber(1, literal(jammedTiles[1]), literal(1)));
                connections.push_back(DC<JammedWaveTileNumber>(jammedWavetileY, 1));
                if(isDirect2LDS)
                {
                    if(useSwappedAccess)
                        graph.coordinates.addElement(
                            Flatten(), {jammedWavetileY, nThrY, iThrY}, {iMacY});
                    else
                        graph.coordinates.addElement(
                            Flatten(), {jammedWavetileY, iThrY, nThrY}, {iMacY});
                }
                else
                    graph.coordinates.addElement(
                        Flatten(), {jammedWavetileY, nThrY, iThrY}, {iMacY});
            }
            else
            {
                if(isDirect2LDS)
                {
                    if(useSwappedAccess)
                        graph.coordinates.addElement(Flatten(), {nThrY, iThrY}, {iMacY});
                    else
                        graph.coordinates.addElement(Flatten(), {iThrY, nThrY}, {iMacY});
                }
                else
                    graph.coordinates.addElement(Flatten(), {nThrY, iThrY}, {iMacY});
            }
        }

        /**
         * @brief Create an internal tile backed by a ThreadTile.
         */
        int createInternalTile(KernelGraph&         graph,
                               VariableType         varType,
                               int                  macTileTag,
                               CommandParametersPtr params,
                               ContextPtr           context)
        {
            return createInternalTile(graph, varType, macTileTag, {1, 1}, false, params, context);
        }

        int createInternalTile(KernelGraph&                     graph,
                               VariableType                     varType,
                               int                              macTileTag,
                               std::vector<unsigned int> const& numWaveTiles,
                               bool                             splitStore,
                               CommandParametersPtr             params,
                               ContextPtr                       context)
        {
            auto macTile = graph.coordinates.getNode<MacroTile>(macTileTag);

            std::vector<int> sizes = {macTile.sizes.at(0) / static_cast<int>(numWaveTiles[0]),
                                      macTile.sizes.at(1) / static_cast<int>(numWaveTiles[1])};

            if(macTile.memoryType == MemoryType::LDS)
            {
                auto internalTile       = MacroTile(sizes, MemoryType::VGPR, macTile.subTileSizes);
                internalTile.layoutType = macTile.layoutType;
                if(splitStore)
                    internalTile.memoryType = MemoryType::WAVE_SPLIT;
                return graph.coordinates.addElement(internalTile);
            }

            auto workgroupSizes = context->kernel()->workgroupSize();

            auto numWorkitems = product(workgroupSizes);
            auto numElements  = product(sizes);
            AssertFatal(
                numElements >= numWorkitems && numElements % numWorkitems == 0,
                "The number of elements must be an integer multiple of the number of workitems",
                ShowValue(numElements),
                ShowValue(numWorkitems));
            auto numElementsPerWorkitem = static_cast<int>(numElements / numWorkitems);
            auto thrTileM               = numElementsPerWorkitem;
            auto thrTileN               = 1;

            auto useSwappedAccess = params->transposeMemoryAccess[macTile.layoutType];
            if(splitStore)
                useSwappedAccess = false;

            // Load multiple smaller-precision (< 32-bit) elements into contiguous VGPRs
            bool packed     = false;
            uint packFactor = bitsPerRegister / DataTypeInfo::Get(varType).elementBits;

            if(auto packedVariableType = DataTypeInfo::Get(varType).packedVariableType())
            {
                packFactor = DataTypeInfo::Get(*packedVariableType).packing;
            }

            if(params->packMultipleElementsInto1VGPR && packFactor > 1
               && thrTileM % packFactor == 0)
            {
                thrTileM /= packFactor;
                thrTileN = packFactor;
                packed   = true;
            }

            auto direct2LDS         = macTile.memoryType == MemoryType::WAVE_Direct2LDS;
            auto useWiderDirect2LDS = direct2LDS
                                      && context->targetArchitecture().HasCapability(
                                          GPUCapability::HasWiderDirectToLds);

            // Enable the use of longer word instructions if possible
            if(params->enableLongDwordInstructions && (packed || packFactor <= 1)
               && (!direct2LDS || useWiderDirect2LDS))
            {
                auto maxWidth = std::min(context->kernelOptions().storeGlobalWidth,
                                         context->kernelOptions().loadLocalWidth);

                auto numDwordsPerElement = DataTypeInfo::Get(varType).registerCount;
                auto macTileM            = macTile.sizes[0];
                auto macTileN            = macTile.sizes[1];

                auto macTileFastMovingDimSize = !useSwappedAccess ? macTileM : macTileN;

                updateThreadTileForLongDwords(
                    thrTileM, thrTileN, maxWidth, macTileFastMovingDimSize, numDwordsPerElement);
            }

            if(!useSwappedAccess)
                std::swap(thrTileM, thrTileN);

            auto internalTile       = MacroTile(sizes, MemoryType::VGPR, {thrTileM, thrTileN});
            internalTile.layoutType = macTile.layoutType;
            if(splitStore)
                internalTile.memoryType = MemoryType::WAVE_SPLIT;

            auto internalTileTag = graph.coordinates.addElement(internalTile);
            Log::debug("  createInternalTile({}): {}x{} {} {}; subTileSizes {}x{}; packed {} ({})",
                       internalTileTag,
                       sizes[0],
                       sizes[1],
                       toString(internalTile.layoutType),
                       toString(varType.dataType),
                       thrTileM,
                       thrTileN,
                       packed,
                       packFactor);

            return internalTileTag;
        }

        /**
         * @brief Add coordinate-transforms for loading a MacroTile
         * from global into a ThreadTile.
         */
        void loadMacroTile_VGPR(KernelGraph&                     graph,
                                std::vector<DeferredConnection>& connections,
                                int                              userTag,
                                int                              macTileTag,
                                std::vector<int> const&          sdim,
                                std::vector<unsigned int> const& jammedTiles,
                                CommandParametersPtr             params,
                                ContextPtr                       context,
                                bool                             isDirect2LDS)
        {
            auto workgroupSizes = context->kernel()->workgroupSize();

            auto [nMacX, iMacX, nMacY, iMacY]
                = addLoadMacroTileCT(graph, connections, macTileTag, sdim);

            auto macTile          = *graph.coordinates.get<MacroTile>(macTileTag);
            auto useSwappedAccess = params->transposeMemoryAccess[macTile.layoutType];

            addLoadThreadTileCT(graph,
                                connections,
                                macTileTag,
                                iMacX,
                                iMacY,
                                workgroupSizes,
                                jammedTiles,
                                useSwappedAccess,
                                isDirect2LDS);

            graph.coordinates.addElement(DataFlow(), {userTag}, {macTileTag});
        }

        /**
         * @brief Add coordinate-transforms for loading a MacroTile
         * from global memory into a WaveTile.
         */
        void loadMacroTile_WAVE(KernelGraph&                     graph,
                                std::vector<DeferredConnection>& connections,
                                int                              userTag,
                                int                              macTileTag,
                                std::vector<int> const&          sdim,
                                VariableType const&              varType,
                                std::vector<unsigned int> const& jammedTiles,
                                CommandParametersPtr             params,
                                ContextPtr                       context)

        {
            auto wavefrontSize = context->kernel()->wavefront_size();

            auto [nMacX, iMacX, nMacY, iMacY]
                = addLoadMacroTileCT(graph, connections, macTileTag, sdim);

            addLoadWaveTileCT(graph,
                              connections,
                              macTileTag,
                              iMacX,
                              iMacY,
                              varType,
                              wavefrontSize,
                              true,
                              jammedTiles,
                              params,
                              context);

            graph.coordinates.addElement(DataFlow(), {userTag}, {macTileTag});
        }

        void loadMacroTile_SWIZZLE(KernelGraph&                     graph,
                                   std::vector<DeferredConnection>& connections,
                                   int                              loadTag,
                                   int                              userTag,
                                   int                              macTileTag,
                                   std::vector<int> const&          sdim,
                                   VariableType const&              varType,
                                   std::vector<unsigned int> const& jammedTiles,
                                   CommandParametersPtr             params,
                                   ContextPtr                       context)
        {
            auto wavefrontSize = context->kernel()->wavefront_size();

            auto [nMacX, iMacX, nMacY, iMacY]
                = addLoadMacroTileCT(graph, connections, macTileTag, sdim);

            addLoadSwizzleTileCT(graph,
                                 connections,
                                 macTileTag,
                                 iMacX,
                                 iMacY,
                                 varType,
                                 wavefrontSize,
                                 jammedTiles,
                                 params);

            graph.coordinates.addElement(DataFlow(), {userTag}, {macTileTag});
        }

        /**
         * @brief Add coordinate-transforms for storing a MacroTile
         * from a ThreadTile into global.
         */
        void storeMacroTile_VGPR(KernelGraph&                     graph,
                                 std::vector<DeferredConnection>& connections,
                                 int                              userTag,
                                 int                              macTileTag,
                                 std::vector<int> const&          sdim,
                                 std::vector<unsigned int> const& jammedTiles,
                                 CommandParametersPtr             params,
                                 ContextPtr                       context)
        {
            auto workgroupSizes = context->kernel()->workgroupSize();
            auto macTile        = *graph.coordinates.get<MacroTile>(macTileTag);

            auto useSwappedAccess = params->transposeMemoryAccess[macTile.layoutType];

            auto [nMacX, iMacX, nMacY, iMacY]
                = addStoreMacroTileCT(graph, connections, macTileTag, sdim, jammedTiles);

            addStoreThreadTileCT(graph,
                                 connections,
                                 macTileTag,
                                 iMacX,
                                 iMacY,
                                 workgroupSizes,
                                 jammedTiles,
                                 useSwappedAccess);

            graph.coordinates.addElement(DataFlow(), {macTileTag}, {userTag});
        }

        /**
         * @brief Add coordinate-transforms for storing a MacroTile
         * from a WaveTile into global memory.
         */
        void storeMacroTile_WAVE(KernelGraph&                     graph,
                                 std::vector<DeferredConnection>& connections,
                                 int                              userTag,
                                 int                              macTileTag,
                                 std::vector<int> const&          sdim,
                                 std::vector<unsigned int> const& jammedTiles,
                                 CommandParametersPtr             params,
                                 ContextPtr                       context)

        {
            auto wavefrontSize = context->kernel()->wavefront_size();

            auto [nMacX, iMacX, nMacY, iMacY]
                = addStoreMacroTileCT(graph, connections, macTileTag, sdim);

            addStoreWaveTileCT(
                graph, connections, macTileTag, iMacX, iMacY, wavefrontSize, jammedTiles, params);

            graph.coordinates.addElement(DataFlow(), {macTileTag}, {userTag});
        }

        /**
         * @brief Add coordinate-transforms for storing a MacroTile
         * from a WaveTile into global memory.
         */
        void storeMacroTile_WAVE_SPLIT(KernelGraph&                     graph,
                                       std::vector<DeferredConnection>& connections,
                                       int                              userTag,
                                       int                              tileTag,
                                       std::vector<int> const&          sdim,
                                       std::vector<unsigned int> const& jammedTiles,
                                       ContextPtr                       context)

        {
            auto wavefrontSize  = context->kernel()->wavefront_size();
            auto workgroupSizes = context->kernel()->workgroupSize();

            auto tile = graph.coordinates.getNode<MacroTile>(tileTag);

            auto iMacXStoreGlobal = graph.coordinates.addElement(tile.tileIndex(0));
            auto iMacYStoreGlobal = graph.coordinates.addElement(tile.tileIndex(1));

            auto [nMacX, iMacX, nMacY, iMacY]
                = addStoreMacroTileCT(graph, connections, tileTag, sdim, jammedTiles);

            if(jammedTiles[0] > 1)
            {
                auto jammedWavetileX = graph.coordinates.addElement(
                    JammedWaveTileNumber(0, literal(jammedTiles[0]), literal(1)));
                connections.push_back(DC<JammedWaveTileNumber>(jammedWavetileX, 0));
                graph.coordinates.addElement(
                    Flatten(), {jammedWavetileX, iMacXStoreGlobal}, {iMacX});
            }
            else
            {
                graph.coordinates.addElement(PassThrough(), {iMacXStoreGlobal}, {iMacX});
            }

            if(jammedTiles[1] > 1)
            {
                auto jammedWavetileY = graph.coordinates.addElement(
                    JammedWaveTileNumber(1, literal(jammedTiles[1]), literal(1)));
                connections.push_back(DC<JammedWaveTileNumber>(jammedWavetileY, 1));
                graph.coordinates.addElement(
                    Flatten(), {jammedWavetileY, iMacYStoreGlobal}, {iMacY});
            }
            else
            {
                graph.coordinates.addElement(PassThrough(), {iMacYStoreGlobal}, {iMacY});
            }

            addStoreAccumulatorTileCT(graph,
                                      connections,
                                      tileTag,
                                      iMacXStoreGlobal,
                                      iMacYStoreGlobal,
                                      wavefrontSize,
                                      workgroupSizes);

            graph.coordinates.addElement(DataFlow(), {tileTag}, {userTag});
        }

        /**
         * @brief Tile re-writer.
         */
        struct LowerTileVisitor : public BaseGraphVisitor
        {
            using BaseGraphVisitor::visitEdge;
            using BaseGraphVisitor::visitOperation;

            LowerTileVisitor(CommandParametersPtr params, ContextPtr context)

                : BaseGraphVisitor(context)
                , m_params(params)
                , m_kernel(context->kernel())
            {
            }

            virtual void visitEdge(KernelGraph&              graph,
                                   KernelGraph const&        original,
                                   GraphReindexer&           reindexer,
                                   int                       tag,
                                   ConstructMacroTile const& edge) override
            {
                // NOP: don't need this edge anymore
            }

            virtual void visitEdge(KernelGraph&             graph,
                                   KernelGraph const&       original,
                                   GraphReindexer&          reindexer,
                                   int                      tag,
                                   DestructMacroTile const& edge) override
            {
                // NOP: don't need this edge anymore
            }

            virtual void visitOperation(KernelGraph&       graph,
                                        KernelGraph const& original,
                                        GraphReindexer&    reindexer,
                                        int                tag,
                                        Exchange const&    exchange) override
            {
                auto logger = rocRoller::Log::getLogger();

                logger->debug("KernelGraph::ExchangeVisitor::Exchange({})", tag);

                auto originalMacTileTag = original.mapper.get<MacroTile>(tag);
                auto macTileTag         = reindexer.coordinates.at(originalMacTileTag);

                copyOperation(graph, original, reindexer, tag);

                auto macTile = graph.coordinates.getNode<MacroTile>(macTileTag);

                AssertFatal(macTile.rank == 2, "Rank /= 2 not implemented yet.");

                logger->debug(" MacroTile({}), Size: {}", macTileTag, macTile.sizes);

                std::vector<DeferredConnection> connections;

                if(macTile.memoryType == MemoryType::WAVE_SWIZZLE)
                {
                    std::optional<int> maybeWaveTileTag;
                    auto tags = graph.coordinates.getOutputNodeIndices<DataFlowEdge>(macTileTag);
                    for(auto tag : tags)
                    {
                        if(graph.coordinates.get<WaveTile>(tag))
                        {
                            maybeWaveTileTag = tag;
                            break;
                        }
                    }
                    AssertFatal(maybeWaveTileTag, "wavetile element not found");
                    auto waveTileTag = *maybeWaveTileTag;
                    auto waveTile    = *graph.coordinates.get<WaveTile>(waveTileTag);
                    auto iWaveX      = graph.coordinates.addElement(waveTile.tileIndex(0));
                    auto iWaveY      = graph.coordinates.addElement(waveTile.tileIndex(1));

                    uint const nSIMDBlock   = macTile.miTileSizes[2];
                    uint const nSIMDIndex   = 4 / nSIMDBlock;
                    uint const lanesPerSIMD = 16;

                    auto SIMDBlock = graph.coordinates.addElement(
                        Adhoc("SIMDBlock", literal(nSIMDBlock), nullptr));
                    auto SIMDIndex = graph.coordinates.addElement(
                        Adhoc("SIMDIndex", literal(nSIMDIndex), nullptr));
                    auto laneInSIMD
                        = graph.coordinates.addElement(Lane(literal(lanesPerSIMD), nullptr));

                    uint const numElements   = waveTile.elements();
                    auto       wavefrontSize = m_context->kernel()->wavefront_size();
                    uint const wfs           = static_cast<uint>(wavefrontSize);
                    uint const numVgpr       = numElements / wfs;
                    uint const nVgprIndex    = macTile.miTileSizes[2];
                    uint const nVgprBlock    = numVgpr / nVgprIndex;

                    auto vgprBlock = graph.coordinates.addElement(
                        VGPRBlockNumber(literal(nVgprBlock), literal(1u)));
                    auto vgprIndex = graph.coordinates.addElement(
                        VGPRBlockIndex(literal(nVgprIndex), literal(1u)));

                    connections.push_back(DC<WaveTile>(waveTileTag));
                    connections.push_back(DC<Adhoc>(SIMDBlock, 0));
                    connections.push_back(DC<Adhoc>(SIMDIndex, 1));
                    connections.push_back(DC<Lane>(laneInSIMD));
                    connections.push_back(DC<VGPRBlockNumber>(vgprBlock));
                    connections.push_back(DC<VGPRBlockIndex>(vgprIndex));

                    graph.coordinates.addElement(
                        Flatten(), {vgprIndex, SIMDIndex, laneInSIMD}, {iWaveX});
                    graph.coordinates.addElement(Flatten(), {vgprBlock, SIMDBlock}, {iWaveY});
                    graph.coordinates.addElement(Flatten(), {iWaveX, iWaveY}, {waveTileTag});
                }
                else
                    Throw<FatalError>("Exchange: MacroTile memory type not supported yet.",
                                      ShowValue(macTile.memoryType));

                auto exchangeTag = reindexer.control.at(tag);
                for(auto& dc : connections)
                {
                    graph.mapper.connect(exchangeTag, dc.coordinate, dc.connectionSpec);
                }
            }

            virtual void visitOperation(KernelGraph&       graph,
                                        KernelGraph const& original,
                                        GraphReindexer&    reindexer,
                                        int                tag,
                                        LoadTiled const&   oload) override
            {
                auto logger = rocRoller::Log::getLogger();

                logger->debug("KernelGraph::LowerTileVisitor::LoadTiled({})", tag);

                auto originalUserTag = original.mapper.get<User>(tag);
                auto originalTileTag = original.mapper.get<MacroTile>(tag);
                auto LDSTileTag      = original.mapper.get<LDS>(tag);
                auto isDirect2LDS    = (LDSTileTag != -1);
                auto userTag         = reindexer.coordinates.at(originalUserTag);
                auto tileTag         = reindexer.coordinates.at(originalTileTag);

                auto sdims
                    = original.coordinates.getOutputNodeIndices(originalUserTag, CT::isEdge<Split>)
                          .to<std::vector>();
                for(int i = 0; i < sdims.size(); i++)
                    sdims[i] = reindexer.coordinates.at(sdims[i]);
                AssertFatal(sdims.size() >= 2);

                copyOperation(graph, original, reindexer, tag);

                auto tile = graph.coordinates.getNode<MacroTile>(tileTag);

                AssertFatal(tile.rank == 2, "Rank /= 2 not implemented yet.");

                logger->debug("  User({}), MacroTile({}), Size: {}", userTag, tileTag, tile.sizes);

                std::vector<DeferredConnection> connections;

                auto loadTag               = reindexer.control.at(tag);
                auto varType               = getVariableType(graph, loadTag);
                auto wavetilesPerWavefront = m_params->getWaveTilesPerWavefront();
                switch(tile.memoryType)
                {
                case MemoryType::VGPR:
                    loadMacroTile_VGPR(graph,
                                       connections,
                                       userTag,
                                       tileTag,
                                       sdims,
                                       {1, 1},
                                       m_params,
                                       m_context,
                                       isDirect2LDS);
                    break;
                case MemoryType::WAVE:
                    loadMacroTile_WAVE(graph,
                                       connections,
                                       userTag,
                                       tileTag,
                                       sdims,
                                       varType,
                                       wavetilesPerWavefront,
                                       m_params,
                                       m_context);
                    break;
                case MemoryType::WAVE_SWIZZLE:
                    loadMacroTile_SWIZZLE(graph,
                                          connections,
                                          loadTag,
                                          userTag,
                                          tileTag,
                                          sdims,
                                          varType,
                                          wavetilesPerWavefront,
                                          m_params,
                                          m_context);
                    break;
                default:
                    Throw<FatalError>("LoadTiled: MacroTile memory type not supported yet.",
                                      ShowValue(tile.memoryType));
                }

                for(auto& dc : connections)
                {
                    graph.mapper.connect(loadTag, dc.coordinate, dc.connectionSpec);
                }
            }

            virtual void visitOperation(KernelGraph&       graph,
                                        KernelGraph const& original,
                                        GraphReindexer&    reindexer,
                                        int                tag,
                                        LoadLDSTile const& oload) override
            {
                auto logger = rocRoller::Log::getLogger();

                logger->debug("KernelGraph::LowerTileVisitor::LoadLDSTile({})", tag);

                auto originalLDSTag  = original.mapper.get<LDS>(tag);
                auto originalTileTag = original.mapper.get<MacroTile>(tag);

                copyOperation(graph, original, reindexer, tag);

                auto ldsTag  = reindexer.coordinates.at(originalLDSTag);
                auto tileTag = reindexer.coordinates.at(originalTileTag);
                auto tile    = graph.coordinates.getNode<MacroTile>(tileTag);

                AssertFatal(tile.rank == 2, "Rank /= 2 not implemented yet.");

                auto workgroupSizes        = m_context->kernel()->workgroupSize();
                auto wavefrontSize         = m_context->kernel()->wavefront_size();
                auto wavetilesPerWavefront = m_params->getWaveTilesPerWavefront();
                bool useSwappedAccess      = m_params->transposeMemoryAccess[tile.layoutType];

                logger->debug("  LDS({}), MacroTile({}), MacroTile size: {}x{}, SubTile size: "
                              "{}x{}, MemoryType {}, LayoutType {}, useSwappedAccess {}",
                              ldsTag,
                              tileTag,
                              tile.sizes[0],
                              tile.sizes[1],
                              tile.subTileSizes[0],
                              tile.subTileSizes[1],
                              toString(tile.memoryType),
                              toString(tile.layoutType),
                              useSwappedAccess);

                std::vector<DeferredConnection> connections;

                auto loadTag = reindexer.control.at(tag);

                auto iMacX = graph.coordinates.addElement(tile.tileIndex(0));
                auto iMacY = graph.coordinates.addElement(tile.tileIndex(1));

                if(tile.memoryType == MemoryType::WAVE)
                {
                    auto              varType     = getVariableType(graph, loadTag);
                    std::vector<uint> jammedTiles = wavetilesPerWavefront;
                    addLoadWaveTileCT(graph,
                                      connections,
                                      tileTag,
                                      iMacX,
                                      iMacY,
                                      varType,
                                      wavefrontSize,
                                      true,
                                      jammedTiles,
                                      m_params,
                                      m_context);
                }
                else if(tile.memoryType == MemoryType::WAVE_SPLIT)
                {
                    useSwappedAccess = false;
                    addLoadAccumulatorTileCT(
                        graph, connections, tileTag, iMacX, iMacY, wavefrontSize, workgroupSizes);
                }
                else
                {
                    std::vector<uint> jammedTiles = {1, 1};
                    addLoadThreadTileCT(graph,
                                        connections,
                                        tileTag,
                                        iMacX,
                                        iMacY,
                                        workgroupSizes,
                                        jammedTiles,
                                        false);
                }

                if(useSwappedAccess)
                    graph.coordinates.addElement(Tile(), {ldsTag}, {iMacX, iMacY});
                else
                    graph.coordinates.addElement(Tile(), {ldsTag}, {iMacY, iMacX});

                for(auto& dc : connections)
                {
                    graph.mapper.connect(loadTag, dc.coordinate, dc.connectionSpec);
                }
            }

            virtual void visitOperation(KernelGraph&       graph,
                                        KernelGraph const& original,
                                        GraphReindexer&    reindexer,
                                        int                tag,
                                        StoreTiled const&  ostore) override
            {
                auto logger = rocRoller::Log::getLogger();

                logger->debug("KernelGraph::LowerTileVisitor::StoreTiled({})", tag);

                auto originalUserTag = original.mapper.get<User>(tag);
                auto originalTileTag = original.mapper.get<MacroTile>(tag);

                copyOperation(graph, original, reindexer, tag);

                auto userTag = reindexer.coordinates.at(originalUserTag);
                auto tileTag = reindexer.coordinates.at(originalTileTag);
                auto user    = graph.coordinates.getNode<User>(userTag);
                auto tile    = graph.coordinates.getNode<MacroTile>(tileTag);

                AssertFatal(tile.rank == 2, ShowValue(tile.rank), "Rank /= 2 not implemented yet.");

                logger->debug("  User({}), MacroTile({}), MacroTile size: {}x{}, MemoryType {}",
                              userTag,
                              tileTag,
                              tile.sizes[0],
                              tile.sizes[1],
                              toString(tile.memoryType));

                auto sdims
                    = original.coordinates.getInputNodeIndices(originalUserTag, CT::isEdge<Join>)
                          .to<std::vector>();
                for(int i = 0; i < sdims.size(); i++)
                    sdims[i] = reindexer.coordinates.at(sdims[i]);
                AssertFatal(sdims.size() >= 2);

                std::vector<DeferredConnection> connections;

                auto storeTag              = reindexer.control.at(tag);
                auto wavetilesPerWavefront = m_params->getWaveTilesPerWavefront();

                switch(tile.memoryType)
                {
                case MemoryType::VGPR:
                    storeMacroTile_VGPR(graph,
                                        connections,
                                        userTag,
                                        tileTag,
                                        sdims,
                                        wavetilesPerWavefront,
                                        m_params,
                                        m_context);
                    break;
                case MemoryType::WAVE:
                    storeMacroTile_WAVE(graph,
                                        connections,
                                        userTag,
                                        tileTag,
                                        sdims,
                                        wavetilesPerWavefront,
                                        m_params,
                                        m_context);
                    break;
                case MemoryType::WAVE_SPLIT:
                    storeMacroTile_WAVE_SPLIT(graph,
                                              connections,
                                              userTag,
                                              tileTag,
                                              sdims,
                                              wavetilesPerWavefront,
                                              m_context);
                    break;
                default:
                    Throw<FatalError>("StoreTiled: MacroTile memory type not supported yet.");
                }

                for(auto& dc : connections)
                {
                    graph.mapper.connect(storeTag, dc.coordinate, dc.connectionSpec);
                }
            }

            virtual void visitOperation(KernelGraph&        graph,
                                        KernelGraph const&  original,
                                        GraphReindexer&     reindexer,
                                        int                 tag,
                                        StoreLDSTile const& ostore) override
            {
                auto logger = rocRoller::Log::getLogger();

                logger->debug("KernelGraph::LowerTileVisitor::StoreLDSTile({})", tag);

                auto originalLDSTag  = original.mapper.get<LDS>(tag);
                auto originalTileTag = original.mapper.get<MacroTile>(tag);

                copyOperation(graph, original, reindexer, tag);

                auto ldsTag       = reindexer.coordinates.at(originalLDSTag);
                auto tileTag      = reindexer.coordinates.at(originalTileTag);
                auto tile         = graph.coordinates.getNode<MacroTile>(tileTag);
                auto ldsTile      = graph.coordinates.getNode<LDS>(ldsTag);
                auto isDirect2LDS = ldsTile.isDirect2LDS;
                AssertFatal(tile.rank == 2, "Rank /= 2 not implemented yet.");

                auto workgroupSizes        = m_context->kernel()->workgroupSize();
                auto wavefrontSize         = m_context->kernel()->wavefront_size();
                auto wavetilesPerWavefront = m_params->getWaveTilesPerWavefront();
                auto useWaveAccess         = tile.memoryType == MemoryType::WAVE
                                     || tile.memoryType == MemoryType::WAVE_LDS;
                bool useSwappedAccess = m_params->transposeMemoryAccess[tile.layoutType];
                // XXX debug
                logger->debug("  LDS({}), MacroTile({}), MacroTile size: {}x{}, SubTile size: "
                              "{}x{}, MemoryType {}, LayoutType {}, useSwappedAccess {}",
                              ldsTag,
                              tileTag,
                              tile.sizes[0],
                              tile.sizes[1],
                              tile.subTileSizes[0],
                              tile.subTileSizes[1],
                              toString(tile.memoryType),
                              toString(tile.layoutType),
                              useSwappedAccess);

                std::vector<DeferredConnection> connections;

                auto iMacX = graph.coordinates.addElement(tile.tileIndex(0));
                auto iMacY = graph.coordinates.addElement(tile.tileIndex(1));

                if(tile.memoryType == MemoryType::VGPR)
                {
                    // We are storing entire workgroup tiles
                    std::vector<uint> jammedTiles = {1, 1};
                    addStoreThreadTileCT(graph,
                                         connections,
                                         tileTag,
                                         iMacX,
                                         iMacY,
                                         workgroupSizes,
                                         jammedTiles,
                                         useSwappedAccess,
                                         isDirect2LDS);
                }
                else
                {
                    // We are storing single wavefront tiles.  This
                    // currently assumes that epilogue blocks are
                    // serialized.
                    std::vector<uint> jammedTiles = {1, 1};
                    addStoreWaveTileCT(graph,
                                       connections,
                                       tileTag,
                                       iMacX,
                                       iMacY,
                                       wavefrontSize,
                                       jammedTiles,
                                       m_params);
                }

                if(useSwappedAccess)
                    graph.coordinates.addElement(Flatten(), {iMacX, iMacY}, {ldsTag});
                else
                    graph.coordinates.addElement(Flatten(), {iMacY, iMacX}, {ldsTag});

                auto storeTag = reindexer.control.at(tag);
                for(auto& dc : connections)
                {
                    graph.mapper.connect(storeTag, dc.coordinate, dc.connectionSpec);
                }
            }

        private:
            AssemblyKernelPtr    m_kernel;
            CommandParametersPtr m_params;
        };

        KernelGraph LowerTile::apply(KernelGraph const& graph)
        {
            TIMER(t, "KernelGraph::lowerTile");
            auto visitor = LowerTileVisitor(m_params, m_context);
            return rewrite(graph, visitor);
        }
    }
}
