/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025 AMD ROCm(TM) Software
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

#include <rocRoller/KernelGraph/Transforms/SwizzleScale.hpp>

#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/Simplify.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/KernelOptions_detail.hpp>

// TODO: Extract most of this to a detail header file and add tests

namespace rocRoller
{
    namespace KernelGraph
    {
        namespace CT = rocRoller::KernelGraph::CoordinateGraph;
        using GD     = rocRoller::Graph::Direction;
        using namespace ControlGraph;
        using namespace CoordinateGraph;
        namespace Expression = rocRoller::Expression;
        using namespace Expression;

        // map<loadTag, {multiplyTag, macroTileTag}>
        std::map<int, std::pair<int, int>>
            collectScaleLoadInfo(KernelGraph& graph, NaryArgument arg, int loopTag)
        {
            std::map<int, std::pair<int, int>> scaleLoads;
            auto loopBodies = graph.control.getOutputNodeIndices<Body>(loopTag).to<std::vector>();
            for(auto loopBodyTag : loopBodies)
            {
                std::unordered_map<int, int> scaleTileForMultiply;
                for(auto const multiplyTag : filter(graph.control.isElemType<Multiply>(),
                                                    graph.control.depthFirstVisit(loopBodyTag)))
                {
                    auto [scaleMacTag, scaleMac] = graph.getDimension<MacroTile>(
                        multiplyTag, Connections::typeArgument<MacroTile>(arg));

                    scaleTileForMultiply[scaleMacTag] = multiplyTag;
                }

                auto isLoad = [&](int tag) {
                    auto const& elem = graph.control.getElement(tag);
                    return isOperation<LoadTiled>(elem) || isOperation<LoadLDSTile>(elem);
                };

                for(auto const loadTag : filter(isLoad, graph.control.depthFirstVisit(loopBodyTag)))
                {
                    auto tileTag = graph.mapper.get<MacroTile>(loadTag);
                    if(scaleTileForMultiply.contains(tileTag))
                    {
                        // TODO: skip the swizzle pass for scale loaded via LDS.
                        if(isOperation<LoadLDSTile>(graph.control.getElement(loadTag)))
                            return std::map<int, std::pair<int, int>>();
                        scaleLoads[loadTag] = {scaleTileForMultiply[tileTag], tileTag};
                    }
                }
            }

            return scaleLoads;
        }

        void orderExchangesBeforeMultipliesInLoopBody(KernelGraph&       graph,
                                                      ContextPtr         context,
                                                      NaryArgument       arg,
                                                      std::map<int, int> tileExchangeMap,
                                                      std::map<int, std::pair<int, int>> scaleLoads,
                                                      int                                loopTag)
        {
            auto loopBodies = graph.control.getOutputNodeIndices<Body>(loopTag).to<std::vector>();
            for(auto loopBodyTag : loopBodies)
            {
                for(auto const multiplyTag : filter(graph.control.isElemType<Multiply>(),
                                                    graph.control.depthFirstVisit(loopBodyTag)))
                {

                    auto [tileTag, tile] = graph.getDimension<MacroTile>(
                        multiplyTag, Connections::typeArgument<MacroTile>(arg));

                    if(not tileExchangeMap.contains(tileTag))
                        continue;

                    Log::debug("Adding exchange-before-multiply Sequence edge from {} to {} for {}",
                               tileExchangeMap.at(tileTag),
                               multiplyTag,
                               toString(arg));

                    graph.control.addElement(
                        Sequence(), {tileExchangeMap.at(tileTag)}, {multiplyTag});
                }
            }
        }

        std::map<int, std::map<int, int>>
            filterLoadUnrollColouring(UnrollColouring const&                    colouring,
                                      std::map<int, std::pair<int, int>> const& scaleLoads)
        {
            AssertFatal(!scaleLoads.empty(), "Scale loads are not found");

            std::map<int, std::map<int, int>> rv;
            for(auto load = scaleLoads.cbegin(); load != scaleLoads.cend(); load++)
            {
                auto unrollMap = colouring.operationColour.at(load->first);
                rv.insert(std::make_pair(load->first, unrollMap));
            }

            return rv;
        }

        std::vector<DeferredConnection> addExchangeCT(KernelGraph& graph,
                                                      ContextPtr   context,
                                                      int          macTileTag,
                                                      int          waveTileTag,
                                                      NaryArgument arg)
        {
            auto macTile = graph.coordinates.getNode<MacroTile>(macTileTag);
            AssertFatal(macTile.memoryType == MemoryType::WAVE_SWIZZLE,
                        "Exchange: MacroTile memory type not supported yet.");
            AssertFatal(macTile.subTileSizes.at(0) == 64 || macTile.subTileSizes.at(0) == 32,
                        ShowValue(macTile.subTileSizes.at(0)));

            std::vector<DeferredConnection> connections;

            auto waveTile = graph.coordinates.get<WaveTile>(waveTileTag).value();
            auto iWaveX   = graph.coordinates.addElement(waveTile.tileIndex(0));
            auto iWaveY   = graph.coordinates.addElement(waveTile.tileIndex(1));

            auto       wavefrontSize = context->kernel()->wavefront_size();
            uint const lanesPerSIMD  = 16;
            uint const nSIMDsPerWave = wavefrontSize / lanesPerSIMD;
            uint const nSIMDIndex    = macTile.subTileSizes.at(0) / lanesPerSIMD;
            uint const nSIMDBlock    = nSIMDsPerWave / nSIMDIndex;
            auto       SIMDBlock
                = graph.coordinates.addElement(Adhoc("SIMDBlock", literal(nSIMDBlock), nullptr));
            auto SIMDIndex
                = graph.coordinates.addElement(Adhoc("SIMDIndex", literal(nSIMDIndex), nullptr));
            auto laneInSIMD = graph.coordinates.addElement(Lane(literal(lanesPerSIMD), nullptr));

            uint const numElements       = waveTile.elements();
            uint const activeLanesInWave = static_cast<uint>(wavefrontSize);
            uint const numVgpr           = numElements / activeLanesInWave;
            uint const nVgprIndex
                = std::min(nSIMDIndex, static_cast<uint>(macTile.miTileSizes.at(2)));
            // Minimal swizzle tile size 64x4 or 32x8 = 256
            uint const numElementsPerMinimalSwizzleTile = 256;
            uint const nVgprBlock = numElementsPerMinimalSwizzleTile / macTile.subTileSizes.at(0)
                                    / nSIMDBlock / nVgprIndex;
            uint const nBlocks = numVgpr / nVgprBlock / nVgprIndex;
            auto       vgprBlock
                = graph.coordinates.addElement(VGPRBlockNumber(literal(nVgprBlock), literal(1u)));
            auto vgprIndex
                = graph.coordinates.addElement(VGPRBlockIndex(literal(nVgprIndex), literal(1u)));
            auto block
                = graph.coordinates.addElement(Adhoc("Block", literal(nBlocks), literal(1u)));
            auto vgpr = graph.coordinates.addElement(VGPR(literal(numVgpr), literal(1u)));

            graph.coordinates.addElement(Tile(), {vgpr}, {block, vgprBlock, vgprIndex});

            uint const nSIMDIndexBlock = nVgprIndex;
            uint const nSIMDIndexIndex = nSIMDIndex / nSIMDIndexBlock;
            auto       SIMDIndexBlock  = graph.coordinates.addElement(
                Adhoc("SIMDIndexBlock", literal(nSIMDIndexBlock), nullptr));
            auto SIMDIndexIndex = graph.coordinates.addElement(
                Adhoc("SIMDIndexIndex", literal(nSIMDIndexIndex), nullptr));

            graph.coordinates.addElement(Flatten(), {SIMDIndexBlock, SIMDIndexIndex}, {SIMDIndex});

            connections.push_back(DC<WaveTile>(waveTileTag));
            connections.push_back(DC<Adhoc>(SIMDBlock, 0));
            connections.push_back(DC<Adhoc>(SIMDIndex, 1));
            connections.push_back(DC<Adhoc>(SIMDIndexBlock, 2));
            connections.push_back(DC<Adhoc>(SIMDIndexIndex, 3));
            connections.push_back(DC<Lane>(laneInSIMD));
            connections.push_back(DC<VGPRBlockNumber>(vgprBlock));
            connections.push_back(DC<VGPRBlockIndex>(vgprIndex));
            connections.push_back(DC<VGPR>(vgpr));

            if(arg == NaryArgument::LHS_SCALE)
            {
                if(macTile.subTileSizes.at(0) == 64)
                {
                    graph.coordinates.addElement(
                        Flatten(), {vgprIndex, SIMDIndexIndex, laneInSIMD}, {iWaveX});
                    graph.coordinates.addElement(
                        Flatten(), {block, SIMDBlock, vgprBlock, SIMDIndexBlock}, {iWaveY});
                    graph.coordinates.addElement(Flatten(), {iWaveX, iWaveY}, {waveTileTag});
                }
                else if(macTile.subTileSizes.at(0) == 32 && macTile.miTileSizes.at(0) == 16)
                {
                    graph.coordinates.addElement(Flatten(), {vgprIndex, laneInSIMD}, {iWaveX});
                    graph.coordinates.addElement(
                        Flatten(), {block, vgprBlock, SIMDBlock, SIMDIndex}, {iWaveY});
                    graph.coordinates.addElement(Flatten(), {iWaveX, iWaveY}, {waveTileTag});
                }
                else if(macTile.subTileSizes.at(0) == 32 && macTile.miTileSizes.at(0) == 32)
                {
                    graph.coordinates.addElement(Flatten(), {SIMDIndex, laneInSIMD}, {iWaveX});
                    graph.coordinates.addElement(
                        Flatten(), {block, vgprIndex, SIMDBlock, vgprBlock}, {iWaveY});
                    graph.coordinates.addElement(Flatten(), {iWaveX, iWaveY}, {waveTileTag});
                }
            }
            if(arg == NaryArgument::RHS_SCALE)
            {
                if(macTile.subTileSizes.at(0) == 64)
                {
                    graph.coordinates.addElement(
                        Flatten(), {vgprIndex, SIMDIndexIndex, laneInSIMD}, {iWaveY});
                    graph.coordinates.addElement(
                        Flatten(), {block, SIMDBlock, vgprBlock, SIMDIndexBlock}, {iWaveX});
                    graph.coordinates.addElement(Flatten(), {iWaveY, iWaveX}, {waveTileTag});
                }
                else if(macTile.subTileSizes.at(0) == 32 && macTile.miTileSizes.at(0) == 16)
                {
                    graph.coordinates.addElement(Flatten(), {vgprIndex, laneInSIMD}, {iWaveY});
                    graph.coordinates.addElement(
                        Flatten(), {block, vgprBlock, SIMDBlock, SIMDIndex}, {iWaveX});
                    graph.coordinates.addElement(Flatten(), {iWaveY, iWaveX}, {waveTileTag});
                }
                else if(macTile.subTileSizes.at(0) == 32 && macTile.miTileSizes.at(0) == 32)
                {
                    graph.coordinates.addElement(Flatten(), {SIMDIndex, laneInSIMD}, {iWaveY});
                    graph.coordinates.addElement(
                        Flatten(), {block, vgprIndex, SIMDBlock, vgprBlock}, {iWaveX});
                    graph.coordinates.addElement(Flatten(), {iWaveY, iWaveX}, {waveTileTag});
                }
            }

            return connections;
        }

        std::tuple<std::vector<DeferredConnection>,
                   std::vector<DeferredConnection>,
                   std::map<int, int>>
            addSwizzleLoadCT(KernelGraph& graph, ContextPtr context, int tag, NaryArgument arg)
        {
            AssertFatal(arg == NaryArgument::LHS_SCALE || arg == NaryArgument::RHS_SCALE);

            std::vector<DeferredConnection> connections;

            auto wavefrontSize = context->kernel()->wavefront_size();

            auto existingMacTile
                = graph.coordinates.getNode<MacroTile>(graph.mapper.get<MacroTile>(tag));
            AssertFatal(existingMacTile.subTileSizes.size() == 4, "Invalid tile specification");

            // create new macrotile
            auto macTile    = MacroTile(existingMacTile.sizes,
                                     existingMacTile.layoutType,
                                     existingMacTile.swizzleTileSizes,
                                     MemoryType::WAVE_SWIZZLE,
                                     existingMacTile.subTileSizes);
            auto macTileTag = graph.coordinates.addElement(macTile);
            connections.push_back(DC<MacroTile>(macTileTag));

            auto iMac0 = graph.coordinates.addElement(macTile.tileIndex(0));
            auto iMac1 = graph.coordinates.addElement(macTile.tileIndex(1));

            auto isLoadTiled = graph.control.get<LoadTiled>(tag).has_value();
            if(isLoadTiled)
            {
                auto userTag = graph.mapper.get<User>(tag);
                AssertFatal(userTag != -1, "User coordinate associated with LoadTiled not found");

                // copy user
                auto user = graph.coordinates.addElement(graph.coordinates.getElement(userTag));
                connections.push_back(DC<User>(user));

                // copy sdims
                auto existingSDims
                    = graph.coordinates.getOutputNodeIndices(userTag, CT::isEdge<Split>)
                          .to<std::vector>();
                std::vector<int> sDims;
                for(int i = 0; i < existingSDims.size(); i++)
                {
                    sDims.push_back(graph.coordinates.addElement(
                        graph.coordinates.getElement(existingSDims[i])));
                }
                graph.coordinates.addElement(Split(), std::vector<int>{user}, sDims);

                auto numTiles0 = tileCeilDivide(graph.coordinates.get<SubDimension>(sDims[0])->size,
                                                macTile.sizes[0]);

                auto numTiles1 = tileCeilDivide(graph.coordinates.get<SubDimension>(sDims[1])->size,
                                                macTile.sizes[1]);

                connections.push_back(DC<SubDimension>(sDims[0], 0));
                connections.push_back(DC<SubDimension>(sDims[1], 1));

                auto nMac0 = graph.coordinates.addElement(macTile.tileNumber(0, numTiles0));
                auto nMac1 = graph.coordinates.addElement(macTile.tileNumber(1, numTiles1));

                connections.push_back(DC<MacroTileNumber>(nMac0, 0));
                connections.push_back(DC<MacroTileNumber>(nMac1, 1));

                graph.coordinates.addElement(Tile(), {sDims[0]}, {nMac0, iMac0});
                graph.coordinates.addElement(Tile(), {sDims[1]}, {nMac1, iMac1});

                auto existingMacTileNum0 = graph.mapper.get<MacroTileNumber>(tag, 0);
                AssertFatal(existingMacTileNum0 != -1,
                            "MacroTileNumber 0 coordinate associated with LoadTiled not found");
                auto location = graph.coordinates.getLocation(existingMacTileNum0);
                for(auto const& output : location.outgoing)
                {
                    auto edge = graph.coordinates.getElement(output);
                    auto outTags
                        = graph.coordinates.getNeighbours<Graph::Direction::Downstream>(output);
                    graph.coordinates.addElement(edge, std::vector<int>{nMac0}, outTags);
                }

                auto existingMacTileNum1 = graph.mapper.get<MacroTileNumber>(tag, 1);
                AssertFatal(existingMacTileNum1 != -1,
                            "MacroTileNumber 1 coordinate associated with LoadTiled not found");
                location = graph.coordinates.getLocation(existingMacTileNum1);
                for(auto const& output : location.outgoing)
                {
                    auto edge = graph.coordinates.getElement(output);
                    auto outTags
                        = graph.coordinates.getNeighbours<Graph::Direction::Downstream>(output);
                    graph.coordinates.addElement(edge, std::vector<int>{nMac1}, outTags);
                }

                graph.coordinates.addElement(DataFlow(), {user}, {macTileTag});
            }

            auto isLoadLDSTile = graph.control.get<LoadLDSTile>(tag).has_value();
            if(isLoadLDSTile)
            {
                auto ldsTag = graph.mapper.get<LDS>(tag);
                AssertFatal(ldsTag != -1, "LDS coordinate associated with LoadLDSTile not found");

                // copy lds
                auto lds = graph.coordinates.addElement(graph.coordinates.getElement(ldsTag));
                graph.coordinates.addElement(View(), {lds}, {ldsTag});
                if(arg == NaryArgument::LHS_SCALE)
                    graph.coordinates.addElement(Tile(), {lds}, {iMac0, iMac1});
                if(arg == NaryArgument::RHS_SCALE)
                    graph.coordinates.addElement(Tile(), {lds}, {iMac1, iMac0});
                connections.push_back(DC<LDS>(lds));
            }

            auto waveTile    = WaveTile(macTile);
            auto waveTileTag = graph.coordinates.addElement(waveTile);

            connections.push_back(DC<WaveTile>(waveTileTag));

            auto nWave0 = graph.coordinates.addElement(waveTile.tileNumber(0));
            auto nWave1 = graph.coordinates.addElement(waveTile.tileNumber(1));
            auto iWave0 = graph.coordinates.addElement(waveTile.tileIndex(0));
            auto iWave1 = graph.coordinates.addElement(waveTile.tileIndex(1));

            graph.coordinates.addElement(Tile(), {iMac0}, {nWave0, iWave0});
            graph.coordinates.addElement(Tile(), {iMac1}, {nWave1, iWave1});

            graph.coordinates.addElement(Tile(), {waveTileTag}, {iWave0, iWave1});

            connections.push_back(DC<WaveTileNumber>(nWave0, 0));
            connections.push_back(DC<WaveTileNumber>(nWave1, 1));

            uint const nLaneInSIMD   = 16;
            uint const nSIMDsPerWave = wavefrontSize / nLaneInSIMD;
            uint const nSIMDIndex    = macTile.subTileSizes.at(0) / nLaneInSIMD;
            auto       SIMDIndex
                = graph.coordinates.addElement(Adhoc("SIMDIndex", literal(nSIMDIndex), nullptr));
            auto laneInSIMD = graph.coordinates.addElement(Lane(literal(nLaneInSIMD), nullptr));

            uint const nSIMDBlock = nSIMDsPerWave / nSIMDIndex;
            auto       SIMDBlock
                = graph.coordinates.addElement(Adhoc("SIMDBlock", literal(nSIMDBlock), nullptr));

            uint const numElements       = waveTile.elements();
            uint const activeLanesInWave = static_cast<uint>(wavefrontSize);
            uint const numVgpr           = numElements / activeLanesInWave;
            uint const nVgprIndex
                = std::min(nSIMDIndex, static_cast<uint>(macTile.miTileSizes.at(2)));
            // Minimal swizzle tile size 64x4 or 32x8 = 256
            uint const numElementsPerMinimalSwizzleTile = 256;
            uint const nVgprBlock = numElementsPerMinimalSwizzleTile / macTile.subTileSizes.at(0)
                                    / nSIMDBlock / nVgprIndex;
            uint const nBlocks = numVgpr / nVgprBlock / nVgprIndex;
            auto       vgprBlock
                = graph.coordinates.addElement(VGPRBlockNumber(literal(nVgprBlock), literal(1u)));
            auto vgprIndex
                = graph.coordinates.addElement(VGPRBlockIndex(literal(nVgprIndex), literal(1u)));
            auto vgpr = graph.coordinates.addElement(VGPR(literal(numVgpr), literal(1u)));
            auto block
                = graph.coordinates.addElement(Adhoc("Block", literal(nBlocks), literal(1u)));
            graph.coordinates.addElement(Flatten(), {block, vgprBlock, vgprIndex}, {vgpr});
            connections.push_back(DC<VGPRBlockNumber>(vgprBlock));
            connections.push_back(DC<VGPRBlockIndex>(vgprIndex));
            connections.push_back(DC<VGPR>(vgpr));

            auto activeLanesInWaveLiteral = literal(activeLanesInWave);

            auto wave  = graph.coordinates.addElement(Wavefront(-1));
            auto wave0 = graph.coordinates.addElement(Wavefront(0));
            auto wave1 = graph.coordinates.addElement(Wavefront(1));
            graph.coordinates.addElement(Flatten(), {wave0, wave1}, {wave});

            auto workitem = graph.coordinates.addElement(Workitem(0));
            auto lane = graph.coordinates.addElement(Lane(activeLanesInWaveLiteral, literal(1u)));
            graph.coordinates.addElement(Flatten(), {wave, lane}, {workitem});
            graph.coordinates.addElement(Flatten(), {SIMDBlock, SIMDIndex, laneInSIMD}, {lane});

            std::map<int, int> unrolls;

            auto existingUnroll0 = graph.mapper.get<Unroll>(tag, 0);
            auto existingUnroll1 = graph.mapper.get<Unroll>(tag, 1);
            auto existingUnroll2 = graph.mapper.get<Unroll>(tag, 2);

            if(arg == NaryArgument::LHS_SCALE)
            {
                graph.coordinates.addElement(Tile(), {iWave0}, {SIMDIndex, laneInSIMD});
                graph.coordinates.addElement(
                    Tile(), {iWave1}, {block, SIMDBlock, vgprBlock, vgprIndex});

                if(existingUnroll0 != -1)
                {
                    auto factor = macTile.subTileSizes.at(0) / existingMacTile.subTileSizes.at(0);

                    auto jammedWaveTile0 = graph.coordinates.addElement(JammedWaveTileNumber(
                        0, literal(getUnrollSize(graph, existingUnroll0) / factor), literal(1)));
                    graph.coordinates.addElement(Tile(), {nWave0}, {wave0, jammedWaveTile0});
                    connections.push_back(DC<JammedWaveTileNumber>(jammedWaveTile0, 0));

                    auto unroll0 = existingUnroll0;
                    graph.coordinates.addElement(PassThrough(), {jammedWaveTile0}, {unroll0});
                    connections.push_back(DC<Unroll>(unroll0, 0));
                    unrolls[existingUnroll0] = unroll0;
                }
                if(existingUnroll1 != -1)
                {
                    auto factor = macTile.subTileSizes.at(2) / existingMacTile.subTileSizes.at(2);

                    auto unroll1 = existingUnroll1;
                    graph.coordinates.addElement(PassThrough(), {nWave1}, {unroll1});
                    connections.push_back(DC<Unroll>(unroll1, 1));
                    unrolls[existingUnroll1] = unroll1;
                }
                if(existingUnroll2 != -1)
                {
                    unrolls[existingUnroll2] = existingUnroll2;
                }
            }

            if(arg == NaryArgument::RHS_SCALE)
            {
                graph.coordinates.addElement(Tile(), {iWave1}, {SIMDIndex, laneInSIMD});
                graph.coordinates.addElement(
                    Tile(), {iWave0}, {block, SIMDBlock, vgprBlock, vgprIndex});

                if(existingUnroll1 != -1)
                {
                    auto factor = macTile.subTileSizes.at(1) / existingMacTile.subTileSizes.at(1);

                    auto jammedWaveTile1 = graph.coordinates.addElement(JammedWaveTileNumber(
                        1, literal(getUnrollSize(graph, existingUnroll1) / factor), literal(1)));
                    graph.coordinates.addElement(Tile(), {nWave1}, {wave1, jammedWaveTile1});
                    connections.push_back(DC<JammedWaveTileNumber>(jammedWaveTile1, 1));

                    auto unroll1 = existingUnroll1;
                    graph.coordinates.addElement(PassThrough(), {jammedWaveTile1}, {unroll1});
                    connections.push_back(DC<Unroll>(unroll1, 1));
                    unrolls[existingUnroll1] = unroll1;
                }
                if(existingUnroll0 != -1)
                {
                    auto factor = macTile.subTileSizes.at(2) / existingMacTile.subTileSizes.at(2);

                    auto unroll0 = existingUnroll0;
                    graph.coordinates.addElement(PassThrough(), {nWave0}, {unroll0});
                    connections.push_back(DC<Unroll>(unroll0, 0));
                    unrolls[existingUnroll0] = unroll0;
                }
                if(existingUnroll2 != -1)
                {
                    unrolls[existingUnroll2] = existingUnroll2;
                }
            }

            auto exchangeConnections = addExchangeCT(graph, context, macTileTag, waveTileTag, arg);

            return {connections, exchangeConnections, unrolls};
        }

        std::pair<int, int> getOuterMergeFactors(KernelGraph const& graph, int macTileTag)
        {
            auto macTile = graph.coordinates.getNode<MacroTile>(macTileTag);
            auto waveM   = macTile.subTileSizes.at(0);
            auto waveN   = macTile.subTileSizes.at(1);
            auto waveK   = macTile.subTileSizes.at(2);

            AssertFatal(
                waveM == waveN, "waveM is not equal to waveN", ShowValue(waveM), ShowValue(waveN));

            auto waveSwizzleM = macTile.swizzleTileSizes.at(0);
            auto waveSwizzleN = macTile.swizzleTileSizes.at(1);
            auto waveSwizzleK = macTile.swizzleTileSizes.at(2);

            AssertFatal(waveSwizzleM == waveSwizzleN,
                        "waveSwizzleM is not equal to waveSwizzleN",
                        ShowValue(waveSwizzleM),
                        ShowValue(waveSwizzleN));

            return std::make_pair(waveSwizzleM / waveM, waveSwizzleK / waveK);
        }

        std::pair<int, int> getInnerMergeFactors(KernelGraph const& graph, int macTileTag)
        {
            auto macTile = graph.coordinates.getNode<MacroTile>(macTileTag);
            auto waveM   = macTile.subTileSizes.at(0);
            auto waveN   = macTile.subTileSizes.at(1);
            auto waveK   = macTile.subTileSizes.at(2);

            AssertFatal(
                waveM == waveN, "waveM is not equal to waveN", ShowValue(waveM), ShowValue(waveN));

            auto const waveSwizzleM = macTile.swizzleTileSizes.at(0);
            auto const waveSwizzleN = macTile.swizzleTileSizes.at(1);
            AssertFatal(waveSwizzleM == waveSwizzleN,
                        "waveSwizzleM is not equal to waveSwizzleN",
                        ShowValue(waveSwizzleM),
                        ShowValue(waveSwizzleN));
            // Minimal swizzle tile size 64x4 or 32x8 = 256
            uint const numElementsPerMinimalSwizzleTile = 256;
            auto const waveSwizzleK = numElementsPerMinimalSwizzleTile / waveSwizzleM;

            return std::make_pair(waveSwizzleM / waveM, waveSwizzleK / waveK);
        }

        std::map<int, std::vector<std::pair<int, int>>>
            findMergeableLoads(KernelGraph const&                        graph,
                               std::map<int, std::pair<int, int>> const& scaleLoads,
                               std::map<int, std::map<int, int>>&        loadUnrollMap,
                               NaryArgument                              arg)
        {
            AssertFatal(!scaleLoads.empty() && !loadUnrollMap.empty());

            std::map<int, std::vector<std::pair<int, int>>> mergeables;

            auto mergeLoadsByUnroll = [&](int fastDim,
                                          int slowDim0,
                                          int slowDim1,
                                          int innerFactor,
                                          int outerFactor,
                                          int indexFactor = 0) {
                AssertFatal(innerFactor <= outerFactor);
                if(innerFactor <= 1 && outerFactor <= 1)
                    return;

                // (slowDimVal1, slowDimVal0, fastDimVal, load)
                std::map<int, std::map<int, std::map<int, int>>> unrollLoadMap;
                for(auto const load : loadUnrollMap)
                {
                    auto unrollMap = loadUnrollMap[load.first];
                    AssertFatal(
                        unrollMap.contains(fastDim), ShowValue(load.first), ShowValue(fastDim));
                    AssertFatal(
                        unrollMap.contains(slowDim0), ShowValue(load.first), ShowValue(slowDim0));
                    AssertFatal(slowDim1 == -1 || unrollMap.contains(slowDim1),
                                ShowValue(load.first),
                                ShowValue(slowDim1));
                    int slowDimVal1 = (slowDim1 == -1) ? 0 : unrollMap[slowDim1];
                    for(auto const unroll : unrollMap)
                    {
                        if(unroll.first == fastDim)
                        {
                            unrollLoadMap[slowDimVal1][unrollMap[slowDim0]][unrollMap[fastDim]]
                                = load.first;
                        }
                    }
                }

                for(auto const sDim1 : unrollLoadMap)
                {
                    for(auto const sDim0 : sDim1.second)
                    {
                        int mergeOp = -1;
                        for(auto const fDim : sDim0.second)
                        {
                            if(fDim.first % outerFactor == 0)
                            {
                                mergeOp = fDim.second;
                                loadUnrollMap[mergeOp][fastDim] /= outerFactor;
                            }
                            else
                            {
                                AssertFatal(mergeOp != -1);
                                auto order = graph.control.compareNodes(
                                    rocRoller::UpdateCache, mergeOp, fDim.second);
                                AssertFatal(graph.control.compareNodes(
                                                rocRoller::UpdateCache, mergeOp, fDim.second)
                                                == NodeOrdering::LeftFirst,
                                            ShowValue(mergeOp),
                                            ShowValue(fDim.second),
                                            ShowValue(order));
                                loadUnrollMap.erase(fDim.second);

                                int index  = 0;
                                int factor = fDim.first * indexFactor;
                                if(fDim.first % innerFactor == 0)
                                    index = fDim.first / innerFactor;

                                // insertion order matters here
                                mergeables[mergeOp].push_back(
                                    std::make_pair(fDim.second, index + factor));
                                if(mergeables.count(fDim.second) > 0)
                                {
                                    for(auto& pair : mergeables[fDim.second])
                                    {
                                        if(pair.second > 0)
                                            pair.second += factor;
                                    }
                                    mergeables[mergeOp].insert(mergeables[mergeOp].end(),
                                                               mergeables[fDim.second].begin(),
                                                               mergeables[fDim.second].end());
                                    mergeables.erase(fDim.second);
                                }
                            }
                        }
                    }
                }
            };

            auto sampleLoad = loadUnrollMap.begin()->first;
            auto unroll0    = graph.mapper.get<Unroll>(sampleLoad, 0);
            auto unroll1    = graph.mapper.get<Unroll>(sampleLoad, 1);
            auto unroll2    = graph.mapper.get<Unroll>(sampleLoad, 2);

            // if unroll2 is -1, this returns 1.
            auto unrollKSize = getUnrollSize(graph, unroll2);

            auto sampleTile                    = scaleLoads.begin()->second.second;
            auto [innerFactorMN, innerFactorK] = getInnerMergeFactors(graph, sampleTile);
            auto [outerFactorMN, outerFactorK] = getOuterMergeFactors(graph, sampleTile);
            AssertFatal(
                innerFactorMN == outerFactorMN, ShowValue(innerFactorMN), ShowValue(outerFactorMN));
            AssertFatal(outerFactorMN % innerFactorMN == 0,
                        ShowValue(innerFactorMN),
                        ShowValue(outerFactorMN));

            if(arg == NaryArgument::LHS_SCALE)
            {
                // A : M x K
                auto xUnrollSize    = getUnrollSize(graph, unroll0);
                auto macKUnrollSize = getUnrollSize(graph, unroll1);

                // merge scale loads
                if(xUnrollSize % innerFactorMN == 0 && macKUnrollSize % innerFactorK == 0)
                {
                    AssertFatal((macKUnrollSize * unrollKSize) % outerFactorK == 0,
                                ShowValue(macKUnrollSize),
                                ShowValue(unrollKSize),
                                ShowValue(outerFactorK));

                    mergeLoadsByUnroll(unroll0, unroll1, unroll2, innerFactorMN, outerFactorMN);
                    if(macKUnrollSize % outerFactorK == 0)
                        mergeLoadsByUnroll(unroll1, unroll0, unroll2, innerFactorK, outerFactorK);
                    else
                    {
                        mergeLoadsByUnroll(unroll1, unroll0, unroll2, innerFactorK, macKUnrollSize);
                        mergeLoadsByUnroll(unroll2,
                                           unroll1,
                                           unroll0,
                                           outerFactorK / macKUnrollSize,
                                           outerFactorK / macKUnrollSize,
                                           macKUnrollSize / innerFactorK);
                    }
                }
            }

            if(arg == NaryArgument::RHS_SCALE)
            {
                // B : K x N
                auto yUnrollSize    = getUnrollSize(graph, unroll1);
                auto macKUnrollSize = getUnrollSize(graph, unroll0);

                // merge scale loads
                if(yUnrollSize % innerFactorMN == 0 && macKUnrollSize % innerFactorK == 0)
                {
                    AssertFatal((macKUnrollSize * unrollKSize) % outerFactorK == 0,
                                ShowValue(macKUnrollSize),
                                ShowValue(unrollKSize),
                                ShowValue(outerFactorK));

                    mergeLoadsByUnroll(unroll1, unroll0, unroll2, innerFactorMN, outerFactorMN);
                    if(macKUnrollSize % outerFactorK == 0)
                        mergeLoadsByUnroll(unroll0, unroll1, unroll2, innerFactorK, outerFactorK);
                    else
                    {
                        mergeLoadsByUnroll(unroll0, unroll1, unroll2, innerFactorK, macKUnrollSize);
                        mergeLoadsByUnroll(unroll2,
                                           unroll0,
                                           unroll1,
                                           outerFactorK / macKUnrollSize,
                                           outerFactorK / macKUnrollSize,
                                           macKUnrollSize / innerFactorK);
                    }
                }
            }

            return mergeables;
        }

        void
            swizzleScaleLoads(KernelGraph& graph, ContextPtr context, NaryArgument arg, int loopTag)
        {
            auto scaleLoads = collectScaleLoadInfo(graph, arg, loopTag);

            if(scaleLoads.empty())
            {
                // TODO: Change this to let RR know that the SwizzleScale transform was applied but didn't do anything
                Log::debug("Unable to find SwizzleScale candidates");
                return;
            }

            auto colouring = colourByUnrollValue(graph);

            auto loadUnrollMap = filterLoadUnrollColouring(colouring, scaleLoads);
            if(loadUnrollMap.empty())
                return;

            auto mergeables = findMergeableLoads(graph, scaleLoads, loadUnrollMap, arg);

            if(mergeables.empty())
                return;

            auto sampleLoad = mergeables.begin()->first;
            auto [loadConnections, exchangeConnections, unrollReindexMap]
                = addSwizzleLoadCT(graph, context, sampleLoad, arg);

            std::map<int, int> tileExchangeMap;

            for(auto const load : mergeables)
            {
                // add coordinate connections for LoadTiled
                for(auto const& dc : loadConnections)
                {
                    graph.mapper.connect(load.first, dc.coordinate, dc.connectionSpec);
                }

                // make a copy of MacroTile for separate register tagging
                if(load.first != sampleLoad)
                    duplicateMacroTile(graph, load.first);

                // add exchange node after load
                auto exchange
                    = graph.control.addElement(Exchange(getVariableType(graph, load.first)));
                auto topOp = getTopSetCoordinate(graph, load.first);
                graph.control.addElement(Sequence(), {topOp}, {exchange});

                // add coordinate connections for Exchange
                for(auto const& dc : exchangeConnections)
                {
                    graph.mapper.connect(exchange, dc.coordinate, dc.connectionSpec);
                }

                // Since the load tile size (e.g. 64x4, 64x8, 64x12, 64x16) can be
                // greater than equal to the exchange tile size (64x4),
                // add index edge to point to the register allocation (subset).
                auto tileTag = graph.mapper.get<MacroTile>(load.first);
                auto tile    = graph.coordinates.getNode<MacroTile>(tileTag);
                AssertFatal(tile.miTileSizes.size() == 4, ShowValue(tile.miTileSizes.size()));

                auto exchangeTileTag = graph.coordinates.addElement(tile);
                graph.coordinates.addElement(Index(0), {exchangeTileTag}, {tileTag});
                graph.mapper.connect<MacroTile>(exchange, exchangeTileTag);

                auto destMacTileTag = context->kernelOptions()->scaleSkipPermlane
                                          ? exchangeTileTag
                                          : graph.coordinates.addElement(MacroTile());

                graph.mapper.connect(exchange, destMacTileTag, NaryArgument::DEST);

                auto createNode
                    = [&context](int idx) -> rocRoller::KernelGraph::CoordinateGraph::Edge {
                    if(context->kernelOptions()->scaleSkipPermlane)
                        return Segment(idx);

                    return Index(idx);
                };

                // add index edge to point to exchange output tile.
                int index = 0;

                graph.coordinates.addElement(
                    createNode(index++), {scaleLoads.at(load.first).second}, {destMacTileTag});

                tileExchangeMap[scaleLoads.at(load.first).second] = exchange;

                // merge the loads
                for(auto const merge : load.second)
                {
                    auto mergeTopOp = getTopSetCoordinate(graph, merge.first);
                    auto ordering   = graph.control.compareNodes(
                        rocRoller::UseCacheIfAvailable, topOp, mergeTopOp);
                    AssertFatal(ordering == NodeOrdering::LeftFirst);
                    auto replaceOp = graph.control.addElement(NOP());
                    if(merge.second > 0)
                    {
                        exchange = graph.control.addElement(
                            Exchange(getVariableType(graph, load.first)));
                        graph.control.addElement(Sequence(), {replaceOp}, {exchange});

                        // add coordinate connections for Exchange
                        for(auto const& dc : exchangeConnections)
                        {
                            graph.mapper.connect(exchange, dc.coordinate, dc.connectionSpec);
                        }

                        // Since the load tile size (e.g. 64x4, 64x8, 64x12, 64x16) can be
                        // greater than equal to the exchange tile size (64x4),
                        // add index edge to point to the register allocation (subset).
                        exchangeTileTag = graph.coordinates.addElement(tile);
                        graph.coordinates.addElement(
                            Index(merge.second), {exchangeTileTag}, {tileTag});
                        graph.mapper.connect<MacroTile>(exchange, exchangeTileTag);

                        destMacTileTag = context->kernelOptions()->scaleSkipPermlane
                                             ? exchangeTileTag
                                             : graph.coordinates.addElement(MacroTile());
                        graph.mapper.connect(exchange, destMacTileTag, NaryArgument::DEST);

                        // reset the index
                        index = 0;
                    }
                    replaceWith(graph, mergeTopOp, replaceOp, false);
                    purgeNodeAndChildren(graph, mergeTopOp);

                    graph.coordinates.addElement(
                        createNode(index++), {scaleLoads.at(merge.first).second}, {destMacTileTag});

                    tileExchangeMap[scaleLoads.at(merge.first).second] = exchange;
                }

                // Reindex the SetCoordinate chain value and its Unroll coordinate connection
                auto tag                = load.first;
                auto remainingReindexes = unrollReindexMap;
                while(not remainingReindexes.empty())
                {
                    auto maybeSetCoordinate = findContainingOperation<SetCoordinate>(tag, graph);
                    if(not maybeSetCoordinate)
                    {
                        std::string missingUnrolls = "{";
                        for(auto const [unroll, _] : remainingReindexes)
                            missingUnrolls + std::to_string(unroll) + " ";

                        AssertFatal(maybeSetCoordinate,
                                    "Cannot reindex, SetCoordinate not found for unroll(s):",
                                    ShowValue(tag),
                                    ShowValue(missingUnrolls));
                    }
                    tag = maybeSetCoordinate.value();

                    auto unroll = graph.mapper.get<Unroll>(tag);
                    AssertFatal(unroll > 0,
                                "SetCoordinate is not connected to the Unroll dimension",
                                ShowValue(tag),
                                ShowValue(unroll));

                    auto newOp
                        = SetCoordinate(Expression::literal(loadUnrollMap[load.first][unroll]));
                    graph.control.setElement(tag, newOp);

                    auto newUnroll = unrollReindexMap.at(unroll);
                    graph.mapper.disconnect<Unroll>(tag, unroll);
                    graph.mapper.connect<Unroll>(tag, newUnroll);

                    AssertFatal(remainingReindexes.contains(unroll),
                                "Expected unroll to be in remaining reindexes: ",
                                ShowValue(unroll));
                    remainingReindexes.erase(unroll);
                }
            }

            orderExchangesBeforeMultipliesInLoopBody(
                graph, context, arg, tileExchangeMap, scaleLoads, loopTag);
        }

        KernelGraph SwizzleScale::apply(KernelGraph const& original)
        {
            if(!m_params->swizzleScale)
                return original;

            // TODO: enable SwizzleScale when transA == T or transB == N
            AssertFatal(m_params->transposeMemoryAccess[LayoutType::MATRIX_A]
                            && !m_params->transposeMemoryAccess[LayoutType::MATRIX_B],
                        "Non-TN is not supported by SwizzleScale");

            auto newGraph = original;

            auto const rootTag = newGraph.control.roots().only().value();

            auto findNamedLoopsBelow = [&](auto startTag, auto name) {
                std::vector<int> loopTags;
                for(auto const loop : filter(newGraph.control.isElemType<ForLoopOp>(),
                                             newGraph.control.depthFirstVisit(startTag)))
                {
                    auto forloop = newGraph.control.get<ForLoopOp>(loop).value();
                    if(forloop.loopName == name)
                    {
                        loopTags.push_back(loop);
                    }
                }
                return loopTags;
            };

            // Support kernels with multiple distinct KLoops
            auto kLoopTags = findNamedLoopsBelow(rootTag, KLOOP);
            AssertFatal(not kLoopTags.empty(), "Kernel must contain at least one KLoop");

            for(auto const kLoopTag : kLoopTags)
            {
                swizzleScaleLoads(newGraph, m_context, NaryArgument::LHS_SCALE, kLoopTag);
                swizzleScaleLoads(newGraph, m_context, NaryArgument::RHS_SCALE, kLoopTag);

                auto kLoopTailTags = findNamedLoopsBelow(kLoopTag, KLOOPTAIL);
                AssertFatal(kLoopTailTags.size() <= 1, "Each KLoop can have at most one KLoopTail");
                if(not kLoopTailTags.empty())
                {
                    auto const kLoopTailTag = kLoopTailTags[0];
                    swizzleScaleLoads(newGraph, m_context, NaryArgument::LHS_SCALE, kLoopTailTag);
                    swizzleScaleLoads(newGraph, m_context, NaryArgument::RHS_SCALE, kLoopTailTag);
                }
            }

            removeRedundantSequenceEdges(newGraph);

            return newGraph;
        }
    }
}
