// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/CommandSolution_fwd.hpp>
#include <rocRoller/DataTypes/DataTypes_Utils.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/KernelGraph/Transforms/AddTransposeLoadCT.hpp>
#include <rocRoller/KernelGraph/Transforms/LowerTile.hpp>
#include <rocRoller/KernelGraph/Transforms/LowerTile_details.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>
#include <rocRoller/KernelOptions_detail.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Operations/Operations.hpp>
#include <rocRoller/Scheduling/LDSModel.hpp>
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

        namespace LowerTileDetails
        {
            bool isTileOfSubDwordTypeWithNonContiguousVGPRBlocks(DataType            type,
                                                                 MatrixMultiplySizes mi)
            {
                if(isF16(type))
                {
                    return ((mi.m == 16) && (mi.n == 16) && (mi.k == 32))
                           || ((mi.m == 32) && (mi.n == 32) && (mi.k == 16));
                }
                else if(isF8(type))
                {
                    return ((mi.m == 16) && (mi.n == 16) && (mi.k == 128))
                           || ((mi.m == 32) && (mi.n == 32) && (mi.k == 64));
                }
                else if(isF6(type))
                {
                    return ((mi.m == 16) && (mi.n == 16) && (mi.k == 128))
                           || ((mi.m == 32) && (mi.n == 32) && (mi.k == 64));
                }
                else if(isF4(type))
                {
                    return ((mi.m == 16) && (mi.n == 16) && (mi.k == 128))
                           || ((mi.m == 32) && (mi.n == 32) && (mi.k == 64));
                }
                return false;
            }
        }

        using namespace LowerTileDetails;

        /**
         * Parameters for LDS bank swizzle column permutation.
         * Permutes K-column assignments per tile row to eliminate
         * ds_read_b128 bank conflicts.  See docs/src/LDSSwizzling.md.
         */
        /// Number of dwordx4 columns per LDS bank row.
        unsigned int getColumnsPerBankRow(ContextPtr context)
        {
            auto gfx      = context->targetArchitecture().target().gfx;
            auto numBanks = Scheduling::LDSModel::getNumLDSBanks(
                gfx, {Scheduling::LDSModel::LdsDirection::Read}, 4 /*dwordx4*/);
            return numBanks / 4u; // each dwordx4 spans 4 banks
        }

        using LDSSwizzleParams = LDSSwizzleDetail::LDSSwizzleParams;

        /**
         * Check whether LDS bank swizzle is active for a given layout type.
         */
        bool isLDSSwizzleActive(LDSBankSwizzleMode swizzleMode,
                                LayoutType         layoutType,
                                DataType           dataType)
        {
            if(isScaleType(dataType))
                return false;

            return swizzleMode == LDSBankSwizzleMode::Swizzle
                   && (layoutType == LayoutType::MATRIX_A || layoutType == LayoutType::MATRIX_B);
        }

        ConstraintStatus NoConstructDestructMT(const KernelGraph& k)
        {
            TIMER(t, "Constraint::NoConstructDestructMT");

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

        std::tuple<int, int, int, int>
            addLoadMacroTileCTPreTiled(KernelGraph&                     graph,
                                       std::vector<DeferredConnection>& connections,
                                       int                              macTileTag,
                                       std::vector<int> const&          sdim,
                                       bool                             updatePreTiledSubDimStrides)
        {
            namespace CT = rocRoller::KernelGraph::CoordinateGraph;
            using ET     = rocRoller::Expression::EvaluationTime;

            AssertFatal(sdim.size() == 4, "Expected 4 sub-dimensions for Pretiled MacroTile CT.");

            auto tile = graph.coordinates.getNode<MacroTile>(macTileTag);

            auto sdimX       = sdim[0];
            auto sdimY       = sdim[1];
            auto sdimPTXTile = sdim[2];
            auto sdimPTYTile = sdim[3];

            auto sizeX = graph.coordinates.get<SubDimension>(sdimX)->size;
            auto sizeY = graph.coordinates.get<SubDimension>(sdimY)->size;

            auto sizePTTileXExpr = graph.coordinates.get<SubDimension>(sdimPTXTile)->size;
            AssertFatal(
                evaluationTimes(sizePTTileXExpr)[ET::Translate],
                "Size of pre-tile tile must be known at translate time (ie, must be literal).");

            auto sizePTTileYExpr = graph.coordinates.get<SubDimension>(sdimPTYTile)->size;
            AssertFatal(
                evaluationTimes(sizePTTileYExpr)[ET::Translate],
                "Size of pre-tile tile must be known at translate time (ie, must be literal).");

            auto sizePTTileX = getUnsignedInt(evaluate(sizePTTileXExpr));
            auto sizePTTileY = getUnsignedInt(evaluate(sizePTTileYExpr));

            if(updatePreTiledSubDimStrides)
            {
                auto const strideDataType = DataType::UInt32;

                auto stridePTTileXExpr = graph.coordinates.get<SubDimension>(sdimPTXTile)->stride;
                auto stridePTTileX     = getUnsignedInt(evaluate(stridePTTileXExpr));
                auto columnMajor       = stridePTTileX == 1u;

                auto subDimX = graph.coordinates.get<SubDimension>(sdimX).value();
                auto subDimY = graph.coordinates.get<SubDimension>(sdimY).value();

                auto tileSizeX = literal(sizePTTileX, strideDataType);
                auto tileSizeY = literal(sizePTTileY, strideDataType);

                if(columnMajor)
                {
                    subDimX.stride = tileSizeX * tileSizeY;
                    subDimY.stride = (subDimX.size / tileSizeX) * tileSizeX * tileSizeY;
                }
                else
                {
                    subDimX.stride = (subDimY.size / tileSizeY) * tileSizeX * tileSizeY;
                    subDimY.stride = tileSizeX * tileSizeY;
                }

                graph.coordinates.setElement(sdimX, subDimX);
                graph.coordinates.setElement(sdimY, subDimY);
            }

            auto numPTTilesX = tileCeilDivide(sizeX, sizePTTileX);
            auto numPTTilesY = tileCeilDivide(sizeY, sizePTTileY);

            Log::debug("PreTiled CT: numPTTilesX: {}", toString(numPTTilesX));
            Log::debug("PreTiled CT: numPTTilesY: {}", toString(numPTTilesY));
            Log::debug("PreTiled CT: sizePTTileXExpr: {}", toString(sizePTTileXExpr));
            Log::debug("PreTiled CT: sizePTTileYExpr: {}", toString(sizePTTileYExpr));

            auto nPTX = graph.coordinates.addElement(CT::WaveTileNumber(0, numPTTilesX, nullptr));
            auto nPTY = graph.coordinates.addElement(CT::WaveTileNumber(1, numPTTilesY, nullptr));
            auto iPTX
                = graph.coordinates.addElement(CT::WaveTileIndex(0, literal(sizePTTileX), nullptr));
            auto iPTY
                = graph.coordinates.addElement(CT::WaveTileIndex(1, literal(sizePTTileY), nullptr));

            graph.coordinates.addElement(PassThrough(), {sdimX}, {nPTX});
            graph.coordinates.addElement(PassThrough(), {sdimY}, {nPTY});

            graph.coordinates.addElement(PassThrough(), {sdimPTXTile}, {iPTX});
            graph.coordinates.addElement(PassThrough(), {sdimPTYTile}, {iPTY});

            connections.push_back(DC<SubDimension>(sdimPTXTile, 0));
            connections.push_back(DC<SubDimension>(sdimPTYTile, 1));

            auto numTilesX = tileCeilDivide(sizeX, tile.sizes[0]);
            auto numTilesY = tileCeilDivide(sizeY, tile.sizes[1]);

            Log::debug("PreTiled CT: numTilesX: {}", toString(numTilesX));
            Log::debug("PreTiled CT: numTilesY: {}", toString(numTilesY));

            auto nX = graph.coordinates.addElement(tile.tileNumber(0, numTilesX));
            auto nY = graph.coordinates.addElement(tile.tileNumber(1, numTilesY));
            auto iX = graph.coordinates.addElement(tile.tileIndex(0));
            auto iY = graph.coordinates.addElement(tile.tileIndex(1));

            AssertFatal(tile.sizes[0] % sizePTTileX == 0,
                        "Pre-tile size mismatch: tile.sizes[0] must be divisible by sizePTTileX.",
                        ShowValue(tile.sizes[0]),
                        ShowValue(sizePTTileX));
            AssertFatal(tile.sizes[1] % sizePTTileY == 0,
                        "Pre-tile size mismatch: tile.sizes[1] must be divisible by sizePTTileY.",
                        ShowValue(tile.sizes[1]),
                        ShowValue(sizePTTileY));
            AssertFatal(tile.sizes[0] / sizePTTileX > 0,
                        "Bad pre-tile size: tile.sizes[0] must be greater than sizePTTileX.",
                        ShowValue(tile.sizes[0]),
                        ShowValue(sizePTTileX));
            AssertFatal(tile.sizes[1] / sizePTTileY > 0,
                        "Bad pre-tile size: tile.sizes[1] must be greater than sizePTTileY.",
                        ShowValue(tile.sizes[1]),
                        ShowValue(sizePTTileY));

            auto numPTTilesPerTileX = tile.sizes[0] / sizePTTileX;
            auto numPTTilesPerTileY = tile.sizes[1] / sizePTTileY;

            auto nPTSubX = graph.coordinates.addElement(
                CT::WaveTileNumber(0, literal(numPTTilesPerTileX), nullptr));
            auto nPTSubY = graph.coordinates.addElement(
                CT::WaveTileNumber(1, literal(numPTTilesPerTileY), nullptr));

            graph.coordinates.addElement(Tile(), {nPTX}, {nX, nPTSubX});
            graph.coordinates.addElement(Tile(), {nPTY}, {nY, nPTSubY});

            graph.coordinates.addElement(Flatten(), {nPTSubX, iPTX}, {iX});
            graph.coordinates.addElement(Flatten(), {nPTSubY, iPTY}, {iY});

            connections.push_back(DC<MacroTileNumber>(nX, 0));
            connections.push_back(DC<MacroTileNumber>(nY, 1));

            return {nX, iX, nY, iY};
        }

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
                               std::vector<int> const&          sdim,
                               bool                             updatePreTiledSubDimStrides)
        {
            if(sdim.size() == 4)
                return addLoadMacroTileCTPreTiled(
                    graph, connections, macTileTag, sdim, updatePreTiledSubDimStrides);

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
         * from row/column coordinates `iWaveX` and `iWaveY` for
         * MI instructions that need data along fast-moving
         * dimension non-contiguously across VGPRBlocks.
         *
         * The `lane` and `element` parameters are existing
         * coordinates corresponding to a Lane coordiante and VGPR
         * coordinate (which should be thought of as which
         * element/item is being addressed).
         */
        void addLoadWaveTileViaNonContiguousVGPRBlocksCT(
            KernelGraph&                     graph,
            std::vector<DeferredConnection>& connections,
            int                              iWaveX,
            int                              iWaveY,
            int                              lane,
            int                              element,
            MatrixMultiplySizes              mi,
            uint                             bitsPerElement,
            int                              wavefrontSize)

        {
            uint const lanesPerSIMD = 16;
            uint const simdsPerWave = wavefrontSize / lanesPerSIMD;

            uint const simdsPerSGroup = mi.m / lanesPerSIMD;
            // We should find a name for this 2x factor between wave32 & wave64.
            uint const numVBlocks = wavefrontSize == 64 ? (bitsPerElement == 8 ? 2 : 1)
                                                        : (bitsPerElement == 8 ? 4 : 2);

            uint const elementsPerVGPRBlock = ((mi.m * mi.k) / wavefrontSize) / numVBlocks;

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

            connections.push_back(DC<VGPRBlockNumber>(elementBlockNumber));
            connections.push_back(DC<VGPRBlockIndex>(elementBlockIndex));

            graph.coordinates.addElement(Tile(), {iWaveX}, {simdBlockIndex, laneInSIMD});

            graph.coordinates.addElement(
                Tile(), {iWaveY}, {elementBlockNumber, simdBlockNumber, elementBlockIndex});

            graph.coordinates.addElement(Flatten(), {simdBlockNumber, simdBlockIndex}, {SIMD});
            graph.coordinates.addElement(Flatten(), {SIMD, laneInSIMD}, {lane});
            graph.coordinates.addElement(
                Flatten(), {elementBlockNumber, elementBlockIndex}, {element});
        }

        void addLoadSwizzleTileCT(KernelGraph&                     graph,
                                  std::vector<DeferredConnection>& connections,
                                  int                              macTileTag,
                                  int                              iMacX,
                                  int                              iMacY,
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

            uint const nLaneInSIMD   = 16;
            uint const nSIMDsPerWave = wavefrontSize / nLaneInSIMD;
            uint const nSIMDIndex    = macTile.subTileSizes.at(0) / nLaneInSIMD;
            auto       SIMDIndex
                = graph.coordinates.addElement(Adhoc("SIMDIndex", literal(nSIMDIndex), nullptr));
            auto laneInSIMD = graph.coordinates.addElement(Lane(literal(nLaneInSIMD), nullptr));

            graph.coordinates.addElement(Tile(), {iWaveX}, {SIMDIndex, laneInSIMD});

            uint const nSIMDBlock = nSIMDsPerWave / nSIMDIndex;
            auto       SIMDBlock
                = graph.coordinates.addElement(Adhoc("SIMDBlock", literal(nSIMDBlock), nullptr));

            uint const numElements       = waveTile.elements();
            uint       activeLanesInWave = static_cast<uint>(wavefrontSize);
            uint const numVgpr           = numElements / activeLanesInWave;
            uint const nVgprIndex
                = std::min(nSIMDIndex, static_cast<uint>(macTile.miTileSizes.at(2)));
            uint const nVgprBlock
                = numElements / macTile.subTileSizes.at(0) / nSIMDBlock / nVgprIndex;
            auto vgprBlock
                = graph.coordinates.addElement(VGPRBlockNumber(literal(nVgprBlock), literal(1u)));
            auto vgprIndex
                = graph.coordinates.addElement(VGPRBlockIndex(literal(nVgprIndex), literal(1u)));
            auto vgpr = graph.coordinates.addElement(VGPR(literal(numVgpr), literal(1u)));
            graph.coordinates.addElement(Flatten(), {vgprBlock, vgprIndex}, {vgpr});
            connections.push_back(DC<VGPRBlockNumber>(vgprBlock));
            connections.push_back(DC<VGPRBlockIndex>(vgprIndex));
            connections.push_back(DC<VGPR>(vgpr));

            graph.coordinates.addElement(Tile(), {iWaveY}, {SIMDBlock, vgpr});

            auto activeLanesInWaveLiteral = literal(activeLanesInWave);

            auto wave  = graph.coordinates.addElement(Wavefront(-1));
            auto waveX = graph.coordinates.addElement(Wavefront(0));
            auto waveY = graph.coordinates.addElement(Wavefront(1));
            graph.coordinates.addElement(Flatten(), {waveX, waveY}, {wave});

            auto workitem = graph.coordinates.addElement(Workitem(0));
            auto lane = graph.coordinates.addElement(Lane(activeLanesInWaveLiteral, literal(1u)));
            connections.push_back(DC<Lane>(lane));
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

        void addStoreSwizzleTileCT(KernelGraph&                     graph,
                                   std::vector<DeferredConnection>& connections,
                                   int                              macTileTag,
                                   int                              iMacX,
                                   int                              iMacY,
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

            graph.coordinates.addElement(Flatten(), {nWaveX, iWaveX}, {iMacX});
            graph.coordinates.addElement(Flatten(), {nWaveY, iWaveY}, {iMacY});

            graph.coordinates.addElement(Flatten(), {iWaveX, iWaveY}, {waveTileTag});

            connections.push_back(DC<WaveTileNumber>(nWaveX, 0));
            connections.push_back(DC<WaveTileNumber>(nWaveY, 1));

            uint const nLaneInSIMD   = 16;
            uint const nSIMDsPerWave = wavefrontSize / nLaneInSIMD;
            uint const nSIMDIndex    = macTile.subTileSizes.at(0) / nLaneInSIMD;
            auto       SIMDIndex
                = graph.coordinates.addElement(Adhoc("SIMDIndex", literal(nSIMDIndex), nullptr));
            auto laneInSIMD = graph.coordinates.addElement(Lane(literal(nLaneInSIMD), nullptr));

            graph.coordinates.addElement(Flatten(), {SIMDIndex, laneInSIMD}, {iWaveX});

            uint const nSIMDBlock = nSIMDsPerWave / nSIMDIndex;
            auto       SIMDBlock
                = graph.coordinates.addElement(Adhoc("SIMDBlock", literal(nSIMDBlock), nullptr));

            uint const numElements       = waveTile.elements();
            uint       activeLanesInWave = static_cast<uint>(wavefrontSize);
            uint const numVgpr           = numElements / activeLanesInWave;
            uint const nVgprIndex
                = macTile.subTileSizes.at(2) / (activeLanesInWave / macTile.subTileSizes.at(0));
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

            graph.coordinates.addElement(Flatten(), {SIMDBlock, vgpr}, {iWaveY});

            auto activeLanesInWaveLiteral = literal(activeLanesInWave);

            auto wave  = graph.coordinates.addElement(Wavefront(-1));
            auto waveX = graph.coordinates.addElement(Wavefront(0));
            auto waveY = graph.coordinates.addElement(Wavefront(1));
            graph.coordinates.addElement(Tile(), {wave}, {waveX, waveY});

            auto workitem = graph.coordinates.addElement(Workitem(0));
            auto lane = graph.coordinates.addElement(Lane(activeLanesInWaveLiteral, literal(1u)));
            connections.push_back(DC<Lane>(lane));
            graph.coordinates.addElement(Tile(), {workitem}, {wave, lane});
            graph.coordinates.addElement(Tile(), {lane}, {SIMDBlock, SIMDIndex, laneInSIMD});

            auto jammedWavetileX = graph.coordinates.addElement(
                JammedWaveTileNumber(0, literal(jammedTiles[0]), literal(1)));
            graph.coordinates.addElement(Flatten(), {jammedWavetileX, waveX}, {nWaveX});
            connections.push_back(DC<JammedWaveTileNumber>(jammedWavetileX, 0));

            auto jammedWavetileY = graph.coordinates.addElement(
                JammedWaveTileNumber(1, literal(jammedTiles[1]), literal(1)));
            graph.coordinates.addElement(Flatten(), {jammedWavetileY, waveY}, {nWaveY});
            connections.push_back(DC<JammedWaveTileNumber>(jammedWavetileY, 1));

            graph.coordinates.addElement(DataFlow(), {waveTileTag}, {macTileTag});
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
                               DataType const&                  dataType,
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

            uint numElements       = waveTile.elements();
            uint activeLanesInWave = static_cast<uint>(wavefrontSize);
            uint numGPR            = numElements / activeLanesInWave;

            const MatrixMultiplySizes mi{.m = macTile.subTileSizes[0],
                                         .n = macTile.subTileSizes[1],
                                         .k = macTile.subTileSizes[2]};
            uint                      K_L = mi.k / (activeLanesInWave / mi.m);

            auto activeLanesInWaveLiteral = literal(activeLanesInWave);

            auto lane = graph.coordinates.addElement(Lane(activeLanesInWaveLiteral, nullptr));
            auto vgpr = graph.coordinates.addElement(VGPR(literal(numGPR), nullptr));

            graph.coordinates.addElement(Flatten(), {waveX, waveY}, {wave});
            graph.coordinates.addElement(Flatten(), {wave, lane}, {workitem});

            connections.push_back(DC<Lane>(lane));
            connections.push_back(DC<VGPR>(vgpr));

            auto bitsPerElement = DataTypeInfo::Get(dataType).elementBits;

            const auto& arch = context->targetArchitecture();

            bool isTransposeLayout = params->transposeMemoryAccess[waveTile.layout];

            switch(waveTile.layout)
            {
            case LayoutType::MATRIX_A:
            {
                auto jammedWavetileX = graph.coordinates.addElement(
                    JammedWaveTileNumber(0, literal(jammedTiles[0]), literal(1)));
                connections.push_back(DC<JammedWaveTileNumber>(jammedWavetileX, 0));

                if(isTileOfSubDwordTypeWithNonContiguousVGPRBlocks(dataType, mi))
                {
                    AssertFatal((wavefrontSize == 64 || wavefrontSize == 32));

                    if(isFromLDS && isTransposableTile(arch, mi, dataType) && !isTransposeLayout)
                    {
                        Log::debug("Adding transpose-load CT for A macTileTag {}", macTileTag);
                        addTransposeLoadWaveTileCT(context,
                                                   connections,
                                                   graph,
                                                   macTileTag,
                                                   iWaveX,
                                                   iWaveY,
                                                   lane,
                                                   vgpr,
                                                   mi,
                                                   bitsPerElement,
                                                   activeLanesInWave);
                    }
                    else
                    {
                        addLoadWaveTileViaNonContiguousVGPRBlocksCT(graph,
                                                                    connections,
                                                                    iWaveX,
                                                                    iWaveY,
                                                                    lane,
                                                                    vgpr,
                                                                    mi,
                                                                    bitsPerElement,
                                                                    activeLanesInWave);
                    }
                }
                else
                {
                    AssertFatal(
                        K_L != 0,
                        "Invalid operand: cannot divide by zero to compute the BlockNumber");
                    auto blockNumber = graph.coordinates.addElement(
                        VGPRBlockNumber(literal(mi.k / K_L), nullptr));
                    auto blockIndex
                        = graph.coordinates.addElement(VGPRBlockIndex(literal(K_L), nullptr));

                    connections.push_back(DC<VGPRBlockNumber>(blockNumber));
                    connections.push_back(DC<VGPRBlockIndex>(blockIndex));

                    graph.coordinates.addElement(Tile(), {iWaveY}, {blockNumber, blockIndex});

                    graph.coordinates.addElement(Flatten(), {blockNumber, iWaveX}, {lane});
                    graph.coordinates.addElement(PassThrough(), {blockIndex}, {vgpr});
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

                if(isTileOfSubDwordTypeWithNonContiguousVGPRBlocks(dataType, mi))
                {
                    AssertFatal((wavefrontSize == 64 || wavefrontSize == 32));

                    if(isFromLDS && isTransposableTile(arch, mi, dataType) && isTransposeLayout)
                    {
                        Log::debug("Adding transpose-load CT for B macTileTag {}", macTileTag);
                        addTransposeLoadWaveTileCT(context,
                                                   connections,
                                                   graph,
                                                   macTileTag,
                                                   iWaveY,
                                                   iWaveX,
                                                   lane,
                                                   vgpr,
                                                   mi,
                                                   bitsPerElement,
                                                   activeLanesInWave);
                    }
                    else
                    {
                        addLoadWaveTileViaNonContiguousVGPRBlocksCT(graph,
                                                                    connections,
                                                                    iWaveY,
                                                                    iWaveX,
                                                                    lane,
                                                                    vgpr,
                                                                    mi,
                                                                    bitsPerElement,
                                                                    activeLanesInWave);
                    }
                }
                else
                {
                    AssertFatal(
                        K_L != 0,
                        "Invalid operand: cannot divide by zero to compute the BlockNumber");
                    auto blockNumber = graph.coordinates.addElement(
                        VGPRBlockNumber(literal(mi.k / K_L), nullptr));
                    auto blockIndex
                        = graph.coordinates.addElement(VGPRBlockIndex(literal(K_L), nullptr));

                    connections.push_back(DC<VGPRBlockNumber>(blockNumber));
                    connections.push_back(DC<VGPRBlockIndex>(blockIndex));

                    graph.coordinates.addElement(Tile(), {iWaveX}, {blockNumber, blockIndex});

                    graph.coordinates.addElement(Flatten(), {blockNumber, iWaveY}, {lane});
                    graph.coordinates.addElement(PassThrough(), {blockIndex}, {vgpr});
                }

                if(params->swizzleScale)
                    graph.coordinates.addElement(Tile(), {nWaveY}, {waveY, jammedWavetileY});
                else
                    graph.coordinates.addElement(Tile(), {nWaveY}, {jammedWavetileY, waveY});
            }
            break;

            case LayoutType::MATRIX_ACCUMULATOR:
            {
                auto numGPRsPerLaneInWave
                    = activeLanesInWave == 64 ? 4 : 8; // A wave can have 256 GPR in total
                uint simdsPerWave = activeLanesInWave / 16u; // Each SIMD consists of 16 lanes
                auto simdsPerWaveLiteral = literal(simdsPerWave);
                auto unitStride          = literal(1u);

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
                auto nLblk = graph.coordinates.addElement(Adhoc(
                    "LANEBlockNumber", literal(activeLanesInWave / simdsPerWave), unitStride));
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
         * When `rightmostFastest` is true, both of these orders are
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
                                 bool                               rightmostFastest,
                                 bool                               isGlobalToLDS,
                                 bool                               ldsSwizzle,
                                 unsigned int                       columnsPerBankRow)
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

            if(macTile.layoutType == LayoutType::MATRIX_A
               || macTile.layoutType == LayoutType::MATRIX_B
               || macTile.layoutType == LayoutType::MATRIX_ACCUMULATOR
               || macTile.layoutType == LayoutType::SCRATCH)
            {
                // GEMM layouts: use rightmostFastest to control threadtile orientation
                if(rightmostFastest)
                {
                    elementNumberX = graph.coordinates.addElement(
                        ElementNumber(0, literal(thrTile.sizes.at(0))));
                    elementNumberY = graph.coordinates.addElement(
                        ElementNumber(1, literal(thrTile.sizes.at(1))));

                    graph.coordinates.addElement(PassThrough(), {iThrX}, {elementNumberX});
                    graph.coordinates.addElement(PassThrough(), {iThrY}, {elementNumberY});

                    graph.coordinates.addElement(Flatten(), {nThrX, nThrY}, {workitemX});
                }
                else
                {
                    elementNumberX = graph.coordinates.addElement(
                        ElementNumber(0, literal(thrTile.sizes.at(1))));
                    elementNumberY = graph.coordinates.addElement(
                        ElementNumber(1, literal(thrTile.sizes.at(0))));

                    graph.coordinates.addElement(PassThrough(), {iThrX}, {elementNumberY});
                    graph.coordinates.addElement(PassThrough(), {iThrY}, {elementNumberX});

                    graph.coordinates.addElement(Flatten(), {nThrY, nThrX}, {workitemX});
                }
            }
            else
            {
                elementNumberX
                    = graph.coordinates.addElement(ElementNumber(0, literal(thrTile.sizes.at(0))));
                elementNumberY
                    = graph.coordinates.addElement(ElementNumber(1, literal(thrTile.sizes.at(1))));

                graph.coordinates.addElement(PassThrough(), {iThrX}, {elementNumberX});
                graph.coordinates.addElement(PassThrough(), {iThrY}, {elementNumberY});

                graph.coordinates.addElement(PassThrough(), {nThrX}, {workitemX});

                auto workitemY
                    = graph.coordinates.addElement(Workitem(1, literal(workgroupSizes.at(1))));
                graph.coordinates.addElement(PassThrough(), {nThrY}, {workitemY});
            }

            connections.push_back(DC<ElementNumber>(elementNumberX, 0));
            connections.push_back(DC<ElementNumber>(elementNumberY, 1));

            // LoadTiled swizzle for Direct2LDS MATRIX_B: permute K-column
            // (nThrX, dim 0) based on N-row (nThrY).
            auto loadTiledSwizzleNThrX = nThrX;
            if(ldsSwizzle && macTile.layoutType == LayoutType::MATRIX_B)
            {
                auto numColumns = thrTile.wsizes.at(0);
                auto params     = LDSSwizzleParams::fromColumnCount(numColumns, columnsPerBankRow);

                if(!params.noConflicts())
                {

                    loadTiledSwizzleNThrX = LDSSwizzleDetail::addLoadTiledSwizzleEdges(
                        graph, nThrX, nThrY, 0, params);

                    Log::getLogger()->debug(
                        "LoadTiled swizzle B: K={} R={}", params.numColumns, params.rowsPerBankRow);
                }
            }

            if(jammedTiles.size() > 0 && jammedTiles[0] > 1)
            {
                auto jammedWavetileX = graph.coordinates.addElement(
                    JammedWaveTileNumber(0, literal(jammedTiles[0]), literal(1)));
                connections.push_back(DC<JammedWaveTileNumber>(jammedWavetileX, 0));
                if(isGlobalToLDS)
                {
                    if(rightmostFastest)
                        graph.coordinates.addElement(
                            Tile(), {iMacX}, {jammedWavetileX, iThrX, loadTiledSwizzleNThrX});
                    else
                        graph.coordinates.addElement(
                            Tile(), {iMacX}, {jammedWavetileX, loadTiledSwizzleNThrX, iThrX});
                }
                else
                    graph.coordinates.addElement(
                        Tile(), {iMacX}, {jammedWavetileX, loadTiledSwizzleNThrX, iThrX});
            }
            else
            {
                if(isGlobalToLDS)
                {
                    if(rightmostFastest)
                        graph.coordinates.addElement(
                            Tile(), {iMacX}, {iThrX, loadTiledSwizzleNThrX});
                    else
                        graph.coordinates.addElement(
                            Tile(), {iMacX}, {loadTiledSwizzleNThrX, iThrX});
                }
                else
                    graph.coordinates.addElement(Tile(), {iMacX}, {loadTiledSwizzleNThrX, iThrX});
            }

            // LoadTiled swizzle for Direct2LDS MATRIX_A: permute K-column
            // (nThrY, dim 1) based on M-row (nThrX).
            auto loadTiledSwizzleNThrY = nThrY;
            if(ldsSwizzle && macTile.layoutType == LayoutType::MATRIX_A)
            {
                auto numColumns = thrTile.wsizes.at(1);
                auto params     = LDSSwizzleParams::fromColumnCount(numColumns, columnsPerBankRow);

                if(!params.noConflicts())
                {

                    auto swappedCol
                        = graph.coordinates.addElement(ThreadTileNumber(1, literal(numColumns)));
                    graph.coordinates.addElement(
                        PairSwap{params.rowsPerBankRow}, {swappedCol}, {nThrY, nThrX});

                    loadTiledSwizzleNThrY
                        = graph.coordinates.addElement(ThreadTileNumber(1, literal(numColumns)));
                    graph.coordinates.addElement(
                        Rotate{params.numColumns, params.rowsPerBankRow, false},
                        {loadTiledSwizzleNThrY},
                        {swappedCol, nThrX});

                    Log::getLogger()->debug(
                        "LoadTiled swizzle A: nThrY={} nThrX={} loadTiledSwizzleNThrY={} K={} R={}",
                        nThrY,
                        nThrX,
                        loadTiledSwizzleNThrY,
                        params.numColumns,
                        params.rowsPerBankRow);
                }
            }

            if(jammedTiles.size() > 1 && jammedTiles[1] > 1)
            {
                auto jammedWavetileY = graph.coordinates.addElement(
                    JammedWaveTileNumber(1, literal(jammedTiles[1]), literal(1)));
                connections.push_back(DC<JammedWaveTileNumber>(jammedWavetileY, 1));
                if(isGlobalToLDS)
                {
                    if(rightmostFastest)
                        graph.coordinates.addElement(
                            Tile(), {iMacY}, {jammedWavetileY, loadTiledSwizzleNThrY, iThrY});
                    else
                        graph.coordinates.addElement(
                            Tile(), {iMacY}, {jammedWavetileY, iThrY, loadTiledSwizzleNThrY});
                }
                else
                    graph.coordinates.addElement(
                        Tile(), {iMacY}, {jammedWavetileY, loadTiledSwizzleNThrY, iThrY});
            }
            else
            {
                if(isGlobalToLDS)
                {
                    if(rightmostFastest)
                        graph.coordinates.addElement(
                            Tile(), {iMacY}, {loadTiledSwizzleNThrY, iThrY});
                    else
                        graph.coordinates.addElement(
                            Tile(), {iMacY}, {iThrY, loadTiledSwizzleNThrY});
                }
                else
                    graph.coordinates.addElement(Tile(), {iMacY}, {loadTiledSwizzleNThrY, iThrY});
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

            uint numElements       = waveTile.sizes[0] * waveTile.sizes[1];
            uint activeLanesInWave = static_cast<uint>(wavefrontSize);
            uint numGPR            = numElements / activeLanesInWave;

            auto numGPRsPerLaneInWave = activeLanesInWave == 64 ? 4 : 8;
            uint simdsPerWave         = activeLanesInWave / 16u;
            auto simdsPerWaveLiteral  = literal(simdsPerWave);

            auto nRowBlocks = literal(waveTile.sizes[0] / simdsPerWave);
            auto nColBlocks = literal(waveTile.sizes[1] / simdsPerWave);

            auto nWaveX = graph.coordinates.addElement(waveTile.tileNumber(0));
            auto nWaveY = graph.coordinates.addElement(waveTile.tileNumber(1));
            auto iWaveX = graph.coordinates.addElement(waveTile.tileIndex(0));
            auto iWaveY = graph.coordinates.addElement(waveTile.tileIndex(1));

            auto activeLanesInWaveLiteral = literal(activeLanesInWave);
            auto unitStride               = literal(1u);

            auto nVblk = graph.coordinates.addElement(
                VGPRBlockNumber(literal(numGPR / numGPRsPerLaneInWave), unitStride));
            auto iVblk = graph.coordinates.addElement(
                VGPRBlockIndex(literal(numGPRsPerLaneInWave), unitStride));
            auto nLblk = graph.coordinates.addElement(
                Adhoc("LANEBlockNumber", literal(activeLanesInWave / simdsPerWave), unitStride));
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

            auto lane = graph.coordinates.addElement(Lane(activeLanesInWaveLiteral, unitStride));
            auto vgpr = graph.coordinates.addElement(VGPR(literal(numGPR), unitStride));

            connections.push_back(DC<WaveTile>(waveTileTag));
            connections.push_back(DC<VGPRBlockNumber>(nVblk));
            connections.push_back(DC<VGPRBlockIndex>(iVblk));
            connections.push_back(DC<Lane>(lane));
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
                                  bool                               rightmostFastest,
                                  bool                               isGlobalToLDS)
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

            if(macTile.layoutType == LayoutType::MATRIX_A
               || macTile.layoutType == LayoutType::MATRIX_B
               || macTile.layoutType == LayoutType::MATRIX_ACCUMULATOR
               || macTile.layoutType == LayoutType::SCRATCH)
            {
                if(rightmostFastest)
                {
                    elementNumberX = graph.coordinates.addElement(
                        ElementNumber(0, literal(thrTile.sizes.at(0))));
                    elementNumberY = graph.coordinates.addElement(
                        ElementNumber(1, literal(thrTile.sizes.at(1))));

                    connections.push_back(DC<ElementNumber>(elementNumberX, 0));
                    connections.push_back(DC<ElementNumber>(elementNumberY, 1));

                    graph.coordinates.addElement(PassThrough(), {elementNumberX}, {iThrX});
                    graph.coordinates.addElement(PassThrough(), {elementNumberY}, {iThrY});

                    graph.coordinates.addElement(Tile(), {workitemX}, {nThrX, nThrY});
                }
                else
                {
                    elementNumberX = graph.coordinates.addElement(
                        ElementNumber(0, literal(thrTile.sizes.at(1))));
                    elementNumberY = graph.coordinates.addElement(
                        ElementNumber(1, literal(thrTile.sizes.at(0))));

                    connections.push_back(DC<ElementNumber>(elementNumberX, 0));
                    connections.push_back(DC<ElementNumber>(elementNumberY, 1));

                    graph.coordinates.addElement(PassThrough(), {elementNumberY}, {iThrX});
                    graph.coordinates.addElement(PassThrough(), {elementNumberX}, {iThrY});

                    graph.coordinates.addElement(Tile(), {workitemX}, {nThrY, nThrX});
                }
            }
            else
            {
                elementNumberX
                    = graph.coordinates.addElement(ElementNumber(0, literal(thrTile.sizes.at(0))));
                elementNumberY
                    = graph.coordinates.addElement(ElementNumber(1, literal(thrTile.sizes.at(1))));

                connections.push_back(DC<ElementNumber>(elementNumberX, 0));
                connections.push_back(DC<ElementNumber>(elementNumberY, 1));

                graph.coordinates.addElement(PassThrough(), {elementNumberX}, {iThrX});
                graph.coordinates.addElement(PassThrough(), {elementNumberY}, {iThrY});

                auto workitemY
                    = graph.coordinates.addElement(Workitem(1, literal(workgroupSizes.at(1))));

                graph.coordinates.addElement(PassThrough(), {workitemX}, {nThrX});
                graph.coordinates.addElement(PassThrough(), {workitemY}, {nThrY});
            }

            if(jammedTiles.size() > 0 && jammedTiles[0] > 1)
            {
                auto jammedWavetileX = graph.coordinates.addElement(
                    JammedWaveTileNumber(0, literal(jammedTiles[0]), literal(1)));
                connections.push_back(DC<JammedWaveTileNumber>(jammedWavetileX, 0));
                if(rightmostFastest && isGlobalToLDS)
                    graph.coordinates.addElement(
                        Flatten(), {jammedWavetileX, iThrX, nThrX}, {iMacX});
                else
                    graph.coordinates.addElement(
                        Flatten(), {jammedWavetileX, nThrX, iThrX}, {iMacX});
            }
            else
            {
                if(rightmostFastest && isGlobalToLDS)
                    graph.coordinates.addElement(Flatten(), {iThrX, nThrX}, {iMacX});
                else
                    graph.coordinates.addElement(Flatten(), {nThrX, iThrX}, {iMacX});
            }

            // Note: No column swizzle on the store side.  For D2L the
            // LDS write address (M0) must remain un-swizzled because the
            // hardware writes sequentially at M0 + lane*16.  The swizzle
            // is applied only on the load side (addLoadThreadTileCT) where
            // it permutes the global read offset (v16/offen) so that each
            // lane fetches from a different global column, producing a
            // swizzled data layout in LDS without shifting M0.
            auto loadTiledSwizzleNThrY = nThrY;

            if(jammedTiles.size() > 1 && jammedTiles[1] > 1)
            {
                auto jammedWavetileY = graph.coordinates.addElement(
                    JammedWaveTileNumber(1, literal(jammedTiles[1]), literal(1)));
                connections.push_back(DC<JammedWaveTileNumber>(jammedWavetileY, 1));
                if(isGlobalToLDS)
                {
                    if(rightmostFastest)
                        graph.coordinates.addElement(
                            Flatten(), {jammedWavetileY, loadTiledSwizzleNThrY, iThrY}, {iMacY});
                    else
                        graph.coordinates.addElement(
                            Flatten(), {jammedWavetileY, iThrY, loadTiledSwizzleNThrY}, {iMacY});
                }
                else
                    graph.coordinates.addElement(
                        Flatten(), {jammedWavetileY, nThrY, iThrY}, {iMacY});
            }
            else
            {
                if(isGlobalToLDS)
                {
                    if(rightmostFastest)
                        graph.coordinates.addElement(
                            Flatten(), {loadTiledSwizzleNThrY, iThrY}, {iMacY});
                    else
                        graph.coordinates.addElement(
                            Flatten(), {iThrY, loadTiledSwizzleNThrY}, {iMacY});
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

            auto rightmostFastest = params->transposeMemoryAccess[macTile.layoutType];
            if(splitStore)
                rightmostFastest = false;

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
            auto update = params->enableLongDwordInstructions;
            update      = update && (packed || packFactor <= 1);
            update      = update && (!direct2LDS || useWiderDirect2LDS);
            if(update)
            {
                auto maxWidth = std::min(context->kernelOptions()->storeGlobalWidth,
                                         context->kernelOptions()->loadLocalWidth);

                auto numDwordsPerElement = DataTypeInfo::Get(varType).registerCount;
                auto macTileM            = macTile.sizes[0];
                auto macTileN            = macTile.sizes[1];

                auto macTileFastMovingDimSize = !rightmostFastest ? macTileM : macTileN;

                auto avoidDWordX2 = direct2LDS;
                updateThreadTileForLongDwords(thrTileM,
                                              thrTileN,
                                              maxWidth,
                                              macTileFastMovingDimSize,
                                              numDwordsPerElement,
                                              avoidDWordX2);
            }

            if(!rightmostFastest)
                std::swap(thrTileM, thrTileN);

            auto memoryType{MemoryType::VGPR};
            if(macTile.memoryType == MemoryType::WAVE_Direct2LDS)
            {
                memoryType = macTile.memoryType;
            }

            auto internalTile       = MacroTile(sizes, memoryType, {thrTileM, thrTileN});
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
                                bool                             isGlobalToLDS,
                                bool                             ldsSwizzle)
        {
            auto workgroupSizes = context->kernel()->workgroupSize();

            auto [nMacX, iMacX, nMacY, iMacY]
                = addLoadMacroTileCT(graph, connections, macTileTag, sdim);

            auto macTile          = *graph.coordinates.get<MacroTile>(macTileTag);
            auto rightmostFastest = params->transposeMemoryAccess[macTile.layoutType];

            addLoadThreadTileCT(graph,
                                connections,
                                macTileTag,
                                iMacX,
                                iMacY,
                                workgroupSizes,
                                jammedTiles,
                                rightmostFastest,
                                isGlobalToLDS,
                                ldsSwizzle,
                                ldsSwizzle ? getColumnsPerBankRow(context) : 0u);

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
                                DataType const&                  dataType,
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
                              dataType,
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
                                   std::vector<unsigned int> const& jammedTiles,
                                   CommandParametersPtr             params,
                                   ContextPtr                       context)
        {
            auto wavefrontSize = context->kernel()->wavefront_size();

            auto [nMacX, iMacX, nMacY, iMacY]
                = addLoadMacroTileCT(graph, connections, macTileTag, sdim);

            addLoadSwizzleTileCT(
                graph, connections, macTileTag, iMacX, iMacY, wavefrontSize, jammedTiles, params);

            graph.coordinates.addElement(DataFlow(), {userTag}, {macTileTag});
        }

        void storeMacroTile_SWIZZLE(KernelGraph&                     graph,
                                    std::vector<DeferredConnection>& connections,
                                    int                              storeTag,
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

            addStoreSwizzleTileCT(
                graph, connections, macTileTag, iMacX, iMacY, wavefrontSize, jammedTiles, params);

            graph.coordinates.addElement(DataFlow(), {macTileTag}, {userTag});
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

            auto rightmostFastest = params->transposeMemoryAccess[macTile.layoutType];

            auto [nMacX, iMacX, nMacY, iMacY]
                = addStoreMacroTileCT(graph, connections, macTileTag, sdim, jammedTiles);

            addStoreThreadTileCT(graph,
                                 connections,
                                 macTileTag,
                                 iMacX,
                                 iMacY,
                                 workgroupSizes,
                                 jammedTiles,
                                 rightmostFastest,
                                 /*isGlobalToLDS=*/false);

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
                // LDS bank swizzle currently targets CDNA4 (64 LDS banks).
                auto swzMode = context->kernelOptions()->ldsSwizzleMode;
                AssertFatal(swzMode == LDSBankSwizzleMode::None
                                || context->targetArchitecture().target().isCDNA4GPU(),
                            "LDS bank swizzle is only supported on CDNA4 architectures",
                            ShowValue(swzMode));
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

                    auto       wavefrontSize = m_context->kernel()->wavefront_size();
                    uint const lanesPerSIMD  = 16;
                    uint const nSIMDsPerWave = wavefrontSize / lanesPerSIMD;
                    uint const nSIMDIndex    = macTile.subTileSizes.at(0) / lanesPerSIMD;
                    uint const nSIMDBlock    = nSIMDsPerWave / nSIMDIndex;
                    auto       SIMDBlock     = graph.coordinates.addElement(
                        Adhoc("SIMDBlock", literal(nSIMDBlock), nullptr));
                    auto SIMDIndex = graph.coordinates.addElement(
                        Adhoc("SIMDIndex", literal(nSIMDIndex), nullptr));
                    auto laneInSIMD
                        = graph.coordinates.addElement(Lane(literal(lanesPerSIMD), nullptr));

                    uint const numElements       = waveTile.elements();
                    uint const activeLanesInWave = static_cast<uint>(wavefrontSize);
                    uint const numVgpr           = numElements / activeLanesInWave;
                    uint const nVgprIndex
                        = std::min(nSIMDIndex, static_cast<uint>(macTile.miTileSizes.at(2)));
                    uint const nVgprBlock
                        = numElements / macTile.subTileSizes.at(0) / nSIMDBlock / nVgprIndex;
                    auto vgprBlock = graph.coordinates.addElement(
                        VGPRBlockNumber(literal(nVgprBlock), literal(1u)));
                    auto vgprIndex = graph.coordinates.addElement(
                        VGPRBlockIndex(literal(nVgprIndex), literal(1u)));
                    auto vgpr = graph.coordinates.addElement(VGPR(literal(numVgpr), literal(1u)));

                    graph.coordinates.addElement(Flatten(), {vgprBlock, vgprIndex}, {vgpr});

                    uint const nSIMDIndexBlock = nVgprIndex;
                    uint const nSIMDIndexIndex = nSIMDIndex / nSIMDIndexBlock;
                    auto       SIMDIndexBlock  = graph.coordinates.addElement(
                        Adhoc("SIMDIndexBlock", literal(nSIMDIndexBlock), nullptr));
                    auto SIMDIndexIndex = graph.coordinates.addElement(
                        Adhoc("SIMDIndexIndex", literal(nSIMDIndexIndex), nullptr));
                    graph.coordinates.addElement(
                        Flatten(), {SIMDIndexBlock, SIMDIndexIndex}, {SIMDIndex});

                    connections.push_back(DC<WaveTile>(waveTileTag));
                    connections.push_back(DC<Adhoc>(SIMDBlock, 0));
                    connections.push_back(DC<Adhoc>(SIMDIndex, 1));
                    connections.push_back(DC<Adhoc>(SIMDIndexBlock, 2));
                    connections.push_back(DC<Adhoc>(SIMDIndexIndex, 3));
                    connections.push_back(DC<Lane>(laneInSIMD));
                    connections.push_back(DC<VGPRBlockNumber>(vgprBlock));
                    connections.push_back(DC<VGPRBlockIndex>(vgprIndex));
                    connections.push_back(DC<VGPR>(vgpr));

                    graph.coordinates.addElement(
                        Flatten(), {vgprIndex, SIMDIndexIndex, laneInSIMD}, {iWaveX});
                    graph.coordinates.addElement(
                        Flatten(), {SIMDBlock, vgprBlock, SIMDIndexBlock}, {iWaveY});
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
                {
                    bool vgprSwizzle
                        = isLDSSwizzleActive(m_context->kernelOptions()->ldsSwizzleMode,
                                             tile.layoutType,
                                             varType.dataType);
                    loadMacroTile_VGPR(graph,
                                       connections,
                                       userTag,
                                       tileTag,
                                       sdims,
                                       {1, 1},
                                       m_params,
                                       m_context,
                                       /*isGlobalToLDS=*/vgprSwizzle,
                                       /*ldsSwizzle=*/vgprSwizzle);
                }
                break;
                case MemoryType::WAVE:
                case MemoryType::WAVE_FROM_GLOBAL:
                    loadMacroTile_WAVE(graph,
                                       connections,
                                       userTag,
                                       tileTag,
                                       sdims,
                                       varType.dataType,
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
                                          wavetilesPerWavefront,
                                          m_params,
                                          m_context);
                    break;
                case MemoryType::WAVE_Direct2LDS:
                {
                    bool d2lSwizzle = isLDSSwizzleActive(m_context->kernelOptions()->ldsSwizzleMode,
                                                         tile.layoutType,
                                                         varType.dataType);
                    loadMacroTile_VGPR(graph,
                                       connections,
                                       userTag,
                                       tileTag,
                                       sdims,
                                       {1, 1},
                                       m_params,
                                       m_context,
                                       /*isGlobalToLDS=*/true,
                                       /*ldsSwizzle=*/d2lSwizzle);
                }
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
                bool rightmostFastest      = m_params->transposeMemoryAccess[tile.layoutType];

                logger->debug("  LDS({}), MacroTile({}), MacroTile size: {}x{}, SubTile size: "
                              "{}x{}, MemoryType {}, LayoutType {}, rightmostFastest {}",
                              ldsTag,
                              tileTag,
                              tile.sizes[0],
                              tile.sizes[1],
                              tile.subTileSizes[0],
                              tile.subTileSizes[1],
                              toString(tile.memoryType),
                              toString(tile.layoutType),
                              rightmostFastest);

                std::vector<DeferredConnection> connections;

                auto loadTag = reindexer.control.at(tag);

                auto iMacX = graph.coordinates.addElement(tile.tileIndex(0));
                auto iMacY = graph.coordinates.addElement(tile.tileIndex(1));

                if(tile.memoryType == MemoryType::WAVE
                   || tile.memoryType == MemoryType::WAVE_FROM_GLOBAL)
                {
                    auto              varType     = getVariableType(graph, loadTag);
                    std::vector<uint> jammedTiles = wavetilesPerWavefront;
                    addLoadWaveTileCT(graph,
                                      connections,
                                      tileTag,
                                      iMacX,
                                      iMacY,
                                      varType.dataType,
                                      wavefrontSize,
                                      true,
                                      jammedTiles,
                                      m_params,
                                      m_context);
                }
                else if(tile.memoryType == MemoryType::WAVE_SPLIT)
                {
                    rightmostFastest = false;
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
                                        false,
                                        false,
                                        false,
                                        0u);
                }

                // LoadLDSTile swizzle: insert PairSwap + Rotate(inv) edges
                // that un-permute the K-column so the wave reads the
                // correct logical element.
                //   {colChunk, elemInChunk} --Flatten--> {colCoord}
                //   {rotatedCol} --Rotate(inv)--> {colChunk, rowCoord}
                //   {rawColChunk} --PairSwap--> {rotatedCol, rowCoord}
                //   {rawColElem} --Tile--> {rawColChunk, elemInChunk}
                //   ldsTag --Tile--> {rawColElem, non-K-dim}
                auto tileIMacY = iMacY;
                auto tileIMacX = iMacX;
                {
                    auto lrVarType = getVariableType(graph, loadTag);
                    bool isA       = tile.layoutType == LayoutType::MATRIX_A;
                    bool doSwizzle = isLDSSwizzleActive(m_context->kernelOptions()->ldsSwizzleMode,
                                                        tile.layoutType,
                                                        lrVarType.dataType);

                    if(doSwizzle)
                    {
                        int  kDim        = isA ? 1 : 0;
                        auto elementBits = DataTypeInfo::Get(lrVarType.dataType).elementBits;
                        auto tileK       = static_cast<unsigned int>(tile.sizes.at(kDim));
                        auto params      = LDSSwizzleParams::compute(
                            tileK, elementBits, getColumnsPerBankRow(m_context));

                        if(!params.noConflicts())
                        {
                            int rowCoord = isA ? iMacX : iMacY;
                            int colCoord = isA ? iMacY : iMacX;

                            auto rawColElem = LDSSwizzleDetail::addLoadLDSTileSwizzleEdges(
                                graph, colCoord, rowCoord, kDim, params);

                            if(isA)
                                tileIMacY = rawColElem;
                            else
                                tileIMacX = rawColElem;
                        }

                        logger->debug("LoadLDSTile swizzle {}: K={} E={} R={}",
                                      toString(tile.layoutType),
                                      params.numColumns,
                                      params.elementsPerChunk,
                                      params.rowsPerBankRow);
                    }
                }

                if(rightmostFastest)
                    graph.coordinates.addElement(Tile(), {ldsTag}, {tileIMacX, tileIMacY});
                else
                    graph.coordinates.addElement(Tile(), {ldsTag}, {tileIMacY, tileIMacX});

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
                case MemoryType::WAVE_SWIZZLE:
                    storeMacroTile_SWIZZLE(graph,
                                           connections,
                                           storeTag,
                                           userTag,
                                           tileTag,
                                           sdims,
                                           wavetilesPerWavefront,
                                           m_params,
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

                auto ldsTag  = reindexer.coordinates.at(originalLDSTag);
                auto tileTag = reindexer.coordinates.at(originalTileTag);
                auto tile    = graph.coordinates.getNode<MacroTile>(tileTag);
                auto ldsTile = graph.coordinates.getNode<LDS>(ldsTag);
                AssertFatal(tile.rank == 2, "Rank /= 2 not implemented yet.");

                auto workgroupSizes        = m_context->kernel()->workgroupSize();
                auto wavefrontSize         = m_context->kernel()->wavefront_size();
                auto wavetilesPerWavefront = m_params->getWaveTilesPerWavefront();
                auto useWaveAccess         = tile.memoryType == MemoryType::WAVE
                                     || tile.memoryType == MemoryType::WAVE_LDS;
                bool rightmostFastest = m_params->transposeMemoryAccess[tile.layoutType];
                // XXX debug
                logger->debug("  LDS({}), MacroTile({}), MacroTile size: {}x{}, SubTile size: "
                              "{}x{}, MemoryType {}, LayoutType {}, rightmostFastest {}",
                              ldsTag,
                              tileTag,
                              tile.sizes[0],
                              tile.sizes[1],
                              tile.subTileSizes[0],
                              tile.subTileSizes[1],
                              toString(tile.memoryType),
                              toString(tile.layoutType),
                              rightmostFastest);

                std::vector<DeferredConnection> connections;

                auto iMacX = graph.coordinates.addElement(tile.tileIndex(0));
                auto iMacY = graph.coordinates.addElement(tile.tileIndex(1));

                bool ldsSwizzle = isLDSSwizzleActive(m_context->kernelOptions()->ldsSwizzleMode,
                                                     tile.layoutType,
                                                     ostore.varType.dataType);

                logger->debug("StoreLDSTile: tag={} tileTag={} memoryType={} layoutType={} "
                              "ldsSwizzle={}",
                              tag,
                              tileTag,
                              toString(tile.memoryType),
                              toString(tile.layoutType),
                              ldsSwizzle);

                if(tile.memoryType == MemoryType::VGPR)
                {
                    // We are storing entire workgroup tiles.
                    // When LDS swizzle is active, use isGlobalToLDS=true so the
                    // store coordinate decomposition matches the load side,
                    // producing the same LDS layout as D2LDS.
                    std::vector<uint> jammedTiles = {1, 1};
                    addStoreThreadTileCT(graph,
                                         connections,
                                         tileTag,
                                         iMacX,
                                         iMacY,
                                         workgroupSizes,
                                         jammedTiles,
                                         rightmostFastest,
                                         /*isGlobalToLDS=*/ldsSwizzle);
                }
                else if(tile.memoryType == MemoryType::WAVE_Direct2LDS)
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
                                         rightmostFastest,
                                         /*isGlobalToLDS=*/true);
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

                if(rightmostFastest)
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
            auto visitor = LowerTileVisitor(m_params, m_context);
            return rewrite(graph, visitor);
        }

        namespace LDSSwizzleDetail
        {
            using namespace CoordinateGraph;
            using namespace Expression;

            LDSSwizzleParams LDSSwizzleParams::compute(unsigned int tileK,
                                                       unsigned int elementBits,
                                                       unsigned int colsPerBankRow)
            {
                AssertFatal(elementBits > 0 && tileK > 0,
                            "LDSSwizzleParams::compute requires non-zero tileK and elementBits",
                            ShowValue(tileK),
                            ShowValue(elementBits));
                auto elems = 128u / elementBits;
                auto cols  = tileK / elems;
                AssertFatal(tileK % elems == 0,
                            "tileK must be a multiple of elements per dwordx4 chunk",
                            ShowValue(tileK),
                            ShowValue(elems));
                AssertFatal(cols > 0 && (cols & (cols - 1)) == 0 && colsPerBankRow % cols == 0,
                            "LDS swizzle requires power-of-2 numColumns",
                            ShowValue(cols));
                return {cols, colsPerBankRow / cols, elems, colsPerBankRow};
            }

            LDSSwizzleParams LDSSwizzleParams::fromColumnCount(unsigned int numCols,
                                                               unsigned int colsPerBankRow)
            {
                AssertFatal(numCols > 0 && (numCols & (numCols - 1)) == 0
                                && colsPerBankRow % numCols == 0,
                            "LDS swizzle requires power-of-2 numColumns",
                            ShowValue(numCols));
                return {numCols, colsPerBankRow / numCols, 0u, colsPerBankRow};
            }

            bool LDSSwizzleParams::noConflicts() const
            {
                return numColumns >= columnsPerBankRow;
            }

            int addLoadTiledSwizzleEdges(KernelGraph&            graph,
                                         int                     colCoord,
                                         int                     rowCoord,
                                         int                     kDim,
                                         LDSSwizzleParams const& params)
            {
                auto swappedCol = graph.coordinates.addElement(
                    ThreadTileNumber(kDim, literal(params.numColumns)));
                graph.coordinates.addElement(
                    PairSwap{params.rowsPerBankRow}, {swappedCol}, {colCoord, rowCoord});

                auto rotatedCol = graph.coordinates.addElement(
                    ThreadTileNumber(kDim, literal(params.numColumns)));
                graph.coordinates.addElement(
                    Rotate{params.numColumns, params.rowsPerBankRow, false},
                    {rotatedCol},
                    {swappedCol, rowCoord});

                return rotatedCol;
            }

            int addLoadLDSTileSwizzleEdges(KernelGraph&            graph,
                                           int                     colCoord,
                                           int                     rowCoord,
                                           int                     kDim,
                                           LDSSwizzleParams const& params)
            {
                auto ePC = params.elementsPerChunk;

                auto colChunk = graph.coordinates.addElement(
                    MacroTileIndex(kDim, literal(params.numColumns), literal(1u)));
                auto elemInChunk
                    = graph.coordinates.addElement(MacroTileIndex(kDim, literal(ePC), literal(1u)));
                graph.coordinates.addElement(Flatten(), {colChunk, elemInChunk}, {colCoord});

                auto rotatedCol = graph.coordinates.addElement(
                    MacroTileIndex(kDim, literal(params.numColumns), literal(1u)));
                graph.coordinates.addElement(Rotate{params.numColumns, params.rowsPerBankRow, true},
                                             {rotatedCol},
                                             {colChunk, rowCoord});

                auto rawColChunk = graph.coordinates.addElement(
                    MacroTileIndex(kDim, literal(params.numColumns), literal(1u)));
                graph.coordinates.addElement(
                    PairSwap{params.rowsPerBankRow}, {rawColChunk}, {rotatedCol, rowCoord});

                auto rawColElem = graph.coordinates.addElement(
                    MacroTileIndex(kDim, literal(params.numColumns * ePC), literal(1u)));
                graph.coordinates.addElement(Tile(), {rawColElem}, {rawColChunk, elemInChunk});

                return rawColElem;
            }
        }
    }
}
