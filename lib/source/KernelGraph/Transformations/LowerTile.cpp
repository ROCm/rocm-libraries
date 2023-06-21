
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/Transforms/LowerTile.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Operations/Operations.hpp>

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
            auto macTile = graph.coordinates.getNode<MacroTile>(macTileTag);
            auto sdimX   = sdim[0];
            auto sdimY   = sdim[1];

            connections.push_back(DC<SubDimension>(sdimX, 0));
            connections.push_back(DC<SubDimension>(sdimY, 1));

            auto nMacX = graph.coordinates.addElement(macTile.tileNumber(0));
            auto nMacY = graph.coordinates.addElement(macTile.tileNumber(1));
            auto iMacX = graph.coordinates.addElement(macTile.tileIndex(0));
            auto iMacY = graph.coordinates.addElement(macTile.tileIndex(1));

            connections.push_back(DC<MacroTileNumber>(nMacX, 0));
            connections.push_back(DC<MacroTileNumber>(nMacY, 1));

            graph.coordinates.addElement(Tile(), {sdimX}, {nMacX, iMacX});
            graph.coordinates.addElement(Tile(), {sdimY}, {nMacY, iMacY});

            return {nMacX, iMacX, nMacY, iMacY};
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
                               int                              wavefrontSize,
                               std::vector<unsigned int> const& jammedTiles)
        {
            auto macTile = graph.coordinates.getNode<MacroTile>(macTileTag);

            AssertFatal(macTile.subTileSizes.size() == 4, "Invalid tile specification.");

            auto m = macTile.subTileSizes[0];
            auto n = macTile.subTileSizes[1];
            auto k = macTile.subTileSizes[2];

            std::vector<int> tileSize;
            if(macTile.layoutType == LayoutType::MATRIX_A)
                tileSize = {m, k};
            if(macTile.layoutType == LayoutType::MATRIX_B)
                tileSize = {k, n};
            if(macTile.layoutType == LayoutType::MATRIX_ACCUMULATOR)
                tileSize = {m, n};

            auto workitem    = graph.coordinates.addElement(Workitem(0));
            auto waveTile    = WaveTile(tileSize, macTile.layoutType);
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

            uint numElements = product(tileSize);
            uint wfs         = static_cast<uint>(wavefrontSize);
            uint numVgpr     = numElements / wfs;

            auto wavefrontSizeLiteral = literal(wfs);

            auto lane = graph.coordinates.addElement(Lane(wavefrontSizeLiteral, nullptr));
            auto vgpr = graph.coordinates.addElement(VGPR(literal(numVgpr), nullptr));

            graph.coordinates.addElement(Flatten(), {waveX, waveY}, {wave});
            graph.coordinates.addElement(Flatten(), {wave, lane}, {workitem});

            connections.push_back(DC<VGPR>(vgpr));

            int jammedWavetileX = -1;
            if(jammedTiles[0] >= 1)
            {
                jammedWavetileX = graph.coordinates.addElement(
                    JammedWaveTileNumber(0, literal(jammedTiles[0]), literal(1)));
                connections.push_back(DC<JammedWaveTileNumber>(jammedWavetileX, 0));
            }
            int jammedWavetileY = -1;
            if(jammedTiles[1] >= 1)
            {
                jammedWavetileY = graph.coordinates.addElement(
                    JammedWaveTileNumber(1, literal(jammedTiles[1]), literal(1)));
                connections.push_back(DC<JammedWaveTileNumber>(jammedWavetileY, 1));
            }

            switch(waveTile.layout)
            {
            case LayoutType::MATRIX_A:
            {
                auto blockNumber = graph.coordinates.addElement(
                    Adhoc("BlockNumber", literal(static_cast<uint>(wfs / m)), nullptr));
                auto blockIndex = graph.coordinates.addElement(
                    Adhoc("BlockIndex", literal(static_cast<uint>(m)), nullptr));

                graph.coordinates.addElement(Flatten(), {blockNumber, blockIndex}, {lane});

                graph.coordinates.addElement(Tile(), {iWaveY}, {blockNumber, vgpr});
                graph.coordinates.addElement(PassThrough(), {iWaveX}, {blockIndex});

                if(jammedTiles[0] > 1)
                    graph.coordinates.addElement(Tile(), {nWaveX}, {waveX, jammedWavetileX});
                else
                    graph.coordinates.addElement(PassThrough(), {nWaveX}, {waveX});
            }
            break;

            case LayoutType::MATRIX_B:
            {
                auto blockNumber = graph.coordinates.addElement(
                    Adhoc("BlockNumber", literal(static_cast<uint>(wfs / m)), nullptr));
                auto blockIndex = graph.coordinates.addElement(
                    Adhoc("BlockIndex", literal(static_cast<uint>(m)), nullptr));

                graph.coordinates.addElement(Flatten(), {blockNumber, blockIndex}, {lane});

                graph.coordinates.addElement(Tile(), {iWaveX}, {blockNumber, vgpr});
                graph.coordinates.addElement(PassThrough(), {iWaveY}, {blockIndex});

                if(jammedTiles[1] > 1)
                    graph.coordinates.addElement(Tile(), {nWaveY}, {waveY, jammedWavetileY});
                else
                    graph.coordinates.addElement(PassThrough(), {nWaveY}, {waveY});
            }
            break;

            case LayoutType::MATRIX_ACCUMULATOR:
            {
                // MFMA accumulator tile size
                uint mts            = 4u;
                auto mfma_tile_size = literal(mts);
                auto unitStride     = literal(1u);

                auto nRowBlocks = literal(waveTile.sizes[0] / mts);
                auto nColBlocks = literal(waveTile.sizes[1] / mts);

                auto nVblk = graph.coordinates.addElement(
                    VGPRBlockNumber(literal(numVgpr / mts), unitStride));
                auto iVblk
                    = graph.coordinates.addElement(VGPRBlockIndex(mfma_tile_size, unitStride));
                auto nLblk = graph.coordinates.addElement(
                    Adhoc("LANEBlockNumber", literal(wfs / mts), unitStride));
                auto iLblk = graph.coordinates.addElement(
                    Adhoc("LANEBlockIndex", mfma_tile_size, unitStride));
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

                if(jammedTiles[0] > 1)
                    graph.coordinates.addElement(Tile(), {nWaveX}, {waveX, jammedWavetileX});
                else
                    graph.coordinates.addElement(PassThrough(), {nWaveX}, {waveX});

                if(jammedTiles[1] > 1)
                    graph.coordinates.addElement(Tile(), {nWaveY}, {waveY, jammedWavetileY});
                else
                    graph.coordinates.addElement(PassThrough(), {nWaveY}, {waveY});
            }
            break;

            default:
                Throw<FatalError>("Not implemented yet.");
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
                                 bool                               useSwappedAccess)
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
                   || macTile.layoutType == LayoutType::MATRIX_B)
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
                   || macTile.layoutType == LayoutType::MATRIX_B)
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

            graph.coordinates.addElement(Tile(), {iMacX}, {nThrX, iThrX});
            graph.coordinates.addElement(Tile(), {iMacY}, {nThrY, iThrY});
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
                                      std::array<unsigned int, 3> const& workgroupSizes)
        {
            auto macTile = graph.coordinates.getNode<MacroTile>(macTileTag);
            auto thrTile = ThreadTile(macTile);

            auto nThrX
                = graph.coordinates.addElement(ThreadTileNumber(0, literal(thrTile.sizes.at(0))));
            auto nThrY
                = graph.coordinates.addElement(ThreadTileNumber(1, literal(thrTile.wsizes.at(1))));
            auto iThrX
                = graph.coordinates.addElement(ThreadTileIndex(0, literal(thrTile.wsizes.at(0))));
            auto iThrY
                = graph.coordinates.addElement(ThreadTileIndex(1, literal(thrTile.sizes.at(1))));

            auto workitemX
                = graph.coordinates.addElement(Workitem(0, literal(workgroupSizes.at(0))));

            auto elementNumberX
                = graph.coordinates.addElement(ElementNumber(0, literal(thrTile.sizes.at(0))));
            auto elementNumberY
                = graph.coordinates.addElement(ElementNumber(1, literal(thrTile.sizes.at(1))));

            graph.coordinates.addElement(Flatten(), {nThrY, iThrX}, {workitemX});
            graph.coordinates.addElement(PassThrough(), {nThrX}, {elementNumberX});
            graph.coordinates.addElement(PassThrough(), {iThrY}, {elementNumberY});

            connections.push_back(DC<ElementNumber>(elementNumberX, 0));
            connections.push_back(DC<ElementNumber>(elementNumberY, 1));

            graph.coordinates.addElement(Tile(), {iMacX}, {nThrX, iThrX});
            graph.coordinates.addElement(Tile(), {iMacY}, {nThrY, iThrY});
        }

        /**
         * @brief Store version of addLoadMacroTileCT.
         */
        std::tuple<int, int, int, int>
            addStoreMacroTileCT(KernelGraph&                     graph,
                                std::vector<DeferredConnection>& connections,
                                int                              macTileTag,
                                std::vector<int> const&          sdim)
        {
            auto macTile = graph.coordinates.getNode<MacroTile>(macTileTag);
            auto sdimX   = sdim[0];
            auto sdimY   = sdim[1];

            connections.push_back(DC<SubDimension>(sdimX, 0));
            connections.push_back(DC<SubDimension>(sdimY, 1));

            auto nMacX = graph.coordinates.addElement(macTile.tileNumber(0));
            auto nMacY = graph.coordinates.addElement(macTile.tileNumber(1));
            auto iMacX = graph.coordinates.addElement(macTile.tileIndex(0));
            auto iMacY = graph.coordinates.addElement(macTile.tileIndex(1));

            connections.push_back(DC<MacroTileNumber>(nMacX, 0));
            connections.push_back(DC<MacroTileNumber>(nMacY, 1));

            graph.coordinates.addElement(Flatten(), {nMacX, iMacX}, {sdim[0]});
            graph.coordinates.addElement(Flatten(), {nMacY, iMacY}, {sdim[1]});

            return {nMacX, iMacX, nMacY, iMacY};
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
                                std::vector<unsigned int> const& jammedTiles)
        {
            auto macTile = graph.coordinates.getNode<MacroTile>(macTileTag);

            AssertFatal(macTile.layoutType == LayoutType::MATRIX_ACCUMULATOR,
                        "Store must be from accumulator.");

            auto workitem    = graph.coordinates.addElement(Workitem(0));
            auto waveTile    = WaveTile(macTile.subTileSizes, LayoutType::MATRIX_ACCUMULATOR);
            auto waveTileTag = graph.coordinates.addElement(waveTile);

            graph.coordinates.addElement(PassThrough(), {waveTileTag}, {macTileTag});

            uint numElements = waveTile.sizes[0] * waveTile.sizes[1];
            uint wfs         = static_cast<uint>(wavefrontSize);
            uint num_agpr    = numElements / wfs;

            // MFMA accumulator tile size
            uint mts            = 4u;
            auto mfma_tile_size = literal(mts);

            auto nRowBlocks = literal(waveTile.sizes[0] / mts);
            auto nColBlocks = literal(waveTile.sizes[1] / mts);

            auto nWaveX = graph.coordinates.addElement(waveTile.tileNumber(0));
            auto nWaveY = graph.coordinates.addElement(waveTile.tileNumber(1));
            auto iWaveX = graph.coordinates.addElement(waveTile.tileIndex(0));
            auto iWaveY = graph.coordinates.addElement(waveTile.tileIndex(1));

            auto wavefrontSizeLiteral = literal(static_cast<uint>(wavefrontSize));
            auto unitStride           = literal(1u);

            auto nVblk = graph.coordinates.addElement(
                VGPRBlockNumber(literal(num_agpr / mts), unitStride));
            auto iVblk = graph.coordinates.addElement(VGPRBlockIndex(mfma_tile_size, unitStride));
            auto nLblk = graph.coordinates.addElement(
                Adhoc("LANEBlockNumber", literal(wfs / mts), unitStride));
            auto iLblk
                = graph.coordinates.addElement(Adhoc("LANEBlockIndex", mfma_tile_size, unitStride));
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
            auto vgpr = graph.coordinates.addElement(VGPR(literal(num_agpr), unitStride));

            connections.push_back(DC<WaveTile>(waveTileTag));
            connections.push_back(DC<VGPRBlockNumber>(nVblk));
            connections.push_back(DC<VGPRBlockIndex>(iVblk));
            connections.push_back(DC<VGPR>(vgpr));

            graph.coordinates.addElement(Tile(), {vgpr}, {nVblk, iVblk});
            graph.coordinates.addElement(Tile(), {lane}, {nLblk, iLblk});
            graph.coordinates.addElement(Flatten(), {nVblk, nLblk}, {block});
            graph.coordinates.addElement(Tile(), {block}, {rowBlock, colBlock});

            int jammedWavetileX = -1;
            if(jammedTiles[0] >= 1)
            {
                jammedWavetileX = graph.coordinates.addElement(
                    JammedWaveTileNumber(0, literal(jammedTiles[0]), literal(1)));
                connections.push_back(DC<JammedWaveTileNumber>(jammedWavetileX, 0));
            }
            int jammedWavetileY = -1;
            if(jammedTiles[1] >= 1)
            {
                jammedWavetileY = graph.coordinates.addElement(
                    JammedWaveTileNumber(1, literal(jammedTiles[1]), literal(1)));
                connections.push_back(DC<JammedWaveTileNumber>(jammedWavetileY, 1));
            }

            if(jammedTiles[0] > 1)
                graph.coordinates.addElement(Flatten(), {waveX, jammedWavetileX}, {nWaveX});
            else
                graph.coordinates.addElement(PassThrough(), {waveX}, {nWaveX});
            if(jammedTiles[1] > 1)
                graph.coordinates.addElement(Flatten(), {waveY, jammedWavetileY}, {nWaveY});
            else
                graph.coordinates.addElement(PassThrough(), {waveY}, {nWaveY});

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
                                  bool                               useSwappedAccess)
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
                   || macTile.layoutType == LayoutType::MATRIX_B)
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
                   || macTile.layoutType == LayoutType::MATRIX_B)
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

            graph.coordinates.addElement(Flatten(), {nThrX, iThrX}, {iMacX});
            graph.coordinates.addElement(Flatten(), {nThrY, iThrY}, {iMacY});
        }

        /**
         * @brief Store version of addLoadAccumulatorTileCT.
         */
        void addStoreAccumulatorTileCT(KernelGraph&                       graph,
                                       std::vector<DeferredConnection>&   connections,
                                       int                                macTileTag,
                                       int                                iMacX,
                                       int                                iMacY,
                                       std::array<unsigned int, 3> const& workgroupSizes)
        {
            auto macTile = graph.coordinates.getNode<MacroTile>(macTileTag);

            auto thrTile = ThreadTile(macTile);

            auto nThrX
                = graph.coordinates.addElement(ThreadTileNumber(0, literal(thrTile.sizes.at(0))));
            auto nThrY
                = graph.coordinates.addElement(ThreadTileNumber(1, literal(thrTile.wsizes.at(1))));
            auto iThrX
                = graph.coordinates.addElement(ThreadTileIndex(0, literal(thrTile.wsizes.at(0))));
            auto iThrY
                = graph.coordinates.addElement(ThreadTileIndex(1, literal(thrTile.sizes.at(1))));

            auto workitemX
                = graph.coordinates.addElement(Workitem(0, literal(workgroupSizes.at(0))));

            auto elementNumberX
                = graph.coordinates.addElement(ElementNumber(0, literal(thrTile.sizes.at(0))));
            auto elementNumberY
                = graph.coordinates.addElement(ElementNumber(1, literal(thrTile.sizes.at(1))));

            connections.push_back(DC<ElementNumber>(elementNumberX, 0));
            connections.push_back(DC<ElementNumber>(elementNumberY, 1));

            graph.coordinates.addElement(Tile(), {workitemX}, {nThrY, iThrX});
            graph.coordinates.addElement(PassThrough(), {elementNumberX}, {nThrX});
            graph.coordinates.addElement(PassThrough(), {elementNumberY}, {iThrY});

            graph.coordinates.addElement(Flatten(), {nThrX, iThrX}, {iMacX});
            graph.coordinates.addElement(Flatten(), {nThrY, iThrY}, {iMacY});
        }

        /**
         * @brief Translates TypeAndSubDimension connections to
         * LDSTypeAndSubDimension connections.
         *
         * The direction (ie, from global, into LDS, from LDS etc) is
         * added to each TypeAndSubDimension connection.
         *
         * Optionally copies other connections without modifying them.
         */
        void addLDSDirection(std::vector<DeferredConnection>&       connections,
                             std::vector<DeferredConnection> const& original,
                             Connections::LDSLoadStore              direction,
                             bool                                   passthrough = false)
        {
            for(auto& dc : original)
            {
                if(std::holds_alternative<Connections::TypeAndSubDimension>(dc.connectionSpec))
                {
                    auto ts = std::get<Connections::TypeAndSubDimension>(dc.connectionSpec);
                    auto connection
                        = Connections::LDSTypeAndSubDimension{ts.id, ts.subdimension, direction};
                    connections.push_back({connection, dc.coordinate});
                    if(passthrough)
                    {
                        // TODO This is needed because the
                        // lowerMatrixMultiply pass looks for
                        // WaveTiles.
                        connections.push_back(dc);
                    }
                }
            }
        }

        /**
         * @brief Create internal tile for WAVE formats.
         */
        int createInternalTile(KernelGraph& graph, int opTag, int macTileTag, ContextPtr context)
        {
            auto macTile = graph.coordinates.getNode<MacroTile>(macTileTag);

            if(macTile.memoryType == MemoryType::LDS)
            {
                auto internalTile
                    = MacroTile(macTile.sizes, MemoryType::VGPR, macTile.subTileSizes);
                internalTile.layoutType = macTile.layoutType;
                return graph.coordinates.addElement(internalTile);
            }

            auto workgroupSizes = context->kernel()->workgroupSize();

            auto numWorkitems           = product(workgroupSizes);
            auto numElements            = product(macTile.sizes);
            auto numElementsPerWorkitem = static_cast<int>(numElements / numWorkitems);
            auto thrTileM               = numElementsPerWorkitem;
            auto thrTileN               = 1;

            auto vtype = getVariableType(graph, opTag);

            auto useSwappedAccess
                = context->kernelOptions().transposeMemoryAccess[macTile.layoutType];

            // Load multiple smaller-precision (< 32-bit) elements into one VGPR
            auto packFactor = bytesPerRegister / DataTypeInfo::Get(vtype).elementSize;
            bool packed     = false;
            if(context->kernelOptions().packMultipleElementsInto1VGPR && packFactor > 1
               && thrTileM % packFactor == 0)
            {
                thrTileM = thrTileM / packFactor;
                thrTileN = packFactor;

                packed = true;
            }

            // Enable the use of longer word instructions if possible
            if(context->kernelOptions().enableLongDwordInstructions && (packed || packFactor <= 1))
            {
                auto maxWidth = std::min(context->kernelOptions().storeGlobalWidth,
                                         context->kernelOptions().loadLocalWidth);

                auto numDwordsPerElement = DataTypeInfo::Get(vtype).registerCount;

                updateThreadTileForLongDwords(thrTileM, thrTileN, maxWidth, numDwordsPerElement);
            }

            if(!useSwappedAccess)
                std::swap(thrTileM, thrTileN);

            auto internalTile = MacroTile(macTile.sizes, MemoryType::VGPR, {thrTileM, thrTileN});
            internalTile.layoutType = macTile.layoutType;

            return graph.coordinates.addElement(internalTile);
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
                                ContextPtr                       context)

        {
            auto workgroupSizes = context->kernel()->workgroupSize();

            auto [nMacX, iMacX, nMacY, iMacY]
                = addLoadMacroTileCT(graph, connections, macTileTag, sdim);

            addLoadThreadTileCT(graph, connections, macTileTag, iMacX, iMacY, workgroupSizes, true);

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
                                std::vector<unsigned int> const& jammedTiles,
                                ContextPtr                       context)

        {
            auto wavefrontSize = context->kernel()->wavefront_size();

            auto [nMacX, iMacX, nMacY, iMacY]
                = addLoadMacroTileCT(graph, connections, macTileTag, sdim);

            addLoadWaveTileCT(
                graph, connections, macTileTag, iMacX, iMacY, wavefrontSize, jammedTiles);

            graph.coordinates.addElement(DataFlow(), {userTag}, {macTileTag});
        }

        /**
         * @brief Add coordinate-transforms for loading from global
         * memory into a WaveTile through LDS.
         */
        void loadMacroTile_LDS(KernelGraph&                     graph,
                               std::vector<DeferredConnection>& connections,
                               int                              loadTag,
                               int                              userTag,
                               int                              macTileTag,
                               std::vector<int> const&          sdim,
                               std::vector<unsigned int> const& jammedTiles,
                               ContextPtr                       context)

        {
            auto macTile = graph.coordinates.getNode<MacroTile>(macTileTag);

            auto workgroupSizes = context->kernel()->workgroupSize();
            auto wavefrontSize  = context->kernel()->wavefront_size();

            auto useWaveAccess = macTile.memoryType == MemoryType::WAVE_LDS;
            auto useSwappedAccess
                = useWaveAccess
                  && context->kernelOptions().transposeMemoryAccess[macTile.layoutType];

            auto internalMacTileTag = createInternalTile(graph, loadTag, macTileTag, context);
            auto ldsTag             = graph.coordinates.addElement(LDS());

            auto internalTile  = *graph.coordinates.get<MacroTile>(internalMacTileTag);
            auto iMacXStoreLDS = graph.coordinates.addElement(internalTile.tileIndex(0));
            auto iMacYStoreLDS = graph.coordinates.addElement(internalTile.tileIndex(1));
            auto iMacXLoadLDS  = graph.coordinates.addElement(internalTile.tileIndex(0));
            auto iMacYLoadLDS  = graph.coordinates.addElement(internalTile.tileIndex(1));

            // Load from global into VGPRs
            {
                std::vector<DeferredConnection> ldsConnections;

                auto [nMacX, iMacX, nMacY, iMacY]
                    = addLoadMacroTileCT(graph, ldsConnections, macTileTag, sdim);

                addLDSDirection(
                    connections, ldsConnections, Connections::LDSLoadStore::LOAD_FROM_GLOBAL, true);

                addLoadThreadTileCT(graph,
                                    ldsConnections,
                                    internalMacTileTag,
                                    iMacX,
                                    iMacY,
                                    workgroupSizes,
                                    useSwappedAccess || !useWaveAccess);

                addLDSDirection(
                    connections, ldsConnections, Connections::LDSLoadStore::LOAD_FROM_GLOBAL);

                connections.push_back(
                    LDSDC<User>(userTag, Connections::LDSLoadStore::LOAD_FROM_GLOBAL));
                connections.push_back(LDSDC<MacroTile>(
                    internalMacTileTag, Connections::LDSLoadStore::LOAD_FROM_GLOBAL));
            }

            // Store from VGPRs into LDS
            {
                std::vector<DeferredConnection> ldsConnections;

                addStoreThreadTileCT(graph,
                                     ldsConnections,
                                     internalMacTileTag,
                                     iMacXStoreLDS,
                                     iMacYStoreLDS,
                                     workgroupSizes,
                                     useSwappedAccess || !useWaveAccess);

                addLDSDirection(
                    connections, ldsConnections, Connections::LDSLoadStore::STORE_INTO_LDS);

                connections.push_back(LDSDC<MacroTile>(internalMacTileTag,
                                                       Connections::LDSLoadStore::STORE_INTO_LDS));
                connections.push_back(
                    LDSDC<LDS>(ldsTag, Connections::LDSLoadStore::STORE_INTO_LDS));
            }

            if(useSwappedAccess || !useWaveAccess)
                graph.coordinates.addElement(Flatten(), {iMacXStoreLDS, iMacYStoreLDS}, {ldsTag});
            else
                graph.coordinates.addElement(Flatten(), {iMacYStoreLDS, iMacXStoreLDS}, {ldsTag});

            // Load from LDS into VGPRs
            {
                std::vector<DeferredConnection> ldsConnections;

                if(macTile.memoryType == MemoryType::WAVE_LDS)
                {
                    addLoadWaveTileCT(graph,
                                      ldsConnections,
                                      macTileTag,
                                      iMacXLoadLDS,
                                      iMacYLoadLDS,
                                      wavefrontSize,
                                      jammedTiles);
                }
                else
                {
                    addLoadThreadTileCT(graph,
                                        ldsConnections,
                                        internalMacTileTag,
                                        iMacXLoadLDS,
                                        iMacYLoadLDS,
                                        workgroupSizes,
                                        true);
                }

                addLDSDirection(
                    connections, ldsConnections, Connections::LDSLoadStore::LOAD_FROM_LDS, true);

                connections.push_back(DC<MacroTile>(macTileTag));
                connections.push_back(LDSDC<LDS>(ldsTag, Connections::LDSLoadStore::LOAD_FROM_LDS));
            }

            if(useSwappedAccess || !useWaveAccess)
                graph.coordinates.addElement(Tile(), {ldsTag}, {iMacXLoadLDS, iMacYLoadLDS});
            else
                graph.coordinates.addElement(Tile(), {ldsTag}, {iMacYLoadLDS, iMacXLoadLDS});

            graph.coordinates.addElement(DataFlow(), {userTag}, {ldsTag});
            graph.coordinates.addElement(DataFlow(), {ldsTag}, {macTileTag});
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
                                 ContextPtr                       context)

        {
            auto workgroupSizes = context->kernel()->workgroupSize();

            auto [nMacX, iMacX, nMacY, iMacY]
                = addStoreMacroTileCT(graph, connections, macTileTag, sdim);

            addStoreThreadTileCT(
                graph, connections, macTileTag, iMacX, iMacY, workgroupSizes, true);

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
                                 ContextPtr                       context)

        {
            auto wavefrontSize = context->kernel()->wavefront_size();

            auto [nMacX, iMacX, nMacY, iMacY]
                = addStoreMacroTileCT(graph, connections, macTileTag, sdim);

            addStoreWaveTileCT(
                graph, connections, macTileTag, iMacX, iMacY, wavefrontSize, jammedTiles);

            graph.coordinates.addElement(DataFlow(), {macTileTag}, {userTag});
        }

        /**
         * @brief Add coordinate-transforms for storing into global
         * memory from a WaveTile through LDS.
         */
        void storeMacroTile_WAVE_LDS(KernelGraph&                     graph,
                                     std::vector<DeferredConnection>& connections,
                                     int                              storeTag,
                                     int                              userTag,
                                     int                              macTileTag,
                                     std::vector<int> const&          sdim,
                                     std::vector<unsigned int> const& jammedTiles,
                                     ContextPtr                       context)

        {
            auto macTile = graph.coordinates.getNode<MacroTile>(macTileTag);
            auto user    = graph.coordinates.getNode<User>(userTag);

            AssertFatal(macTile.layoutType == LayoutType::MATRIX_ACCUMULATOR);

            auto workgroupSizes = context->kernel()->workgroupSize();
            auto wavefrontSize  = context->kernel()->wavefront_size();

            auto internalMacTileTag = createInternalTile(graph, storeTag, macTileTag, context);
            auto ldsTag             = graph.coordinates.addElement(LDS());

            auto internalTile  = *graph.coordinates.get<MacroTile>(internalMacTileTag);
            auto iMacXStoreLDS = graph.coordinates.addElement(internalTile.tileIndex(0));
            auto iMacYStoreLDS = graph.coordinates.addElement(internalTile.tileIndex(1));
            auto iMacXLoadLDS  = graph.coordinates.addElement(internalTile.tileIndex(0));
            auto iMacYLoadLDS  = graph.coordinates.addElement(internalTile.tileIndex(1));

            // Store from VGPRs to LDS
            {
                std::vector<DeferredConnection> ldsConnections;

                addStoreWaveTileCT(graph,
                                   ldsConnections,
                                   macTileTag,
                                   iMacXStoreLDS,
                                   iMacYStoreLDS,
                                   wavefrontSize,
                                   jammedTiles);

                addLDSDirection(
                    connections, ldsConnections, Connections::LDSLoadStore::STORE_INTO_LDS, true);

                connections.push_back(DC<MacroTile>(macTileTag));
                connections.push_back(
                    LDSDC<LDS>(ldsTag, Connections::LDSLoadStore::STORE_INTO_LDS));
            }

            graph.coordinates.addElement(Flatten(), {iMacXStoreLDS, iMacYStoreLDS}, {ldsTag});

            // Load from LDS to VGPRs
            {
                std::vector<DeferredConnection> ldsConnections;

                addLoadAccumulatorTileCT(graph,
                                         ldsConnections,
                                         internalMacTileTag,
                                         iMacXLoadLDS,
                                         iMacYLoadLDS,
                                         workgroupSizes);

                addLDSDirection(
                    connections, ldsConnections, Connections::LDSLoadStore::LOAD_FROM_LDS);

                connections.push_back(
                    LDSDC<MacroTile>(internalMacTileTag, Connections::LDSLoadStore::LOAD_FROM_LDS));
                connections.push_back(LDSDC<LDS>(ldsTag, Connections::LDSLoadStore::LOAD_FROM_LDS));
            }

            graph.coordinates.addElement(Tile(), {ldsTag}, {iMacXLoadLDS, iMacYLoadLDS});

            // Store from VGPRs to global
            {
                std::vector<DeferredConnection> ldsConnections;

                auto [nMacX, iMacX, nMacY, iMacY]
                    = addStoreMacroTileCT(graph, connections, internalMacTileTag, sdim);

                addStoreAccumulatorTileCT(
                    graph, ldsConnections, internalMacTileTag, iMacX, iMacY, workgroupSizes);

                addLDSDirection(connections,
                                ldsConnections,
                                Connections::LDSLoadStore::STORE_INTO_GLOBAL,
                                true);

                connections.push_back(LDSDC<MacroTile>(
                    internalMacTileTag, Connections::LDSLoadStore::STORE_INTO_GLOBAL));
                connections.push_back(
                    LDSDC<User>(userTag, Connections::LDSLoadStore::STORE_INTO_GLOBAL));
            }

            graph.coordinates.addElement(DataFlow(), {macTileTag}, {ldsTag});
            graph.coordinates.addElement(DataFlow(), {ldsTag}, {userTag});
        }

        /**
         * @brief Tile re-writer.
         */
        struct LowerTileVisitor : public BaseGraphVisitor
        {
            using BaseGraphVisitor::visitEdge;
            using BaseGraphVisitor::visitOperation;

            LowerTileVisitor(std::shared_ptr<CommandParameters> params, ContextPtr context)
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
                                        LoadTiled const&   oload) override
            {
                auto logger = rocRoller::Log::getLogger();

                logger->debug("KernelGraph::LowerTileVisitor::LoadTiled({})", tag);

                auto originalUserTag    = original.mapper.get<User>(tag);
                auto originalMacTileTag = original.mapper.get<MacroTile>(tag);
                auto userTag            = reindexer.coordinates.at(originalUserTag);
                auto macTileTag         = reindexer.coordinates.at(originalMacTileTag);

                auto sdims
                    = original.coordinates
                          .getInputNodeIndices(originalMacTileTag, CT::isEdge<ConstructMacroTile>)
                          .to<std::vector>();
                for(int i = 0; i < sdims.size(); i++)
                    sdims[i] = reindexer.coordinates.at(sdims[i]);

                copyOperation(graph, original, reindexer, tag);

                auto macTile = graph.coordinates.getNode<MacroTile>(macTileTag);

                AssertFatal(macTile.rank == 2, "Rank /= 2 not implemented yet.");

                logger->debug("  User({}), MacroTile({}), MacroTile size: {}x{}",
                              userTag,
                              macTileTag,
                              macTile.sizes[0],
                              macTile.sizes[1]);

                std::vector<DeferredConnection> connections;

                auto loadTag               = reindexer.control.at(tag);
                auto wavetilesPerWorkgroup = m_params->getWaveTilesPerWorkgroup();

                switch(macTile.memoryType)
                {
                case MemoryType::VGPR:
                    loadMacroTile_VGPR(graph, connections, userTag, macTileTag, sdims, m_context);
                    break;
                case MemoryType::WAVE:
                    loadMacroTile_WAVE(graph,
                                       connections,
                                       userTag,
                                       macTileTag,
                                       sdims,
                                       wavetilesPerWorkgroup,
                                       m_context);
                    break;
                case MemoryType::LDS:
                case MemoryType::WAVE_LDS:
                    loadMacroTile_LDS(graph,
                                      connections,
                                      loadTag,
                                      userTag,
                                      macTileTag,
                                      sdims,
                                      wavetilesPerWorkgroup,
                                      m_context);
                    break;
                default:
                    Throw<FatalError>("LoadTiled: MacroTile memory type not supported yet.");
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
                                        StoreTiled const&  ostore) override
            {
                auto logger = rocRoller::Log::getLogger();

                logger->debug("KernelGraph::LowerTileVisitor::StoreTiled({})", tag);

                auto originalUserTag    = original.mapper.get<User>(tag);
                auto originalMacTileTag = original.mapper.get<MacroTile>(tag);
                auto userTag            = reindexer.coordinates.at(originalUserTag);
                auto macTileTag         = reindexer.coordinates.at(originalMacTileTag);

                auto sdims
                    = original.coordinates
                          .getOutputNodeIndices(originalMacTileTag, CT::isEdge<DestructMacroTile>)
                          .to<std::vector>();
                for(int i = 0; i < sdims.size(); i++)
                    sdims[i] = reindexer.coordinates.at(sdims[i]);

                copyOperation(graph, original, reindexer, tag);

                auto macTile = graph.coordinates.getNode<MacroTile>(macTileTag);

                AssertFatal(macTile.rank == 2, "Rank /= 2 not implemented yet.");

                logger->debug("  User({}), MacroTile({}), MacroTile size: {}x{}",
                              userTag,
                              macTileTag,
                              macTile.sizes[0],
                              macTile.sizes[1]);

                std::vector<DeferredConnection> connections;

                auto storeTag              = reindexer.control.at(tag);
                auto wavetilesPerWorkgroup = m_params->getWaveTilesPerWorkgroup();

                switch(macTile.memoryType)
                {
                case MemoryType::VGPR:
                    storeMacroTile_VGPR(graph, connections, userTag, macTileTag, sdims, m_context);
                    break;
                case MemoryType::WAVE:
                    storeMacroTile_WAVE(graph,
                                        connections,
                                        userTag,
                                        macTileTag,
                                        sdims,
                                        wavetilesPerWorkgroup,
                                        m_context);
                    break;
                case MemoryType::WAVE_LDS:
                    storeMacroTile_WAVE_LDS(graph,
                                            connections,
                                            storeTag,
                                            userTag,
                                            macTileTag,
                                            sdims,
                                            wavetilesPerWorkgroup,
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

        private:
            std::shared_ptr<AssemblyKernel>    m_kernel;
            std::shared_ptr<CommandParameters> m_params;
        };

        KernelGraph LowerTile::apply(KernelGraph const& graph)
        {
            TIMER(t, "KernelGraph::lowerTile");
            auto visitor = LowerTileVisitor(m_params, m_context);
            return rewrite(graph, visitor);
        }
    }
}
