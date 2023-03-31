
#include <rocRoller/KernelGraph/Utils.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        namespace Expression = rocRoller::Expression;
        namespace CT         = rocRoller::KernelGraph::CoordinateGraph;

        using GD = Graph::Direction;

        using namespace CoordinateGraph;
        using namespace ControlGraph;
        using namespace Expression;

        /***********************************
         * Helpers
         */

        /**
         * Create a range-based for loop.
         */
        std::pair<int, int> rangeFor(KernelGraph&              graph,
                                     Expression::ExpressionPtr size,
                                     const std::string&        loopName)
        {
            auto unitStride   = Expression::literal(1u);
            auto rangeK       = graph.coordinates.addElement(Linear(size, unitStride));
            auto dimK         = graph.coordinates.addElement(ForLoop(size, unitStride));
            auto sizeDataType = Expression::resultVariableType(size);
            auto exprK        = std::make_shared<Expression::Expression>(
                DataFlowTag{rangeK, Register::Type::Scalar, sizeDataType});

            auto forK  = graph.control.addElement(ForLoopOp{exprK < size, loopName});
            auto initK = graph.control.addElement(
                Assign{Register::Type::Scalar, Expression::literal(0, sizeDataType)});
            auto incrementK
                = graph.control.addElement(Assign{Register::Type::Scalar, exprK + unitStride});

            graph.coordinates.addElement(DataFlow(), {rangeK}, {dimK});
            graph.control.addElement(Initialize(), {forK}, {initK});
            graph.control.addElement(ForLoopIncrement(), {forK}, {incrementK});

            graph.mapper.connect<Dimension>(forK, rangeK);
            graph.mapper.connect(initK, rangeK, NaryArgument::DEST);
            graph.mapper.connect(incrementK, rangeK, NaryArgument::DEST);

            return {dimK, forK};
        }

        void updateLoadLDSMacroTile(KernelGraph&      graph,
                                    MacroTile const&  macTile,
                                    int               loadTag,
                                    std::vector<int>& sdims,
                                    int               K,
                                    int               lds,
                                    bool              useSwappedAccess)
        {
            // given that the loadMacroTile has already lowered the macrotile for LoadTiled
            // before it is transformed to LoadLDSTile

            if(macTile.layoutType == LayoutType::MATRIX_A)
            {
                // remove passthrough between A row block and x-workgroup
                // remove x-workgroup
                auto aTileNumX   = graph.mapper.get<MacroTileNumber>(loadTag, 0);
                auto aWorkgroupX = graph.mapper.get<Workgroup>(loadTag, 0);
                graph.coordinates.deleteElement(std::vector<int>{aTileNumX},
                                                std::vector<int>{aWorkgroupX},
                                                CT::isEdge<PassThrough>);
                graph.mapper.disconnect<Workgroup>(loadTag, aWorkgroupX, 0);
                graph.coordinates.deleteElement(aWorkgroupX);

                // remove edge between A column block and K if it's not an unroll split.
                auto aTileNumY = graph.mapper.get<MacroTileNumber>(loadTag, 1);

                //TODO: Clean this up once there is a non-templated version of getOutputNodeIndices.
                auto outputNodes
                    = graph.coordinates
                          .getOutputNodeIndices(aTileNumY, [](Edge const& a) { return true; })
                          .to<std::vector>();
                if(outputNodes.size() == 1)
                {
                    graph.coordinates.deleteElement(std::vector<int>{aTileNumY},
                                                    outputNodes,
                                                    [](Edge const& a) { return true; });
                }
            }
            else if(macTile.layoutType == LayoutType::MATRIX_B)
            {
                // remove passthrough between B column block and y-workgroup
                // remove y-workgroup
                auto bTileNumY   = graph.mapper.get<MacroTileNumber>(loadTag, 1);
                auto bWorkgroupY = graph.mapper.get<Workgroup>(loadTag, 1);
                graph.coordinates.deleteElement(std::vector<int>{bTileNumY},
                                                std::vector<int>{bWorkgroupY},
                                                CT::isEdge<PassThrough>);
                graph.mapper.disconnect<Workgroup>(loadTag, bWorkgroupY, 1);
                graph.coordinates.deleteElement(bWorkgroupY);

                // remove edge between B row block and K if it's not an unroll split.
                auto bTileNumX = graph.mapper.get<MacroTileNumber>(loadTag, 0);

                auto outputNodes
                    = graph.coordinates
                          .getOutputNodeIndices(bTileNumX, [](Edge const& a) { return true; })
                          .to<std::vector>();

                if(outputNodes.size() == 1)
                {
                    graph.coordinates.deleteElement(std::vector<int>{bTileNumX},
                                                    outputNodes,
                                                    [](Edge const& a) { return true; });
                }
            }
            else
            {
                auto tileNumX   = graph.mapper.get<MacroTileNumber>(loadTag, 0);
                auto workgroupX = graph.mapper.get<Workgroup>(loadTag, 0);
                graph.mapper.disconnect<Workgroup>(loadTag, workgroupX, 0);
                graph.coordinates.deleteElement(std::vector<int>{tileNumX},
                                                std::vector<int>{workgroupX},
                                                CT::isEdge<PassThrough>);
                graph.coordinates.deleteElement(workgroupX);

                auto tileNumY   = graph.mapper.get<MacroTileNumber>(loadTag, 1);
                auto workgroupY = graph.mapper.get<Workgroup>(loadTag, 1);
                graph.mapper.disconnect<Workgroup>(loadTag, workgroupY, 1);
                graph.coordinates.deleteElement(std::vector<int>{tileNumY},
                                                std::vector<int>{workgroupY},
                                                CT::isEdge<PassThrough>);
                graph.coordinates.deleteElement(workgroupY);
            }

            std::vector<int> iMac;
            for(size_t i = 0; i < sdims.size(); ++i)
            {
                auto mac = graph.coordinates.getOutputNodeIndices(sdims[i], CT::isEdge<Tile>)
                               .to<std::vector>();
                iMac.push_back(mac[1]);
                graph.coordinates.deleteElement(std::vector<int>{sdims[i]}, mac, CT::isEdge<Tile>);
                graph.mapper.disconnect<MacroTileNumber>(loadTag, mac[0], i);
                graph.coordinates.deleteElement(mac[0]);
            }

            if(useSwappedAccess)
                graph.coordinates.addElement(Tile(), {lds}, {iMac[0], iMac[1]});
            else
                graph.coordinates.addElement(Tile(), {lds}, {iMac[1], iMac[0]});
        }

        void loadWaveMacroTile(KernelGraph&                     graph,
                               MacroTile const&                 macTile,
                               int                              loadTag,
                               int                              iMacX,
                               int                              iMacY,
                               int                              userTag,
                               int                              wavefrontSize,
                               std::vector<unsigned int> const& wavetilesPerWorkgroup)
        {
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
            graph.mapper.connect<WaveTile>(loadTag, waveTileTag);

            auto nWaveX = graph.coordinates.addElement(waveTile.tileNumber(0));
            auto nWaveY = graph.coordinates.addElement(waveTile.tileNumber(1));
            auto iWaveX = graph.coordinates.addElement(waveTile.tileIndex(0));
            auto iWaveY = graph.coordinates.addElement(waveTile.tileIndex(1));

            graph.coordinates.addElement(Tile(), {iMacX}, {nWaveX, iWaveX});
            graph.coordinates.addElement(Tile(), {iMacY}, {nWaveY, iWaveY});

            graph.mapper.connect<WaveTileNumber>(loadTag, nWaveX, 0);
            graph.mapper.connect<WaveTileNumber>(loadTag, nWaveY, 1);

            auto waveX = graph.coordinates.addElement(Wavefront(0));
            auto waveY = graph.coordinates.addElement(Wavefront(1));
            auto wave  = graph.coordinates.addElement(Wavefront(-1));

            uint numElements = product(tileSize);
            uint wfs         = static_cast<uint>(wavefrontSize);
            uint numVgpr     = numElements / wfs;

            auto wavefrontSizeLiteral = Expression::literal(wfs);

            auto lane = graph.coordinates.addElement(Lane(wavefrontSizeLiteral, nullptr));
            auto vgpr = graph.coordinates.addElement(VGPR(literal(numVgpr), nullptr));

            graph.coordinates.addElement(Flatten(), {waveX, waveY}, {wave});
            graph.coordinates.addElement(Flatten(), {wave, lane}, {workitem});

            graph.mapper.connect<VGPR>(loadTag, vgpr);

            int jammedWavetileX = -1;
            if(wavetilesPerWorkgroup[0] >= 1)
            {
                jammedWavetileX = graph.coordinates.addElement(
                    JammedWaveTileNumber(0, literal(wavetilesPerWorkgroup[0]), literal(1)));
                graph.mapper.connect<JammedWaveTileNumber>(loadTag, jammedWavetileX, 0);
            }
            int jammedWavetileY = -1;
            if(wavetilesPerWorkgroup[1] >= 1)
            {
                jammedWavetileY = graph.coordinates.addElement(
                    JammedWaveTileNumber(1, literal(wavetilesPerWorkgroup[1]), literal(1)));
                graph.mapper.connect<JammedWaveTileNumber>(loadTag, jammedWavetileY, 1);
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

                if(wavetilesPerWorkgroup[0] > 1)
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

                if(wavetilesPerWorkgroup[1] > 1)
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

                graph.mapper.connect<VGPRBlockNumber>(loadTag, nVblk);
                graph.mapper.connect<VGPRBlockIndex>(loadTag, iVblk);

                graph.coordinates.addElement(Tile(), {iWaveX}, {rowBlock, iVblk});
                graph.coordinates.addElement(Tile(), {iWaveY}, {colBlock, iLblk});

                graph.coordinates.addElement(Flatten(), {rowBlock, colBlock}, {block});
                graph.coordinates.addElement(Tile(), {block}, {nVblk, nLblk});

                graph.coordinates.addElement(Flatten(), {nVblk, iVblk}, {vgpr});
                graph.coordinates.addElement(Flatten(), {nLblk, iLblk}, {lane});

                if(wavetilesPerWorkgroup[0] > 1)
                    graph.coordinates.addElement(Tile(), {nWaveX}, {waveX, jammedWavetileX});
                else
                    graph.coordinates.addElement(PassThrough(), {nWaveX}, {waveX});

                if(wavetilesPerWorkgroup[1] > 1)
                    graph.coordinates.addElement(Tile(), {nWaveY}, {waveY, jammedWavetileY});
                else
                    graph.coordinates.addElement(PassThrough(), {nWaveY}, {waveY});
            }
            break;

            default:
                Throw<FatalError>("Not implemented yet.");
            }
        }

        std::vector<DeferredConnection>
            loadMacroTileFromLDS(KernelGraph&                       graph,
                                 int                                ldsTag,
                                 int                                macTileTag,
                                 std::array<unsigned int, 3> const& workgroupSizes)
        {
            auto macTile = graph.coordinates.getNode<MacroTile>(macTileTag);

            rocRoller::Log::getLogger()->debug(
                "KernelGraph::Utils::loadMacroTileFromLDS(): LDS({}), MacroTile({})",
                ldsTag,
                macTileTag);
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::Utils::loadMacroTileFromLDS(): MacroTile size: {}x{}",
                macTile.sizes[0],
                macTile.sizes[1]);

            auto workitemX
                = graph.coordinates.addElement(Workitem(0, literal(workgroupSizes.at(0))));

            auto thrTile = ThreadTile(macTile);

            auto iMacX = graph.coordinates.addElement(macTile.tileIndex(0));
            auto iMacY = graph.coordinates.addElement(macTile.tileIndex(1));
            auto nThrX
                = graph.coordinates.addElement(ThreadTileNumber(0, literal(thrTile.sizes.at(0))));
            auto nThrY
                = graph.coordinates.addElement(ThreadTileNumber(1, literal(thrTile.wsizes.at(1))));
            auto iThrX
                = graph.coordinates.addElement(ThreadTileIndex(0, literal(thrTile.wsizes.at(0))));
            auto iThrY
                = graph.coordinates.addElement(ThreadTileIndex(1, literal(thrTile.sizes.at(1))));

            graph.coordinates.addElement(Tile(), {ldsTag}, {iMacX, iMacY});

            graph.coordinates.addElement(Tile(), {iMacX}, {nThrX, iThrX});
            graph.coordinates.addElement(Tile(), {iMacY}, {nThrY, iThrY});

            graph.coordinates.addElement(Flatten(), {nThrY, iThrX}, {workitemX});

            auto elementNumberX
                = graph.coordinates.addElement(ElementNumber(0, literal(thrTile.sizes.at(0))));
            auto elementNumberY
                = graph.coordinates.addElement(ElementNumber(1, literal(thrTile.sizes.at(1))));

            graph.coordinates.addElement(PassThrough(), {nThrX}, {elementNumberX});
            graph.coordinates.addElement(PassThrough(), {iThrY}, {elementNumberY});

            std::vector<DeferredConnection> connections;
            connections.push_back(DC<ElementNumber>(elementNumberX, 0));
            connections.push_back(DC<ElementNumber>(elementNumberY, 1));

            // LDS --DataFlow--> macrotile
            graph.coordinates.addElement(DataFlow(), {ldsTag}, {macTileTag});

            return connections;
        }

        void loadMacroTileForLDS(KernelGraph&                       graph,
                                 int                                loadTag,
                                 int                                userTag,
                                 int                                macTileTag,
                                 std::vector<int>&                  sdim,
                                 int                                K,
                                 std::array<unsigned int, 3> const& workgroupSizes,
                                 int                                unroll,
                                 bool                               useSwappedAccess)
        {
            auto macTile = graph.coordinates.getNode<MacroTile>(macTileTag);
            auto user    = graph.coordinates.getNode<User>(userTag);

            rocRoller::Log::getLogger()->debug(
                "KernelGraph::Utils::loadMacroTileForLDS(): User({}), MacroTile({})",
                userTag,
                macTileTag);
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::Utils::loadMacroTileForLDS(): MacroTile size: {}x{}",
                macTile.sizes[0],
                macTile.sizes[1]);

            AssertFatal(
                macTile.rank == 2, "Rank != 2 not implemented yet.", ShowValue(macTile.rank));

            auto sdimX = sdim[0];
            auto sdimY = sdim[1];
            graph.mapper.connect<SubDimension>(loadTag, sdimX, 0);
            graph.mapper.connect<SubDimension>(loadTag, sdimY, 1);

            auto nMacX = graph.coordinates.addElement(macTile.tileNumber(0));
            auto nMacY = graph.coordinates.addElement(macTile.tileNumber(1));
            auto iMacX = graph.coordinates.addElement(macTile.tileIndex(0));
            auto iMacY = graph.coordinates.addElement(macTile.tileIndex(1));

            graph.mapper.connect<MacroTileNumber>(loadTag, nMacX, 0);
            graph.mapper.connect<MacroTileNumber>(loadTag, nMacY, 1);

            graph.coordinates.addElement(Tile(), {sdimX}, {nMacX, iMacX});
            graph.coordinates.addElement(Tile(), {sdimY}, {nMacY, iMacY});

            auto workitemX
                = graph.coordinates.addElement(Workitem(0, literal(workgroupSizes.at(0))));

            auto thrTile = ThreadTile(macTile);

            auto nThrX
                = graph.coordinates.addElement(ThreadTileNumber(0, literal(thrTile.wsizes.at(0))));
            auto nThrY
                = graph.coordinates.addElement(ThreadTileNumber(1, literal(thrTile.wsizes.at(1))));
            auto iThrX
                = graph.coordinates.addElement(ThreadTileIndex(0, literal(thrTile.sizes.at(0))));
            auto iThrY
                = graph.coordinates.addElement(ThreadTileIndex(1, literal(thrTile.sizes.at(1))));

            if(useSwappedAccess)
            {
                auto elementNumberX
                    = graph.coordinates.addElement(ElementNumber(0, literal(thrTile.sizes.at(0))));
                auto elementNumberY
                    = graph.coordinates.addElement(ElementNumber(1, literal(thrTile.sizes.at(1))));

                graph.coordinates.addElement(PassThrough(), {iThrX}, {elementNumberX});
                graph.coordinates.addElement(PassThrough(), {iThrY}, {elementNumberY});

                graph.mapper.connect<ElementNumber>(loadTag, elementNumberX, 0);
                graph.mapper.connect<ElementNumber>(loadTag, elementNumberY, 1);

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
                auto elementNumberX
                    = graph.coordinates.addElement(ElementNumber(0, literal(thrTile.sizes.at(1))));
                auto elementNumberY
                    = graph.coordinates.addElement(ElementNumber(1, literal(thrTile.sizes.at(0))));

                graph.coordinates.addElement(PassThrough(), {iThrX}, {elementNumberY});
                graph.coordinates.addElement(PassThrough(), {iThrY}, {elementNumberX});

                graph.mapper.connect<ElementNumber>(loadTag, elementNumberX, 0);
                graph.mapper.connect<ElementNumber>(loadTag, elementNumberY, 1);

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

            graph.coordinates.addElement(Tile(), {iMacX}, {nThrX, iThrX});
            graph.coordinates.addElement(Tile(), {iMacY}, {nThrY, iThrY});

            if(macTile.layoutType == LayoutType::MATRIX_A)
            {
                auto workgroupX = graph.coordinates.addElement(Workgroup(0));
                graph.mapper.connect<Workgroup>(loadTag, workgroupX, 0);
                graph.coordinates.addElement(PassThrough(), {nMacX}, {workgroupX});
                // A row block is x-workgroup, column block is for loop index
                if(unroll >= 0)
                {
                    graph.coordinates.addElement(Split(), {nMacY}, {K, unroll});
                }
                else
                {
                    graph.coordinates.addElement(PassThrough(), {nMacY}, {K});
                }
            }
            else if(macTile.layoutType == LayoutType::MATRIX_B)
            {
                auto workgroupY = graph.coordinates.addElement(Workgroup(1));
                graph.mapper.connect<Workgroup>(loadTag, workgroupY, 1);
                // B row block is for loop index, column block is y-workgroup
                if(unroll >= 0)
                {
                    graph.coordinates.addElement(Split(), {nMacX}, {K, unroll});
                }
                else
                {
                    graph.coordinates.addElement(PassThrough(), {nMacX}, {K});
                }
                graph.coordinates.addElement(PassThrough(), {nMacY}, {workgroupY});
            }
            else
            {
                auto workgroupX = graph.coordinates.addElement(Workgroup(0));
                graph.coordinates.addElement(PassThrough(), {nMacX}, {workgroupX});
                graph.mapper.connect<Workgroup>(loadTag, workgroupX, 0);

                auto workgroupY = graph.coordinates.addElement(Workgroup(1));
                graph.coordinates.addElement(PassThrough(), {nMacY}, {workgroupY});
                graph.mapper.connect<Workgroup>(loadTag, workgroupY, 1);
            }
        }

        void loadMacroTile(KernelGraph&                       graph,
                           int                                loadTag,
                           int                                userTag,
                           int                                macTileTag,
                           std::vector<int>&                  sdim,
                           std::array<unsigned int, 3> const& workgroupSizes,
                           int                                wavefrontSize,
                           std::vector<unsigned int> const&   wavetilesPerWorkgroup)

        {
            auto macTile = graph.coordinates.getNode<MacroTile>(macTileTag);
            auto user    = graph.coordinates.getNode<User>(userTag);

            rocRoller::Log::getLogger()->debug(
                "KernelGraph::Utils::loadMacroTile(): User({}), MacroTile({})",
                userTag,
                macTileTag);
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::Utils::loadMacroTile(): MacroTile size: {}x{}",
                macTile.sizes[0],
                macTile.sizes[1]);

            AssertFatal(macTile.rank == 2, "Rank /= 2 not implemented yet.");

            auto sdimX = sdim[0];
            auto sdimY = sdim[1];
            graph.mapper.connect<SubDimension>(loadTag, sdimX, 0);
            graph.mapper.connect<SubDimension>(loadTag, sdimY, 1);

            auto nMacX = graph.coordinates.addElement(macTile.tileNumber(0));
            auto nMacY = graph.coordinates.addElement(macTile.tileNumber(1));
            auto iMacX = graph.coordinates.addElement(macTile.tileIndex(0));
            auto iMacY = graph.coordinates.addElement(macTile.tileIndex(1));

            graph.mapper.connect<MacroTileNumber>(loadTag, nMacX, 0);
            graph.mapper.connect<MacroTileNumber>(loadTag, nMacY, 1);

            auto workgroupX = graph.coordinates.addElement(Workgroup(0));
            auto workgroupY = graph.coordinates.addElement(Workgroup(1));

            graph.mapper.connect<Workgroup>(loadTag, workgroupX, 0);
            graph.mapper.connect<Workgroup>(loadTag, workgroupY, 1);

            graph.coordinates.addElement(Tile(), {sdimX}, {nMacX, iMacX});
            graph.coordinates.addElement(Tile(), {sdimY}, {nMacY, iMacY});

            graph.coordinates.addElement(PassThrough(), {nMacX}, {workgroupX});
            graph.coordinates.addElement(PassThrough(), {nMacY}, {workgroupY});

            switch(macTile.memoryType)
            {
            case MemoryType::VGPR:
            case MemoryType::LDS:
            {
                auto thrTile = ThreadTile(macTile);

                auto nThrX = graph.coordinates.addElement(
                    ThreadTileNumber(0, literal(thrTile.wsizes.at(0))));
                auto nThrY = graph.coordinates.addElement(
                    ThreadTileNumber(1, literal(thrTile.wsizes.at(1))));
                auto iThrX = graph.coordinates.addElement(
                    ThreadTileIndex(0, literal(thrTile.sizes.at(0))));
                auto iThrY = graph.coordinates.addElement(
                    ThreadTileIndex(1, literal(thrTile.sizes.at(1))));

                auto workitemX
                    = graph.coordinates.addElement(Workitem(0, literal(thrTile.wsizes.at(0))));
                auto workitemY
                    = graph.coordinates.addElement(Workitem(1, literal(thrTile.wsizes.at(1))));

                auto elementNumberX
                    = graph.coordinates.addElement(ElementNumber(0, literal(thrTile.sizes.at(0))));
                auto elementNumberY
                    = graph.coordinates.addElement(ElementNumber(1, literal(thrTile.sizes.at(1))));

                graph.mapper.connect<ElementNumber>(loadTag, elementNumberX, 0);
                graph.mapper.connect<ElementNumber>(loadTag, elementNumberY, 1);

                graph.coordinates.addElement(Tile(), {iMacX}, {nThrX, iThrX});
                graph.coordinates.addElement(Tile(), {iMacY}, {nThrY, iThrY});

                graph.coordinates.addElement(PassThrough(), {nThrX}, {workitemX});
                graph.coordinates.addElement(PassThrough(), {nThrY}, {workitemY});

                graph.coordinates.addElement(PassThrough(), {iThrX}, {elementNumberX});
                graph.coordinates.addElement(PassThrough(), {iThrY}, {elementNumberY});

                // User -> DataFlow() -> LDS gets added in addLDSOps
            }
            break;

            case MemoryType::WAVE:
            case MemoryType::WAVE_LDS:
                loadWaveMacroTile(graph,
                                  macTile,
                                  loadTag,
                                  iMacX,
                                  iMacY,
                                  userTag,
                                  wavefrontSize,
                                  wavetilesPerWorkgroup);
                // User -> DataFlow() -> LDS gets added in addWaveLDSOps
                break;

            default:
                Throw<FatalError>("Load : MacroTile memory type not supported yet.");
            }
        }

        void updateStoreLDSMacroTile(KernelGraph&      graph,
                                     MacroTile const&  macTile,
                                     int               storeTag,
                                     std::vector<int>& sdims,
                                     int               lds)
        {
            // given that the storeMacroTile has already lowered the macrotile for StoreTiled
            // before it is transformed to StoreLDSTile

            // remove macrotile numbers and workgroups
            auto tileNumX   = graph.mapper.get<MacroTileNumber>(storeTag, 0);
            auto workgroupX = graph.mapper.get<Workgroup>(storeTag, 0);
            graph.mapper.disconnect<Workgroup>(storeTag, workgroupX, 0);
            graph.coordinates.deleteElement(
                std::vector<int>{workgroupX}, std::vector<int>{tileNumX}, CT::isEdge<PassThrough>);
            graph.coordinates.deleteElement(workgroupX);

            auto tileNumY   = graph.mapper.get<MacroTileNumber>(storeTag, 1);
            auto workgroupY = graph.mapper.get<Workgroup>(storeTag, 1);
            graph.mapper.disconnect<Workgroup>(storeTag, workgroupY, 1);
            graph.coordinates.deleteElement(
                std::vector<int>{workgroupY}, std::vector<int>{tileNumY}, CT::isEdge<PassThrough>);
            graph.coordinates.deleteElement(workgroupY);

            std::vector<int> iMac;
            for(size_t i = 0; i < sdims.size(); ++i)
            {
                auto mac = graph.coordinates.getInputNodeIndices(sdims[i], CT::isEdge<Flatten>)
                               .to<std::vector>();
                iMac.push_back(mac[1]);
                graph.coordinates.deleteElement(
                    mac, std::vector<int>{sdims[i]}, CT::isEdge<Flatten>);
                graph.mapper.disconnect<MacroTileNumber>(storeTag, mac[0], i);
                graph.coordinates.deleteElement(mac[0]);
            }

            graph.coordinates.addElement(Flatten(), {iMac[0], iMac[1]}, {lds});
        }

        void storeWaveMacroTile(KernelGraph&                     graph,
                                MacroTile const&                 macTile,
                                int                              storeTag,
                                int                              iMacX,
                                int                              iMacY,
                                int                              workitem,
                                int                              userTag,
                                int                              wavefrontSize,
                                std::vector<unsigned int> const& wavetilesPerWorkgroup)
        {
            AssertFatal(macTile.layoutType == LayoutType::MATRIX_ACCUMULATOR,
                        "Store must be from accumulator.");

            auto waveTile    = WaveTile(macTile.subTileSizes, LayoutType::MATRIX_ACCUMULATOR);
            auto waveTileTag = graph.coordinates.addElement(waveTile);

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

            graph.mapper.connect<WaveTile>(storeTag, waveTileTag);
            graph.mapper.connect<VGPRBlockNumber>(storeTag, nVblk);
            graph.mapper.connect<VGPRBlockIndex>(storeTag, iVblk);
            graph.mapper.connect<VGPR>(storeTag, vgpr);

            graph.coordinates.addElement(Tile(), {vgpr}, {nVblk, iVblk});
            graph.coordinates.addElement(Tile(), {lane}, {nLblk, iLblk});
            graph.coordinates.addElement(Flatten(), {nVblk, nLblk}, {block});
            graph.coordinates.addElement(Tile(), {block}, {rowBlock, colBlock});

            int jammedWavetileX = -1;
            if(wavetilesPerWorkgroup[0] >= 1)
            {
                jammedWavetileX = graph.coordinates.addElement(
                    JammedWaveTileNumber(0, literal(wavetilesPerWorkgroup[0]), literal(1)));
                graph.mapper.connect<JammedWaveTileNumber>(storeTag, jammedWavetileX, 0);
            }
            int jammedWavetileY = -1;
            if(wavetilesPerWorkgroup[1] >= 1)
            {
                jammedWavetileY = graph.coordinates.addElement(
                    JammedWaveTileNumber(1, literal(wavetilesPerWorkgroup[1]), literal(1)));
                graph.mapper.connect<JammedWaveTileNumber>(storeTag, jammedWavetileY, 1);
            }

            if(wavetilesPerWorkgroup[0] > 1)
                graph.coordinates.addElement(Flatten(), {waveX, jammedWavetileX}, {nWaveX});
            else
                graph.coordinates.addElement(PassThrough(), {waveX}, {nWaveX});
            if(wavetilesPerWorkgroup[1] > 1)
                graph.coordinates.addElement(Flatten(), {waveY, jammedWavetileY}, {nWaveY});
            else
                graph.coordinates.addElement(PassThrough(), {waveY}, {nWaveY});

            graph.coordinates.addElement(Flatten(), {rowBlock, iVblk}, {iWaveX});
            graph.coordinates.addElement(Flatten(), {colBlock, iLblk}, {iWaveY});

            graph.coordinates.addElement(Tile(), {workitem}, {wave, lane});
        }

        void storeMacroTileIntoLDS(KernelGraph&                       graph,
                                   int                                storeTag,
                                   int                                ldsTag,
                                   int                                macTileTag,
                                   std::array<unsigned int, 3> const& workgroupSizes,
                                   bool                               useSwappedAccess)
        {
            auto macTile = graph.coordinates.getNode<MacroTile>(macTileTag);
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::Utils::storeMacroTileIntoLDS(): LDS({}), MacroTile({})",
                ldsTag,
                macTileTag);
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::Utils::storeMacroTileIntoLDS(): MacroTile size: {}x{}",
                macTile.sizes[0],
                macTile.sizes[1]);

            auto workitemX
                = graph.coordinates.addElement(Workitem(0, literal(workgroupSizes.at(0))));

            auto thrTile = ThreadTile(macTile);

            auto iMacX = graph.coordinates.addElement(macTile.tileIndex(0));
            auto iMacY = graph.coordinates.addElement(macTile.tileIndex(1));

            auto nThrX
                = graph.coordinates.addElement(ThreadTileNumber(0, literal(thrTile.wsizes.at(0))));
            auto nThrY
                = graph.coordinates.addElement(ThreadTileNumber(1, literal(thrTile.wsizes.at(1))));
            auto iThrX
                = graph.coordinates.addElement(ThreadTileIndex(0, literal(thrTile.sizes.at(0))));
            auto iThrY
                = graph.coordinates.addElement(ThreadTileIndex(1, literal(thrTile.sizes.at(1))));

            if(useSwappedAccess)
            {
                auto elementNumberX
                    = graph.coordinates.addElement(ElementNumber(0, literal(thrTile.sizes.at(0))));
                auto elementNumberY
                    = graph.coordinates.addElement(ElementNumber(1, literal(thrTile.sizes.at(1))));

                graph.coordinates.addElement(PassThrough(), {elementNumberX}, {iThrX});
                graph.coordinates.addElement(PassThrough(), {elementNumberY}, {iThrY});

                graph.mapper.connect<ElementNumber>(storeTag, elementNumberX, 0);
                graph.mapper.connect<ElementNumber>(storeTag, elementNumberY, 1);

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

                graph.coordinates.addElement(Flatten(), {iMacX, iMacY}, {ldsTag});
            }
            else
            {
                auto elementNumberX
                    = graph.coordinates.addElement(ElementNumber(0, literal(thrTile.sizes.at(1))));
                auto elementNumberY
                    = graph.coordinates.addElement(ElementNumber(1, literal(thrTile.sizes.at(0))));

                graph.coordinates.addElement(PassThrough(), {elementNumberY}, {iThrX});
                graph.coordinates.addElement(PassThrough(), {elementNumberX}, {iThrY});

                graph.mapper.connect<ElementNumber>(storeTag, elementNumberX, 0);
                graph.mapper.connect<ElementNumber>(storeTag, elementNumberY, 1);

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

                graph.coordinates.addElement(Flatten(), {iMacY, iMacX}, {ldsTag});
            }

            graph.coordinates.addElement(Flatten(), {nThrX, iThrX}, {iMacX});
            graph.coordinates.addElement(Flatten(), {nThrY, iThrY}, {iMacY});

            //macrotile --DataFlow--> LDS
            graph.coordinates.addElement(DataFlow(), {macTileTag}, {ldsTag});
        }

        std::vector<DeferredConnection>
            storeMacroTileForLDS(KernelGraph&                       graph,
                                 int                                userTag,
                                 int                                macTileTag,
                                 std::vector<int>&                  sdims,
                                 std::array<unsigned int, 3> const& workgroupSizes)
        {
            auto macTile = graph.coordinates.getNode<MacroTile>(macTileTag);
            auto user    = graph.coordinates.getNode<User>(userTag);

            rocRoller::Log::getLogger()->debug(
                "KernelGraph::Utils::storeMacroTileForLDS(): User({}), MacroTile({})",
                userTag,
                macTileTag);
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::Utils::storeMacroTileForLDS(): MacroTile size: {}x{}",
                macTile.sizes[0],
                macTile.sizes[1]);

            AssertRecoverable(macTile.rank >= 0 && sdims.size() == (size_t)macTile.rank,
                              "Tensor size mismatch.");
            AssertRecoverable(macTile.rank == 2, "Rank /= 2 not implemented yet.");

            auto nMacX = graph.coordinates.addElement(macTile.tileNumber(0));
            auto nMacY = graph.coordinates.addElement(macTile.tileNumber(1));
            auto iMacX = graph.coordinates.addElement(macTile.tileIndex(0));
            auto iMacY = graph.coordinates.addElement(macTile.tileIndex(1));

            std::vector<DeferredConnection> connections;
            connections.push_back(DC<MacroTileNumber>(nMacX, 0));
            connections.push_back(DC<MacroTileNumber>(nMacY, 1));

            auto workgroupX = graph.coordinates.addElement(Workgroup(0));
            auto workgroupY = graph.coordinates.addElement(Workgroup(1));

            connections.push_back(DC<Workgroup>(workgroupX, 0));
            connections.push_back(DC<Workgroup>(workgroupY, 1));

            graph.coordinates.addElement(Flatten(), {nMacX, iMacX}, {sdims[0]});
            graph.coordinates.addElement(Flatten(), {nMacY, iMacY}, {sdims[1]});

            graph.coordinates.addElement(PassThrough(), {workgroupX}, {nMacX});
            graph.coordinates.addElement(PassThrough(), {workgroupY}, {nMacY});

            auto workitemX
                = graph.coordinates.addElement(Workitem(0, literal(workgroupSizes.at(0))));

            auto thrTile = ThreadTile(macTile);

            auto nThrX
                = graph.coordinates.addElement(ThreadTileNumber(0, literal(thrTile.sizes.at(0))));
            auto nThrY
                = graph.coordinates.addElement(ThreadTileNumber(1, literal(thrTile.wsizes.at(1))));
            auto iThrX
                = graph.coordinates.addElement(ThreadTileIndex(0, literal(thrTile.wsizes.at(0))));
            auto iThrY
                = graph.coordinates.addElement(ThreadTileIndex(1, literal(thrTile.sizes.at(1))));

            graph.coordinates.addElement(Flatten(), {nThrX, iThrX}, {iMacX});
            graph.coordinates.addElement(Flatten(), {nThrY, iThrY}, {iMacY});

            auto elementNumberX
                = graph.coordinates.addElement(ElementNumber(0, literal(thrTile.sizes.at(0))));
            auto elementNumberY
                = graph.coordinates.addElement(ElementNumber(1, literal(thrTile.sizes.at(1))));

            graph.coordinates.addElement(PassThrough(), {elementNumberX}, {nThrX});
            graph.coordinates.addElement(PassThrough(), {elementNumberY}, {iThrY});

            connections.push_back(DC<ElementNumber>(elementNumberX, 0));
            connections.push_back(DC<ElementNumber>(elementNumberY, 1));

            graph.coordinates.addElement(Tile(), {workitemX}, {nThrY, iThrX});

            return connections;
        }

        void storeMacroTile(KernelGraph&                       graph,
                            int                                storeTag,
                            int                                userTag,
                            int                                macTileTag,
                            std::vector<int>&                  sdims,
                            std::array<unsigned int, 3> const& workgroupSizes,
                            int                                wavefrontSize,
                            std::vector<unsigned int> const&   wavetilesPerWorkgroup)
        {
            auto macTile = graph.coordinates.getNode<MacroTile>(macTileTag);
            auto user    = graph.coordinates.getNode<User>(userTag);

            rocRoller::Log::getLogger()->debug(
                "KernelGraph::Utils::storeMacroTile(): User({}), MacroTile({})",
                userTag,
                macTileTag);
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::Utils::storeMacroTile(): MacroTile size: {}x{}",
                macTile.sizes[0],
                macTile.sizes[1]);

            AssertRecoverable(macTile.rank >= 0 && sdims.size() == (size_t)macTile.rank,
                              "Tensor size mismatch.");
            AssertRecoverable(macTile.rank == 2, "Rank /= 2 not implemented yet.");

            auto nMacX = graph.coordinates.addElement(macTile.tileNumber(0));
            auto nMacY = graph.coordinates.addElement(macTile.tileNumber(1));
            auto iMacX = graph.coordinates.addElement(macTile.tileIndex(0));
            auto iMacY = graph.coordinates.addElement(macTile.tileIndex(1));

            graph.mapper.connect<MacroTileNumber>(storeTag, nMacX, 0);
            graph.mapper.connect<MacroTileNumber>(storeTag, nMacY, 1);

            auto workgroupX = graph.coordinates.addElement(Workgroup(0));
            auto workgroupY = graph.coordinates.addElement(Workgroup(1));

            graph.mapper.connect<Workgroup>(storeTag, workgroupX, 0);
            graph.mapper.connect<Workgroup>(storeTag, workgroupY, 1);

            graph.coordinates.addElement(Flatten(), {nMacX, iMacX}, {sdims[0]});
            graph.coordinates.addElement(Flatten(), {nMacY, iMacY}, {sdims[1]});

            graph.coordinates.addElement(PassThrough(), {workgroupX}, {nMacX});
            graph.coordinates.addElement(PassThrough(), {workgroupY}, {nMacY});

            switch(macTile.memoryType)
            {
            case MemoryType::VGPR:
            case MemoryType::LDS:
            {
                auto thrTile = ThreadTile(macTile);

                auto nThrX = graph.coordinates.addElement(
                    ThreadTileNumber(0, literal(thrTile.wsizes.at(0))));
                auto nThrY = graph.coordinates.addElement(
                    ThreadTileNumber(1, literal(thrTile.wsizes.at(1))));
                auto iThrX = graph.coordinates.addElement(
                    ThreadTileIndex(0, literal(thrTile.sizes.at(0))));
                auto iThrY = graph.coordinates.addElement(
                    ThreadTileIndex(1, literal(thrTile.sizes.at(1))));

                auto workitemX
                    = graph.coordinates.addElement(Workitem(0, literal(thrTile.wsizes.at(0))));
                auto workitemY
                    = graph.coordinates.addElement(Workitem(1, literal(thrTile.wsizes.at(1))));

                auto elementNumberX
                    = graph.coordinates.addElement(ElementNumber(0, literal(thrTile.sizes.at(0))));
                auto elementNumberY
                    = graph.coordinates.addElement(ElementNumber(1, literal(thrTile.sizes.at(1))));

                graph.mapper.connect<ElementNumber>(storeTag, elementNumberX, 0);
                graph.mapper.connect<ElementNumber>(storeTag, elementNumberY, 1);

                graph.coordinates.addElement(Flatten(), {nThrX, iThrX}, {iMacX});
                graph.coordinates.addElement(Flatten(), {nThrY, iThrY}, {iMacY});

                graph.coordinates.addElement(PassThrough(), {workitemX}, {nThrX});
                graph.coordinates.addElement(PassThrough(), {workitemY}, {nThrY});

                graph.coordinates.addElement(PassThrough(), {elementNumberX}, {iThrX});
                graph.coordinates.addElement(PassThrough(), {elementNumberY}, {iThrY});
            }
            break;

            case MemoryType::WAVE:
            case MemoryType::WAVE_LDS:
            {
                auto workitem
                    = graph.coordinates.addElement(Workitem(0, literal(workgroupSizes.at(0))));

                storeWaveMacroTile(graph,
                                   macTile,
                                   storeTag,
                                   iMacX,
                                   iMacY,
                                   workitem,
                                   userTag,
                                   wavefrontSize,
                                   wavetilesPerWorkgroup);
            }
            break;

            default:
                Throw<FatalError>("Store : MacroTile memory type not supported yet.");
            }
        }

        void addConnectionsMultiply(KernelGraph& graph, int waveMult, int loadATag, int loadBTag)
        {
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::Utils::addConnectionsMultiply(): Multiply({})", waveMult);

            auto loadA = graph.control.getElement(loadATag);
            auto loadB = graph.control.getElement(loadBTag);
            AssertFatal(isOperation<LoadTiled>(loadA) && isOperation<LoadTiled>(loadB),
                        "Both operands should be LoadTiled");

            // LoadTiled A
            auto userATag = graph.mapper.get<User>(loadATag);
            AssertFatal(userATag > 0, "User dimension not found");
            graph.mapper.connect<User>(waveMult, userATag, 0);

            // LoadTiled B
            auto userBTag = graph.mapper.get<User>(loadBTag);
            AssertFatal(userBTag > 0, "User dimension not found");
            graph.mapper.connect<User>(waveMult, userBTag, 1);

            AssertFatal(userATag > 0 && userBTag > 0, "User dimensions not found");

            auto [waveATag, waveA] = graph.getDimension<WaveTile>(loadATag);
            auto [waveBTag, waveB] = graph.getDimension<WaveTile>(loadBTag);

            auto macroTileA = graph.mapper.get<MacroTile>(loadATag);
            auto macroTileB = graph.mapper.get<MacroTile>(loadBTag);

            graph.mapper.connect(
                waveMult, macroTileA, Connections::typeArgument<MacroTile>(NaryArgument::LHS));
            graph.mapper.connect(
                waveMult, macroTileB, Connections::typeArgument<MacroTile>(NaryArgument::RHS));
            graph.mapper.connect(
                waveMult, waveATag, Connections::typeArgument<WaveTile>(NaryArgument::LHS));
            graph.mapper.connect(
                waveMult, waveBTag, Connections::typeArgument<WaveTile>(NaryArgument::RHS));
        }

        std::pair<Expression::ExpressionPtr, Expression::ExpressionPtr>
            getForLoopIncrement(KernelGraph const& graph, int forLoop)
        {
            // Find the ForLoopIcrement calculation

            // Grab all for loop increments from current for loop.
            // ForLoops coming from Compute Index may have more than one loop increment.
            // The forLoopIncrement that satifies all of the following conditions will be the
            // Increment that actually updates the iterator.
            auto loopIncrements
                = graph.control.getOutputNodeIndices<ForLoopIncrement>(forLoop).to<std::vector>();
            for(auto const& increment : loopIncrements)
            {
                auto loopIncrementOp = graph.control.getNode<Assign>(increment);

                //Ensure that the forLoopIncrement has an add expression
                if(!(std::holds_alternative<Expression::Add>(*loopIncrementOp.expression)))
                    continue;
                auto addExpr = std::get<Expression::Add>(*loopIncrementOp.expression);

                auto connections = graph.mapper.getConnections(increment);
                //Iterator should have one connection, if it doesn't it's not connected to coordinate.
                if(connections.size() != 1)
                    continue;
                auto dim_tag = connections[0].coordinate;
                //Iterator should have a DataFlow expression as its LHS
                if(!(std::holds_alternative<Expression::DataFlowTag>(*addExpr.lhs)))
                    continue;
                //LHS should also be the loop iterator data flow tag.
                if(std::get<Expression::DataFlowTag>(*addExpr.lhs).tag != dim_tag)
                    continue;
                //If all else is true and the first connection of the forLoop is the dim_tag
                //Then we have the loopIncrement that we were searching for.
                if(graph.mapper.getConnections(forLoop)[0].coordinate != dim_tag)
                    continue;
                return {addExpr.lhs, addExpr.rhs};
            }
            // There should be a loopIncrement that satisfies the above conditions
            // if not then throw an error.
            throw FatalError("No forLoopIncrement for supplied forLoop.");
        }

        int replaceWith(KernelGraph& graph, int op, int newOp, bool includeBody)
        {
            auto location = graph.control.getLocation(op);
            for(auto const& input : location.incoming)
            {
                auto edge = graph.control.getElement(input);
                int  parent
                    = *graph.control.getNeighbours<Graph::Direction::Upstream>(input).begin();
                graph.control.deleteElement(input);
                if(graph.control.getInputNodeIndices<Body>(newOp).to<std::unordered_set>().count(
                       parent)
                   == 0)
                {
                    graph.control.addElement(edge, {parent}, {newOp});
                }
            }
            for(auto const& output : location.outgoing)
            {
                auto edge = graph.control.getElement(output);
                if(std::holds_alternative<ControlEdge>(edge))
                {
                    auto cedge = std::get<ControlEdge>(edge);
                    if(std::holds_alternative<Sequence>(cedge)
                       || (includeBody && std::holds_alternative<Body>(cedge)))
                    {
                        int child
                            = *graph.control.getNeighbours<Graph::Direction::Downstream>(output)
                                   .begin();
                        graph.control.deleteElement(output);
                        if(graph.control.getOutputNodeIndices<Body>(newOp)
                               .to<std::unordered_set>()
                               .count(child)
                           == 0)
                        {
                            graph.control.addElement(edge, {newOp}, {child});
                        }
                    }
                }
            }

            return newOp;
        }

        bool needsComputeIndex(Operation const& op)
        {
            if(std::holds_alternative<StoreTiled>(op) || std::holds_alternative<StoreLDSTile>(op)
               || std::holds_alternative<LoadTiled>(op) || std::holds_alternative<LoadLDSTile>(op))
                return true;
            return false;
        }

        std::vector<int> findComputeIndexCandidates(KernelGraph const& kgraph, int start)
        {
            std::vector<int> rv;

            return kgraph.control
                .findNodes(
                    start,
                    [&](int tag) -> bool {
                        auto elem = kgraph.control.getElement(tag);
                        if(!std::holds_alternative<Operation>(elem))
                            return false;
                        auto op = std::get<Operation>(elem);
                        return needsComputeIndex(op);
                    },
                    GD::Downstream)
                .to<std::vector>();
        }

        void purgeFor(KernelGraph& kgraph, int loop)
        {
            // Purge loop dimension and iterator
            for(auto const& c : kgraph.mapper.getConnections(loop))
            {
                int iterator = c.coordinate;
                // TODO THIS IS A FRAGILE WAY OF DETECTING "NO MORE REFERENCES"
                if(kgraph.mapper.getCoordinateConnections(iterator).size() <= 3)
                {
                    auto dataflow = *only(
                        kgraph.coordinates.getNeighbours<Graph::Direction::Downstream>(iterator));
                    auto forLoop = *only(
                        kgraph.coordinates.getNeighbours<Graph::Direction::Downstream>(dataflow));
                    kgraph.coordinates.deleteElement(iterator);
                    kgraph.coordinates.deleteElement(dataflow);
                    kgraph.coordinates.deleteElement(forLoop);
                }
                // XXX THIS LEAVES SOME DANGLING COORDS; IS THIS STILL TRUE?
            }

            // Purge loop
            purgeNodeAndChildren(kgraph, loop);
        }

        void purgeNodeAndChildren(KernelGraph& kgraph, int node)
        {
            for(auto const& reap : kgraph.control.depthFirstVisit(node).to<std::vector>())
            {
                kgraph.control.deleteElement(reap);
                kgraph.mapper.purge(reap);
            }
            kgraph.mapper.purge(node);
        }

        bool isHardwareCoordinate(int tag, KernelGraph const& kgraph)
        {
            return kgraph.coordinates.get<VGPR>(tag) || kgraph.coordinates.get<Workitem>(tag)
                   || kgraph.coordinates.get<Workgroup>(tag);
        }

        bool isLoopishCoordinate(int tag, KernelGraph const& kgraph)
        {
            return kgraph.coordinates.get<ForLoop>(tag) || kgraph.coordinates.get<Unroll>(tag);
        }

        bool isStorageCoordinate(int tag, KernelGraph const& kgraph)
        {
            return kgraph.coordinates.get<LDS>(tag) || kgraph.coordinates.get<User>(tag);
        }

        std::pair<int, Graph::Direction> getOperationTarget(int tag, KernelGraph const& kgraph)
        {
            auto elem = kgraph.control.getElement(tag);
            return std::visit(
                rocRoller::overloaded{
                    [&](StoreTiled const& op) -> std::pair<int, Graph::Direction> {
                        return {kgraph.mapper.get<User>(tag), GD::Upstream};
                    },
                    [&](LoadTiled const& op) -> std::pair<int, Graph::Direction> {
                        return {kgraph.mapper.get<User>(tag), GD::Downstream};
                    },
                    [&](StoreLDSTile const& op) -> std::pair<int, Graph::Direction> {
                        return {kgraph.mapper.get<LDS>(tag), GD::Upstream};
                    },
                    [&](LoadLDSTile const& op) -> std::pair<int, Graph::Direction> {
                        return {kgraph.mapper.get<LDS>(tag), GD::Downstream};
                    },
                    [&](Assign const& op) -> std::pair<int, Graph::Direction> {
                        return {kgraph.mapper.getConnections(tag)[0].coordinate, GD::Downstream};
                    },
                    [&](auto const& op) -> std::pair<int, Graph::Direction> {
                        Throw<FatalError>(
                            "Operation is not a load, store, or assign: ", tag, " ", toString(op));
                        return {0, GD::Downstream};
                    }},
                std::get<Operation>(elem));
        }

        std::pair<std::vector<int>, std::unordered_set<int>> findRequiredCoordinates(
            int target, Graph::Direction direction, KernelGraph const& kgraph)
        {
            namespace CT = rocRoller::KernelGraph::CoordinateGraph;

            // TODO: Design a better way of binding storage to coordinates
            auto maybeLDS = kgraph.coordinates.get<LDS>(target);
            if(maybeLDS)
            {
                // If target is LDS; it might be a duplicated LDS
                // node.  For the purposes of figuring out required
                // coordinates, use the parent LDS as the target
                // instead.
                auto maybeParentLDS
                    = only(kgraph.coordinates.getInputNodeIndices(target, CT::isEdge<PassThrough>));
                if(maybeParentLDS)
                    target = *maybeParentLDS;
            }

            auto dontWalkPastLoopOrStorageNodes = [&](int tag) -> bool {
                auto element = kgraph.coordinates.getElement(tag);
                auto edge    = std::get<CT::Edge>(element);

                bool isCT   = std::holds_alternative<CT::CoordinateTransformEdge>(edge);
                bool follow = true;
                for(auto neighbour : kgraph.coordinates.getNeighbours(tag, opposite(direction)))
                {
                    if(neighbour == target)
                        continue;
                    if(isLoopishCoordinate(neighbour, kgraph))
                        follow = false;
                    if(isStorageCoordinate(neighbour, kgraph))
                        follow = false;
                }
                return isCT && follow;
            };

            // From the target coordinate, walk the graph but stop at loop
            // or storage nodes.  This will result in a list of nodes that
            // are used in the coordinate transform to compute indexes for
            // the target coordinate.
            auto candidates
                = kgraph.coordinates
                      .depthFirstVisit(target, dontWalkPastLoopOrStorageNodes, direction)
                      .to<std::vector>();

            // Internal nodes in the coordinate transform are computed as
            // part of the transform, so just keep leaf nodes and/or
            // hardware/loop coordinates.
            std::vector<int> required;
            std::copy_if(
                candidates.cbegin(), candidates.cend(), std::back_inserter(required), [&](int tag) {
                    bool isLeaf = kgraph.coordinates.getNeighbours(tag, direction)
                                      .to<std::vector>()
                                      .empty();
                    bool isLeafy
                        = isHardwareCoordinate(tag, kgraph) || isLoopishCoordinate(tag, kgraph);
                    return isLeaf || isLeafy;
                });

            std::unordered_set<int> path;
            if(direction == Graph::Direction::Downstream)
            {
                path = kgraph.coordinates
                           .path<Graph::Direction::Upstream>(required, std::vector<int>{target})
                           .to<std::unordered_set>();
            }
            else
            {
                path = kgraph.coordinates
                           .path<Graph::Direction::Downstream>(required, std::vector<int>{target})
                           .to<std::unordered_set>();
            }

            return {required, path};
        }

        std::optional<std::pair<int, Graph::Direction>>
            findStorageNeighbour(int tag, KernelGraph const& graph)
        {
            using rt = std::pair<int, Graph::Direction>;
            auto neighbourTag
                = only(graph.coordinates.getNeighbours(tag, Graph::Direction::Upstream));
            if(neighbourTag && isStorageCoordinate(*neighbourTag, graph))
            {
                return rt{*neighbourTag, Graph::Direction::Downstream};
            }
            neighbourTag = only(graph.coordinates.getNeighbours(tag, Graph::Direction::Downstream));
            if(neighbourTag && isStorageCoordinate(*neighbourTag, graph))
            {
                return rt{*neighbourTag, Graph::Direction::Upstream};
            }
            return {};
        }

        int duplicateControlNode(KernelGraph& graph, int tag)
        {
            auto op = graph.control.addElement(graph.control.getElement(tag));
            for(auto const& c : graph.mapper.getConnections(tag))
            {
                graph.mapper.connect(op, c.coordinate, c.connection);
            }

            return op;
        }

        void
            updateThreadTileForLongDwords(int& t_m, int& t_n, int maxWidth, int numDwordsPerElement)
        {
            auto numDwordsPerWorkitem = t_m * numDwordsPerElement;

            std::vector<int> potentialFactors = {4, 3, 2, 1};

            auto start      = potentialFactors.begin();
            auto end        = potentialFactors.end();
            auto factorPred = [numDwordsPerWorkitem, maxWidth](int factor) {
                return factor <= maxWidth && numDwordsPerWorkitem % factor == 0;
            };
            auto it = std::find_if(start, end, factorPred);

            if(it != potentialFactors.end())
            {
                auto dwordFactor = *it / numDwordsPerElement;
                AssertFatal(dwordFactor >= 1, "dword factor can't be less than 1");

                t_m = t_m / dwordFactor;
                t_n = t_n * dwordFactor;
            }
        }

        int getTopSetCoordinate(KernelGraph& graph, int load)
        {
            int tag = load;

            while(true)
            {
                auto parent = only(graph.control.getInputNodeIndices<Body>(tag));
                if(!parent)
                    break;

                if(graph.control.get<SetCoordinate>(*parent))
                    tag = *parent;
                else
                    break;

                auto setCoord = graph.control.get<SetCoordinate>(tag);
                AssertFatal(graph.mapper.get<Unroll>(tag), "SetCoordinate needs Unroll dimension");
            }
            return tag;
        }

        std::set<int> getTopSetCoordinates(KernelGraph& graph, std::vector<int> loads)
        {
            std::set<int> retval;
            for(auto& load : loads)
            {
                retval.insert(getTopSetCoordinate(graph, load));
            }
            return retval;
        }

        int getSetCoordinateForDim(KernelGraph& graph, int dim, int load)
        {
            int tag = load;

            while(true)
            {
                auto parent = only(graph.control.getInputNodeIndices<Body>(tag));
                AssertFatal(parent, "Dimension was not found in the parents.");
                AssertFatal(graph.control.get<SetCoordinate>(*parent),
                            "Dimension was not found in the parents.");
                tag           = *parent;
                auto setCoord = graph.control.get<SetCoordinate>(tag);
                AssertFatal(graph.mapper.get<Unroll>(tag), "SetCoordinate needs Unroll dimension");
                if(graph.mapper.get<Unroll>(tag) == dim)
                {
                    return tag;
                }
            }
        }

        std::vector<int> getLoadsForUnroll(KernelGraph&     graph,
                                           int              unrollCoord,
                                           std::vector<int> loads,
                                           int              unroll)
        {
            std::vector<int> retval;
            for(auto& load : loads)
            {
                int  tag      = getSetCoordinateForDim(graph, unrollCoord, load);
                auto setCoord = graph.control.get<SetCoordinate>(tag);
                AssertFatal(evaluationTimes(
                                setCoord->value)[rocRoller::Expression::EvaluationTime::Translate],
                            "Unroll value should be a literal");
                if(unroll == getUnsignedInt(evaluate(setCoord->value)))
                {
                    retval.push_back(load);
                }
            }
            return retval;
        }
    }
}
