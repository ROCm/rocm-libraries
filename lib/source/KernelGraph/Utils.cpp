
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
            auto unit_stride  = Expression::literal(1u);
            auto rangeK       = graph.coordinates.addElement(Linear(size, unit_stride));
            auto dimK         = graph.coordinates.addElement(ForLoop(size, unit_stride));
            auto sizeDataType = Expression::resultVariableType(size);
            auto exprK        = std::make_shared<Expression::Expression>(
                DataFlowTag{rangeK, Register::Type::Scalar, sizeDataType});

            auto forK  = graph.control.addElement(ForLoopOp{exprK < size, loopName});
            auto initK = graph.control.addElement(
                Assign{Register::Type::Scalar, Expression::literal(0, sizeDataType)});
            auto incrementK
                = graph.control.addElement(Assign{Register::Type::Scalar, exprK + unit_stride});

            graph.coordinates.addElement(DataFlow(), {rangeK}, {dimK});
            graph.control.addElement(Initialize(), {forK}, {initK});
            graph.control.addElement(ForLoopIncrement(), {forK}, {incrementK});

            graph.mapper.connect<Dimension>(forK, rangeK);
            graph.mapper.connect(initK, rangeK, NaryArgument::DEST);
            graph.mapper.connect(incrementK, rangeK, NaryArgument::DEST);

            return {dimK, forK};
        }

        void updateLoadLDSMacroTile(KernelGraph&      graph,
                                    MacroTile const&  mac_tile,
                                    int               load_tag,
                                    std::vector<int>& sdims,
                                    int               K,
                                    int               lds)
        {
            // given that the loadMacroTile has already lowered the macrotile for LoadTiled
            // before it is transformed to LoadLDSTile

            if(mac_tile.layoutType == LayoutType::MATRIX_A)
            {
                // remove passthrough between A row block and x-workgroup
                // remove x-workgroup
                auto a_tilenum_x   = graph.mapper.get<MacroTileNumber>(load_tag, 0);
                auto a_workgroup_x = graph.mapper.get<Workgroup>(load_tag, 0);
                graph.coordinates.deleteElement(std::vector<int>{a_tilenum_x},
                                                std::vector<int>{a_workgroup_x},
                                                CT::isEdge<PassThrough>);
                graph.mapper.disconnect<Workgroup>(load_tag, a_workgroup_x, 0);
                graph.coordinates.deleteElement(a_workgroup_x);

                // remove passthrough between A column block and K
                auto a_tilenum_y = graph.mapper.get<MacroTileNumber>(load_tag, 1);
                graph.coordinates.deleteElement(
                    std::vector<int>{a_tilenum_y}, std::vector<int>{K}, CT::isEdge<PassThrough>);
            }
            else if(mac_tile.layoutType == LayoutType::MATRIX_B)
            {
                // remove passthrough between B column block and y-workgroup
                // remove y-workgroup
                auto b_tilenum_y   = graph.mapper.get<MacroTileNumber>(load_tag, 1);
                auto b_workgroup_y = graph.mapper.get<Workgroup>(load_tag, 1);
                graph.coordinates.deleteElement(std::vector<int>{b_tilenum_y},
                                                std::vector<int>{b_workgroup_y},
                                                CT::isEdge<PassThrough>);
                graph.mapper.disconnect<Workgroup>(load_tag, b_workgroup_y, 1);
                graph.coordinates.deleteElement(b_workgroup_y);

                // remove passthrough between B row block and K
                auto b_tilenum_x = graph.mapper.get<MacroTileNumber>(load_tag, 0);
                graph.coordinates.deleteElement(
                    std::vector<int>{b_tilenum_x}, std::vector<int>{K}, CT::isEdge<PassThrough>);
            }
            else
            {
                auto tilenum_x   = graph.mapper.get<MacroTileNumber>(load_tag, 0);
                auto workgroup_x = graph.mapper.get<Workgroup>(load_tag, 0);
                graph.mapper.disconnect<Workgroup>(load_tag, workgroup_x, 0);
                graph.coordinates.deleteElement(std::vector<int>{tilenum_x},
                                                std::vector<int>{workgroup_x},
                                                CT::isEdge<PassThrough>);
                graph.coordinates.deleteElement(workgroup_x);

                auto tilenum_y   = graph.mapper.get<MacroTileNumber>(load_tag, 1);
                auto workgroup_y = graph.mapper.get<Workgroup>(load_tag, 1);
                graph.mapper.disconnect<Workgroup>(load_tag, workgroup_y, 1);
                graph.coordinates.deleteElement(std::vector<int>{tilenum_y},
                                                std::vector<int>{workgroup_y},
                                                CT::isEdge<PassThrough>);
                graph.coordinates.deleteElement(workgroup_y);
            }

            std::vector<int> i_mac;
            for(size_t i = 0; i < sdims.size(); ++i)
            {
                auto mac = graph.coordinates.getOutputNodeIndices(sdims[i], CT::isEdge<Tile>)
                               .to<std::vector>();
                i_mac.push_back(mac[1]);
                graph.coordinates.deleteElement(std::vector<int>{sdims[i]}, mac, CT::isEdge<Tile>);
                graph.mapper.disconnect<MacroTileNumber>(load_tag, mac[0], i);
                graph.coordinates.deleteElement(mac[0]);
            }

            graph.coordinates.addElement(Tile(), {lds}, {i_mac[0], i_mac[1]});
        }

        void loadWaveMacroTile(KernelGraph&                     graph,
                               MacroTile const&                 mac_tile,
                               int                              load_tag,
                               int                              i_mac_x,
                               int                              i_mac_y,
                               int                              user_tag,
                               int                              wavefrontSize,
                               std::vector<unsigned int> const& wavetilesPerWorkgroup)
        {
            AssertFatal(mac_tile.subTileSizes.size() == 4, "Invalid tile specification.");

            auto m = mac_tile.subTileSizes[0];
            auto n = mac_tile.subTileSizes[1];
            auto k = mac_tile.subTileSizes[2];

            std::vector<int> tileSize;
            if(mac_tile.layoutType == LayoutType::MATRIX_A)
                tileSize = {m, k};
            if(mac_tile.layoutType == LayoutType::MATRIX_B)
                tileSize = {k, n};
            if(mac_tile.layoutType == LayoutType::MATRIX_ACCUMULATOR)
                tileSize = {m, n};

            auto workitem      = graph.coordinates.addElement(Workitem(0));
            auto wave_tile     = WaveTile(tileSize, mac_tile.layoutType);
            auto wave_tile_tag = graph.coordinates.addElement(wave_tile);
            graph.mapper.connect<WaveTile>(load_tag, wave_tile_tag);

            auto n_wave_x = graph.coordinates.addElement(wave_tile.tileNumber(0));
            auto n_wave_y = graph.coordinates.addElement(wave_tile.tileNumber(1));
            auto i_wave_x = graph.coordinates.addElement(wave_tile.tileIndex(0));
            auto i_wave_y = graph.coordinates.addElement(wave_tile.tileIndex(1));

            graph.coordinates.addElement(Tile(), {i_mac_x}, {n_wave_x, i_wave_x});
            graph.coordinates.addElement(Tile(), {i_mac_y}, {n_wave_y, i_wave_y});

            graph.mapper.connect<WaveTileNumber>(load_tag, n_wave_x, 0);
            graph.mapper.connect<WaveTileNumber>(load_tag, n_wave_y, 1);

            auto wave_x = graph.coordinates.addElement(Wavefront(0));
            auto wave_y = graph.coordinates.addElement(Wavefront(1));
            auto wave   = graph.coordinates.addElement(Wavefront(-1));

            uint num_elements = product(tileSize);
            uint wfs          = static_cast<uint>(wavefrontSize);
            uint num_vgpr     = num_elements / wfs;

            auto wavefront_size = Expression::literal(wfs);

            auto lane = graph.coordinates.addElement(Lane(wavefront_size, nullptr));
            auto vgpr = graph.coordinates.addElement(VGPR(literal(num_vgpr), nullptr));

            graph.coordinates.addElement(Flatten(), {wave_x, wave_y}, {wave});
            graph.coordinates.addElement(Flatten(), {wave, lane}, {workitem});

            graph.mapper.connect<VGPR>(load_tag, vgpr);

            int jammed_wavetile_x = -1;
            if(wavetilesPerWorkgroup[0] >= 1)
            {
                jammed_wavetile_x = graph.coordinates.addElement(
                    JammedWaveTileNumber(0, literal(wavetilesPerWorkgroup[0]), literal(1)));
                graph.mapper.connect<JammedWaveTileNumber>(load_tag, jammed_wavetile_x, 0);
            }
            int jammed_wavetile_y = -1;
            if(wavetilesPerWorkgroup[1] >= 1)
            {
                jammed_wavetile_y = graph.coordinates.addElement(
                    JammedWaveTileNumber(1, literal(wavetilesPerWorkgroup[1]), literal(1)));
                graph.mapper.connect<JammedWaveTileNumber>(load_tag, jammed_wavetile_y, 1);
            }

            switch(wave_tile.layout)
            {
            case LayoutType::MATRIX_A:
            {
                auto block_number = graph.coordinates.addElement(
                    Adhoc("BlockNumber", literal(static_cast<uint>(wfs / m)), nullptr));
                auto block_index = graph.coordinates.addElement(
                    Adhoc("BlockIndex", literal(static_cast<uint>(m)), nullptr));

                graph.coordinates.addElement(Flatten(), {block_number, block_index}, {lane});

                graph.coordinates.addElement(Tile(), {i_wave_y}, {block_number, vgpr});
                graph.coordinates.addElement(PassThrough(), {i_wave_x}, {block_index});

                if(wavetilesPerWorkgroup[0] > 1)
                    graph.coordinates.addElement(Tile(), {n_wave_x}, {wave_x, jammed_wavetile_x});
                else
                    graph.coordinates.addElement(PassThrough(), {n_wave_x}, {wave_x});
            }
            break;

            case LayoutType::MATRIX_B:
            {
                auto block_number = graph.coordinates.addElement(
                    Adhoc("BlockNumber", literal(static_cast<uint>(wfs / m)), nullptr));
                auto block_index = graph.coordinates.addElement(
                    Adhoc("BlockIndex", literal(static_cast<uint>(m)), nullptr));

                graph.coordinates.addElement(Flatten(), {block_number, block_index}, {lane});

                graph.coordinates.addElement(Tile(), {i_wave_x}, {block_number, vgpr});
                graph.coordinates.addElement(PassThrough(), {i_wave_y}, {block_index});

                if(wavetilesPerWorkgroup[1] > 1)
                    graph.coordinates.addElement(Tile(), {n_wave_y}, {wave_y, jammed_wavetile_y});
                else
                    graph.coordinates.addElement(PassThrough(), {n_wave_y}, {wave_y});
            }
            break;

            case LayoutType::MATRIX_ACCUMULATOR:
            {
                // MFMA accumulator tile size
                uint mts            = 4u;
                auto mfma_tile_size = literal(mts);
                auto unit_stride    = literal(1u);

                auto n_row_blocks = literal(wave_tile.sizes[0] / mts);
                auto n_col_blocks = literal(wave_tile.sizes[1] / mts);

                auto n_vblk = graph.coordinates.addElement(
                    VGPRBlockNumber(literal(num_vgpr / mts), unit_stride));
                auto i_vblk
                    = graph.coordinates.addElement(VGPRBlockIndex(mfma_tile_size, unit_stride));
                auto n_lblk = graph.coordinates.addElement(
                    Adhoc("LANEBlockNumber", literal(wfs / mts), unit_stride));
                auto i_lblk = graph.coordinates.addElement(
                    Adhoc("LANEBlockIndex", mfma_tile_size, unit_stride));
                auto block = graph.coordinates.addElement(
                    Adhoc("LinearBlock", literal(num_elements / 16u), unit_stride));
                auto row_block
                    = graph.coordinates.addElement(Adhoc("RowBlock", n_row_blocks, unit_stride));
                auto col_block
                    = graph.coordinates.addElement(Adhoc("ColBlock", n_col_blocks, unit_stride));

                graph.mapper.connect<VGPRBlockNumber>(load_tag, n_vblk);
                graph.mapper.connect<VGPRBlockIndex>(load_tag, i_vblk);

                graph.coordinates.addElement(Tile(), {i_wave_x}, {row_block, i_vblk});
                graph.coordinates.addElement(Tile(), {i_wave_y}, {col_block, i_lblk});

                graph.coordinates.addElement(Flatten(), {row_block, col_block}, {block});
                graph.coordinates.addElement(Tile(), {block}, {n_vblk, n_lblk});

                graph.coordinates.addElement(Flatten(), {n_vblk, i_vblk}, {vgpr});
                graph.coordinates.addElement(Flatten(), {n_lblk, i_lblk}, {lane});

                if(wavetilesPerWorkgroup[0] > 1)
                    graph.coordinates.addElement(Tile(), {n_wave_x}, {wave_x, jammed_wavetile_x});
                else
                    graph.coordinates.addElement(PassThrough(), {n_wave_x}, {wave_x});

                if(wavetilesPerWorkgroup[1] > 1)
                    graph.coordinates.addElement(Tile(), {n_wave_y}, {wave_y, jammed_wavetile_y});
                else
                    graph.coordinates.addElement(PassThrough(), {n_wave_y}, {wave_y});
            }
            break;

            default:
                Throw<FatalError>("Not implemented yet.");
            }
        }

        void loadMacroTileFromLDS(KernelGraph&                       graph,
                                  int                                load_tag,
                                  int                                lds_tag,
                                  int                                mac_tile_tag,
                                  std::array<unsigned int, 3> const& workgroupSizes)
        {
            auto mac_tile = graph.coordinates.getNode<MacroTile>(mac_tile_tag);

            rocRoller::Log::getLogger()->debug(
                "KernelGraph::Utils::loadMacroTileFromLDS(): LDS({}), MacroTile({})",
                lds_tag,
                mac_tile_tag);
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::Utils::loadMacroTileFromLDS(): MacroTile size: {}x{}",
                mac_tile.sizes[0],
                mac_tile.sizes[1]);

            auto workitem_x
                = graph.coordinates.addElement(Workitem(0, literal(workgroupSizes.at(0))));

            auto thr_tile = ThreadTile(mac_tile);

            auto i_mac_x = graph.coordinates.addElement(mac_tile.tileIndex(0));
            auto i_mac_y = graph.coordinates.addElement(mac_tile.tileIndex(1));
            auto n_thr_x
                = graph.coordinates.addElement(ThreadTileNumber(0, literal(thr_tile.sizes.at(0))));
            auto n_thr_y
                = graph.coordinates.addElement(ThreadTileNumber(1, literal(thr_tile.wsizes.at(1))));
            auto i_thr_x
                = graph.coordinates.addElement(ThreadTileIndex(0, literal(thr_tile.wsizes.at(0))));
            auto i_thr_y
                = graph.coordinates.addElement(ThreadTileIndex(1, literal(thr_tile.sizes.at(1))));

            graph.coordinates.addElement(Tile(), {lds_tag}, {i_mac_x, i_mac_y});

            graph.coordinates.addElement(Tile(), {i_mac_x}, {n_thr_x, i_thr_x});
            graph.coordinates.addElement(Tile(), {i_mac_y}, {n_thr_y, i_thr_y});

            graph.coordinates.addElement(Flatten(), {n_thr_y, i_thr_x}, {workitem_x});

            auto element_number_x
                = graph.coordinates.addElement(ElementNumber(0, literal(thr_tile.sizes.at(0))));
            auto element_number_y
                = graph.coordinates.addElement(ElementNumber(1, literal(thr_tile.sizes.at(1))));

            graph.coordinates.addElement(PassThrough(), {n_thr_x}, {element_number_x});
            graph.coordinates.addElement(PassThrough(), {i_thr_y}, {element_number_y});

            graph.mapper.connect<ElementNumber>(load_tag, element_number_x, 0);
            graph.mapper.connect<ElementNumber>(load_tag, element_number_y, 1);

            // LDS --DataFlow--> macrotile
            graph.coordinates.addElement(DataFlow(), {lds_tag}, {mac_tile_tag});
        }

        void loadMacroTileForLDS(KernelGraph&                       graph,
                                 int                                load_tag,
                                 int                                user_tag,
                                 int                                mac_tile_tag,
                                 std::vector<int>&                  sdim,
                                 int                                K,
                                 std::array<unsigned int, 3> const& workgroupSizes,
                                 std::shared_ptr<Context>           context)
        {
            auto mac_tile = graph.coordinates.getNode<MacroTile>(mac_tile_tag);
            auto user     = graph.coordinates.getNode<User>(user_tag);

            rocRoller::Log::getLogger()->debug(
                "KernelGraph::Utils::loadMacroTileForLDS(): User({}), MacroTile({})",
                user_tag,
                mac_tile_tag);
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::Utils::loadMacroTileForLDS(): MacroTile size: {}x{}",
                mac_tile.sizes[0],
                mac_tile.sizes[1]);

            AssertFatal(mac_tile.rank == 2, "Rank /= 2 not implemented yet.");

            auto sdim_x = sdim[0];
            auto sdim_y = sdim[1];
            graph.mapper.connect<SubDimension>(load_tag, sdim_x, 0);
            graph.mapper.connect<SubDimension>(load_tag, sdim_y, 1);

            auto n_mac_x = graph.coordinates.addElement(mac_tile.tileNumber(0));
            auto n_mac_y = graph.coordinates.addElement(mac_tile.tileNumber(1));
            auto i_mac_x = graph.coordinates.addElement(mac_tile.tileIndex(0));
            auto i_mac_y = graph.coordinates.addElement(mac_tile.tileIndex(1));

            graph.mapper.connect<MacroTileNumber>(load_tag, n_mac_x, 0);
            graph.mapper.connect<MacroTileNumber>(load_tag, n_mac_y, 1);

            graph.coordinates.addElement(Tile(), {sdim_x}, {n_mac_x, i_mac_x});
            graph.coordinates.addElement(Tile(), {sdim_y}, {n_mac_y, i_mac_y});

            auto thr_tile = ThreadTile(mac_tile);

            auto element_number_x
                = graph.coordinates.addElement(ElementNumber(0, literal(thr_tile.sizes.at(0))));
            auto element_number_y
                = graph.coordinates.addElement(ElementNumber(1, literal(thr_tile.sizes.at(1))));

            graph.mapper.connect<ElementNumber>(load_tag, element_number_x, 0);
            graph.mapper.connect<ElementNumber>(load_tag, element_number_y, 1);

            int n_thr_x, n_thr_y, i_thr_x, i_thr_y;
            if((mac_tile.layoutType == LayoutType::MATRIX_A
                && context->kernelOptions().transposeMemoryAccessA)
               || (mac_tile.layoutType == LayoutType::MATRIX_B
                   && context->kernelOptions().transposeMemoryAccessB)
               || context->kernelOptions().transposeMemoryAccessOther)
            {
                n_thr_x = graph.coordinates.addElement(
                    ThreadTileNumber(0, literal(thr_tile.sizes.at(0))));
                n_thr_y = graph.coordinates.addElement(
                    ThreadTileNumber(1, literal(thr_tile.wsizes.at(1))));
                i_thr_x = graph.coordinates.addElement(
                    ThreadTileIndex(0, literal(thr_tile.wsizes.at(0))));
                i_thr_y = graph.coordinates.addElement(
                    ThreadTileIndex(1, literal(thr_tile.sizes.at(1))));

                graph.coordinates.addElement(PassThrough(), {n_thr_x}, {element_number_x});
                graph.coordinates.addElement(PassThrough(), {i_thr_y}, {element_number_y});

                auto workitem_x
                    = graph.coordinates.addElement(Workitem(0, literal(workgroupSizes.at(0))));
                if(mac_tile.layoutType == LayoutType::MATRIX_A
                   || mac_tile.layoutType == LayoutType::MATRIX_B)
                {
                    graph.coordinates.addElement(Flatten(), {n_thr_y, i_thr_x}, {workitem_x});
                }
                else
                {
                    graph.coordinates.addElement(PassThrough(), {i_thr_x}, {workitem_x});

                    auto workitem_y
                        = graph.coordinates.addElement(Workitem(1, literal(workgroupSizes.at(1))));
                    graph.coordinates.addElement(PassThrough(), {n_thr_y}, {workitem_y});
                }
            }
            else
            {
                n_thr_x = graph.coordinates.addElement(
                    ThreadTileNumber(0, literal(thr_tile.wsizes.at(0))));
                n_thr_y = graph.coordinates.addElement(
                    ThreadTileNumber(1, literal(thr_tile.wsizes.at(1))));
                i_thr_x = graph.coordinates.addElement(
                    ThreadTileIndex(0, literal(thr_tile.sizes.at(0))));
                i_thr_y = graph.coordinates.addElement(
                    ThreadTileIndex(1, literal(thr_tile.sizes.at(1))));

                graph.coordinates.addElement(PassThrough(), {i_thr_x}, {element_number_x});
                graph.coordinates.addElement(PassThrough(), {i_thr_y}, {element_number_y});

                auto workitem_x
                    = graph.coordinates.addElement(Workitem(0, literal(workgroupSizes.at(0))));
                if(mac_tile.layoutType == LayoutType::MATRIX_A
                   || mac_tile.layoutType == LayoutType::MATRIX_B)
                {
                    graph.coordinates.addElement(Flatten(), {n_thr_y, n_thr_x}, {workitem_x});
                }
                else
                {
                    graph.coordinates.addElement(PassThrough(), {n_thr_x}, {workitem_x});

                    auto workitem_y
                        = graph.coordinates.addElement(Workitem(1, literal(workgroupSizes.at(1))));
                    graph.coordinates.addElement(PassThrough(), {n_thr_y}, {workitem_y});
                }
            }

            graph.coordinates.addElement(Tile(), {i_mac_x}, {n_thr_x, i_thr_x});
            graph.coordinates.addElement(Tile(), {i_mac_y}, {n_thr_y, i_thr_y});

            if(mac_tile.layoutType == LayoutType::MATRIX_A)
            {
                auto workgroup_x = graph.coordinates.addElement(Workgroup(0));
                graph.mapper.connect<Workgroup>(load_tag, workgroup_x, 0);
                graph.coordinates.addElement(PassThrough(), {n_mac_x}, {workgroup_x});
                // A row block is x-workgroup, column block is for loop index
                graph.coordinates.addElement(PassThrough(), {n_mac_y}, {K});
            }
            else if(mac_tile.layoutType == LayoutType::MATRIX_B)
            {
                auto workgroup_y = graph.coordinates.addElement(Workgroup(1));
                graph.mapper.connect<Workgroup>(load_tag, workgroup_y, 1);
                // B row block is for loop index, column block is y-workgroup
                graph.coordinates.addElement(PassThrough(), {n_mac_x}, {K});
                graph.coordinates.addElement(PassThrough(), {n_mac_y}, {workgroup_y});
            }
            else
            {
                auto workgroup_x = graph.coordinates.addElement(Workgroup(0));
                graph.coordinates.addElement(PassThrough(), {n_mac_x}, {workgroup_x});
                graph.mapper.connect<Workgroup>(load_tag, workgroup_x, 0);

                auto workgroup_y = graph.coordinates.addElement(Workgroup(1));
                graph.coordinates.addElement(PassThrough(), {n_mac_y}, {workgroup_y});
                graph.mapper.connect<Workgroup>(load_tag, workgroup_y, 1);
            }
        }

        void loadMacroTile(KernelGraph&                       graph,
                           int                                load_tag,
                           int                                user_tag,
                           int                                mac_tile_tag,
                           std::vector<int>&                  sdim,
                           std::array<unsigned int, 3> const& workgroupSizes,
                           int                                wavefrontSize,
                           std::vector<unsigned int> const&   wavetilesPerWorkgroup)

        {
            auto mac_tile = graph.coordinates.getNode<MacroTile>(mac_tile_tag);
            auto user     = graph.coordinates.getNode<User>(user_tag);

            rocRoller::Log::getLogger()->debug(
                "KernelGraph::Utils::loadMacroTile(): User({}), MacroTile({})",
                user_tag,
                mac_tile_tag);
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::Utils::loadMacroTile(): MacroTile size: {}x{}",
                mac_tile.sizes[0],
                mac_tile.sizes[1]);

            AssertFatal(mac_tile.rank == 2, "Rank /= 2 not implemented yet.");

            auto sdim_x = sdim[0];
            auto sdim_y = sdim[1];
            graph.mapper.connect<SubDimension>(load_tag, sdim_x, 0);
            graph.mapper.connect<SubDimension>(load_tag, sdim_y, 1);

            auto n_mac_x = graph.coordinates.addElement(mac_tile.tileNumber(0));
            auto n_mac_y = graph.coordinates.addElement(mac_tile.tileNumber(1));
            auto i_mac_x = graph.coordinates.addElement(mac_tile.tileIndex(0));
            auto i_mac_y = graph.coordinates.addElement(mac_tile.tileIndex(1));

            graph.mapper.connect<MacroTileNumber>(load_tag, n_mac_x, 0);
            graph.mapper.connect<MacroTileNumber>(load_tag, n_mac_y, 1);

            auto workgroup_x = graph.coordinates.addElement(Workgroup(0));
            auto workgroup_y = graph.coordinates.addElement(Workgroup(1));

            graph.mapper.connect<Workgroup>(load_tag, workgroup_x, 0);
            graph.mapper.connect<Workgroup>(load_tag, workgroup_y, 1);

            graph.coordinates.addElement(Tile(), {sdim_x}, {n_mac_x, i_mac_x});
            graph.coordinates.addElement(Tile(), {sdim_y}, {n_mac_y, i_mac_y});

            graph.coordinates.addElement(PassThrough(), {n_mac_x}, {workgroup_x});
            graph.coordinates.addElement(PassThrough(), {n_mac_y}, {workgroup_y});

            switch(mac_tile.memoryType)
            {
            case MemoryType::VGPR:
            case MemoryType::LDS:
            {
                auto thr_tile = ThreadTile(mac_tile);

                auto n_thr_x = graph.coordinates.addElement(
                    ThreadTileNumber(0, literal(thr_tile.wsizes.at(0))));
                auto n_thr_y = graph.coordinates.addElement(
                    ThreadTileNumber(1, literal(thr_tile.wsizes.at(1))));
                auto i_thr_x = graph.coordinates.addElement(
                    ThreadTileIndex(0, literal(thr_tile.sizes.at(0))));
                auto i_thr_y = graph.coordinates.addElement(
                    ThreadTileIndex(1, literal(thr_tile.sizes.at(1))));

                auto workitem_x
                    = graph.coordinates.addElement(Workitem(0, literal(thr_tile.wsizes.at(0))));
                auto workitem_y
                    = graph.coordinates.addElement(Workitem(1, literal(thr_tile.wsizes.at(1))));

                auto element_number_x
                    = graph.coordinates.addElement(ElementNumber(0, literal(thr_tile.sizes.at(0))));
                auto element_number_y
                    = graph.coordinates.addElement(ElementNumber(1, literal(thr_tile.sizes.at(1))));

                graph.mapper.connect<ElementNumber>(load_tag, element_number_x, 0);
                graph.mapper.connect<ElementNumber>(load_tag, element_number_y, 1);

                graph.coordinates.addElement(Tile(), {i_mac_x}, {n_thr_x, i_thr_x});
                graph.coordinates.addElement(Tile(), {i_mac_y}, {n_thr_y, i_thr_y});

                graph.coordinates.addElement(PassThrough(), {n_thr_x}, {workitem_x});
                graph.coordinates.addElement(PassThrough(), {n_thr_y}, {workitem_y});

                graph.coordinates.addElement(PassThrough(), {i_thr_x}, {element_number_x});
                graph.coordinates.addElement(PassThrough(), {i_thr_y}, {element_number_y});

                // User -> DataFlow() -> LDS gets added in addLDSOps
            }
            break;

            case MemoryType::WAVE:
            case MemoryType::WAVE_LDS:
                loadWaveMacroTile(graph,
                                  mac_tile,
                                  load_tag,
                                  i_mac_x,
                                  i_mac_y,
                                  user_tag,
                                  wavefrontSize,
                                  wavetilesPerWorkgroup);
                // User -> DataFlow() -> LDS gets added in addWaveLDSOps
                break;

            default:
                Throw<FatalError>("Load : MacroTile memory type not supported yet.");
            }
        }

        void updateStoreLDSMacroTile(KernelGraph&      graph,
                                     MacroTile const&  mac_tile,
                                     int               store_tag,
                                     std::vector<int>& sdims,
                                     int               lds)
        {
            // given that the storeMacroTile has already lowered the macrotile for StoreTiled
            // before it is transformed to StoreLDSTile

            // remove macrotile numbers and workgroups
            auto tilenum_x   = graph.mapper.get<MacroTileNumber>(store_tag, 0);
            auto workgroup_x = graph.mapper.get<Workgroup>(store_tag, 0);
            graph.mapper.disconnect<Workgroup>(store_tag, workgroup_x, 0);
            graph.coordinates.deleteElement(std::vector<int>{workgroup_x},
                                            std::vector<int>{tilenum_x},
                                            CT::isEdge<PassThrough>);
            graph.coordinates.deleteElement(workgroup_x);

            auto tilenum_y   = graph.mapper.get<MacroTileNumber>(store_tag, 1);
            auto workgroup_y = graph.mapper.get<Workgroup>(store_tag, 1);
            graph.mapper.disconnect<Workgroup>(store_tag, workgroup_y, 1);
            graph.coordinates.deleteElement(std::vector<int>{workgroup_y},
                                            std::vector<int>{tilenum_y},
                                            CT::isEdge<PassThrough>);
            graph.coordinates.deleteElement(workgroup_y);

            std::vector<int> i_mac;
            for(size_t i = 0; i < sdims.size(); ++i)
            {
                auto mac = graph.coordinates.getInputNodeIndices(sdims[i], CT::isEdge<Flatten>)
                               .to<std::vector>();
                i_mac.push_back(mac[1]);
                graph.coordinates.deleteElement(
                    mac, std::vector<int>{sdims[i]}, CT::isEdge<Flatten>);
                graph.mapper.disconnect<MacroTileNumber>(store_tag, mac[0], i);
                graph.coordinates.deleteElement(mac[0]);
            }

            graph.coordinates.addElement(Flatten(), {i_mac[0], i_mac[1]}, {lds});
        }

        void storeWaveMacroTile(KernelGraph&                     graph,
                                MacroTile const&                 mac_tile,
                                int                              store_tag,
                                int                              i_mac_x,
                                int                              i_mac_y,
                                int                              workitem,
                                int                              user_tag,
                                int                              wavefrontSize,
                                std::vector<unsigned int> const& wavetilesPerWorkgroup)
        {
            AssertFatal(mac_tile.layoutType == LayoutType::MATRIX_ACCUMULATOR,
                        "Store must be from accumulator.");

            auto wave_tile     = WaveTile(mac_tile.subTileSizes, LayoutType::MATRIX_ACCUMULATOR);
            auto wave_tile_tag = graph.coordinates.addElement(wave_tile);

            uint num_elements = wave_tile.sizes[0] * wave_tile.sizes[1];
            uint wfs          = static_cast<uint>(wavefrontSize);
            uint num_agpr     = num_elements / wfs;

            // MFMA accumulator tile size
            uint mts            = 4u;
            auto mfma_tile_size = literal(mts);

            auto n_row_blocks = literal(wave_tile.sizes[0] / mts);
            auto n_col_blocks = literal(wave_tile.sizes[1] / mts);

            auto n_wave_x = graph.coordinates.addElement(wave_tile.tileNumber(0));
            auto n_wave_y = graph.coordinates.addElement(wave_tile.tileNumber(1));
            auto i_wave_x = graph.coordinates.addElement(wave_tile.tileIndex(0));
            auto i_wave_y = graph.coordinates.addElement(wave_tile.tileIndex(1));

            auto wavefront_size = Expression::literal(static_cast<uint>(wavefrontSize));
            auto unit_stride    = literal(1u);

            auto n_vblk = graph.coordinates.addElement(
                VGPRBlockNumber(literal(num_agpr / mts), unit_stride));
            auto i_vblk = graph.coordinates.addElement(VGPRBlockIndex(mfma_tile_size, unit_stride));
            auto n_lblk = graph.coordinates.addElement(
                Adhoc("LANEBlockNumber", literal(wfs / mts), unit_stride));
            auto i_lblk = graph.coordinates.addElement(
                Adhoc("LANEBlockIndex", mfma_tile_size, unit_stride));
            auto block = graph.coordinates.addElement(
                Adhoc("LinearBlock", literal(num_elements / 16u), unit_stride));
            auto row_block
                = graph.coordinates.addElement(Adhoc("RowBlock", n_row_blocks, unit_stride));
            auto col_block
                = graph.coordinates.addElement(Adhoc("ColBlock", n_col_blocks, unit_stride));

            graph.coordinates.addElement(Flatten(), {n_wave_x, i_wave_x}, {i_mac_x});
            graph.coordinates.addElement(Flatten(), {n_wave_y, i_wave_y}, {i_mac_y});

            auto wave_x = graph.coordinates.addElement(Wavefront(0));
            auto wave_y = graph.coordinates.addElement(Wavefront(1));
            auto wave   = graph.coordinates.addElement(Wavefront(-1));

            graph.coordinates.addElement(Tile(), {wave}, {wave_x, wave_y});

            auto lane = graph.coordinates.addElement(Lane(wavefront_size, unit_stride));
            auto vgpr = graph.coordinates.addElement(VGPR(literal(num_agpr), unit_stride));

            graph.mapper.connect<WaveTile>(store_tag, wave_tile_tag);
            graph.mapper.connect<VGPRBlockNumber>(store_tag, n_vblk);
            graph.mapper.connect<VGPRBlockIndex>(store_tag, i_vblk);
            graph.mapper.connect<VGPR>(store_tag, vgpr);

            graph.coordinates.addElement(Tile(), {vgpr}, {n_vblk, i_vblk});
            graph.coordinates.addElement(Tile(), {lane}, {n_lblk, i_lblk});
            graph.coordinates.addElement(Flatten(), {n_vblk, n_lblk}, {block});
            graph.coordinates.addElement(Tile(), {block}, {row_block, col_block});

            int jammed_wavetile_x = -1;
            if(wavetilesPerWorkgroup[0] >= 1)
            {
                jammed_wavetile_x = graph.coordinates.addElement(
                    JammedWaveTileNumber(0, literal(wavetilesPerWorkgroup[0]), literal(1)));
                graph.mapper.connect<JammedWaveTileNumber>(store_tag, jammed_wavetile_x, 0);
            }
            int jammed_wavetile_y = -1;
            if(wavetilesPerWorkgroup[1] >= 1)
            {
                jammed_wavetile_y = graph.coordinates.addElement(
                    JammedWaveTileNumber(1, literal(wavetilesPerWorkgroup[1]), literal(1)));
                graph.mapper.connect<JammedWaveTileNumber>(store_tag, jammed_wavetile_y, 1);
            }

            if(wavetilesPerWorkgroup[0] > 1)
                graph.coordinates.addElement(Flatten(), {wave_x, jammed_wavetile_x}, {n_wave_x});
            else
                graph.coordinates.addElement(PassThrough(), {wave_x}, {n_wave_x});
            if(wavetilesPerWorkgroup[1] > 1)
                graph.coordinates.addElement(Flatten(), {wave_y, jammed_wavetile_y}, {n_wave_y});
            else
                graph.coordinates.addElement(PassThrough(), {wave_y}, {n_wave_y});

            graph.coordinates.addElement(Flatten(), {row_block, i_vblk}, {i_wave_x});
            graph.coordinates.addElement(Flatten(), {col_block, i_lblk}, {i_wave_y});

            graph.coordinates.addElement(Tile(), {workitem}, {wave, lane});
        }

        void storeMacroTileIntoLDS(KernelGraph&                       graph,
                                   int                                store_tag,
                                   int                                lds_tag,
                                   int                                mac_tile_tag,
                                   std::array<unsigned int, 3> const& workgroupSizes,
                                   std::shared_ptr<Context>           context)
        {
            auto mac_tile = graph.coordinates.getNode<MacroTile>(mac_tile_tag);
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::Utils::storeMacroTileIntoLDS(): LDS({}), MacroTile({})",
                lds_tag,
                mac_tile_tag);
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::Utils::storeMacroTileIntoLDS(): MacroTile size: {}x{}",
                mac_tile.sizes[0],
                mac_tile.sizes[1]);

            auto thr_tile = ThreadTile(mac_tile);

            auto i_mac_x = graph.coordinates.addElement(mac_tile.tileIndex(0));
            auto i_mac_y = graph.coordinates.addElement(mac_tile.tileIndex(1));

            auto element_number_x
                = graph.coordinates.addElement(ElementNumber(0, literal(thr_tile.sizes.at(0))));
            auto element_number_y
                = graph.coordinates.addElement(ElementNumber(1, literal(thr_tile.sizes.at(1))));

            graph.mapper.connect<ElementNumber>(store_tag, element_number_x, 0);
            graph.mapper.connect<ElementNumber>(store_tag, element_number_y, 1);

            int n_thr_x, n_thr_y, i_thr_x, i_thr_y;
            if((mac_tile.layoutType == LayoutType::MATRIX_A
                && context->kernelOptions().transposeMemoryAccessA)
               || (mac_tile.layoutType == LayoutType::MATRIX_B
                   && context->kernelOptions().transposeMemoryAccessB)
               || context->kernelOptions().transposeMemoryAccessOther)
            {
                n_thr_x = graph.coordinates.addElement(
                    ThreadTileNumber(0, literal(thr_tile.sizes.at(0))));
                n_thr_y = graph.coordinates.addElement(
                    ThreadTileNumber(1, literal(thr_tile.wsizes.at(1))));
                i_thr_x = graph.coordinates.addElement(
                    ThreadTileIndex(0, literal(thr_tile.wsizes.at(0))));
                i_thr_y = graph.coordinates.addElement(
                    ThreadTileIndex(1, literal(thr_tile.sizes.at(1))));

                graph.coordinates.addElement(PassThrough(), {element_number_x}, {n_thr_x});
                graph.coordinates.addElement(PassThrough(), {element_number_y}, {i_thr_y});

                auto workitem_x
                    = graph.coordinates.addElement(Workitem(0, literal(workgroupSizes.at(0))));

                if(mac_tile.layoutType == LayoutType::MATRIX_A
                   || mac_tile.layoutType == LayoutType::MATRIX_B)
                {
                    graph.coordinates.addElement(Tile(), {workitem_x}, {n_thr_y, i_thr_x});
                }
                else
                {
                    auto workitem_y
                        = graph.coordinates.addElement(Workitem(1, literal(workgroupSizes.at(1))));

                    graph.coordinates.addElement(PassThrough(), {workitem_x}, {i_thr_x});
                    graph.coordinates.addElement(PassThrough(), {workitem_y}, {n_thr_y});
                }
            }
            else
            {
                n_thr_x = graph.coordinates.addElement(
                    ThreadTileNumber(0, literal(thr_tile.wsizes.at(0))));
                n_thr_y = graph.coordinates.addElement(
                    ThreadTileNumber(1, literal(thr_tile.wsizes.at(1))));
                i_thr_x = graph.coordinates.addElement(
                    ThreadTileIndex(0, literal(thr_tile.sizes.at(0))));
                i_thr_y = graph.coordinates.addElement(
                    ThreadTileIndex(1, literal(thr_tile.sizes.at(1))));

                graph.coordinates.addElement(PassThrough(), {element_number_x}, {i_thr_x});
                graph.coordinates.addElement(PassThrough(), {element_number_y}, {i_thr_y});

                auto workitem_x
                    = graph.coordinates.addElement(Workitem(0, literal(workgroupSizes.at(0))));

                if(mac_tile.layoutType == LayoutType::MATRIX_A
                   || mac_tile.layoutType == LayoutType::MATRIX_B)
                {
                    graph.coordinates.addElement(Tile(), {workitem_x}, {n_thr_y, n_thr_x});
                }
                else
                {
                    auto workitem_y
                        = graph.coordinates.addElement(Workitem(1, literal(workgroupSizes.at(1))));

                    graph.coordinates.addElement(PassThrough(), {workitem_x}, {n_thr_x});
                    graph.coordinates.addElement(PassThrough(), {workitem_y}, {n_thr_y});
                }
            }

            graph.coordinates.addElement(Flatten(), {n_thr_x, i_thr_x}, {i_mac_x});
            graph.coordinates.addElement(Flatten(), {n_thr_y, i_thr_y}, {i_mac_y});

            graph.coordinates.addElement(Flatten(), {i_mac_x, i_mac_y}, {lds_tag});

            //macrotile --DataFlow--> LDS
            graph.coordinates.addElement(DataFlow(), {mac_tile_tag}, {lds_tag});
        }

        void storeMacroTileForLDS(KernelGraph&                       graph,
                                  int                                store_tag,
                                  int                                user_tag,
                                  int                                mac_tile_tag,
                                  std::vector<int>&                  sdims,
                                  std::array<unsigned int, 3> const& workgroupSizes)
        {
            auto mac_tile = graph.coordinates.getNode<MacroTile>(mac_tile_tag);
            auto user     = graph.coordinates.getNode<User>(user_tag);

            rocRoller::Log::getLogger()->debug(
                "KernelGraph::Utils::storeMacroTileForLDS(): User({}), MacroTile({})",
                user_tag,
                mac_tile_tag);
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::Utils::storeMacroTileForLDS(): MacroTile size: {}x{}",
                mac_tile.sizes[0],
                mac_tile.sizes[1]);

            AssertRecoverable(mac_tile.rank >= 0 && sdims.size() == (size_t)mac_tile.rank,
                              "Tensor size mismatch.");
            AssertRecoverable(mac_tile.rank == 2, "Rank /= 2 not implemented yet.");

            auto n_mac_x = graph.coordinates.addElement(mac_tile.tileNumber(0));
            auto n_mac_y = graph.coordinates.addElement(mac_tile.tileNumber(1));
            auto i_mac_x = graph.coordinates.addElement(mac_tile.tileIndex(0));
            auto i_mac_y = graph.coordinates.addElement(mac_tile.tileIndex(1));

            graph.mapper.connect<MacroTileNumber>(store_tag, n_mac_x, 0);
            graph.mapper.connect<MacroTileNumber>(store_tag, n_mac_y, 1);

            auto workgroup_x = graph.coordinates.addElement(Workgroup(0));
            auto workgroup_y = graph.coordinates.addElement(Workgroup(1));

            graph.mapper.connect<Workgroup>(store_tag, workgroup_x, 0);
            graph.mapper.connect<Workgroup>(store_tag, workgroup_y, 1);

            graph.coordinates.addElement(Flatten(), {n_mac_x, i_mac_x}, {sdims[0]});
            graph.coordinates.addElement(Flatten(), {n_mac_y, i_mac_y}, {sdims[1]});

            graph.coordinates.addElement(PassThrough(), {workgroup_x}, {n_mac_x});
            graph.coordinates.addElement(PassThrough(), {workgroup_y}, {n_mac_y});

            auto workitem_x
                = graph.coordinates.addElement(Workitem(0, literal(workgroupSizes.at(0))));

            auto thr_tile = ThreadTile(mac_tile);

            auto n_thr_x
                = graph.coordinates.addElement(ThreadTileNumber(0, literal(thr_tile.sizes.at(0))));
            auto n_thr_y
                = graph.coordinates.addElement(ThreadTileNumber(1, literal(thr_tile.wsizes.at(1))));
            auto i_thr_x
                = graph.coordinates.addElement(ThreadTileIndex(0, literal(thr_tile.wsizes.at(0))));
            auto i_thr_y
                = graph.coordinates.addElement(ThreadTileIndex(1, literal(thr_tile.sizes.at(1))));

            graph.coordinates.addElement(Flatten(), {n_thr_x, i_thr_x}, {i_mac_x});
            graph.coordinates.addElement(Flatten(), {n_thr_y, i_thr_y}, {i_mac_y});

            auto element_number_x
                = graph.coordinates.addElement(ElementNumber(0, literal(thr_tile.sizes.at(0))));
            auto element_number_y
                = graph.coordinates.addElement(ElementNumber(1, literal(thr_tile.sizes.at(1))));

            graph.coordinates.addElement(PassThrough(), {element_number_x}, {n_thr_x});
            graph.coordinates.addElement(PassThrough(), {element_number_y}, {i_thr_y});

            graph.mapper.connect<ElementNumber>(store_tag, element_number_x, 0);
            graph.mapper.connect<ElementNumber>(store_tag, element_number_y, 1);

            graph.coordinates.addElement(Tile(), {workitem_x}, {n_thr_y, i_thr_x});
        }

        void storeMacroTile(KernelGraph&                       graph,
                            int                                store_tag,
                            int                                user_tag,
                            int                                mac_tile_tag,
                            std::vector<int>&                  sdims,
                            std::array<unsigned int, 3> const& workgroupSizes,
                            int                                wavefrontSize,
                            std::vector<unsigned int> const&   wavetilesPerWorkgroup)
        {
            auto mac_tile = graph.coordinates.getNode<MacroTile>(mac_tile_tag);
            auto user     = graph.coordinates.getNode<User>(user_tag);

            rocRoller::Log::getLogger()->debug(
                "KernelGraph::Utils::storeMacroTile(): User({}), MacroTile({})",
                user_tag,
                mac_tile_tag);
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::Utils::storeMacroTile(): MacroTile size: {}x{}",
                mac_tile.sizes[0],
                mac_tile.sizes[1]);

            AssertRecoverable(mac_tile.rank >= 0 && sdims.size() == (size_t)mac_tile.rank,
                              "Tensor size mismatch.");
            AssertRecoverable(mac_tile.rank == 2, "Rank /= 2 not implemented yet.");

            auto n_mac_x = graph.coordinates.addElement(mac_tile.tileNumber(0));
            auto n_mac_y = graph.coordinates.addElement(mac_tile.tileNumber(1));
            auto i_mac_x = graph.coordinates.addElement(mac_tile.tileIndex(0));
            auto i_mac_y = graph.coordinates.addElement(mac_tile.tileIndex(1));

            graph.mapper.connect<MacroTileNumber>(store_tag, n_mac_x, 0);
            graph.mapper.connect<MacroTileNumber>(store_tag, n_mac_y, 1);

            auto workgroup_x = graph.coordinates.addElement(Workgroup(0));
            auto workgroup_y = graph.coordinates.addElement(Workgroup(1));

            graph.mapper.connect<Workgroup>(store_tag, workgroup_x, 0);
            graph.mapper.connect<Workgroup>(store_tag, workgroup_y, 1);

            graph.coordinates.addElement(Flatten(), {n_mac_x, i_mac_x}, {sdims[0]});
            graph.coordinates.addElement(Flatten(), {n_mac_y, i_mac_y}, {sdims[1]});

            graph.coordinates.addElement(PassThrough(), {workgroup_x}, {n_mac_x});
            graph.coordinates.addElement(PassThrough(), {workgroup_y}, {n_mac_y});

            switch(mac_tile.memoryType)
            {
            case MemoryType::VGPR:
            case MemoryType::LDS:
            {
                auto thr_tile = ThreadTile(mac_tile);

                auto n_thr_x = graph.coordinates.addElement(
                    ThreadTileNumber(0, literal(thr_tile.wsizes.at(0))));
                auto n_thr_y = graph.coordinates.addElement(
                    ThreadTileNumber(1, literal(thr_tile.wsizes.at(1))));
                auto i_thr_x = graph.coordinates.addElement(
                    ThreadTileIndex(0, literal(thr_tile.sizes.at(0))));
                auto i_thr_y = graph.coordinates.addElement(
                    ThreadTileIndex(1, literal(thr_tile.sizes.at(1))));

                auto workitem_x
                    = graph.coordinates.addElement(Workitem(0, literal(thr_tile.wsizes.at(0))));
                auto workitem_y
                    = graph.coordinates.addElement(Workitem(1, literal(thr_tile.wsizes.at(1))));

                auto element_number_x
                    = graph.coordinates.addElement(ElementNumber(0, literal(thr_tile.sizes.at(0))));
                auto element_number_y
                    = graph.coordinates.addElement(ElementNumber(1, literal(thr_tile.sizes.at(1))));

                graph.mapper.connect<ElementNumber>(store_tag, element_number_x, 0);
                graph.mapper.connect<ElementNumber>(store_tag, element_number_y, 1);

                graph.coordinates.addElement(Flatten(), {n_thr_x, i_thr_x}, {i_mac_x});
                graph.coordinates.addElement(Flatten(), {n_thr_y, i_thr_y}, {i_mac_y});

                graph.coordinates.addElement(PassThrough(), {workitem_x}, {n_thr_x});
                graph.coordinates.addElement(PassThrough(), {workitem_y}, {n_thr_y});

                graph.coordinates.addElement(PassThrough(), {element_number_x}, {i_thr_x});
                graph.coordinates.addElement(PassThrough(), {element_number_y}, {i_thr_y});
            }
            break;

            case MemoryType::WAVE:
            case MemoryType::WAVE_LDS:
            {
                auto workitem
                    = graph.coordinates.addElement(Workitem(0, literal(workgroupSizes.at(0))));

                storeWaveMacroTile(graph,
                                   mac_tile,
                                   store_tag,
                                   i_mac_x,
                                   i_mac_y,
                                   workitem,
                                   user_tag,
                                   wavefrontSize,
                                   wavetilesPerWorkgroup);
            }
            break;

            default:
                Throw<FatalError>("Store : MacroTile memory type not supported yet.");
            }
        }

        void addConnectionsMultiply(KernelGraph& graph, int waveMult)
        {
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::Utils::addConnectionsMultiply(): Multiply({})", waveMult);

            auto loads = graph.control.getOutputNodeIndices<Body>(waveMult).to<std::vector>();
            AssertFatal(loads.size() == 2, "Multiply op needs two operands");
            auto loadA = graph.control.getElement(loads[0]);
            auto loadB = graph.control.getElement(loads[1]);
            AssertFatal(isOperation<LoadTiled>(loadA) && isOperation<LoadTiled>(loadB),
                        "Both operands should be LoadTiled");

            // LoadTiled A
            auto userA_tag = graph.mapper.get<User>(loads[0]);
            AssertFatal(userA_tag > 0, "User dimension not found");
            graph.mapper.connect<User>(waveMult, userA_tag, 0);

            // LoadTiled B
            auto userB_tag = graph.mapper.get<User>(loads[1]);
            AssertFatal(userB_tag > 0, "User dimension not found");
            graph.mapper.connect<User>(waveMult, userB_tag, 1);

            AssertFatal(userA_tag > 0 && userB_tag > 0, "User dimensions not found");

            auto [waveA_tag, waveA] = graph.getDimension<WaveTile>(loads[0]);
            auto [waveB_tag, waveB] = graph.getDimension<WaveTile>(loads[1]);

            auto a = graph.coordinates.getOutputNodeIndices(userA_tag, CT::isEdge<DataFlow>)
                         .to<std::vector>();
            auto b = graph.coordinates.getOutputNodeIndices(userB_tag, CT::isEdge<DataFlow>)
                         .to<std::vector>();
            AssertFatal(a.size() == 1 && b.size() == 1, a.size(), b.size());

            graph.mapper.connect(
                waveMult, a[0], Connections::typeArgument<MacroTile>(NaryArgument::LHS));
            graph.mapper.connect(
                waveMult, b[0], Connections::typeArgument<MacroTile>(NaryArgument::RHS));
            graph.mapper.connect(
                waveMult, waveA_tag, Connections::typeArgument<WaveTile>(NaryArgument::LHS));
            graph.mapper.connect(
                waveMult, waveB_tag, Connections::typeArgument<WaveTile>(NaryArgument::RHS));
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

        /**
         * @brief Replace operation with a scope.  Does not delete the original operation.
         */
        int replaceWithScope(KernelGraph& graph, int op, bool includeBody)
        {
            auto scope = graph.control.addElement(Scope());

            auto location = graph.control.getLocation(op);
            for(auto const& input : location.incoming)
            {
                auto edge = graph.control.getElement(input);
                int  parent
                    = *graph.control.getNeighbours<Graph::Direction::Upstream>(input).begin();
                graph.control.deleteElement(input);
                graph.control.addElement(edge, {parent}, {scope});
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
                        graph.control.addElement(edge, {scope}, {child});
                    }
                }
            }

            return scope;
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
            for(auto const& reap : kgraph.control.depthFirstVisit(loop).to<std::vector>())
            {
                kgraph.control.deleteElement(reap);
                kgraph.mapper.purge(reap);
            }
            kgraph.mapper.purge(loop);
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

        std::vector<int> findRequiredCoordinates(int                target,
                                                 Graph::Direction   direction,
                                                 KernelGraph const& kgraph)
        {
            namespace CT      = rocRoller::KernelGraph::CoordinateGraph;
            auto lookBackward = direction == GD::Downstream ? GD::Upstream : GD::Downstream;

            auto dontWalkPastLoopOrStorageNodes = [&](int tag) -> bool {
                auto element = kgraph.coordinates.getElement(tag);
                auto edge    = std::get<CT::Edge>(element);

                bool isCT   = std::holds_alternative<CT::CoordinateTransformEdge>(edge);
                bool follow = true;
                for(auto neighbour : kgraph.coordinates.getNeighbours(tag, lookBackward))
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

            return required;
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
    }
}
