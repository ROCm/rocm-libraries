#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/InstructionValues/Register.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        namespace Expression = rocRoller::Expression;
        using namespace Expression;

        /***********************************
         * Helpers
         */

        /**
         * Create a range-based for loop.
         */
        std::pair<int, int> rangeFor(KernelHypergraph& graph, Expression::ExpressionPtr size)
        {
            auto unit_stride = Expression::literal(1u);
            auto rangeK      = graph.coordinates.addElement(CoordGraph::Linear(size, unit_stride));
            auto dimK        = graph.coordinates.addElement(CoordGraph::ForLoop(size, unit_stride));
            auto sizeDataType = Expression::resultVariableType(size);
            auto exprK        = std::make_shared<Expression::Expression>(
                DataFlowTag{rangeK, Register::Type::Scalar, sizeDataType});

            auto forK       = graph.control.addElement(ControlHypergraph::ForLoopOp{exprK < size});
            auto initK      = graph.control.addElement(ControlHypergraph::Assign{
                Register::Type::Scalar, Expression::literal(0, sizeDataType)});
            auto incrementK = graph.control.addElement(
                ControlHypergraph::Assign{Register::Type::Scalar, exprK + unit_stride});

            graph.coordinates.addElement(CoordGraph::DataFlow(), {rangeK}, {dimK});
            graph.control.addElement(ControlHypergraph::Initialize(), {forK}, {initK});
            graph.control.addElement(ControlHypergraph::ForLoopIncrement(), {forK}, {incrementK});

            graph.mapper.connect<CoordGraph::Dimension>(forK, rangeK);
            graph.mapper.connect<CoordGraph::Dimension>(initK, rangeK);
            graph.mapper.connect<CoordGraph::ForLoop>(incrementK, rangeK);

            return {dimK, forK};
        }

        void loadWaveMacroTileFromLDS(KernelHypergraph&            graph,
                                      CoordGraph::MacroTile const& mac_tile,
                                      int                          load_tag,
                                      std::vector<int>&            sdims,
                                      int                          K,
                                      int                          lds)
        {
            // given that the loadWaveMacroTile has already lowered the macrotile for LoadTiled
            // before it is transformed to LoadLDSTile

            if(mac_tile.layoutType == LayoutType::MATRIX_A)
            {
                // remove passthrough between A row block and x-workgroup
                // remove x-workgroup
                auto a_tilenum_x   = graph.mapper.get<CoordGraph::MacroTileNumber>(load_tag, 0);
                auto a_workgroup_x = graph.mapper.get<CoordGraph::Workgroup>(load_tag, 0);
                graph.coordinates.deleteElement(std::vector<int>{a_tilenum_x},
                                                std::vector<int>{a_workgroup_x},
                                                CoordGraph::isEdge<CoordGraph::PassThrough>);
                graph.mapper.disconnect<CoordGraph::Workgroup>(load_tag, a_workgroup_x, 0);
                graph.coordinates.deleteElement(a_workgroup_x);

                // remove passthrough between A column block and K
                auto a_tilenum_y = graph.mapper.get<CoordGraph::MacroTileNumber>(load_tag, 1);
                graph.coordinates.deleteElement(std::vector<int>{a_tilenum_y},
                                                std::vector<int>{K},
                                                CoordGraph::isEdge<CoordGraph::PassThrough>);
            }

            if(mac_tile.layoutType == LayoutType::MATRIX_B)
            {
                // remove passthrough between B column block and y-workgroup
                // remove y-workgroup
                auto b_tilenum_y   = graph.mapper.get<CoordGraph::MacroTileNumber>(load_tag, 1);
                auto b_workgroup_y = graph.mapper.get<CoordGraph::Workgroup>(load_tag, 1);
                graph.coordinates.deleteElement(std::vector<int>{b_tilenum_y},
                                                std::vector<int>{b_workgroup_y},
                                                CoordGraph::isEdge<CoordGraph::PassThrough>);
                graph.mapper.disconnect<CoordGraph::Workgroup>(load_tag, b_workgroup_y, 1);
                graph.coordinates.deleteElement(b_workgroup_y);

                // remove passthrough between B row block and K
                auto b_tilenum_x = graph.mapper.get<CoordGraph::MacroTileNumber>(load_tag, 0);
                graph.coordinates.deleteElement(std::vector<int>{b_tilenum_x},
                                                std::vector<int>{K},
                                                CoordGraph::isEdge<CoordGraph::PassThrough>);
            }

            std::vector<int> i_mac;
            for(size_t i = 0; i < sdims.size(); ++i)
            {
                auto mac = graph.coordinates
                               .getOutputNodeIndices(sdims[i], CoordGraph::isEdge<CoordGraph::Tile>)
                               .to<std::vector>();
                i_mac.push_back(mac[1]);
                graph.coordinates.deleteElement(
                    std::vector<int>{sdims[i]}, mac, CoordGraph::isEdge<CoordGraph::Tile>);
                graph.mapper.disconnect<CoordGraph::MacroTileNumber>(load_tag, mac[0], i);
                graph.coordinates.deleteElement(mac[0]);
            }

            graph.coordinates.addElement(CoordGraph::Tile(), {lds}, {i_mac[0], i_mac[1]});
        }

        void loadWaveMacroTile(KernelHypergraph&                graph,
                               CoordGraph::MacroTile const&     mac_tile,
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

            auto workitem      = graph.coordinates.addElement(CoordGraph::Workitem(0));
            auto wave_tile     = CoordGraph::WaveTile(tileSize, mac_tile.layoutType);
            auto wave_tile_tag = graph.coordinates.addElement(wave_tile);
            graph.mapper.connect<CoordGraph::WaveTile>(load_tag, wave_tile_tag);

            auto n_wave_x = graph.coordinates.addElement(wave_tile.tileNumber(0));
            auto n_wave_y = graph.coordinates.addElement(wave_tile.tileNumber(1));
            auto i_wave_x = graph.coordinates.addElement(wave_tile.tileIndex(0));
            auto i_wave_y = graph.coordinates.addElement(wave_tile.tileIndex(1));

            graph.coordinates.addElement(CoordGraph::Tile(), {i_mac_x}, {n_wave_x, i_wave_x});
            graph.coordinates.addElement(CoordGraph::Tile(), {i_mac_y}, {n_wave_y, i_wave_y});

            graph.mapper.connect<CoordGraph::WaveTileNumber>(load_tag, n_wave_x, 0);
            graph.mapper.connect<CoordGraph::WaveTileNumber>(load_tag, n_wave_y, 1);

            auto wave_x = graph.coordinates.addElement(CoordGraph::Wavefront(0));
            auto wave_y = graph.coordinates.addElement(CoordGraph::Wavefront(1));
            auto wave   = graph.coordinates.addElement(CoordGraph::Wavefront(-1));

            uint num_elements = product(tileSize);
            uint wfs          = static_cast<uint>(wavefrontSize);
            uint num_vgpr     = num_elements / wfs;

            auto wavefront_size = Expression::literal(wfs);

            auto lane = graph.coordinates.addElement(CoordGraph::Lane(wavefront_size, nullptr));
            auto vgpr = graph.coordinates.addElement(CoordGraph::VGPR(literal(num_vgpr), nullptr));

            graph.coordinates.addElement(CoordGraph::Flatten(), {wave_x, wave_y}, {wave});
            graph.coordinates.addElement(CoordGraph::Flatten(), {wave, lane}, {workitem});

            graph.mapper.connect<CoordGraph::VGPR>(load_tag, vgpr);

            auto block_number = graph.coordinates.addElement(
                CoordGraph::Adhoc("BlockNumber", literal(static_cast<uint>(wfs / m)), nullptr));
            auto block_index = graph.coordinates.addElement(
                CoordGraph::Adhoc("BlockIndex", literal(static_cast<uint>(m)), nullptr));

            graph.coordinates.addElement(
                CoordGraph::Flatten(), {block_number, block_index}, {lane});

            int jammed_wavetile_x = -1;
            if(wavetilesPerWorkgroup[0] > 1)
            {
                jammed_wavetile_x = graph.coordinates.addElement(CoordGraph::JammedWaveTileNumber(
                    0, literal(wavetilesPerWorkgroup[0]), literal(1)));
                graph.mapper.connect<CoordGraph::JammedWaveTileNumber>(
                    load_tag, jammed_wavetile_x, 0);
            }
            int jammed_wavetile_y = -1;
            if(wavetilesPerWorkgroup[1] > 1)
            {
                jammed_wavetile_y = graph.coordinates.addElement(CoordGraph::JammedWaveTileNumber(
                    1, literal(wavetilesPerWorkgroup[1]), literal(1)));
                graph.mapper.connect<CoordGraph::JammedWaveTileNumber>(
                    load_tag, jammed_wavetile_y, 1);
            }

            switch(wave_tile.layout)
            {
            case LayoutType::MATRIX_A:
            {
                // TODO: not necessary here, but used for lookup in generator
                graph.coordinates.addElement(
                    CoordGraph::Flatten(), {i_wave_y, i_wave_x}, {wave_tile_tag});
                graph.coordinates.addElement(CoordGraph::Tile(), {i_wave_y}, {block_number, vgpr});
                graph.coordinates.addElement(CoordGraph::PassThrough(), {i_wave_x}, {block_index});

                if(wavetilesPerWorkgroup[0] > 1)
                    graph.coordinates.addElement(
                        CoordGraph::Tile(), {n_wave_x}, {wave_x, jammed_wavetile_x});
                else
                    graph.coordinates.addElement(CoordGraph::PassThrough(), {n_wave_x}, {wave_x});
            }
            break;

            case LayoutType::MATRIX_B:
            {
                // TODO: not necessary here, but used for lookup in generator
                graph.coordinates.addElement(
                    CoordGraph::Flatten(), {i_wave_x, i_wave_y}, {wave_tile_tag});
                graph.coordinates.addElement(CoordGraph::Tile(), {i_wave_x}, {block_number, vgpr});
                graph.coordinates.addElement(CoordGraph::PassThrough(), {i_wave_y}, {block_index});

                if(wavetilesPerWorkgroup[1] > 1)
                    graph.coordinates.addElement(
                        CoordGraph::Tile(), {n_wave_y}, {wave_y, jammed_wavetile_y});
                else
                    graph.coordinates.addElement(CoordGraph::PassThrough(), {n_wave_y}, {wave_y});
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
                    CoordGraph::VGPRBlockNumber(literal(num_vgpr / mts), unit_stride));
                auto i_vblk = graph.coordinates.addElement(
                    CoordGraph::VGPRBlockIndex(mfma_tile_size, unit_stride));
                auto n_lblk = graph.coordinates.addElement(
                    CoordGraph::Adhoc("LANEBlockNumber", literal(wfs / mts), unit_stride));
                auto i_lblk = graph.coordinates.addElement(
                    CoordGraph::Adhoc("LANEBlockIndex", mfma_tile_size, unit_stride));
                auto block = graph.coordinates.addElement(
                    CoordGraph::Adhoc("LinearBlock", literal(num_elements / 16u), unit_stride));
                auto row_block = graph.coordinates.addElement(
                    CoordGraph::Adhoc("RowBlock", n_row_blocks, unit_stride));
                auto col_block = graph.coordinates.addElement(
                    CoordGraph::Adhoc("ColBlock", n_col_blocks, unit_stride));

                graph.mapper.connect<CoordGraph::VGPRBlockNumber>(load_tag, n_vblk);
                graph.mapper.connect<CoordGraph::VGPRBlockIndex>(load_tag, i_vblk);

                // ORDER?
                graph.coordinates.addElement(
                    CoordGraph::Flatten(), {i_wave_x, i_wave_y}, {wave_tile_tag});

                graph.coordinates.addElement(CoordGraph::Tile(), {i_wave_x}, {row_block, i_vblk});
                graph.coordinates.addElement(CoordGraph::Tile(), {i_wave_y}, {col_block, i_lblk});

                graph.coordinates.addElement(
                    CoordGraph::Flatten(), {row_block, col_block}, {block});
                graph.coordinates.addElement(CoordGraph::Tile(), {block}, {n_vblk, n_lblk});

                graph.coordinates.addElement(CoordGraph::Flatten(), {n_vblk, i_vblk}, {vgpr});
                graph.coordinates.addElement(CoordGraph::Flatten(), {n_lblk, i_lblk}, {lane});

                if(wavetilesPerWorkgroup[0] > 1)
                    graph.coordinates.addElement(
                        CoordGraph::Tile(), {n_wave_x}, {wave_x, jammed_wavetile_x});
                else
                    graph.coordinates.addElement(CoordGraph::PassThrough(), {n_wave_x}, {wave_x});

                if(wavetilesPerWorkgroup[1] > 1)
                    graph.coordinates.addElement(
                        CoordGraph::Tile(), {n_wave_y}, {wave_y, jammed_wavetile_y});
                else
                    graph.coordinates.addElement(CoordGraph::PassThrough(), {n_wave_y}, {wave_y});
            }
            break;

            default:
                Throw<FatalError>("Not implemented yet.");
            }
        }

        void loadMacroTileFromLDS(KernelHypergraph&                  graph,
                                  int                                load_tag,
                                  int                                lds_tag,
                                  int                                mac_tile_tag,
                                  std::array<unsigned int, 3> const& workgroupSizes)
        {
            auto mac_tile = graph.coordinates.getNode<CoordGraph::MacroTile>(mac_tile_tag);

            rocRoller::Log::getLogger()->debug(
                "KernelGraph::LowerTileVisitor::loadMacroTileFromLDS(): LDS({}), MacroTile({})",
                lds_tag,
                mac_tile_tag);
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::LowerTileVisitor::loadMacroTileFromLDS(): MacroTile size: {}x{}",
                mac_tile.sizes[0],
                mac_tile.sizes[1]);

            auto i_mac_x = graph.coordinates.addElement(mac_tile.tileIndex(0));
            auto i_mac_y = graph.coordinates.addElement(mac_tile.tileIndex(1));
            graph.coordinates.addElement(CoordGraph::Flatten(), {i_mac_x, i_mac_y}, {mac_tile_tag});

            auto workitem = graph.coordinates.addElement(
                CoordGraph::Workitem(0, literal(workgroupSizes.at(0))));

            auto thr_tile     = CoordGraph::ThreadTile(mac_tile.subTileSizes);
            auto thr_tile_tag = graph.coordinates.addElement(thr_tile);

            auto n_thr_x = graph.coordinates.addElement(thr_tile.tileNumber(0));
            auto n_thr_y = graph.coordinates.addElement(thr_tile.tileNumber(1));
            auto i_thr_x = graph.coordinates.addElement(thr_tile.tileIndex(0));
            auto i_thr_y = graph.coordinates.addElement(thr_tile.tileIndex(1));

            graph.mapper.connect<CoordGraph::ThreadTileIndex>(load_tag, i_thr_x, 0);
            graph.mapper.connect<CoordGraph::ThreadTileIndex>(load_tag, i_thr_y, 1);

            // note that the tile index is the first dimension and tile number is the second dimension
            graph.coordinates.addElement(CoordGraph::Tile(), {i_mac_x}, {i_thr_x, n_thr_x});
            graph.coordinates.addElement(CoordGraph::Tile(), {i_mac_y}, {i_thr_y, n_thr_y});

            graph.coordinates.addElement(CoordGraph::Flatten(), {n_thr_x, n_thr_y}, {workitem});

            graph.coordinates.addElement(CoordGraph::Tile(), {thr_tile_tag}, {i_thr_x, i_thr_y});

            // each workitem and its vgpr contributes towards the offset calculation
            graph.coordinates.addElement(CoordGraph::Tile(), {lds_tag}, {thr_tile_tag, workitem});

            // LDS --DataFlow--> macrotile
            graph.coordinates.addElement(CoordGraph::DataFlow(), {lds_tag}, {mac_tile_tag});
        }

        void loadMacroTileForLDS(KernelHypergraph&                  graph,
                                 int                                load_tag,
                                 int                                user_tag,
                                 int                                mac_tile_tag,
                                 std::vector<int>&                  sdim,
                                 int                                K,
                                 std::array<unsigned int, 3> const& workgroupSizes)
        {
            auto mac_tile = graph.coordinates.getNode<CoordGraph::MacroTile>(mac_tile_tag);
            auto user     = graph.coordinates.getNode<CoordGraph::User>(user_tag);

            rocRoller::Log::getLogger()->debug(
                "KernelGraph::LowerTileVisitor::loadMacroTileForLDS(): User({}), MacroTile({})",
                user_tag,
                mac_tile_tag);
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::LowerTileVisitor::loadMacroTileForLDS(): MacroTile size: {}x{}",
                mac_tile.sizes[0],
                mac_tile.sizes[1]);

            AssertFatal(mac_tile.rank == 2, "Rank /= 2 not implemented yet.");

            auto sdim_x = sdim[0];
            auto sdim_y = sdim[1];
            graph.mapper.connect<CoordGraph::SubDimension>(load_tag, sdim_x, 0);
            graph.mapper.connect<CoordGraph::SubDimension>(load_tag, sdim_y, 1);

            auto n_mac_x = graph.coordinates.addElement(mac_tile.tileNumber(0));
            auto n_mac_y = graph.coordinates.addElement(mac_tile.tileNumber(1));
            auto i_mac_x = graph.coordinates.addElement(mac_tile.tileIndex(0));
            auto i_mac_y = graph.coordinates.addElement(mac_tile.tileIndex(1));

            graph.mapper.connect<CoordGraph::MacroTileNumber>(load_tag, n_mac_x, 0);
            graph.mapper.connect<CoordGraph::MacroTileNumber>(load_tag, n_mac_y, 1);

            graph.coordinates.addElement(CoordGraph::Tile(), {sdim_x}, {n_mac_x, i_mac_x});
            graph.coordinates.addElement(CoordGraph::Tile(), {sdim_y}, {n_mac_y, i_mac_y});

            graph.coordinates.addElement(CoordGraph::Flatten(), {i_mac_x, i_mac_y}, {mac_tile_tag});

            auto thr_tile     = CoordGraph::ThreadTile(mac_tile.subTileSizes);
            auto thr_tile_tag = graph.coordinates.addElement(thr_tile);

            auto n_thr_x = graph.coordinates.addElement(thr_tile.tileNumber(0));
            auto n_thr_y = graph.coordinates.addElement(thr_tile.tileNumber(1));
            auto i_thr_x = graph.coordinates.addElement(thr_tile.tileIndex(0));
            auto i_thr_y = graph.coordinates.addElement(thr_tile.tileIndex(1));

            graph.mapper.connect<CoordGraph::ThreadTileIndex>(load_tag, i_thr_x, 0);
            graph.mapper.connect<CoordGraph::ThreadTileIndex>(load_tag, i_thr_y, 1);

            graph.coordinates.addElement(CoordGraph::Tile(), {thr_tile_tag}, {i_thr_x, i_thr_y});

            auto workitem_x = graph.coordinates.addElement(
                CoordGraph::Workitem(0, literal(workgroupSizes.at(0))));
            graph.coordinates.addElement(CoordGraph::Flatten(), {n_thr_x, n_thr_y}, {workitem_x});

            graph.coordinates.addElement(
                CoordGraph::Tile(), {mac_tile_tag}, {thr_tile_tag, workitem_x});

            if(mac_tile.layoutType == LayoutType::MATRIX_A)
            {
                auto workgroup_x = graph.coordinates.addElement(CoordGraph::Workgroup(0));

                graph.mapper.connect<CoordGraph::Workgroup>(load_tag, workgroup_x, 0);
                graph.coordinates.addElement(CoordGraph::PassThrough(), {n_mac_x}, {workgroup_x});
                // A row block is x-workgroup, column block is for loop index
                graph.coordinates.addElement(CoordGraph::PassThrough(), {n_mac_y}, {K});
            }
            else
            {
                auto workgroup_y = graph.coordinates.addElement(CoordGraph::Workgroup(1));

                graph.mapper.connect<CoordGraph::Workgroup>(load_tag, workgroup_y, 1);
                // B row block is for loop index, column block is y-workgroup
                graph.coordinates.addElement(CoordGraph::PassThrough(), {n_mac_x}, {K});
                graph.coordinates.addElement(CoordGraph::PassThrough(), {n_mac_y}, {workgroup_y});
            }
        }

        void loadMacroTile(KernelHypergraph&                  graph,
                           int                                load_tag,
                           int                                user_tag,
                           int                                mac_tile_tag,
                           std::vector<int>&                  sdim,
                           std::array<unsigned int, 3> const& workgroupSizes,
                           int                                wavefrontSize,
                           std::vector<unsigned int> const&   wavetilesPerWorkgroup)

        {
            auto mac_tile = graph.coordinates.getNode<CoordGraph::MacroTile>(mac_tile_tag);
            auto user     = graph.coordinates.getNode<CoordGraph::User>(user_tag);

            rocRoller::Log::getLogger()->debug(
                "KernelGraph::LowerTileVisitor::loadMacroTile(): User({}), MacroTile({})",
                user_tag,
                mac_tile_tag);
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::LowerTileVisitor::loadMacroTile(): MacroTile size: {}x{}",
                mac_tile.sizes[0],
                mac_tile.sizes[1]);

            AssertFatal(mac_tile.rank == 2, "Rank /= 2 not implemented yet.");

            auto sdim_x = sdim[0];
            auto sdim_y = sdim[1];
            graph.mapper.connect<CoordGraph::SubDimension>(load_tag, sdim_x, 0);
            graph.mapper.connect<CoordGraph::SubDimension>(load_tag, sdim_y, 1);

            auto n_mac_x = graph.coordinates.addElement(mac_tile.tileNumber(0));
            auto n_mac_y = graph.coordinates.addElement(mac_tile.tileNumber(1));
            auto i_mac_x = graph.coordinates.addElement(mac_tile.tileIndex(0));
            auto i_mac_y = graph.coordinates.addElement(mac_tile.tileIndex(1));

            graph.mapper.connect<CoordGraph::MacroTileNumber>(load_tag, n_mac_x, 0);
            graph.mapper.connect<CoordGraph::MacroTileNumber>(load_tag, n_mac_y, 1);

            auto workgroup_x = graph.coordinates.addElement(CoordGraph::Workgroup(0));
            auto workgroup_y = graph.coordinates.addElement(CoordGraph::Workgroup(1));

            graph.mapper.connect<CoordGraph::Workgroup>(load_tag, workgroup_x, 0);
            graph.mapper.connect<CoordGraph::Workgroup>(load_tag, workgroup_y, 1);

            graph.coordinates.addElement(CoordGraph::Flatten(), {i_mac_x, i_mac_y}, {mac_tile_tag});

            graph.coordinates.addElement(CoordGraph::Tile(), {sdim_x}, {n_mac_x, i_mac_x});
            graph.coordinates.addElement(CoordGraph::Tile(), {sdim_y}, {n_mac_y, i_mac_y});

            graph.coordinates.addElement(CoordGraph::PassThrough(), {n_mac_x}, {workgroup_x});
            graph.coordinates.addElement(CoordGraph::PassThrough(), {n_mac_y}, {workgroup_y});

            switch(mac_tile.memoryType)
            {
            case MemoryType::VGPR:
            case MemoryType::LDS:
            {
                auto workitem_x = graph.coordinates.addElement(CoordGraph::Workitem(0));
                auto workitem_y = graph.coordinates.addElement(CoordGraph::Workitem(1));

                auto thr_tile = CoordGraph::ThreadTile(mac_tile.subTileSizes);

                auto n_thr_x = graph.coordinates.addElement(thr_tile.tileNumber(0));
                auto n_thr_y = graph.coordinates.addElement(thr_tile.tileNumber(1));
                auto i_thr_x = graph.coordinates.addElement(thr_tile.tileIndex(0));
                auto i_thr_y = graph.coordinates.addElement(thr_tile.tileIndex(1));

                graph.mapper.connect<CoordGraph::ThreadTileIndex>(load_tag, i_thr_x, 0);
                graph.mapper.connect<CoordGraph::ThreadTileIndex>(load_tag, i_thr_y, 1);

                graph.coordinates.addElement(CoordGraph::Tile(), {i_mac_x}, {n_thr_x, i_thr_x});
                graph.coordinates.addElement(CoordGraph::Tile(), {i_mac_y}, {n_thr_y, i_thr_y});

                graph.coordinates.addElement(CoordGraph::PassThrough(), {n_thr_x}, {workitem_x});
                graph.coordinates.addElement(CoordGraph::PassThrough(), {n_thr_y}, {workitem_y});

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

        void storeWaveMacroTile(KernelHypergraph&                graph,
                                CoordGraph::MacroTile const&     mac_tile,
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

            auto wave_tile
                = CoordGraph::WaveTile(mac_tile.subTileSizes, LayoutType::MATRIX_ACCUMULATOR);
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

            graph.coordinates.addElement(CoordGraph::Join(), {i_wave_x, i_wave_y}, {wave_tile_tag});

            auto wavefront_size = Expression::literal(static_cast<uint>(wavefrontSize));
            auto unit_stride    = literal(1u);

            auto n_vblk = graph.coordinates.addElement(
                CoordGraph::VGPRBlockNumber(literal(num_agpr / mts), unit_stride));
            auto i_vblk = graph.coordinates.addElement(
                CoordGraph::VGPRBlockIndex(mfma_tile_size, unit_stride));
            auto n_lblk = graph.coordinates.addElement(
                CoordGraph::Adhoc("LANEBlockNumber", literal(wfs / mts), unit_stride));
            auto i_lblk = graph.coordinates.addElement(
                CoordGraph::Adhoc("LANEBlockIndex", mfma_tile_size, unit_stride));
            auto block = graph.coordinates.addElement(
                CoordGraph::Adhoc("LinearBlock", literal(num_elements / 16u), unit_stride));
            auto row_block = graph.coordinates.addElement(
                CoordGraph::Adhoc("RowBlock", n_row_blocks, unit_stride));
            auto col_block = graph.coordinates.addElement(
                CoordGraph::Adhoc("ColBlock", n_col_blocks, unit_stride));

            graph.coordinates.addElement(CoordGraph::Flatten(), {n_wave_x, i_wave_x}, {i_mac_x});
            graph.coordinates.addElement(CoordGraph::Flatten(), {n_wave_y, i_wave_y}, {i_mac_y});

            auto wave_x = graph.coordinates.addElement(CoordGraph::Wavefront(0));
            auto wave_y = graph.coordinates.addElement(CoordGraph::Wavefront(1));
            auto wave   = graph.coordinates.addElement(CoordGraph::Wavefront(-1));

            graph.coordinates.addElement(CoordGraph::Tile(), {wave}, {wave_x, wave_y});

            auto lane = graph.coordinates.addElement(CoordGraph::Lane(wavefront_size, unit_stride));
            auto vgpr
                = graph.coordinates.addElement(CoordGraph::VGPR(literal(num_agpr), unit_stride));

            graph.mapper.connect<CoordGraph::WaveTile>(store_tag, wave_tile_tag);
            graph.mapper.connect<CoordGraph::VGPRBlockNumber>(store_tag, n_vblk);
            graph.mapper.connect<CoordGraph::VGPRBlockIndex>(store_tag, i_vblk);
            graph.mapper.connect<CoordGraph::VGPR>(store_tag, vgpr);

            graph.coordinates.addElement(CoordGraph::Tile(), {vgpr}, {n_vblk, i_vblk});
            graph.coordinates.addElement(CoordGraph::Tile(), {lane}, {n_lblk, i_lblk});
            graph.coordinates.addElement(CoordGraph::Flatten(), {n_vblk, n_lblk}, {block});
            graph.coordinates.addElement(CoordGraph::Tile(), {block}, {row_block, col_block});

            int jammed_wavetile_x = -1;
            if(wavetilesPerWorkgroup[0] > 1)
            {
                jammed_wavetile_x = graph.coordinates.addElement(CoordGraph::JammedWaveTileNumber(
                    0, literal(wavetilesPerWorkgroup[0]), literal(1)));
                graph.mapper.connect<CoordGraph::JammedWaveTileNumber>(
                    store_tag, jammed_wavetile_x, 0);
            }
            int jammed_wavetile_y = -1;
            if(wavetilesPerWorkgroup[1] > 1)
            {
                jammed_wavetile_y = graph.coordinates.addElement(CoordGraph::JammedWaveTileNumber(
                    1, literal(wavetilesPerWorkgroup[1]), literal(1)));
                graph.mapper.connect<CoordGraph::JammedWaveTileNumber>(
                    store_tag, jammed_wavetile_y, 1);
            }

            if(wavetilesPerWorkgroup[0] > 1)
                graph.coordinates.addElement(
                    CoordGraph::Flatten(), {wave_x, jammed_wavetile_x}, {n_wave_x});
            else
                graph.coordinates.addElement(CoordGraph::PassThrough(), {wave_x}, {n_wave_x});
            if(wavetilesPerWorkgroup[1] > 1)
                graph.coordinates.addElement(
                    CoordGraph::Flatten(), {wave_y, jammed_wavetile_y}, {n_wave_y});
            else
                graph.coordinates.addElement(CoordGraph::PassThrough(), {wave_y}, {n_wave_y});

            graph.coordinates.addElement(CoordGraph::Flatten(), {row_block, i_vblk}, {i_wave_x});
            graph.coordinates.addElement(CoordGraph::Flatten(), {col_block, i_lblk}, {i_wave_y});

            graph.coordinates.addElement(CoordGraph::Tile(), {workitem}, {wave, lane});
        }

        void storeMacroTileIntoLDS(KernelHypergraph&                  graph,
                                   int                                store_tag,
                                   int                                lds_tag,
                                   int                                mac_tile_tag,
                                   std::array<unsigned int, 3> const& workgroupSizes)
        {
            auto mac_tile = graph.coordinates.getNode<CoordGraph::MacroTile>(mac_tile_tag);
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::LowerTileVisitor::storeMacroTileIntoLDS(): LDS({}), MacroTile({})",
                lds_tag,
                mac_tile_tag);
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::LowerTileVisitor::storeMacroTileIntoLDS(): MacroTile size: {}x{}",
                mac_tile.sizes[0],
                mac_tile.sizes[1]);

            auto workitem_x = graph.coordinates.addElement(
                CoordGraph::Workitem(0, literal(workgroupSizes.at(0))));

            auto i_mac_x = graph.coordinates.addElement(mac_tile.tileIndex(0));
            auto i_mac_y = graph.coordinates.addElement(mac_tile.tileIndex(1));

            auto thr_tile     = CoordGraph::ThreadTile(mac_tile.subTileSizes);
            auto thr_tile_tag = graph.coordinates.addElement(thr_tile);

            auto n_thr_x = graph.coordinates.addElement(thr_tile.tileNumber(0));
            auto n_thr_y = graph.coordinates.addElement(thr_tile.tileNumber(1));
            auto i_thr_x = graph.coordinates.addElement(thr_tile.tileIndex(0));
            auto i_thr_y = graph.coordinates.addElement(thr_tile.tileIndex(1));

            graph.coordinates.addElement(CoordGraph::Flatten(), {i_thr_x, i_thr_y}, {thr_tile_tag});

            graph.coordinates.addElement(CoordGraph::Tile(), {workitem_x}, {n_thr_x, n_thr_y});

            graph.mapper.connect<CoordGraph::ThreadTileIndex>(store_tag, i_thr_x, 0);
            graph.mapper.connect<CoordGraph::ThreadTileIndex>(store_tag, i_thr_y, 1);

            // each workitem and its vgpr contributes towards the offset calculation
            graph.coordinates.addElement(
                CoordGraph::Flatten(), {thr_tile_tag, workitem_x}, {lds_tag});

            //macrotile --DataFlow--> LDS
            graph.coordinates.addElement(CoordGraph::DataFlow(), {mac_tile_tag}, {lds_tag});
        }

        void storeMacroTile(KernelHypergraph&                  graph,
                            int                                store_tag,
                            int                                user_tag,
                            int                                mac_tile_tag,
                            std::vector<int>&                  sdims,
                            std::array<unsigned int, 3> const& workgroupSizes,
                            int                                wavefrontSize,
                            std::vector<unsigned int> const&   wavetilesPerWorkgroup)
        {
            auto mac_tile = graph.coordinates.getNode<CoordGraph::MacroTile>(mac_tile_tag);
            auto user     = graph.coordinates.getNode<CoordGraph::User>(user_tag);

            rocRoller::Log::getLogger()->debug(
                "KernelGraph::LowerTileVisitor::storeMacroTile(): User({}), MacroTile({})",
                user_tag,
                mac_tile_tag);
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::LowerTileVisitor::storeMacroTile(): MacroTile size: {}x{}",
                mac_tile.sizes[0],
                mac_tile.sizes[1]);

            AssertRecoverable(mac_tile.rank >= 0 && sdims.size() == (size_t)mac_tile.rank,
                              "Tensor size mismatch.");
            AssertRecoverable(mac_tile.rank == 2, "Rank /= 2 not implemented yet.");

            auto n_mac_x = graph.coordinates.addElement(mac_tile.tileNumber(0));
            auto n_mac_y = graph.coordinates.addElement(mac_tile.tileNumber(1));
            auto i_mac_x = graph.coordinates.addElement(mac_tile.tileIndex(0));
            auto i_mac_y = graph.coordinates.addElement(mac_tile.tileIndex(1));

            auto workgroup_x = graph.coordinates.addElement(CoordGraph::Workgroup(0));
            auto workgroup_y = graph.coordinates.addElement(CoordGraph::Workgroup(1));

            auto workitem = graph.coordinates.addElement(
                CoordGraph::Workitem(0, literal(workgroupSizes.at(0))));

            graph.coordinates.addElement(CoordGraph::Flatten(), {i_mac_x, i_mac_y}, {mac_tile_tag});

            graph.coordinates.addElement(CoordGraph::Flatten(), {n_mac_x, i_mac_x}, {sdims[0]});
            graph.coordinates.addElement(CoordGraph::Flatten(), {n_mac_y, i_mac_y}, {sdims[1]});

            graph.coordinates.addElement(CoordGraph::PassThrough(), {workgroup_x}, {n_mac_x});
            graph.coordinates.addElement(CoordGraph::PassThrough(), {workgroup_y}, {n_mac_y});

            switch(mac_tile.memoryType)
            {
            case MemoryType::VGPR:
            case MemoryType::LDS:
            {
                auto workitem_x = graph.coordinates.addElement(
                    CoordGraph::Workitem(0, literal(workgroupSizes.at(0))));
                auto workitem_y = graph.coordinates.addElement(
                    CoordGraph::Workitem(1, literal(workgroupSizes.at(1))));

                auto thr_tile     = CoordGraph::ThreadTile(mac_tile.subTileSizes);
                auto thr_tile_tag = graph.coordinates.addElement(thr_tile);

                auto n_thr_x = graph.coordinates.addElement(thr_tile.tileNumber(0));
                auto n_thr_y = graph.coordinates.addElement(thr_tile.tileNumber(1));
                auto i_thr_x = graph.coordinates.addElement(thr_tile.tileIndex(0));
                auto i_thr_y = graph.coordinates.addElement(thr_tile.tileIndex(1));

                graph.mapper.connect<CoordGraph::ThreadTileIndex>(store_tag, i_thr_x, 0);
                graph.mapper.connect<CoordGraph::ThreadTileIndex>(store_tag, i_thr_y, 1);

                graph.coordinates.addElement(
                    CoordGraph::Split(), {thr_tile_tag}, {i_thr_x, i_thr_y});
                graph.coordinates.addElement(CoordGraph::Flatten(), {n_thr_x, i_thr_x}, {i_mac_x});
                graph.coordinates.addElement(CoordGraph::Flatten(), {n_thr_y, i_thr_y}, {i_mac_y});

                graph.coordinates.addElement(CoordGraph::PassThrough(), {workitem_x}, {n_thr_x});
                graph.coordinates.addElement(CoordGraph::PassThrough(), {workitem_y}, {n_thr_y});
            }
            break;

            case MemoryType::WAVE:
                storeWaveMacroTile(graph,
                                   mac_tile,
                                   store_tag,
                                   i_mac_x,
                                   i_mac_y,
                                   workitem,
                                   user_tag,
                                   wavefrontSize,
                                   wavetilesPerWorkgroup);
                break;

            default:
                Throw<FatalError>("Store : MacroTile memory type not supported yet.");
            }
        }

        void addConnectionsMultiply(KernelHypergraph& graph, int waveMult)
        {
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::LowerTileVisitor::addConnectionsMultiply(): Multiply({})", waveMult);

            auto loads = graph.control.getOutputNodeIndices<ControlHypergraph::Body>(waveMult)
                             .to<std::vector>();
            AssertFatal(loads.size() == 2, "Multiply op needs two operands");
            auto loadA = graph.control.getElement(loads[0]);
            auto loadB = graph.control.getElement(loads[1]);

            int sourceA_tag = -1, sourceB_tag = -1;
            // LoadTiled A
            if(isOperation<ControlHypergraph::LoadTiled>(loadA))
            {
                auto userA_tag = graph.mapper.get<CoordGraph::User>(loads[0]);
                AssertFatal(userA_tag > 0, "User dimension not found");
                graph.mapper.connect<CoordGraph::User>(waveMult, userA_tag, 0);
                sourceA_tag = userA_tag;
            }
            // LoadLDSTile A
            else if(isOperation<ControlHypergraph::LoadLDSTile>(loadA))
            {
                auto ldsA_tag = graph.mapper.get<CoordGraph::LDS>(loads[0]);
                AssertFatal(ldsA_tag > 0, "LDS dimension not found");
                graph.mapper.connect<CoordGraph::LDS>(waveMult, ldsA_tag, 0);
                sourceA_tag = ldsA_tag;
            }

            // LoadTiled B
            if(isOperation<ControlHypergraph::LoadTiled>(loadB))
            {
                auto userB_tag = graph.mapper.get<CoordGraph::User>(loads[1]);
                AssertFatal(userB_tag > 0, "User dimension not found");
                graph.mapper.connect<CoordGraph::User>(waveMult, userB_tag, 1);
                sourceB_tag = userB_tag;
            }
            // LoadLDSTile B
            else if(isOperation<ControlHypergraph::LoadLDSTile>(loadB))
            {
                auto ldsB_tag = graph.mapper.get<CoordGraph::LDS>(loads[1]);
                AssertFatal(ldsB_tag > 0, "LDS dimension not found");
                graph.mapper.connect<CoordGraph::LDS>(waveMult, ldsB_tag, 1);
                sourceB_tag = ldsB_tag;
            }

            AssertFatal(sourceA_tag > 0 && sourceB_tag > 0, "User or LDS dimensions not found");

            auto [waveA_tag, waveA] = graph.getDimension<CoordGraph::WaveTile>(loads[0]);
            auto [waveB_tag, waveB] = graph.getDimension<CoordGraph::WaveTile>(loads[1]);

            auto a
                = graph.coordinates
                      .getOutputNodeIndices(sourceA_tag, CoordGraph::isEdge<CoordGraph::DataFlow>)
                      .to<std::vector>();
            auto b
                = graph.coordinates
                      .getOutputNodeIndices(sourceB_tag, CoordGraph::isEdge<CoordGraph::DataFlow>)
                      .to<std::vector>();
            AssertFatal(a.size() == 1 && b.size() == 1, a.size(), b.size());

            // connections are: 0: lhs (A); 1: rhs (B); 2: dst (D)
            graph.mapper.connect<CoordGraph::MacroTile>(waveMult, a[0], 0);
            graph.mapper.connect<CoordGraph::MacroTile>(waveMult, b[0], 1);
            graph.mapper.connect<CoordGraph::WaveTile>(waveMult, waveA_tag, 0);
            graph.mapper.connect<CoordGraph::WaveTile>(waveMult, waveB_tag, 1);
        }
    }
}
