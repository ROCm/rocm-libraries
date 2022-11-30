#include "DataTypes/DataTypes.hpp"
#include "Graph/Hypergraph.hpp"
#include "KernelGraph/ControlHypergraph/ControlEdge.hpp"
#include "KernelGraph/ControlHypergraph/Operation.hpp"
#include "KernelGraph/CoordGraph/CoordinateHypergraph.hpp"
#include "KernelGraph/CoordGraph/Dimension.hpp"
#include "KernelGraph/CoordGraph/Dimension_fwd.hpp"
#include "KernelGraph/CoordGraph/Edge.hpp"
#include "KernelGraph/KernelHypergraph.hpp"
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Operations/Operations.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        namespace Expression = rocRoller::Expression;
        using namespace Expression;

        /*
         * Lower tile ops
         */

        struct LowerTileVisitor : public BaseGraphVisitor
        {
            using BaseGraphVisitor::visitEdge;
            using BaseGraphVisitor::visitOperation;

            LowerTileVisitor(std::shared_ptr<CommandParameters> params,
                             std::shared_ptr<Context>           context)
                : BaseGraphVisitor(context)
                , m_params(params)
                , m_kernel(context->kernel())
            {
            }

            virtual void visitEdge(KernelHypergraph&                     graph,
                                   KernelHypergraph const&               original,
                                   GraphReindexer&                       reindexer,
                                   int                                   tag,
                                   CoordGraph::ConstructMacroTile const& edge) override
            {
                // NOP: don't need this edge anymore
            }

            virtual void visitEdge(KernelHypergraph&                    graph,
                                   KernelHypergraph const&              original,
                                   GraphReindexer&                      reindexer,
                                   int                                  tag,
                                   CoordGraph::DestructMacroTile const& edge) override
            {
                // NOP: don't need this edge anymore
            }

            virtual void visitOperation(KernelHypergraph&                           graph,
                                        KernelHypergraph const&                     original,
                                        GraphReindexer&                             reindexer,
                                        int                                         tag,
                                        ControlHypergraph::TensorContraction const& op) override
            {
                copyOperation(graph, original, reindexer, tag);

                auto new_tag = reindexer.control.at(tag);
                auto new_op  = graph.control.getNode<ControlHypergraph::TensorContraction>(new_tag);
                new_op.a     = reindexer.coordinates.at(op.a);
                new_op.b     = reindexer.coordinates.at(op.b);
                graph.control.setElement(new_tag, new_op);
            }

            virtual void visitOperation(KernelHypergraph&                   graph,
                                        KernelHypergraph const&             original,
                                        GraphReindexer&                     reindexer,
                                        int                                 tag,
                                        ControlHypergraph::ElementOp const& op) override
            {
                copyOperation(graph, original, reindexer, tag);

                auto new_tag = reindexer.control.at(tag);
                auto new_op  = graph.control.getNode<ControlHypergraph::ElementOp>(new_tag);
                new_op.a     = op.a > 0 ? reindexer.coordinates.at(op.a) : op.a;
                new_op.b     = op.b > 0 ? reindexer.coordinates.at(op.b) : op.b;
                graph.control.setElement(new_tag, new_op);
            }

            void loadWaveMacroTile(KernelHypergraph&            graph,
                                   CoordGraph::MacroTile const& mac_tile,
                                   int                          load_tag,
                                   int                          i_mac_x,
                                   int                          i_mac_y,
                                   int                          workitem,
                                   int                          user_tag)
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
                uint wfs          = m_context->kernel()->wavefront_size();
                uint num_vgpr     = num_elements / wfs;

                auto wavefront_size = wavefrontSize();

                auto lane = graph.coordinates.addElement(CoordGraph::Lane(wavefront_size, nullptr));
                auto vgpr
                    = graph.coordinates.addElement(CoordGraph::VGPR(literal(num_vgpr), nullptr));

                graph.coordinates.addElement(CoordGraph::Flatten(), {wave_x, wave_y}, {wave});
                graph.coordinates.addElement(CoordGraph::Flatten(), {wave, lane}, {workitem});

                graph.mapper.connect<CoordGraph::VGPR>(load_tag, vgpr);

                auto block_number = graph.coordinates.addElement(
                    CoordGraph::Adhoc("BlockNumber", literal(static_cast<uint>(wfs / m)), nullptr));
                auto block_index = graph.coordinates.addElement(
                    CoordGraph::Adhoc("BlockIndex", literal(static_cast<uint>(m)), nullptr));

                graph.coordinates.addElement(
                    CoordGraph::Flatten(), {block_number, block_index}, {lane});

                auto wavetilesPerWorkgroup = m_params->getWaveTilesPerWorkgroup();
                auto jammed_wavetile_x
                    = graph.coordinates.addElement(CoordGraph::JammedWaveTileNumber(
                        0, literal(wavetilesPerWorkgroup[0]), literal(1)));
                auto jammed_wavetile_y
                    = graph.coordinates.addElement(CoordGraph::JammedWaveTileNumber(
                        1, literal(wavetilesPerWorkgroup[1]), literal(1)));
                graph.mapper.connect<CoordGraph::JammedWaveTileNumber>(
                    load_tag, jammed_wavetile_x, 0);
                graph.mapper.connect<CoordGraph::JammedWaveTileNumber>(
                    load_tag, jammed_wavetile_y, 1);

                switch(wave_tile.layout)
                {
                case LayoutType::MATRIX_A:
                {
                    // TODO: not necessary here, but used for lookup in generator
                    graph.coordinates.addElement(
                        CoordGraph::Flatten(), {i_wave_y, i_wave_x}, {wave_tile_tag});
                    graph.coordinates.addElement(
                        CoordGraph::Tile(), {i_wave_y}, {block_number, vgpr});
                    graph.coordinates.addElement(
                        CoordGraph::PassThrough(), {i_wave_x}, {block_index});

                    if(wavetilesPerWorkgroup[0] > 1)
                        graph.coordinates.addElement(
                            CoordGraph::Tile(), {n_wave_x}, {wave_x, jammed_wavetile_x});
                    else
                        graph.coordinates.addElement(
                            CoordGraph::PassThrough(), {n_wave_x}, {wave_x});
                }
                break;

                case LayoutType::MATRIX_B:
                {
                    // TODO: not necessary here, but used for lookup in generator
                    graph.coordinates.addElement(
                        CoordGraph::Flatten(), {i_wave_x, i_wave_y}, {wave_tile_tag});
                    graph.coordinates.addElement(
                        CoordGraph::Tile(), {i_wave_x}, {block_number, vgpr});
                    graph.coordinates.addElement(
                        CoordGraph::PassThrough(), {i_wave_y}, {block_index});

                    if(wavetilesPerWorkgroup[1] > 1)
                        graph.coordinates.addElement(
                            CoordGraph::Tile(), {n_wave_y}, {wave_y, jammed_wavetile_y});
                    else
                        graph.coordinates.addElement(
                            CoordGraph::PassThrough(), {n_wave_y}, {wave_y});
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

                    graph.coordinates.addElement(
                        CoordGraph::Tile(), {i_wave_x}, {row_block, i_vblk});
                    graph.coordinates.addElement(
                        CoordGraph::Tile(), {i_wave_y}, {col_block, i_lblk});

                    graph.coordinates.addElement(
                        CoordGraph::Flatten(), {row_block, col_block}, {block});
                    graph.coordinates.addElement(CoordGraph::Tile(), {block}, {n_vblk, n_lblk});

                    graph.coordinates.addElement(CoordGraph::Flatten(), {n_vblk, i_vblk}, {vgpr});
                    graph.coordinates.addElement(CoordGraph::Flatten(), {n_lblk, i_lblk}, {lane});

                    if(wavetilesPerWorkgroup[0] > 1)
                        graph.coordinates.addElement(
                            CoordGraph::Tile(), {n_wave_x}, {wave_x, jammed_wavetile_x});
                    else
                        graph.coordinates.addElement(
                            CoordGraph::PassThrough(), {n_wave_x}, {wave_x});

                    if(wavetilesPerWorkgroup[1] > 1)
                        graph.coordinates.addElement(
                            CoordGraph::Tile(), {n_wave_y}, {wave_y, jammed_wavetile_y});
                    else
                        graph.coordinates.addElement(
                            CoordGraph::PassThrough(), {n_wave_y}, {wave_y});
                }
                break;

                default:
                    Throw<FatalError>("Not implemented yet.");
                }
            }

            void loadMacroTile(KernelHypergraph& graph,
                               int               load_tag,
                               int               user_tag,
                               int               mac_tile_tag,
                               std::vector<int>& sdim)
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

                auto wavefront_size = wavefrontSize();

                AssertRecoverable(mac_tile.rank == 2, "Rank /= 2 not implemented yet.");

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

                auto workitem = graph.coordinates.addElement(CoordGraph::Workitem(0));

                graph.coordinates.addElement(
                    CoordGraph::Flatten(), {i_mac_x, i_mac_y}, {mac_tile_tag});

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

                    auto thr_tile     = CoordGraph::ThreadTile(mac_tile.subTileSizes);
                    auto thr_tile_tag = graph.coordinates.addElement(thr_tile);

                    auto n_thr_x = graph.coordinates.addElement(thr_tile.tileNumber(0));
                    auto n_thr_y = graph.coordinates.addElement(thr_tile.tileNumber(1));
                    auto i_thr_x = graph.coordinates.addElement(thr_tile.tileIndex(0));
                    auto i_thr_y = graph.coordinates.addElement(thr_tile.tileIndex(1));

                    graph.mapper.connect<CoordGraph::ThreadTileIndex>(load_tag, i_thr_x, 0);
                    graph.mapper.connect<CoordGraph::ThreadTileIndex>(load_tag, i_thr_y, 1);

                    graph.coordinates.addElement(
                        CoordGraph::Join(), {i_thr_x, i_thr_y}, {thr_tile_tag});
                    graph.coordinates.addElement(CoordGraph::Tile(), {i_mac_x}, {n_thr_x, i_thr_x});
                    graph.coordinates.addElement(CoordGraph::Tile(), {i_mac_y}, {n_thr_y, i_thr_y});

                    graph.coordinates.addElement(
                        CoordGraph::PassThrough(), {workitem}, {i_thr_x, i_thr_y});
                    graph.coordinates.addElement(
                        CoordGraph::PassThrough(), {n_thr_x}, {workitem_x});
                    graph.coordinates.addElement(
                        CoordGraph::PassThrough(), {n_thr_y}, {workitem_y});

                    // User -> DataFlow() -> LDS gets added in addLDSOps
                }
                break;

                case MemoryType::WAVE:
                    loadWaveMacroTile(
                        graph, mac_tile, load_tag, i_mac_x, i_mac_y, workitem, user_tag);
                    break;

                default:
                    Throw<FatalError>("Load : MacroTile memory type not supported yet.");
                }
            }

            virtual void visitOperation(KernelHypergraph&                   graph,
                                        KernelHypergraph const&             original,
                                        GraphReindexer&                     reindexer,
                                        int                                 tag,
                                        ControlHypergraph::LoadTiled const& oload) override
            {
                auto logger = rocRoller::Log::getLogger();
                logger->debug("KernelGraph::LowerTileVisitor::LoadTiled({})", tag);

                auto original_user     = original.mapper.get<CoordGraph::User>(tag);
                auto original_mac_tile = original.mapper.get<CoordGraph::MacroTile>(tag);
                auto user              = reindexer.coordinates.at(original_user);
                auto mac_tile          = reindexer.coordinates.at(original_mac_tile);

                auto sdims
                    = original.coordinates
                          .getInputNodeIndices(original_mac_tile,
                                               CoordGraph::isEdge<CoordGraph::ConstructMacroTile>)
                          .to<std::vector>();
                for(int i = 0; i < sdims.size(); i++)
                    sdims[i] = reindexer.coordinates.at(sdims[i]);

                copyOperation(graph, original, reindexer, tag);

                auto load = reindexer.control.at(tag);
                loadMacroTile(graph, load, user, mac_tile, sdims);
            }

            void storeWaveMacroTile(KernelHypergraph&      graph,
                                    CoordGraph::MacroTile& mac_tile,
                                    int                    store_tag,
                                    int                    i_mac_x,
                                    int                    i_mac_y,
                                    int                    workitem,
                                    int                    user_tag)
            {
                AssertFatal(mac_tile.layoutType == LayoutType::MATRIX_ACCUMULATOR,
                            "Store must be from accumulator.");

                auto wave_tile
                    = CoordGraph::WaveTile(mac_tile.subTileSizes, LayoutType::MATRIX_ACCUMULATOR);
                auto wave_tile_tag = graph.coordinates.addElement(wave_tile);

                uint num_elements = wave_tile.sizes[0] * wave_tile.sizes[1];
                uint wfs          = m_context->kernel()->wavefront_size();
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

                graph.coordinates.addElement(
                    CoordGraph::Join(), {i_wave_x, i_wave_y}, {wave_tile_tag});

                auto wavefront_size = wavefrontSize();
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

                graph.coordinates.addElement(
                    CoordGraph::Flatten(), {n_wave_x, i_wave_x}, {i_mac_x});
                graph.coordinates.addElement(
                    CoordGraph::Flatten(), {n_wave_y, i_wave_y}, {i_mac_y});

                auto wave_x = graph.coordinates.addElement(CoordGraph::Wavefront(0));
                auto wave_y = graph.coordinates.addElement(CoordGraph::Wavefront(1));
                auto wave   = graph.coordinates.addElement(CoordGraph::Wavefront(-1));

                graph.coordinates.addElement(CoordGraph::Tile(), {wave}, {wave_x, wave_y});

                auto lane
                    = graph.coordinates.addElement(CoordGraph::Lane(wavefront_size, unit_stride));
                auto vgpr = graph.coordinates.addElement(
                    CoordGraph::VGPR(literal(num_agpr), unit_stride));

                graph.mapper.connect<CoordGraph::WaveTile>(store_tag, wave_tile_tag);
                graph.mapper.connect<CoordGraph::VGPRBlockNumber>(store_tag, n_vblk);
                graph.mapper.connect<CoordGraph::VGPRBlockIndex>(store_tag, i_vblk);
                graph.mapper.connect<CoordGraph::VGPR>(store_tag, vgpr);

                graph.coordinates.addElement(CoordGraph::Tile(), {vgpr}, {n_vblk, i_vblk});
                graph.coordinates.addElement(CoordGraph::Tile(), {lane}, {n_lblk, i_lblk});
                graph.coordinates.addElement(CoordGraph::Flatten(), {n_vblk, n_lblk}, {block});
                graph.coordinates.addElement(CoordGraph::Tile(), {block}, {row_block, col_block});

                auto wavetilesPerWorkgroup = m_params->getWaveTilesPerWorkgroup();
                auto jammed_wavetile_x
                    = graph.coordinates.addElement(CoordGraph::JammedWaveTileNumber(
                        0, literal(wavetilesPerWorkgroup[0]), literal(1)));
                auto jammed_wavetile_y
                    = graph.coordinates.addElement(CoordGraph::JammedWaveTileNumber(
                        1, literal(wavetilesPerWorkgroup[1]), literal(1)));
                graph.mapper.connect<CoordGraph::JammedWaveTileNumber>(
                    store_tag, jammed_wavetile_x, 0);
                graph.mapper.connect<CoordGraph::JammedWaveTileNumber>(
                    store_tag, jammed_wavetile_y, 1);

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

                graph.coordinates.addElement(
                    CoordGraph::Flatten(), {row_block, i_vblk}, {i_wave_x});
                graph.coordinates.addElement(
                    CoordGraph::Flatten(), {col_block, i_lblk}, {i_wave_y});

                graph.coordinates.addElement(CoordGraph::Tile(), {workitem}, {wave, lane});
            }

            void storeMacroTile(KernelHypergraph& graph,
                                int               store_tag,
                                int               user_tag,
                                int               mac_tile_tag,
                                std::vector<int>& sdims)
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

                auto workgroup_x    = graph.coordinates.addElement(CoordGraph::Workgroup(0));
                auto workgroup_y    = graph.coordinates.addElement(CoordGraph::Workgroup(1));
                auto workgroupSizes = m_kernel->workgroupSize();

                auto workitem = graph.coordinates.addElement(
                    CoordGraph::Workitem(0, literal(workgroupSizes.at(0))));

                graph.coordinates.addElement(
                    CoordGraph::Flatten(), {i_mac_x, i_mac_y}, {mac_tile_tag});

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
                    graph.coordinates.addElement(
                        CoordGraph::Flatten(), {n_thr_x, i_thr_x}, {i_mac_x});
                    graph.coordinates.addElement(
                        CoordGraph::Flatten(), {n_thr_y, i_thr_y}, {i_mac_y});

                    graph.coordinates.addElement(
                        CoordGraph::PassThrough(), {workitem}, {i_thr_x, i_thr_y});
                    graph.coordinates.addElement(
                        CoordGraph::PassThrough(), {workitem_x}, {n_thr_x});
                    graph.coordinates.addElement(
                        CoordGraph::PassThrough(), {workitem_y}, {n_thr_y});
                }
                break;

                case MemoryType::WAVE:
                    storeWaveMacroTile(
                        graph, mac_tile, store_tag, i_mac_x, i_mac_y, workitem, user_tag);
                    break;

                default:
                    Throw<FatalError>("Store : MacroTile memory type not supported yet.");
                }
            }

            virtual void visitOperation(KernelHypergraph&                    graph,
                                        KernelHypergraph const&              original,
                                        GraphReindexer&                      reindexer,
                                        int                                  tag,
                                        ControlHypergraph::StoreTiled const& ostore) override
            {
                rocRoller::Log::getLogger()->debug("KernelGraph::LowerTileVisitor::StoreTiled({})",
                                                   tag);

                auto original_user     = original.mapper.get<CoordGraph::User>(tag);
                auto original_mac_tile = original.mapper.get<CoordGraph::MacroTile>(tag);
                auto user              = reindexer.coordinates.at(original_user);
                auto mac_tile          = reindexer.coordinates.at(original_mac_tile);

                auto sdims
                    = original.coordinates
                          .getOutputNodeIndices(original_mac_tile,
                                                CoordGraph::isEdge<CoordGraph::DestructMacroTile>)
                          .to<std::vector>();
                for(int i = 0; i < sdims.size(); i++)
                    sdims[i] = reindexer.coordinates.at(sdims[i]);

                copyOperation(graph, original, reindexer, tag);

                auto store = reindexer.control.at(tag);
                storeMacroTile(graph, store, user, mac_tile, sdims);
            }

        private:
            std::shared_ptr<AssemblyKernel>    m_kernel;
            std::shared_ptr<CommandParameters> m_params;
        };

        // Add LDSLoad and LDSStore nodes following a LoadTiled node within
        // the control graph, if the tile has a memory type of LDS.
        void addLDSOps(KernelHypergraph& graph, std::shared_ptr<Context> context, int tag)
        {
            auto user_tag = graph.mapper.get<CoordGraph::User>(tag);
            auto tile_tag = graph.mapper.get<CoordGraph::MacroTile>(tag);
            auto tile     = graph.coordinates.getNode<CoordGraph::MacroTile>(tile_tag);

            if(tile.memoryType == MemoryType::LDS)
            {
                auto lds = graph.coordinates.addElement(CoordGraph::LDS());

                auto storeLDS = graph.control.addElement(ControlHypergraph::StoreLDSTile());
                auto barrier  = graph.control.addElement(ControlHypergraph::Barrier());
                auto loadLDS  = graph.control.addElement(ControlHypergraph::LoadLDSTile());

                graph.mapper.connect<CoordGraph::LDS>(storeLDS, lds);
                graph.mapper.connect<CoordGraph::MacroTile>(storeLDS, tile_tag);

                graph.mapper.connect<CoordGraph::User>(loadLDS, user_tag);
                graph.mapper.connect<CoordGraph::LDS>(loadLDS, lds);
                graph.mapper.connect<CoordGraph::MacroTile>(loadLDS, tile_tag);

                // Add an edge to the coordinate graph to index into the LDS allocation.
                auto workgroupSizes = context->kernel()->workgroupSize();

                auto workitem_x = graph.coordinates.addElement(
                    CoordGraph::Workitem(0, literal(workgroupSizes.at(0))));
                auto workitem_y = graph.coordinates.addElement(
                    CoordGraph::Workitem(1, literal(workgroupSizes.at(1))));

                graph.coordinates.addElement(CoordGraph::DataFlow(), {user_tag}, {tile_tag});
                graph.coordinates.addElement(CoordGraph::DataFlow(), {tile_tag}, {lds});
                graph.coordinates.addElement(
                    CoordGraph::Flatten(), {workitem_x, workitem_y}, {lds});

                // Find all edges leaving from load. Those should be changed to leave from loadLDS.
                auto outgoing_edges = graph.control.getNeighbours<Graph::Direction::Downstream>(tag)
                                          .to<std::vector>();
                for(auto e : outgoing_edges)
                {
                    auto elem = graph.control.getElement(e);
                    auto dst  = graph.control.getNeighbours<Graph::Direction::Downstream>(e)
                                   .to<std::vector>();
                    graph.control.deleteElement(e);
                    graph.control.addElement(e, elem, std::vector<int>{loadLDS}, dst);
                }

                graph.control.addElement(ControlHypergraph::Sequence(), {tag}, {storeLDS});
                graph.control.addElement(ControlHypergraph::Sequence(), {storeLDS}, {barrier});
                graph.control.addElement(ControlHypergraph::Sequence(), {barrier}, {loadLDS});
            }
        }

        /**
         * Lower rank-2 TensorContraction into a matrix multiply.
         */
        void lowerMatrixMultiply(KernelHypergraph&                  graph,
                                 int                                tag,
                                 int                                a,
                                 int                                b,
                                 int                                d,
                                 std::shared_ptr<CommandParameters> params,
                                 std::shared_ptr<Context>           context)
        {
            rocRoller::Log::getLogger()->debug("KernelGraph::matrixMultiply() {}", d);

            auto macrotile_a = graph.coordinates.getNode<CoordGraph::MacroTile>(a);
            auto macrotile_b = graph.coordinates.getNode<CoordGraph::MacroTile>(b);

            // get tensor contraction operands
            auto parents = graph.control.parentNodes(tag).to<std::vector>();
            AssertFatal(parents.size() == 2);
            auto operandA = parents[0];
            auto operandB = parents[1];
            AssertFatal(a == graph.mapper.get<CoordGraph::MacroTile>(operandA));
            AssertFatal(b == graph.mapper.get<CoordGraph::MacroTile>(operandB));

            auto reachable_from_tensor
                = graph.control.depthFirstVisit(tag).to<std::unordered_set>();

            // find loadtiled ops that lead to the tensor contraction op
            // find storetiled ops that follow the tensor contraction op
            // find elementOps that follow the tensor contraction op
            std::vector<int> loadA;
            std::vector<int> loadB;
            std::vector<int> stores;
            std::vector<int> elemops;
            for(auto const index : graph.control.getNodes())
            {
                auto elem = graph.control.getElement(index);
                visit(rocRoller::overloaded{
                          [&](auto op) {},
                          [&](ControlHypergraph::LoadTiled const& load) {
                              auto reachable_from_load
                                  = graph.control.depthFirstVisit(index).to<std::unordered_set>();
                              if(reachable_from_load.find(operandA) != reachable_from_load.end())
                                  loadA.push_back(index);
                              else if(reachable_from_load.find(operandB)
                                      != reachable_from_load.end())
                                  loadB.push_back(index);
                          },
                          [&](ControlHypergraph::StoreTiled const& store) {
                              if(reachable_from_tensor.find(index) != reachable_from_tensor.end())
                                  stores.push_back(index);
                          },
                          [&](ControlHypergraph::ElementOp const& op) {
                              if(reachable_from_tensor.find(index) != reachable_from_tensor.end())
                                  elemops.push_back(index);
                          }},
                      std::get<ControlHypergraph::Operation>(elem));
            }
            AssertFatal(loadA.size() == 1 && loadB.size() == 1);
            AssertFatal(stores.size() <= 1);
            auto storeD = !stores.empty() ? stores[0] : -1;

            // find kernel
            auto root = graph.control.roots().to<std::vector>();
            AssertFatal(root.size() == 1, "More than one Kernel node not supported");
            auto kernel = root[0];

            graph.control.deleteElement<ControlHypergraph::Body>(std::vector<int>{kernel},
                                                                 std::vector<int>{loadA[0]});
            graph.control.deleteElement<ControlHypergraph::Body>(std::vector<int>{kernel},
                                                                 std::vector<int>{loadB[0]});

            auto matK = (graph.coordinates.getNode<CoordGraph::SubDimension>(
                             graph.mapper.get<CoordGraph::SubDimension>(loadA[0], 1)))
                            .size;

            auto macK = literal(static_cast<uint>(macrotile_a.sizes[1])); // M x K

            auto [K, forK] = rangeFor(graph, matK / macK); // num of loop iterations : matK / macK

            // remove passthrough between A column block and y-workgroup
            auto a_tilenum_y   = graph.mapper.get<CoordGraph::MacroTileNumber>(loadA[0], 1);
            auto a_workgroup_y = graph.mapper.get<CoordGraph::Workgroup>(loadA[0], 1);
            graph.coordinates.deleteElement(std::vector<int>{a_tilenum_y},
                                            std::vector<int>{a_workgroup_y},
                                            CoordGraph::isEdge<CoordGraph::PassThrough>);
            graph.coordinates.deleteElement(a_workgroup_y);

            // remove passthrough between B row block and x-workgroup
            auto b_tilenum_x   = graph.mapper.get<CoordGraph::MacroTileNumber>(loadB[0], 0);
            auto b_workgroup_x = graph.mapper.get<CoordGraph::Workgroup>(loadB[0], 0);
            graph.coordinates.deleteElement(std::vector<int>{b_tilenum_x},
                                            std::vector<int>{b_workgroup_x},
                                            CoordGraph::isEdge<CoordGraph::PassThrough>);
            graph.coordinates.deleteElement(b_workgroup_x);

            // A row block is x-workgroup, column block is for loop index
            graph.coordinates.addElement(CoordGraph::PassThrough(), {a_tilenum_y}, {K});

            // B row block is for loop index, column block is y-workgroup
            graph.coordinates.addElement(CoordGraph::PassThrough(), {b_tilenum_x}, {K});

            // TODO : create helper functions to make this lowering modular and readable.
            auto waveMult = graph.control.addElement(ControlHypergraph::Multiply());

            // initialise accumulator
            auto [userA_tag, userA] = graph.getDimension<CoordGraph::User>(loadA[0]);
            auto [userB_tag, userB] = graph.getDimension<CoordGraph::User>(loadB[0]);
            auto [waveA_tag, waveA] = graph.getDimension<CoordGraph::WaveTile>(loadA[0]);
            auto [waveB_tag, waveB] = graph.getDimension<CoordGraph::WaveTile>(loadB[0]);

            // connections are: 0: lhs (A); 1: rhs (B); 2: dst (D)
            graph.mapper.connect<CoordGraph::User>(waveMult, userA_tag, 0);
            graph.mapper.connect<CoordGraph::User>(waveMult, userB_tag, 1);
            graph.mapper.connect<CoordGraph::MacroTile>(waveMult, a, 0);
            graph.mapper.connect<CoordGraph::MacroTile>(waveMult, b, 1);
            graph.mapper.connect<CoordGraph::WaveTile>(waveMult, waveA_tag, 0);
            graph.mapper.connect<CoordGraph::WaveTile>(waveMult, waveB_tag, 1);
            graph.mapper.connect<CoordGraph::MacroTile>(waveMult, d, 2);

            uint num_elements = waveA.sizes[0] * waveB.sizes[1];
            uint wfs          = context->kernel()->wavefront_size();
            uint num_agpr     = num_elements / wfs; // number of output registers per thread

            auto initD = graph.control.addElement(ControlHypergraph::Assign{
                Register::Type::Accumulator, literal(0.f), num_agpr, false});

            graph.control.addElement(ControlHypergraph::Initialize(), {forK}, {initD});
            graph.control.addElement(ControlHypergraph::Body(), {forK}, {waveMult});
            graph.control.addElement(ControlHypergraph::Body(), {waveMult}, {loadA[0]});
            graph.control.addElement(ControlHypergraph::Body(), {waveMult}, {loadB[0]});

            graph.mapper.connect<CoordGraph::MacroTile>(initD, d);

            // connect ops after contraction to for loop, remove contraction and its incoming edges
            auto tensor_outgoing_edges
                = graph.control.getNeighbours<Graph::Direction::Downstream>(tag).to<std::vector>();
            for(auto const e : tensor_outgoing_edges)
            {
                auto elem = graph.control.getElement(e);
                auto dst  = graph.control.getNeighbours<Graph::Direction::Downstream>(e)
                               .to<std::vector>();
                graph.control.deleteElement(e);
                graph.control.addElement(e, elem, std::vector<int>{forK}, dst);
            }
            auto tensor_incoming_edges
                = graph.control.getNeighbours<Graph::Direction::Upstream>(tag).to<std::vector>();
            for(auto const e : tensor_incoming_edges)
                graph.control.deleteElement(e);
            graph.control.deleteElement(tag);

            // Add loops to iterate over wavetiles within a wavefront
            auto wavetilesPerWorkgroup = params->getWaveTilesPerWorkgroup();
            AssertFatal(wavetilesPerWorkgroup.size() > 1);

            auto WaveTilesX = -1, WaveTilesY = -1, forWaveTilesX = -1, forWaveTilesY = -1;

            if(wavetilesPerWorkgroup[0] > 1 || wavetilesPerWorkgroup[1] > 1)
            {
                if(wavetilesPerWorkgroup[0] > 1)
                {
                    std::tie(WaveTilesX, forWaveTilesX)
                        = rangeFor(graph, literal(wavetilesPerWorkgroup[0]));
                }
                if(wavetilesPerWorkgroup[1] > 1)
                {
                    std::tie(WaveTilesY, forWaveTilesY)
                        = rangeFor(graph, literal(wavetilesPerWorkgroup[1]));
                }

                // find other loadtiled ops from kernel that lead to elemops
                auto kernel_outputs = graph.control.childNodes(kernel).to<std::vector>();
                std::vector<int> otherLoads;
                std::vector<int> otherOps;
                for(auto const index : kernel_outputs)
                {
                    auto elem = graph.control.getElement(index);
                    visit(
                        rocRoller::overloaded{
                            [&](auto op) { otherOps.push_back(index); },
                            [&](ControlHypergraph::LoadTiled const& load) {
                                auto reachable_from_load
                                    = graph.control.depthFirstVisit(index).to<std::unordered_set>();
                                for(auto const& elemop : elemops)
                                {
                                    if(reachable_from_load.find(elemop)
                                       != reachable_from_load.end())
                                    {
                                        otherLoads.push_back(index);
                                        break;
                                    }
                                }
                            }},
                        std::get<ControlHypergraph::Operation>(elem));
                }
                AssertFatal(otherLoads.size() == 1);

                // Add edges from inner loop to some kernel outputs : forK and otherLoads
                // need to leave other nodes attached with kernel
                // ex: loadtiled ops that don't lead to elemops
                // ex : loadVGPRs for alpha and beta in GEMM
                int lowerLoop, upperLoop;
                if(wavetilesPerWorkgroup[0] > 1 && wavetilesPerWorkgroup[1] > 1)
                {
                    upperLoop = forWaveTilesX;
                    lowerLoop = forWaveTilesY;
                }
                else if(wavetilesPerWorkgroup[0] > 1)
                {
                    lowerLoop = forWaveTilesX;
                    upperLoop = forWaveTilesX;
                }
                else
                {
                    lowerLoop = forWaveTilesY;
                    upperLoop = forWaveTilesY;
                }

                graph.control.addElement(ControlHypergraph::Body(), {lowerLoop}, {forK});

                for(auto const index : otherLoads)
                {
                    auto e = graph.control.getNeighbours<Graph::Direction::Upstream>(index)
                                 .to<std::vector>()[0];
                    auto elem = graph.control.getElement(e);
                    graph.control.deleteElement(e);
                    graph.control.addElement(
                        e, elem, std::vector<int>{lowerLoop}, std::vector<int>{index});
                }

                for(auto const index : otherOps)
                {
                    auto e = graph.control.getNeighbours<Graph::Direction::Downstream>(index)
                                 .to<std::vector>()[0];
                    auto elem = graph.control.getElement(e);
                    graph.control.deleteElement(e);
                    graph.control.addElement(
                        e, elem, std::vector<int>{index}, std::vector<int>{upperLoop});
                }

                // make nested inner loops (forWaveTilesY inside the forWaveTilesX loop)
                if(wavetilesPerWorkgroup[0] > 1 && wavetilesPerWorkgroup[1] > 1)
                    graph.control.addElement(
                        ControlHypergraph::Body(), {forWaveTilesX}, {forWaveTilesY});

                // Add edges from Kernel to outer loop
                graph.control.addElement(ControlHypergraph::Body(), {kernel}, {upperLoop});

                // Add edges from all JammedWaveTileNumber dimensions to the for loop
                if(WaveTilesX != -1)
                {
                    auto a_jammed_x
                        = graph.mapper.get<CoordGraph::JammedWaveTileNumber>(loadA[0], 0);
                    graph.coordinates.addElement(
                        CoordGraph::PassThrough(), {a_jammed_x}, {WaveTilesX});
                    auto b_jammed_x
                        = graph.mapper.get<CoordGraph::JammedWaveTileNumber>(loadB[0], 0);
                    graph.coordinates.addElement(
                        CoordGraph::PassThrough(), {b_jammed_x}, {WaveTilesX});
                    auto c_jammed_x
                        = graph.mapper.get<CoordGraph::JammedWaveTileNumber>(otherLoads[0], 0);
                    graph.coordinates.addElement(
                        CoordGraph::PassThrough(), {c_jammed_x}, {WaveTilesX});

                    if(storeD > 0)
                    {
                        auto d_jammed_x
                            = graph.mapper.get<CoordGraph::JammedWaveTileNumber>(storeD, 0);
                        graph.coordinates.addElement(
                            CoordGraph::PassThrough(), {WaveTilesX}, {d_jammed_x});
                    }
                }
                if(WaveTilesY != -1)
                {
                    auto a_jammed_y
                        = graph.mapper.get<CoordGraph::JammedWaveTileNumber>(loadA[0], 1);
                    graph.coordinates.addElement(
                        CoordGraph::PassThrough(), {a_jammed_y}, {WaveTilesY});
                    auto b_jammed_y
                        = graph.mapper.get<CoordGraph::JammedWaveTileNumber>(loadB[0], 1);
                    graph.coordinates.addElement(
                        CoordGraph::PassThrough(), {b_jammed_y}, {WaveTilesY});
                    auto c_jammed_y
                        = graph.mapper.get<CoordGraph::JammedWaveTileNumber>(otherLoads[0], 1);
                    graph.coordinates.addElement(
                        CoordGraph::PassThrough(), {c_jammed_y}, {WaveTilesY});

                    if(storeD > 0)
                    {
                        auto d_jammed_y
                            = graph.mapper.get<CoordGraph::JammedWaveTileNumber>(storeD, 1);
                        graph.coordinates.addElement(
                            CoordGraph::PassThrough(), {WaveTilesY}, {d_jammed_y});
                    }
                }
            }
            else
                graph.control.addElement(ControlHypergraph::Body(), {kernel}, {forK});
        }

        KernelHypergraph addComputeIndexOperations(KernelHypergraph const& original)
        {
            auto kgraph = original;
            auto kernel = *kgraph.control.roots().begin();

            // MATRIX_A and MATRIX_B loads within a ForLoop
            auto multiplies
                = kgraph.control.getNodes<ControlHypergraph::Multiply>().to<std::vector>();
            AssertFatal(multiplies.size() <= 1, "More than one Multiply not supported yet.");
            if(multiplies.size() == 1)
            {
                auto forK = kgraph.control
                                .findNodes(
                                    multiplies[0],
                                    [&](int tag) -> bool {
                                        return isOperation<ControlHypergraph::ForLoopOp>(
                                            kgraph.control.getElement(tag));
                                    },
                                    Graph::Direction::Upstream)
                                .to<std::vector>();
                AssertFatal(forK.size() > 0, "Didn't find for loop.");
                auto loads = kgraph.control
                                 .findNodes(
                                     multiplies[0],
                                     [&](int tag) -> bool {
                                         return isOperation<ControlHypergraph::LoadTiled>(
                                             kgraph.control.getElement(tag));
                                     },
                                     Graph::Direction::Downstream)
                                 .to<std::vector>();
                AssertFatal(loads.size() == 2, "More than one Multiply not supported yet.");
                kgraph = addComputeIndexAB(kgraph, forK[0], loads[0], loads[1]);
            }

            // MATRIX_ACCUMULATOR loads anywhere
            auto loadAccums
                = kgraph.control
                      .findNodes(
                          kernel,
                          [&](int tag) -> bool {
                              auto load = kgraph.control.get<ControlHypergraph::LoadTiled>(tag);
                              if(load)
                              {
                                  auto [tile_tag, tile]
                                      = kgraph.getDimension<CoordGraph::MacroTile>(tag);
                                  if(tile.layoutType == LayoutType::MATRIX_ACCUMULATOR)
                                      return true;
                              }
                              return false;
                          },
                          Graph::Direction::Downstream)
                      .to<std::vector>();

            for(auto const tag : loadAccums)
            {
                kgraph = addComputeIndexC(kgraph, tag, tag, false);
            }

            // MATRIX_ACCUMULATOR stores anywhere
            auto storeAccums
                = kgraph.control
                      .findNodes(
                          kernel,
                          [&](int tag) -> bool {
                              auto load = kgraph.control.get<ControlHypergraph::StoreTiled>(tag);
                              if(load)
                              {
                                  auto [tile_tag, tile]
                                      = kgraph.getDimension<CoordGraph::MacroTile>(tag);
                                  if(tile.layoutType == LayoutType::MATRIX_ACCUMULATOR)
                                      return true;
                              }
                              return false;
                          },
                          Graph::Direction::Downstream)
                      .to<std::vector>();

            for(auto const tag : storeAccums)
            {
                kgraph = addComputeIndexC(kgraph, tag, tag, true);
            }

            return kgraph;
        }

        KernelHypergraph lowerTile(KernelHypergraph                   graph,
                                   std::shared_ptr<CommandParameters> params,
                                   std::shared_ptr<Context>           context)
        {
            TIMER(t, "KernelGraph::lowerTile");
            rocRoller::Log::getLogger()->debug("KernelGraph::lowerTile()");

            auto visitor = LowerTileVisitor(params, context);
            auto kgraph  = rewrite(graph, visitor);

            auto contractions
                = kgraph.control.getNodes<ControlHypergraph::TensorContraction>().to<std::vector>();
            AssertFatal(contractions.size() <= 1,
                        "More than one TensorContraction not supported yet.");

            if(contractions.size() == 1)
            {
                auto tag = contractions[0];
                auto op  = kgraph.control.getNode<ControlHypergraph::TensorContraction>(tag);
                auto macrotile_a = kgraph.coordinates.getNode<CoordGraph::MacroTile>(op.a);
                auto macrotile_b = kgraph.coordinates.getNode<CoordGraph::MacroTile>(op.b);
                auto d           = kgraph.mapper.get<CoordGraph::MacroTile>(tag);
                if(macrotile_a.rank == 2 && macrotile_b.rank == 2 && op.aDims == std::vector<int>{1}
                   && op.bDims == std::vector<int>{0})
                {
                    lowerMatrixMultiply(kgraph, tag, op.a, op.b, d, params, context);
                }
                else
                {
                    Throw<FatalError>("General contraction not implemented yet.");
                }
            }

            // Find WAVE tile load/store operations and add ComputeIndex operations
            kgraph = addComputeIndexOperations(kgraph);

            // Add LDS operations to the control graph following LoadTiled nodes.
            // This is done after the control graph is completly built to make
            // it easier to modify the edges coming into and out of the
            // original LoadTiled node.
            for(auto const& tag : kgraph.control.getNodes())
            {
                auto elem = kgraph.control.getElement(tag);
                visit(rocRoller::overloaded{[&](auto op) {},
                                            [&](ControlHypergraph::LoadTiled const& load) {
                                                addLDSOps(kgraph, context, tag);
                                            }},
                      std::get<ControlHypergraph::Operation>(elem));
            }
            return kgraph;
        }

    }
}
