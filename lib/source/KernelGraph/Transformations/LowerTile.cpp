#include "KernelGraph/ControlHypergraph/Operation.hpp"
#include "KernelGraph/CoordGraph/CoordinateHypergraph.hpp"
#include "KernelGraph/CoordGraph/Dimension.hpp"
#include "KernelGraph/CoordGraph/Dimension_fwd.hpp"
#include "KernelGraph/CoordGraph/Edge.hpp"
#include "KernelGraph/CoordinateTransform/Dimension.hpp"
#include "KernelGraph/CoordinateTransform/HyperGraph.hpp"
#include "KernelGraph/KernelHypergraph.hpp"
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Operations/Operations.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        using CoordinateTransform::MacroTile;

        using namespace CoordinateTransform;
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

            virtual void visitEdge(HyperGraph& coordGraph,
                                   ControlGraph::ControlGraph&,
                                   Location const& loc,
                                   DataFlow const& df) override
            {
                // Don't need DataFlow edges to/from MacroTile anymore
                bool drop = false;
                for(auto const& d : loc.srcDims)
                {
                    if(std::holds_alternative<MacroTile>(d))
                    {
                        drop = true;
                        break;
                    }
                }
                for(auto const& d : loc.dstDims)
                {
                    if(std::holds_alternative<MacroTile>(d))
                    {
                        drop = true;
                        break;
                    }
                }
                if(!drop)
                {
                    coordGraph.addEdge(loc.srcDims, loc.dstDims, df);
                }
            }

            void loadMacroTile(HyperGraph&                    coordGraph,
                               ControlGraph::ControlGraph&    controlGraph,
                               Location const&                loc,
                               ControlGraph::LoadTiled const& load,
                               User const&                    user,
                               MacroTile const&               mac_tile)
            {
                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::LowerTileVisitor::loadMacroTile(): size {} by {}",
                    mac_tile.sizes[0],
                    mac_tile.sizes[1]);

                auto wavefront_size = wavefrontSize();

                AssertRecoverable(mac_tile.rank == 2, "Rank /= 2 not implementd yet.");

                auto sdim_x = SubDimension(user.tag, 0);
                auto sdim_y = SubDimension(user.tag, 1);

                auto n_mac_x = mac_tile.tileNumber(0);
                auto n_mac_y = mac_tile.tileNumber(1);
                auto i_mac_x = mac_tile.tileIndex(0);
                auto i_mac_y = mac_tile.tileIndex(1);

                auto workgroup_x = Workgroup(mac_tile.tag, 0);
                auto workgroup_y = Workgroup(mac_tile.tag, 1);

                auto workitem   = Workitem(mac_tile.tag, 0);
                auto workitem_x = Workitem(mac_tile.tag, 0);
                auto workitem_y = Workitem(mac_tile.tag, 1);

                coordGraph.addEdge({i_mac_x, i_mac_y}, {mac_tile}, Flatten());

                coordGraph.addEdge({sdim_x}, {n_mac_x, i_mac_x}, Tile());
                coordGraph.addEdge({sdim_y}, {n_mac_y, i_mac_y}, Tile());

                coordGraph.addEdge({n_mac_x}, {workgroup_x}, PassThrough());
                coordGraph.addEdge({n_mac_y}, {workgroup_y}, PassThrough());

                switch(mac_tile.memoryType)
                {
                case MemoryType::VGPR:
                case MemoryType::LDS:
                {
                    auto thr_tile = ThreadTile(mac_tile.tag, mac_tile.subTileSizes);

                    auto n_thr_x = thr_tile.tileNumber(0);
                    auto n_thr_y = thr_tile.tileNumber(1);
                    auto i_thr_x = thr_tile.tileIndex(0);
                    auto i_thr_y = thr_tile.tileIndex(1);

                    coordGraph.addEdge({i_thr_x, i_thr_y}, {thr_tile}, Join());
                    coordGraph.addEdge({i_mac_x}, {n_thr_x, i_thr_x}, Tile());
                    coordGraph.addEdge({i_mac_y}, {n_thr_y, i_thr_y}, Tile());

                    coordGraph.addEdge({n_thr_x}, {workitem_x}, PassThrough());
                    coordGraph.addEdge({n_thr_y}, {workitem_y}, PassThrough());

                    if(mac_tile.memoryType == MemoryType::VGPR)
                    {
                        auto vgpr = VGPR(mac_tile.tag);
                        coordGraph.addEdge({user}, {vgpr}, DataFlow());
                    }

                    // User -> DataFlow() -> LDS gets added in addLDSOps
                }
                break;

                case MemoryType::WAVE:
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

                    auto wave_tile = WaveTile(mac_tile.tag, tileSize, mac_tile.layoutType);

                    auto n_wave_x = wave_tile.tileNumber(0);
                    auto n_wave_y = wave_tile.tileNumber(1);
                    auto i_wave_x = wave_tile.tileIndex(0);
                    auto i_wave_y = wave_tile.tileIndex(1);

                    coordGraph.addEdge({i_mac_x}, {n_wave_x, i_wave_x}, Tile());
                    coordGraph.addEdge({i_mac_y}, {n_wave_y, i_wave_y}, Tile());

                    auto wave_x = Wavefront(mac_tile.tag, 0);
                    auto wave_y = Wavefront(mac_tile.tag, 1);
                    auto wave   = Wavefront(mac_tile.tag, -1);

                    uint num_elements = product(tileSize);
                    uint wfs          = m_context->kernel()->wavefront_size();
                    uint num_vgpr     = num_elements / wfs;

                    auto lane = Lane(mac_tile.tag, wavefront_size, nullptr, false);
                    auto vgpr = VGPR(mac_tile.tag, literal(num_vgpr), nullptr, false);

                    coordGraph.addEdge({wave_x, wave_y}, {wave}, Flatten());
                    coordGraph.addEdge({wave, lane}, {workitem}, Flatten());

                    auto block_number = Adhoc(
                        "BlockNumber", mac_tile.tag, literal(static_cast<uint>(wfs / m)), nullptr);
                    auto block_index
                        = Adhoc("BlockIndex", mac_tile.tag, literal(static_cast<uint>(m)), nullptr);

                    coordGraph.addEdge({block_number, block_index}, {lane}, Flatten());

                    auto wavetilesPerWorkgroup = m_params->getWaveTilesPerWorkgroup();
                    auto jammed_wavetile_x     = JammedWaveTileNumber(
                        mac_tile.tag, 0, literal(wavetilesPerWorkgroup[0]), literal(1));
                    auto jammed_wavetile_y = JammedWaveTileNumber(
                        mac_tile.tag, 1, literal(wavetilesPerWorkgroup[1]), literal(1));

                    switch(wave_tile.layout)
                    {
                    case LayoutType::MATRIX_A:
                    {
                        coordGraph.addEdge(
                            {i_wave_y, i_wave_x},
                            {wave_tile},
                            Flatten()); // TODO: not necessary here, but used for lookup in generator
                        coordGraph.addEdge({i_wave_y}, {block_number, vgpr}, Tile());
                        coordGraph.addEdge({i_wave_x}, {block_index}, PassThrough());

                        if(wavetilesPerWorkgroup[0] > 1)
                            coordGraph.addEdge({n_wave_x}, {wave_x, jammed_wavetile_x}, Tile());
                        else
                            coordGraph.addEdge({n_wave_x}, {wave_x}, PassThrough());
                    }
                    break;
                    case LayoutType::MATRIX_B:
                    {
                        coordGraph.addEdge(
                            {i_wave_x, i_wave_y},
                            {wave_tile},
                            Flatten()); // TODO: not necessary here, but used for lookup in generator
                        coordGraph.addEdge({i_wave_x}, {block_number, vgpr}, Tile());
                        coordGraph.addEdge({i_wave_y}, {block_index}, PassThrough());

                        if(wavetilesPerWorkgroup[1] > 1)
                            coordGraph.addEdge({n_wave_y}, {wave_y, jammed_wavetile_y}, Tile());
                        else
                            coordGraph.addEdge({n_wave_y}, {wave_y}, PassThrough());
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

                        auto n_vblk = Adhoc("VGPRBlockNumber",
                                            wave_tile.tag,
                                            literal(num_vgpr / mts),
                                            unit_stride,
                                            true);
                        auto i_vblk = Adhoc(
                            "VGPRBlockIndex", wave_tile.tag, mfma_tile_size, unit_stride, true);
                        auto n_lblk = Adhoc("LANEBlockNumber",
                                            wave_tile.tag,
                                            literal(wfs / mts),
                                            unit_stride,
                                            true);
                        auto i_lblk = Adhoc(
                            "LANEBlockIndex", wave_tile.tag, mfma_tile_size, unit_stride, true);
                        auto block = Adhoc("LinearBlock",
                                           wave_tile.tag,
                                           literal(num_elements / 16u),
                                           unit_stride,
                                           true);
                        auto row_block
                            = Adhoc("RowBlock", wave_tile.tag, n_row_blocks, unit_stride, true);
                        auto col_block
                            = Adhoc("ColBlock", wave_tile.tag, n_col_blocks, unit_stride, true);

                        // ORDER?
                        coordGraph.addEdge({i_wave_x, i_wave_y}, {wave_tile}, Flatten());

                        coordGraph.addEdge({i_wave_x}, {row_block, i_vblk}, Tile());
                        coordGraph.addEdge({i_wave_y}, {col_block, i_lblk}, Tile());

                        coordGraph.addEdge({row_block, col_block}, {block}, Flatten());
                        coordGraph.addEdge({block}, {n_vblk, n_lblk}, Tile());

                        coordGraph.addEdge({n_vblk, i_vblk}, {vgpr}, Flatten());
                        coordGraph.addEdge({n_lblk, i_lblk}, {lane}, Flatten());

                        if(wavetilesPerWorkgroup[0] > 1)
                            coordGraph.addEdge({n_wave_x}, {wave_x, jammed_wavetile_x}, Tile());
                        else
                            coordGraph.addEdge({n_wave_x}, {wave_x}, PassThrough());

                        if(wavetilesPerWorkgroup[1] > 1)
                            coordGraph.addEdge({n_wave_y}, {wave_y, jammed_wavetile_y}, Tile());
                        else
                            coordGraph.addEdge({n_wave_y}, {wave_y}, PassThrough());

                        coordGraph.addEdge({wave, lane}, {workitem}, Flatten());
                    }
                    break;
                    default:
                        Throw<FatalError>("Not implemented yet.");
                    }

                    coordGraph.addEdge({user}, {vgpr}, DataFlow());
                }
                break;

                default:
                    Throw<FatalError>("Load : MacroTile memory type not supported yet.");
                }
            }

            virtual void visitEdge(HyperGraph&                 coordGraph,
                                   ControlGraph::ControlGraph& controlGraph,
                                   Location const&             loc,
                                   ConstructMacroTile const&   edge) override
            {
                // NOP: don't need this edge anymore
            }

            virtual void visitOperation(HyperGraph&                    coordGraph,
                                        ControlGraph::ControlGraph&    controlGraph,
                                        Location const&                loc,
                                        ControlGraph::LoadTiled const& load) override
            {
                auto logger = rocRoller::Log::getLogger();
                logger->debug("KernelGraph::LowerTileVisitor::LoadTiled() {}", load.tag);

                // For now, defer adding LoadTiled operations when
                // they are fed into a TensorContraction node.
                bool connect = true;
                auto outputs = loc.controlGraph.getOutputs(getTag(load));
                for(auto const& output : outputs)
                {
                    if(std::holds_alternative<ControlGraph::TensorContraction>(output))
                    {
                        logger->debug("KernelGraph::LowerTileVisitor::LoadTiled NOT CONNECTING {}",
                                      toString(load));
                        connect = false;
                    }
                }

                auto user = loc.coordGraph.getDimension(User(load.tag));
                auto tile = loc.coordGraph.getDimension(MacroTile(load.tag));

                loadMacroTile(coordGraph, controlGraph, loc, load, user, tile);

                if(connect)
                    BaseGraphVisitor::visitOperation(coordGraph, controlGraph, loc, load);
            }

            void storeMacroTile(HyperGraph&                     coordGraph,
                                ControlGraph::ControlGraph&     controlGraph,
                                Location const&                 loc,
                                ControlGraph::StoreTiled const& store,
                                User const&                     user,
                                MacroTile const&                mac_tile)
            {
                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::LowerTileVisitor::storeMacroTile(): size {} by {}",
                    mac_tile.sizes[0],
                    mac_tile.sizes[1]);

                std::vector<Dimension> sdims = {
                    SubDimension(store.tag, 0, true),
                    SubDimension(store.tag, 1, true),
                };

                AssertFatal(mac_tile.rank >= 0 && sdims.size() == (size_t)mac_tile.rank,
                            "Tensor size mismatch.");
                AssertFatal(mac_tile.rank == 2, "Rank != 2 not implemented yet.");

                auto wavefront_size = wavefrontSize();
                auto unit_stride    = literal(1u);

                auto n_mac_x = mac_tile.tileNumber(0, true);
                auto n_mac_y = mac_tile.tileNumber(1, true);
                auto i_mac_x = mac_tile.tileIndex(0, true);
                auto i_mac_y = mac_tile.tileIndex(1, true);

                auto workgroup_x    = Workgroup(mac_tile.tag, 0, true);
                auto workgroup_y    = Workgroup(mac_tile.tag, 1, true);
                auto workgroupSizes = m_kernel->workgroupSize();

                auto workitem   = Workitem(mac_tile.tag, 0, literal(workgroupSizes.at(0)), true);
                auto workitem_x = Workitem(mac_tile.tag, 0, literal(workgroupSizes.at(0)), true);
                auto workitem_y = Workitem(mac_tile.tag, 1, literal(workgroupSizes.at(1)), true);

                coordGraph.addEdge({i_mac_x, i_mac_y}, {mac_tile}, Flatten());

                coordGraph.addEdge({n_mac_x, i_mac_x}, {sdims[0]}, Flatten());
                coordGraph.addEdge({n_mac_y, i_mac_y}, {sdims[1]}, Flatten());

                coordGraph.addEdge({workgroup_x}, {n_mac_x}, PassThrough());
                coordGraph.addEdge({workgroup_y}, {n_mac_y}, PassThrough());

                switch(mac_tile.memoryType)
                {
                case MemoryType::VGPR:
                {
                    auto thr_tile = ThreadTile(mac_tile.tag, mac_tile.subTileSizes, true);
                    auto n_thr_x  = thr_tile.tileNumber(0, true);
                    auto n_thr_y  = thr_tile.tileNumber(1, true);
                    auto i_thr_x  = thr_tile.tileIndex(0, true);
                    auto i_thr_y  = thr_tile.tileIndex(1, true);

                    coordGraph.addEdge({thr_tile}, {i_thr_x, i_thr_y}, Split());
                    coordGraph.addEdge({n_thr_x, i_thr_x}, {i_mac_x}, Flatten());
                    coordGraph.addEdge({n_thr_y, i_thr_y}, {i_mac_y}, Flatten());
                    coordGraph.addEdge({workitem_x}, {n_thr_x}, PassThrough());
                    coordGraph.addEdge({workitem_y}, {n_thr_y}, PassThrough());
                }
                break;

                case MemoryType::WAVE:
                {
                    AssertFatal(mac_tile.layoutType == LayoutType::MATRIX_ACCUMULATOR,
                                "Store must be from accumulator.");

                    auto wave_tile = WaveTile(
                        mac_tile.tag, mac_tile.subTileSizes, LayoutType::MATRIX_ACCUMULATOR, true);

                    uint num_elements = wave_tile.sizes[0] * wave_tile.sizes[1];
                    uint wfs          = m_context->kernel()->wavefront_size();
                    uint num_agpr     = num_elements / wfs;

                    // MFMA accumulator tile size
                    uint mts            = 4u;
                    auto mfma_tile_size = literal(mts);

                    auto n_row_blocks = literal(wave_tile.sizes[0] / mts);
                    auto n_col_blocks = literal(wave_tile.sizes[1] / mts);

                    auto n_wave_x = wave_tile.tileNumber(0, true);
                    auto n_wave_y = wave_tile.tileNumber(1, true);
                    auto i_wave_x = wave_tile.tileIndex(0, true);
                    auto i_wave_y = wave_tile.tileIndex(1, true);

                    coordGraph.addEdge({i_wave_x, i_wave_y}, {wave_tile}, Join());

                    auto n_vblk = Adhoc("VGPRBlockNumber",
                                        wave_tile.tag,
                                        literal(num_agpr / mts),
                                        unit_stride,
                                        true);
                    auto i_vblk
                        = Adhoc("VGPRBlockIndex", wave_tile.tag, mfma_tile_size, unit_stride, true);
                    auto n_lblk = Adhoc(
                        "LANEBlockNumber", wave_tile.tag, literal(wfs / mts), unit_stride, true);
                    auto i_lblk
                        = Adhoc("LANEBlockIndex", wave_tile.tag, mfma_tile_size, unit_stride, true);
                    auto block = Adhoc("LinearBlock",
                                       wave_tile.tag,
                                       literal(num_elements / 16u),
                                       unit_stride,
                                       true);
                    auto row_block
                        = Adhoc("RowBlock", wave_tile.tag, n_row_blocks, unit_stride, true);
                    auto col_block
                        = Adhoc("ColBlock", wave_tile.tag, n_col_blocks, unit_stride, true);

                    coordGraph.addEdge({n_wave_x, i_wave_x}, {i_mac_x}, Flatten());
                    coordGraph.addEdge({n_wave_y, i_wave_y}, {i_mac_y}, Flatten());

                    auto wave_x = Wavefront(mac_tile.tag, 0, true);
                    auto wave_y = Wavefront(mac_tile.tag, 1, true);
                    auto wave   = Wavefront(mac_tile.tag, -1, true);

                    coordGraph.addEdge({wave}, {wave_x, wave_y}, Tile());

                    auto lane = Lane(mac_tile.tag, wavefront_size, unit_stride, true);
                    auto vgpr = VGPR(mac_tile.tag, literal(num_agpr), unit_stride, true);

                    coordGraph.addEdge({vgpr}, {n_vblk, i_vblk}, Tile());
                    coordGraph.addEdge({lane}, {n_lblk, i_lblk}, Tile());
                    coordGraph.addEdge({n_vblk, n_lblk}, {block}, Flatten());
                    coordGraph.addEdge({block}, {row_block, col_block}, Tile());

                    auto wavetilesPerWorkgroup = m_params->getWaveTilesPerWorkgroup();
                    auto jammed_wavetile_x     = JammedWaveTileNumber(
                        mac_tile.tag, 0, literal(wavetilesPerWorkgroup[0]), literal(1), true);
                    auto jammed_wavetile_y = JammedWaveTileNumber(
                        mac_tile.tag, 1, literal(wavetilesPerWorkgroup[1]), literal(1), true);

                    if(wavetilesPerWorkgroup[0] > 1)
                        coordGraph.addEdge({wave_x, jammed_wavetile_x}, {n_wave_x}, Flatten());
                    else
                        coordGraph.addEdge({wave_x}, {n_wave_x}, PassThrough());
                    if(wavetilesPerWorkgroup[1] > 1)
                        coordGraph.addEdge({wave_y, jammed_wavetile_y}, {n_wave_y}, Flatten());
                    else
                        coordGraph.addEdge({wave_y}, {n_wave_y}, PassThrough());

                    coordGraph.addEdge({row_block, i_vblk}, {i_wave_x}, Flatten());
                    coordGraph.addEdge({col_block, i_lblk}, {i_wave_y}, Flatten());

                    coordGraph.addEdge({workitem}, {wave, lane}, Tile());
                    coordGraph.addEdge({vgpr}, {user}, DataFlow());
                }
                break;

                default:
                    Throw<FatalError>("Store : MacroTile memory type not supported yet.");
                }

                controlGraph.addEdge(loc.srcs, {store}, loc.controlEdge);
            }

            virtual void visitEdge(HyperGraph&                 coordGraph,
                                   ControlGraph::ControlGraph& controlGraph,
                                   Location const&             loc,
                                   DestructMacroTile const&    edge) override
            {
                // NOP: don't need this edge anymore
            }

            virtual void visitEdge(HyperGraph&                 coordGraph,
                                   ControlGraph::ControlGraph& controlGraph,
                                   Location const&             loc,
                                   MakeOutput const&           edge) override
            {
                auto is_tiled = std::holds_alternative<MacroTile>(loc.dstDims[0]);
                if(!is_tiled)
                    coordGraph.addEdge(loc.srcDims, loc.dstDims, edge);
            }

            virtual void visitOperation(HyperGraph&                     coordGraph,
                                        ControlGraph::ControlGraph&     controlGraph,
                                        Location const&                 loc,
                                        ControlGraph::StoreTiled const& store) override
            {
                rocRoller::Log::getLogger()->debug("KernelGraph::LowerTileVisitor::StoreTiled()");
                auto user = loc.coordGraph.getDimension(User(store.tag, true));
                auto tile = loc.coordGraph.getDimension(MacroTile(store.tag, 2, true));
                storeMacroTile(coordGraph, controlGraph, loc, store, user, tile);
            }

        private:
            std::shared_ptr<AssemblyKernel>    m_kernel;
            std::shared_ptr<CommandParameters> m_params;
        };

        // TODO : Rename this when graph rearch complete
        struct LowerTileVisitor2 : public BaseGraphVisitor2
        {
            using BaseGraphVisitor2::visitEdge;
            using BaseGraphVisitor2::visitOperation;

            LowerTileVisitor2(std::shared_ptr<CommandParameters> params,
                              std::shared_ptr<Context>           context)
                : BaseGraphVisitor2(context)
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

            void loadWaveMacroTile(KernelHypergraph&      graph,
                                   CoordGraph::MacroTile& mac_tile,
                                   int                    i_mac_x,
                                   int                    i_mac_y,
                                   int                    workitem,
                                   int                    user_tag)
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

                auto n_wave_x = graph.coordinates.addElement(wave_tile.tileNumber(0));
                auto n_wave_y = graph.coordinates.addElement(wave_tile.tileNumber(1));
                auto i_wave_x = graph.coordinates.addElement(wave_tile.tileIndex(0));
                auto i_wave_y = graph.coordinates.addElement(wave_tile.tileIndex(1));

                graph.coordinates.addElement(CoordGraph::Tile(), {i_mac_x}, {n_wave_x, i_wave_x});
                graph.coordinates.addElement(CoordGraph::Tile(), {i_mac_y}, {n_wave_y, i_wave_y});

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
                        CoordGraph::Adhoc("VGPRBlockNumber", literal(num_vgpr / mts), unit_stride));

                    auto i_vblk = graph.coordinates.addElement(
                        CoordGraph::Adhoc("VGPRBlockIndex", mfma_tile_size, unit_stride));
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
                graph.coordinates.addElement(CoordGraph::DataFlow(), {user_tag}, {vgpr});
            }

            void loadMacroTile(KernelHypergraph& graph,
                               int               user_tag,
                               int               mac_tile_tag,
                               std::vector<int>& sdim)
            {
                auto mac_tile = graph.coordinates.getNode<CoordGraph::MacroTile>(mac_tile_tag);
                auto user     = graph.coordinates.getNode<CoordGraph::User>(user_tag);

                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::LowerTileVisitor::loadMacroTile(): size {} by {}",
                    mac_tile.sizes[0],
                    mac_tile.sizes[1]);

                auto wavefront_size = wavefrontSize();

                AssertRecoverable(mac_tile.rank == 2, "Rank /= 2 not implemented yet.");

                auto sdim_x = sdim[0];
                auto sdim_y = sdim[1];

                auto n_mac_x = graph.coordinates.addElement(mac_tile.tileNumber(0));
                auto n_mac_y = graph.coordinates.addElement(mac_tile.tileNumber(1));
                auto i_mac_x = graph.coordinates.addElement(mac_tile.tileIndex(0));
                auto i_mac_y = graph.coordinates.addElement(mac_tile.tileIndex(1));

                auto workgroup_x = graph.coordinates.addElement(CoordGraph::Workgroup(0));
                auto workgroup_y = graph.coordinates.addElement(CoordGraph::Workgroup(1));

                auto workitem   = graph.coordinates.addElement(CoordGraph::Workitem(0));
                auto workitem_x = graph.coordinates.addElement(CoordGraph::Workitem(0));
                auto workitem_y = graph.coordinates.addElement(CoordGraph::Workitem(1));

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
                    auto thr_tile     = CoordGraph::ThreadTile(mac_tile.subTileSizes);
                    auto thr_tile_tag = graph.coordinates.addElement(thr_tile);

                    auto n_thr_x = graph.coordinates.addElement(thr_tile.tileNumber(0));
                    auto n_thr_y = graph.coordinates.addElement(thr_tile.tileNumber(1));
                    auto i_thr_x = graph.coordinates.addElement(thr_tile.tileIndex(0));
                    auto i_thr_y = graph.coordinates.addElement(thr_tile.tileIndex(1));

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

                    if(mac_tile.memoryType == MemoryType::VGPR)
                    {
                        auto vgpr = graph.coordinates.addElement(CoordGraph::VGPR());
                        graph.coordinates.addElement(CoordGraph::DataFlow(), {user_tag}, {vgpr});
                    }

                    // User -> DataFlow() -> LDS gets added in addLDSOps
                }
                break;

                case MemoryType::WAVE:
                    loadWaveMacroTile(graph, mac_tile, i_mac_x, i_mac_y, workitem, user_tag);
                    break;

                default:
                    Throw<FatalError>("Load : MacroTile memory type not supported yet.");
                }
            }

            virtual void visitOperation(KernelHypergraph&                   graph,
                                        KernelHypergraph const&             original,
                                        GraphReindexer&                     reindexer,
                                        int                                 tag,
                                        ControlHypergraph::LoadTiled const& load) override
            {
                auto logger = rocRoller::Log::getLogger();
                logger->debug("KernelGraph::LowerTileVisitor::LoadTiled() {}", tag);

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

                loadMacroTile(graph, user, mac_tile, sdims);

                copyOperation(graph, original, reindexer, tag);
            }

            void storeWaveMacroTile(KernelHypergraph&      graph,
                                    CoordGraph::MacroTile& mac_tile,
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
                    CoordGraph::Adhoc("VGPRBlockNumber", literal(num_agpr / mts), unit_stride));
                auto i_vblk = graph.coordinates.addElement(
                    CoordGraph::Adhoc("VGPRBlockIndex", mfma_tile_size, unit_stride));
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
                graph.coordinates.addElement(CoordGraph::DataFlow(), {vgpr}, {user_tag});
            }

            void storeMacroTile(KernelHypergraph& graph,
                                int               user_tag,
                                int               mac_tile_tag,
                                std::vector<int>& sdims)
            {
                auto mac_tile = graph.coordinates.getNode<CoordGraph::MacroTile>(mac_tile_tag);
                auto user     = graph.coordinates.getNode<CoordGraph::User>(user_tag);

                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::LowerTileVisitor::storeMacroTile(): size {} by {}",
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
                auto workitem_x = graph.coordinates.addElement(
                    CoordGraph::Workitem(0, literal(workgroupSizes.at(0))));
                auto workitem_y = graph.coordinates.addElement(
                    CoordGraph::Workitem(1, literal(workgroupSizes.at(1))));

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
                    auto thr_tile     = CoordGraph::ThreadTile(mac_tile.subTileSizes);
                    auto thr_tile_tag = graph.coordinates.addElement(thr_tile);
                    auto n_thr_x      = graph.coordinates.addElement(thr_tile.tileNumber(0));
                    auto n_thr_y      = graph.coordinates.addElement(thr_tile.tileNumber(1));
                    auto i_thr_x      = graph.coordinates.addElement(thr_tile.tileIndex(0));
                    auto i_thr_y      = graph.coordinates.addElement(thr_tile.tileIndex(1));

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
                    storeWaveMacroTile(graph, mac_tile, i_mac_x, i_mac_y, workitem, user_tag);
                    break;

                default:
                    Throw<FatalError>("Store : MacroTile memory type not supported yet.");
                }
            }

            virtual void visitOperation(KernelHypergraph&                    graph,
                                        KernelHypergraph const&              original,
                                        GraphReindexer&                      reindexer,
                                        int                                  tag,
                                        ControlHypergraph::StoreTiled const& store) override
            {
                rocRoller::Log::getLogger()->debug("KernelGraph::LowerTileVisitor::StoreTiled()");

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

                storeMacroTile(graph, user, mac_tile, sdims);

                copyOperation(graph, original, reindexer, tag);
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
                auto outgoing_edges
                    = graph.control.getNeighbours<Graph::Direction::Downstream>(tag);
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

        // Add LDSLoad and LDSStore nodes following a LoadTiled node within
        // the control graph, if the tile has a memory type of LDS.
        void addLDSOps(HyperGraph&                    coordGraph,
                       ControlGraph::ControlGraph&    controlGraph,
                       std::shared_ptr<Context>       context,
                       const ControlGraph::LoadTiled& load)
        {
            auto user = coordGraph.getDimension(User(load.tag));
            auto tile = coordGraph.getDimension(MacroTile(load.tag));

            if(tile.memoryType == MemoryType::LDS)
            {
                auto ldsTag = controlGraph.allocateTag();
                auto lds    = LDS(ldsTag);

                auto storeLDS = ControlGraph::StoreLDSTile(ldsTag);
                auto barrier  = ControlGraph::Barrier(ldsTag);
                auto loadLDS  = ControlGraph::LoadLDSTile(load.tag);

                // Add an edge to the coordinate graph to index into the LDS allocation.
                auto workgroupSizes = context->kernel()->workgroupSize();

                auto workitem_x = Workitem(ldsTag, 0, literal(workgroupSizes.at(0)), true);
                auto workitem_y = Workitem(ldsTag, 1, literal(workgroupSizes.at(1)), true);

                coordGraph.addEdge({user}, {tile}, DataFlow());
                coordGraph.addEdge({tile}, {lds}, DataFlow());
                coordGraph.addEdge({workitem_x, workitem_y}, {lds}, Flatten());

                // Find all edges leaving from load. Those should be changed to leave from loadLDS.
                auto loadChildren = controlGraph.getOutputs(getTag(load));
                for(auto op : loadChildren)
                {
                    auto removed = controlGraph.removeEdge(load, op);
                    controlGraph.addEdge({loadLDS}, {op}, removed.second);
                }

                controlGraph.addEdge({load}, {storeLDS}, ControlGraph::Sequence());
                controlGraph.addEdge({storeLDS}, {barrier}, ControlGraph::Sequence());
                controlGraph.addEdge({barrier}, {loadLDS}, ControlGraph::Sequence());
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
            //TODO : lower rank-2 TensorContraction into a matrix multiply.
        }

        /**
         * Lower rank-2 TensorContraction into a matrix multiply.
         */
        void lowerMatrixMultiply(KernelGraph&                           graph,
                                 ControlGraph::TensorContraction const& contraction,
                                 MacroTile const&                       a,
                                 MacroTile const&                       b,
                                 MacroTile const&                       d,
                                 std::shared_ptr<CommandParameters>     params,
                                 std::shared_ptr<Context>               context)
        {
            rocRoller::Log::getLogger()->debug("KernelGraph::matrixMultiply() {}", d.tag);

            auto loadA = graph.control.getOperation(getTag(ControlGraph::LoadTiled(a.tag)));
            auto loadB = graph.control.getOperation(getTag(ControlGraph::LoadTiled(b.tag)));

            // TODO: The size of the K dimension should be loaded from the command arguments
            //
            // start of proper way
            //   auto matK = graph.coordinates.getDimension(SubDimension(a.tag, 1)).size;
            // end of proper way
            //
            // Using the proper way, the size is a UInt64 (size_t),
            // and I can't get this to play nicely with the for loop
            // expression.
            //
            // As a workaround, I'm also passing the size of the K
            // dimension as a UInt32.
            //
            // start of workaround
            auto matKArg = context->kernel()->findArgument("UINT_MAT_K");
            auto matK    = std::make_shared<Expression::Expression>(
                std::make_shared<AssemblyKernelArgument>(matKArg));
            // end of workaround

            auto macK = literal(static_cast<uint>(a.sizes[1]));

            auto [K, forK] = rangeFor(graph.coordinates, graph.control, matK / macK);

            // remove passthrough between A column block and y-workgroup
            graph.coordinates.removeEdge({{getTag(a.tileNumber(1))},
                                          {getTag(Workgroup(a.tag, 1))},
                                          EdgeType::CoordinateTransform});

            // remove passthrough between B row block and x-workgroup
            graph.coordinates.removeEdge({{getTag(b.tileNumber(0))},
                                          {getTag(Workgroup(b.tag, 0))},
                                          EdgeType::CoordinateTransform});

            // A row block is x-workgroup, column block is for loop index
            graph.coordinates.addEdge({a.tileNumber(0)}, {Workgroup(a.tag, 0)}, PassThrough());
            graph.coordinates.addEdge({a.tileNumber(1)}, {K}, PassThrough());

            // B row block is for loop index, column block is y-workgroup
            graph.coordinates.addEdge({b.tileNumber(0)}, {K}, PassThrough());
            graph.coordinates.addEdge({b.tileNumber(1)}, {Workgroup(b.tag, 1)}, PassThrough());

            auto waveMult = ControlGraph::Multiply(d.tag, a.tag, b.tag);

            // initialise accumulator
            auto waveA = graph.coordinates.getDimension(WaveTile(a.tag));
            auto waveB = graph.coordinates.getDimension(WaveTile(b.tag));

            uint num_elements = waveA.sizes[0] * waveB.sizes[1];
            uint wfs          = context->kernel()->wavefront_size();
            uint num_agpr     = num_elements / wfs;

            ControlGraph::Operation initD = ControlGraph::Assign{
                -1, d.tag, Register::Type::Accumulator, literal(0.f), num_agpr};
            graph.control.allocateTag(initD);

            // add for loop
            graph.control.addEdge({ControlGraph::Kernel()}, {forK}, ControlGraph::Body());
            graph.control.addEdge({forK}, {initD}, ControlGraph::Initialize());
            graph.control.addEdge({forK}, {waveMult}, ControlGraph::Body());
            graph.control.addEdge({waveMult}, {loadA}, ControlGraph::Body());
            graph.control.addEdge({waveMult}, {loadB}, ControlGraph::Body());

            // connect ops after contraction to for loop, remove contraction
            auto outputs = graph.control.getOutputs(getTag(contraction));
            graph.control.addEdge({forK}, outputs, ControlGraph::Sequence());

            graph.control.removeOperation(getTag(contraction), true);

            // Add loops to iterate over wavetiles within a workgroup
            auto wavetilesPerWorkgroup = params->getWaveTilesPerWorkgroup();

            if(wavetilesPerWorkgroup[0] > 1 || wavetilesPerWorkgroup[1] > 1)
            {
                auto topOfKernel = graph.control.findOperations<ControlGraph::Kernel>();
                AssertFatal(topOfKernel.size() == 1, "More than one Kernel node not supported");
                auto kernelOutputs = graph.control.getOutputs(getTag(topOfKernel[0]));

                // Remove edges from Kernel to kernel outputs
                for(auto& kernelOutput : kernelOutputs)
                {
                    graph.control.removeEdge({ControlGraph::Kernel()}, {kernelOutput});
                }

                CoordinateTransform::ForLoop WaveTilesX(-1), WaveTilesY(-1);
                ControlGraph::ForLoopOp      forWaveTilesX, forWaveTilesY;

                if(wavetilesPerWorkgroup[1] > 1)
                {
                    std::tie(WaveTilesY, forWaveTilesY) = rangeFor(
                        graph.coordinates, graph.control, literal(wavetilesPerWorkgroup[1]));
                }
                if(wavetilesPerWorkgroup[0] > 1)
                {
                    std::tie(WaveTilesX, forWaveTilesX) = rangeFor(
                        graph.coordinates, graph.control, literal(wavetilesPerWorkgroup[0]));
                }

                // Add edges from inner loop to kernel outputs
                if(wavetilesPerWorkgroup[1] > 1)
                {
                    for(auto& kernelOutput : kernelOutputs)
                    {
                        graph.control.addEdge(
                            {forWaveTilesY}, {kernelOutput}, ControlGraph::Body());
                    }
                }
                else
                {
                    for(auto& kernelOutput : kernelOutputs)
                    {
                        graph.control.addEdge(
                            {forWaveTilesX}, {kernelOutput}, ControlGraph::Body());
                    }
                }

                if(wavetilesPerWorkgroup[0] > 1 && wavetilesPerWorkgroup[1] > 1)
                    graph.control.addEdge({forWaveTilesX}, {forWaveTilesY}, ControlGraph::Body());

                // Add edges from Kernel to outer loop
                if(wavetilesPerWorkgroup[0] > 1)
                    graph.control.addEdge(
                        {ControlGraph::Kernel()}, {forWaveTilesX}, ControlGraph::Body());
                else
                    graph.control.addEdge(
                        {ControlGraph::Kernel()}, {forWaveTilesY}, ControlGraph::Body());

                // Add edges from all JammedWaveTileNumber dimensions to the for loop
                for(auto const& dimension : graph.coordinates.getDimensions())
                {
                    if(std::holds_alternative<JammedWaveTileNumber>(dimension))
                    {
                        auto wavetilePerWorkgroup = std::get<JammedWaveTileNumber>(dimension);
                        auto forLoopDim = wavetilePerWorkgroup.dim == 0 ? WaveTilesX : WaveTilesY;
                        if(wavetilePerWorkgroup.output)
                            graph.coordinates.addEdge({forLoopDim}, {dimension}, PassThrough());
                        else
                            graph.coordinates.addEdge({dimension}, {forLoopDim}, PassThrough());
                    }
                }
            }
        }

        KernelGraph lowerTile(KernelGraph                        k,
                              std::shared_ptr<CommandParameters> params,
                              std::shared_ptr<Context>           context)
        {
            TIMER(t, "KernelGraph::lowerTile");
            rocRoller::Log::getLogger()->debug("KernelGraph::lowerTile()");
            auto visitor = LowerTileVisitor(params, context);
            auto kgraph  = rewrite(k, visitor);

            auto contractions = kgraph.control.findOperations<ControlGraph::TensorContraction>();
            AssertFatal(contractions.size() <= 1,
                        "More than one TensorContraction not supported yet.");
            if(contractions.size() == 1)
            {
                auto op = contractions[0];
                auto a  = kgraph.coordinates.getDimension(op.a);
                auto b  = kgraph.coordinates.getDimension(op.b);
                auto d  = MacroTile(op.tag);
                if(a.rank == 2 && b.rank == 2 && op.aDims == std::vector<int>{1}
                   && op.bDims == std::vector<int>{0})
                {
                    lowerMatrixMultiply(kgraph, op, a, b, d, params, context);
                }
                else
                {
                    Throw<FatalError>("General contraction not implemented yet.");
                }
            }

            // Add LDS operations to the control graph following LoadTiled nodes.
            // This is done after the control graph is completly built to make
            // it easier to modify the edges coming into and out of the
            // original LoadTiled node.
            for(auto const& op : kgraph.control.getOperations())
            {
                visit(rocRoller::overloaded{
                          [&](auto op) {},
                          [&](ControlGraph::LoadTiled const& op) {
                              addLDSOps(kgraph.coordinates, kgraph.control, context, op);
                          },
                      },
                      op);
            }

            return kgraph;
        }

        KernelHypergraph lowerTile(KernelHypergraph                   graph,
                                   std::shared_ptr<CommandParameters> params,
                                   std::shared_ptr<Context>           context)
        {
            TIMER(t, "KernelGraph::lowerTile");
            rocRoller::Log::getLogger()->debug("KernelGraph::lowerTile()");

            auto visitor = LowerTileVisitor2(params, context);
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
