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

            LowerTileVisitor(std::shared_ptr<Context> context)
                : BaseGraphVisitor(context)
                , m_kernel(context->kernel())
            {
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

                    switch(wave_tile.layout)
                    {
                    case LayoutType::MATRIX_A:
                    {
                        coordGraph.addEdge({i_wave_y, i_wave_x}, {wave_tile}, Flatten());
                        // TODO: Should this be set here?  Or deferred?
                        coordGraph.addEdge({n_wave_x}, {wave_x}, PassThrough());
                        coordGraph.addEdge({wave_tile}, {lane, vgpr}, Tile());
                    }
                    break;
                    case LayoutType::MATRIX_B:
                    {
                        coordGraph.addEdge({i_wave_x, i_wave_y}, {wave_tile}, Flatten());
                        // TODO: Should this be set here?  Or deferred?
                        coordGraph.addEdge({n_wave_y}, {wave_y}, PassThrough());
                        coordGraph.addEdge({wave_tile}, {lane, vgpr}, Tile());
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

                        coordGraph.addEdge({n_wave_y}, {wave_y}, PassThrough());
                        coordGraph.addEdge({n_wave_x}, {wave_x}, PassThrough());

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
                    Throw<FatalError>("MacroTile memory type not supported yet.");
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

                loadMacroTile(coordGraph, controlGraph, loc, load, load.user, load.tile);

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

                AssertRecoverable(mac_tile.rank >= 0 && sdims.size() == (size_t)mac_tile.rank,
                                  "Tensor size mismatch.");
                AssertRecoverable(mac_tile.rank == 2, "Rank /= 2 not implementd yet.");

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

                    coordGraph.addEdge({wave_y}, {n_wave_y}, PassThrough());
                    coordGraph.addEdge({wave_x}, {n_wave_x}, PassThrough());

                    coordGraph.addEdge({row_block, i_vblk}, {i_wave_x}, Flatten());
                    coordGraph.addEdge({col_block, i_lblk}, {i_wave_y}, Flatten());

                    coordGraph.addEdge({workitem}, {wave, lane}, Tile());
                    coordGraph.addEdge({vgpr}, {user}, DataFlow());
                }
                break;

                default:
                    Throw<FatalError>("MacroTile memory type not supported yet.");
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
                storeMacroTile(coordGraph, controlGraph, loc, store, store.user, store.tile);
            }

        private:
            std::shared_ptr<AssemblyKernel> m_kernel;
        };

        // Add LDSLoad and LDSStore nodes following a LoadTiled node within
        // the control graph, if the tile has a memory type of LDS.
        void addLDSOps(HyperGraph&                    coordGraph,
                       ControlGraph::ControlGraph&    controlGraph,
                       std::shared_ptr<Context>       context,
                       const ControlGraph::LoadTiled& load)
        {
            if(load.tile.memoryType == MemoryType::LDS)
            {
                auto resultTag    = controlGraph.allocateTag();
                auto ldsTag       = controlGraph.allocateTag();
                auto barrierTag   = controlGraph.allocateTag();
                auto lds          = LDS(ldsTag);
                auto newMacroTile = load.tile;

                auto newLoad  = ControlGraph::LoadTiled(resultTag, load.user, newMacroTile);
                auto storeLDS = ControlGraph::StoreLDSTile(ldsTag, lds, newMacroTile);
                auto barrier  = ControlGraph::Barrier(barrierTag);
                auto loadLDS  = ControlGraph::LoadLDSTile(load.tag, newMacroTile, lds);

                // Add an edge to the coordinate graph to index into the LDS allocation.
                auto workgroupSizes = context->kernel()->workgroupSize();

                auto workitem_x = Workitem(ldsTag, 0, literal(workgroupSizes.at(0)), true);
                auto workitem_y = Workitem(ldsTag, 1, literal(workgroupSizes.at(1)), true);

                coordGraph.addEdge({workitem_x, workitem_y}, {lds}, Flatten());

                auto loadParents  = controlGraph.getInputs(getTag(load));
                auto loadChildren = controlGraph.getOutputs(getTag(load));

                // Find all edges going to load. Those should be changed to go to newLoad.
                for(auto op : loadParents)
                {
                    auto removed = controlGraph.removeEdge(op, load);
                    controlGraph.addEdge({op}, {newLoad}, removed.second);
                }

                // Find all edges leaving from load. Those should be changed to leave from newLoad.
                for(auto op : loadChildren)
                {
                    auto removed = controlGraph.removeEdge(load, op);
                    controlGraph.addEdge({loadLDS}, {op}, removed.second);
                }

                controlGraph.removeOperation(getTag(load));

                controlGraph.addEdge({newLoad}, {storeLDS}, ControlGraph::Sequence());
                controlGraph.addEdge({storeLDS}, {barrier}, ControlGraph::Sequence());
                controlGraph.addEdge({barrier}, {loadLDS}, ControlGraph::Sequence());
            }
        }

        /**
         * Lower rank-2 TensorContraction into a matrix multiply.
         */
        void lowerMatrixMultiply(KernelGraph&                           graph,
                                 ControlGraph::TensorContraction const& contraction,
                                 MacroTile const&                       a,
                                 MacroTile const&                       b,
                                 MacroTile const&                       d,
                                 std::shared_ptr<Context>               context)
        {
            rocRoller::Log::getLogger()->debug("KernelGraph::matrixMultiply() {}", d.tag);

            auto loadA = graph.control.getOperation(
                getTag(ControlGraph::LoadTiled(a.tag, User(a.tag), MacroTile(a.tag))));
            auto loadB = graph.control.getOperation(
                getTag(ControlGraph::LoadTiled(b.tag, User(b.tag), MacroTile(b.tag))));

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
        }

        KernelGraph lowerTile(KernelGraph k, std::shared_ptr<Context> context)
        {
            TIMER(t, "KernelGraph::lowerTile");
            rocRoller::Log::getLogger()->debug("KernelGraph::lowerTile()");
            auto visitor = LowerTileVisitor(context);
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
                    lowerMatrixMultiply(kgraph, op, a, b, d, context);
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
    }
}
