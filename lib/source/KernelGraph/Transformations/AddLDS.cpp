
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
        namespace CT = rocRoller::KernelGraph::CoordinateGraph;
        using namespace ControlGraph;
        using namespace CoordinateGraph;
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

            virtual void visitOperation(KernelGraph&             graph,
                                        KernelGraph const&       original,
                                        GraphReindexer&          reindexer,
                                        int                      tag,
                                        TensorContraction const& op) override
            {
                copyOperation(graph, original, reindexer, tag);

                auto new_tag = reindexer.control.at(tag);
                auto new_op  = graph.control.getNode<TensorContraction>(new_tag);
                new_op.a     = reindexer.coordinates.at(op.a);
                new_op.b     = reindexer.coordinates.at(op.b);
                graph.control.setElement(new_tag, new_op);
            }

            virtual void visitOperation(KernelGraph&       graph,
                                        KernelGraph const& original,
                                        GraphReindexer&    reindexer,
                                        int                tag,
                                        LoadTiled const&   oload) override
            {
                auto logger = rocRoller::Log::getLogger();
                logger->debug("KernelGraph::LowerTileVisitor::LoadTiled({})", tag);

                auto original_user     = original.mapper.get<User>(tag);
                auto original_mac_tile = original.mapper.get<MacroTile>(tag);
                auto user              = reindexer.coordinates.at(original_user);
                auto mac_tile          = reindexer.coordinates.at(original_mac_tile);

                auto sdims
                    = original.coordinates
                          .getInputNodeIndices(original_mac_tile, CT::isEdge<ConstructMacroTile>)
                          .to<std::vector>();
                for(int i = 0; i < sdims.size(); i++)
                    sdims[i] = reindexer.coordinates.at(sdims[i]);

                copyOperation(graph, original, reindexer, tag);

                auto load = reindexer.control.at(tag);

                auto workgroupSizes        = m_context->kernel()->workgroupSize();
                auto wavefrontSize         = m_context->kernel()->wavefront_size();
                auto wavetilesPerWorkgroup = m_params->getWaveTilesPerWorkgroup();

                loadMacroTile(graph,
                              load,
                              user,
                              mac_tile,
                              sdims,
                              workgroupSizes,
                              wavefrontSize,
                              wavetilesPerWorkgroup);
            }

            virtual void visitOperation(KernelGraph&       graph,
                                        KernelGraph const& original,
                                        GraphReindexer&    reindexer,
                                        int                tag,
                                        StoreTiled const&  ostore) override
            {
                rocRoller::Log::getLogger()->debug("KernelGraph::LowerTileVisitor::StoreTiled({})",
                                                   tag);

                auto original_user     = original.mapper.get<User>(tag);
                auto original_mac_tile = original.mapper.get<MacroTile>(tag);
                auto user              = reindexer.coordinates.at(original_user);
                auto mac_tile          = reindexer.coordinates.at(original_mac_tile);

                auto sdims
                    = original.coordinates
                          .getOutputNodeIndices(original_mac_tile, CT::isEdge<DestructMacroTile>)
                          .to<std::vector>();
                for(int i = 0; i < sdims.size(); i++)
                    sdims[i] = reindexer.coordinates.at(sdims[i]);

                copyOperation(graph, original, reindexer, tag);

                auto store = reindexer.control.at(tag);

                auto workgroupSizes        = m_context->kernel()->workgroupSize();
                auto wavefrontSize         = m_context->kernel()->wavefront_size();
                auto wavetilesPerWorkgroup = m_params->getWaveTilesPerWorkgroup();

                storeMacroTile(graph,
                               store,
                               user,
                               mac_tile,
                               sdims,
                               workgroupSizes,
                               wavefrontSize,
                               wavetilesPerWorkgroup);
            }

        private:
            std::shared_ptr<AssemblyKernel>    m_kernel;
            std::shared_ptr<CommandParameters> m_params;
        };

        // Add LDSLoad and LDSStore nodes following a LoadTiled node within
        // the control graph, if the tile has a memory type of LDS.
        void addLDSOps(KernelGraph& graph, std::shared_ptr<Context> context, int tag)
        {
            auto user_tag = graph.mapper.get<User>(tag);
            auto tile_tag = graph.mapper.get<MacroTile>(tag);
            auto tile     = graph.coordinates.getNode<MacroTile>(tile_tag);
            auto load     = graph.control.getNode<LoadTiled>(tag);

            if(tile.memoryType == MemoryType::LDS)
            {
                auto lds = graph.coordinates.addElement(LDS());

                auto storeLDS = graph.control.addElement(StoreLDSTile(load.vtype.dataType));
                auto barrier  = graph.control.addElement(Barrier());
                auto loadLDS  = graph.control.addElement(LoadLDSTile(load.vtype));

                graph.mapper.connect<LDS>(storeLDS, lds);
                graph.mapper.connect<MacroTile>(storeLDS, tile_tag);

                graph.mapper.connect<LDS>(loadLDS, lds);
                graph.mapper.connect<MacroTile>(loadLDS, tile_tag);

                auto workgroupSizes = context->kernel()->workgroupSize();
                loadMacroTileFromLDS(graph, loadLDS, lds, tile_tag, workgroupSizes);
                storeMacroTileIntoLDS(graph, storeLDS, lds, tile_tag, workgroupSizes);

                graph.coordinates.addElement(DataFlow(), {user_tag}, {tile_tag});

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

                graph.control.addElement(Sequence(), {tag}, {storeLDS});
                graph.control.addElement(Sequence(), {storeLDS}, {barrier});
                graph.control.addElement(Sequence(), {barrier}, {loadLDS});
            }
        }

        void addLoadWaveLDSOps(KernelGraph&                       graph,
                               std::shared_ptr<CommandParameters> params,
                               std::shared_ptr<Context>           context,
                               int                                tile,
                               int                                load,
                               int                                forK,
                               int                                K,
                               int                                waveMult)
        {
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::LowerTileVisitor::addLoadWaveLDSOps: LoadTiled({})", load);

            auto macrotile = graph.coordinates.getNode<MacroTile>(tile);

            if(macrotile.memoryType == MemoryType::WAVE_LDS)
            {
                // change loadTiled to LoadLDSTile under Multiply
                // and update its macrotile's memory type
                auto vtype = graph.control.getNode<LoadTiled>(load).vtype;
                graph.control.setElement(load, LoadLDSTile(vtype));
                macrotile.memoryType = MemoryType::WAVE;
                graph.coordinates.setElement(tile, macrotile);

                auto user = graph.coordinates.getInputNodeIndices(tile, CT::isEdge<DataFlow>)
                                .to<std::vector>();
                AssertFatal(user.size() == 1);
                graph.coordinates.deleteElement(user, std::vector<int>{tile}, CT::isEdge<DataFlow>);
                auto sdims = graph.coordinates.getOutputNodeIndices(user[0], CT::isEdge<Split>)
                                 .to<std::vector>();

                auto lds = graph.coordinates.addElement(LDS());

                // remove workgroups, macrotile numbers and tile edges from sdims
                loadWaveMacroTileFromLDS(graph, macrotile, load, sdims, K, lds);

                // add new loadTiled node to load a macrotile into VGPRs from global memory under the forK loop
                auto load_macrotile_from_global = graph.control.addElement(LoadTiled(vtype));
                graph.control.addElement(Body(), {forK}, {load_macrotile_from_global});
                graph.mapper.connect<User>(load_macrotile_from_global, user[0]);

                // create an internal macrotile to be loaded by one workgroup
                auto workgroupSizes = context->kernel()->workgroupSize();
                auto numWorkitems   = product(workgroupSizes);
                auto numElements    = product(macrotile.sizes);
                auto numVGPRs       = static_cast<int>(numElements / numWorkitems);
                // TODO : load the two Halfs into 1 VGPR
                //if(vtype == DataType::Half)
                //num_vgprs = num_vgprs / 2;
                auto t_m          = numVGPRs;
                auto t_n          = 1;
                auto internalTile = graph.coordinates.addElement(
                    MacroTile(macrotile.sizes, MemoryType::VGPR, {t_m, t_n}));
                auto internalTileDim       = graph.coordinates.getNode<MacroTile>(internalTile);
                internalTileDim.layoutType = macrotile.layoutType;
                graph.coordinates.setElement(internalTile, internalTileDim);
                graph.mapper.connect<MacroTile>(load_macrotile_from_global, internalTile);

                // user --DataFlow--> internalTile
                graph.coordinates.addElement(DataFlow(), {user[0]}, {internalTile});

                // lower tile LoadTiled : load macrotile from global memory
                loadMacroTileForLDS(graph,
                                    load_macrotile_from_global,
                                    user[0],
                                    internalTile,
                                    sdims,
                                    K,
                                    workgroupSizes);

                // add store from VGPRs to LDS following this new loadTiled under the forK loop
                auto store_macrotile_into_LDS
                    = graph.control.addElement(StoreLDSTile(vtype.dataType));
                auto barrier = graph.control.addElement(Barrier());
                graph.control.addElement(
                    Sequence(), {load_macrotile_from_global}, {store_macrotile_into_LDS});
                graph.control.addElement(Sequence(), {store_macrotile_into_LDS}, {barrier});
                graph.control.addElement(Sequence(), {barrier}, {waveMult});
                graph.mapper.connect<MacroTile>(store_macrotile_into_LDS, internalTile);

                // lower tile StoreLDSTile : store macrotile into LDS
                storeMacroTileIntoLDS(
                    graph, store_macrotile_into_LDS, lds, internalTile, workgroupSizes);

                // LDS --DataFlow--> macrotile
                graph.coordinates.addElement(DataFlow(), {lds}, {tile});

                graph.mapper.connect<LDS>(load, lds);
                graph.mapper.connect<LDS>(load_macrotile_from_global, lds);
                graph.mapper.connect<LDS>(store_macrotile_into_LDS, lds);
            }
        }

        void addStoreWaveLDSOps(KernelGraph&                       graph,
                                std::shared_ptr<CommandParameters> params,
                                std::shared_ptr<Context>           context,
                                int                                tile,
                                int                                store,
                                int                                upperLoop)
        {
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::LowerTileVisitor::addStoreWaveLDSOps: StoreTiled({})", store);

            auto macrotile = graph.coordinates.getNode<MacroTile>(tile);

            if(macrotile.memoryType == MemoryType::WAVE_LDS)
            {
                // change StoreTiled to StoreLDSTile
                // and update its macrotile's memory type
                auto dtype = graph.control.getNode<StoreTiled>(store).dataType;
                graph.control.setElement(store, StoreLDSTile(dtype));
                macrotile.memoryType = MemoryType::WAVE;
                graph.coordinates.setElement(tile, macrotile);

                auto user = graph.coordinates.getOutputNodeIndices(tile, CT::isEdge<DataFlow>)
                                .to<std::vector>();
                AssertFatal(user.size() == 1);
                graph.coordinates.deleteElement(std::vector<int>{tile}, user, CT::isEdge<DataFlow>);
                auto sdims = graph.coordinates.getInputNodeIndices(user[0], CT::isEdge<Join>)
                                 .to<std::vector>();
                AssertFatal(sdims.size() > 1);

                auto lds = graph.coordinates.addElement(LDS());

                // remove workgroups, macrotile numbers and tile edges from sdims
                storeWaveMacroTileIntoLDS(graph, macrotile, store, sdims, lds);

                // macrotile --DataFlow--> LDS
                graph.coordinates.addElement(DataFlow(), {tile}, {lds});
                graph.mapper.connect<LDS>(store, lds);

                auto barrier = graph.control.addElement(Barrier());
                graph.control.addElement(Sequence(), {store}, {barrier});

                // add new loadLDSTile node to load a macrotile into VGPRs from LDS
                auto load_macrotile_from_LDS
                    = graph.control.addElement(LoadLDSTile(VariableType(dtype)));
                graph.mapper.connect<LDS>(load_macrotile_from_LDS, lds);
                graph.control.addElement(Sequence(), {upperLoop}, {load_macrotile_from_LDS});

                // create an internal macrotile to be loaded by one workgroup
                auto workgroupSizes = context->kernel()->workgroupSize();
                auto numWorkitems   = product(workgroupSizes);
                auto numElements    = product(macrotile.sizes);
                auto numVGPRs       = static_cast<int>(numElements / numWorkitems);
                // TODO : load the two Halfs into 1 VGPR
                //if(vtype == DataType::Half)
                //num_vgprs = num_vgprs / 2;
                auto t_m          = numVGPRs;
                auto t_n          = 1;
                auto internalTile = graph.coordinates.addElement(
                    MacroTile(macrotile.sizes, MemoryType::VGPR, {t_m, t_n}));
                auto internalTileDim       = graph.coordinates.getNode<MacroTile>(internalTile);
                internalTileDim.layoutType = macrotile.layoutType;
                graph.coordinates.setElement(internalTile, internalTileDim);
                graph.mapper.connect<MacroTile>(load_macrotile_from_LDS, internalTile);

                // lower tile LoadLDSTile : load macrotile from LDS
                loadMacroTileFromLDS(
                    graph, load_macrotile_from_LDS, lds, internalTile, workgroupSizes);

                // add store from VGPRs to global following this new loadLDSTile
                auto store_macrotile_into_global = graph.control.addElement(StoreTiled(dtype));
                graph.control.addElement(
                    Sequence(), {load_macrotile_from_LDS}, {store_macrotile_into_global});
                graph.mapper.connect<MacroTile>(store_macrotile_into_global, internalTile);
                graph.mapper.connect<User>(store_macrotile_into_global, user[0]);
                // internalTile --DataFlow--> user
                graph.coordinates.addElement(DataFlow(), {internalTile}, {user[0]});

                storeMacroTileForLDS(graph,
                                     store_macrotile_into_global,
                                     user[0],
                                     internalTile,
                                     sdims,
                                     workgroupSizes);
            }
        }

        /**
         * Lower rank-2 TensorContraction into a matrix multiply.
         */
        void lowerMatrixMultiply(KernelGraph&                       graph,
                                 int                                tag,
                                 int                                a,
                                 int                                b,
                                 int                                d,
                                 std::shared_ptr<CommandParameters> params,
                                 std::shared_ptr<Context>           context)
        {
            rocRoller::Log::getLogger()->debug("KernelGraph::matrixMultiply() {}", d);

            auto macrotile_a = graph.coordinates.getNode<MacroTile>(a);
            auto macrotile_b = graph.coordinates.getNode<MacroTile>(b);

            // get tensor contraction operands
            auto parents = graph.control.parentNodes(tag).to<std::vector>();
            AssertFatal(parents.size() == 2);
            auto operandA = parents[0];
            auto operandB = parents[1];
            AssertFatal(a == graph.mapper.get<MacroTile>(operandA));
            AssertFatal(b == graph.mapper.get<MacroTile>(operandB));

            auto reachable_from_tensor
                = graph.control.depthFirstVisit(tag).to<std::unordered_set>();

            // find loadtiled ops that lead to the tensor contraction op
            // find storetiled ops that follow the tensor contraction op
            // find elementOps that follow the tensor contraction op
            std::vector<int> loadA;
            std::vector<int> loadB;
            std::vector<int> stores;
            std::vector<int> assigns;
            for(auto const index : graph.control.getNodes())
            {
                auto elem = graph.control.getElement(index);
                visit(rocRoller::overloaded{
                          [&](auto op) {},
                          [&](LoadTiled const& load) {
                              auto reachable_from_load
                                  = graph.control.depthFirstVisit(index).to<std::unordered_set>();
                              if(reachable_from_load.find(operandA) != reachable_from_load.end())
                                  loadA.push_back(index);
                              else if(reachable_from_load.find(operandB)
                                      != reachable_from_load.end())
                                  loadB.push_back(index);
                          },
                          [&](StoreTiled const& store) {
                              if(reachable_from_tensor.find(index) != reachable_from_tensor.end())
                                  stores.push_back(index);
                          },
                          [&](Assign const& op) {
                              if(reachable_from_tensor.find(index) != reachable_from_tensor.end())
                                  assigns.push_back(index);
                          }},
                      std::get<Operation>(elem));
            }
            AssertFatal(loadA.size() == 1 && loadB.size() == 1);
            AssertFatal(stores.size() <= 1);
            auto storeD = !stores.empty() ? stores[0] : -1;

            // find kernel
            auto root = graph.control.roots().to<std::vector>();
            AssertFatal(root.size() == 1, "More than one Kernel node not supported");
            auto kernel = root[0];

            graph.control.deleteElement<Body>(std::vector<int>{kernel}, std::vector<int>{loadA[0]});
            graph.control.deleteElement<Body>(std::vector<int>{kernel}, std::vector<int>{loadB[0]});

            auto matK = (graph.coordinates.getNode<SubDimension>(
                             graph.mapper.get<SubDimension>(loadA[0], 1)))
                            .size;

            auto macK = literal(static_cast<uint>(macrotile_a.sizes[1])); // M x K

            auto [K, forK] = rangeFor(graph, matK / macK); // num of loop iterations : matK / macK

            // remove passthrough between A column block and y-workgroup
            auto a_tilenum_y   = graph.mapper.get<MacroTileNumber>(loadA[0], 1);
            auto a_workgroup_y = graph.mapper.get<Workgroup>(loadA[0], 1);
            graph.coordinates.deleteElement(std::vector<int>{a_tilenum_y},
                                            std::vector<int>{a_workgroup_y},
                                            CT::isEdge<PassThrough>);
            graph.mapper.disconnect<Workgroup>(loadA[0], a_workgroup_y, 1);
            graph.coordinates.deleteElement(a_workgroup_y);

            // remove passthrough between B row block and x-workgroup
            auto b_tilenum_x   = graph.mapper.get<MacroTileNumber>(loadB[0], 0);
            auto b_workgroup_x = graph.mapper.get<Workgroup>(loadB[0], 0);
            graph.coordinates.deleteElement(std::vector<int>{b_tilenum_x},
                                            std::vector<int>{b_workgroup_x},
                                            CT::isEdge<PassThrough>);
            graph.mapper.disconnect<Workgroup>(loadB[0], b_workgroup_x, 0);
            graph.coordinates.deleteElement(b_workgroup_x);

            // A row block is x-workgroup, column block is for loop index
            graph.coordinates.addElement(PassThrough(), {a_tilenum_y}, {K});

            // B row block is for loop index, column block is y-workgroup
            graph.coordinates.addElement(PassThrough(), {b_tilenum_x}, {K});

            // TODO : create helper functions to make this lowering modular and readable.
            auto waveMult = graph.control.addElement(Multiply());
            // connections are: 0: lhs (A); 1: rhs (B); 2: dst (D)
            graph.mapper.connect<MacroTile>(waveMult, d, 2);

            auto [waveA_tag, waveA] = graph.getDimension<WaveTile>(loadA[0]);
            auto [waveB_tag, waveB] = graph.getDimension<WaveTile>(loadB[0]);
            uint num_elements       = waveA.sizes[0] * waveB.sizes[1];
            uint wfs                = context->kernel()->wavefront_size();
            uint num_agpr           = num_elements / wfs; // number of output registers per thread

            auto initD = graph.control.addElement(
                Assign{Register::Type::Accumulator, literal(0.f), num_agpr});

            graph.mapper.connect<MacroTile>(initD, d);

            graph.control.addElement(Sequence(), {initD}, {forK});
            graph.control.addElement(Body(), {forK}, {waveMult});
            graph.control.addElement(Body(), {waveMult}, {loadA[0]});
            graph.control.addElement(Body(), {waveMult}, {loadB[0]});

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

            auto [WaveTilesX, forWaveTilesX] = rangeFor(graph, literal(wavetilesPerWorkgroup[0]));
            auto [WaveTilesY, forWaveTilesY] = rangeFor(graph, literal(wavetilesPerWorkgroup[1]));

            // find other loadtiled ops from kernel that lead to assigns
            auto             kernel_outputs = graph.control.childNodes(kernel).to<std::vector>();
            std::vector<int> otherLoads;
            std::vector<int> otherOps;
            for(auto const index : kernel_outputs)
            {
                auto elem = graph.control.getElement(index);
                visit(rocRoller::overloaded{
                          [&](auto op) { otherOps.push_back(index); },
                          [&](LoadTiled const& load) {
                              auto reachable_from_load
                                  = graph.control.depthFirstVisit(index).to<std::unordered_set>();
                              for(auto const& assign : assigns)
                              {
                                  if(reachable_from_load.find(assign) != reachable_from_load.end())
                                  {
                                      otherLoads.push_back(index);
                                      break;
                                  }
                              }
                          }},
                      std::get<Operation>(elem));
            }
            AssertFatal(otherLoads.size() <= 1);

            // Add edges from inner loop to some kernel outputs : forK and otherLoads
            // need to leave other nodes attached with kernel
            // ex: loadtiled ops that don't lead to assigns
            // ex : loadVGPRs for alpha and beta in GEMM

            graph.control.addElement(Body(), {forWaveTilesY}, {initD});

            for(auto const index : otherLoads)
            {
                auto e = graph.control.getNeighbours<Graph::Direction::Upstream>(index)
                             .to<std::vector>()[0];
                auto elem = graph.control.getElement(e);
                graph.control.deleteElement(e);
                graph.control.addElement(
                    e, elem, std::vector<int>{forWaveTilesY}, std::vector<int>{index});
            }

            for(auto const index : otherOps)
            {
                auto e = graph.control.getNeighbours<Graph::Direction::Downstream>(index)
                             .to<std::vector>()[0];
                auto elem = graph.control.getElement(e);
                graph.control.deleteElement(e);
                graph.control.addElement(
                    e, elem, std::vector<int>{index}, std::vector<int>{forWaveTilesX});
            }

            // make nested inner loops (forWaveTilesY inside the forWaveTilesX loop)
            graph.control.addElement(Body(), {forWaveTilesX}, {forWaveTilesY});

            // Add edges from Kernel to outer loop
            graph.control.addElement(Body(), {kernel}, {forWaveTilesX});

            // Add edges from all JammedWaveTileNumber dimensions to the for loop
            auto a_jammed_x = graph.mapper.get<JammedWaveTileNumber>(loadA[0], 0);
            graph.coordinates.addElement(PassThrough(), {a_jammed_x}, {WaveTilesX});
            auto b_jammed_x = graph.mapper.get<JammedWaveTileNumber>(loadB[0], 0);
            graph.coordinates.addElement(PassThrough(), {b_jammed_x}, {WaveTilesX});

            auto a_jammed_y = graph.mapper.get<JammedWaveTileNumber>(loadA[0], 1);
            graph.coordinates.addElement(PassThrough(), {a_jammed_y}, {WaveTilesY});
            auto b_jammed_y = graph.mapper.get<JammedWaveTileNumber>(loadB[0], 1);
            graph.coordinates.addElement(PassThrough(), {b_jammed_y}, {WaveTilesY});

            if(otherLoads.size() > 0)
            {
                auto c_jammed_x = graph.mapper.get<JammedWaveTileNumber>(otherLoads[0], 0);
                graph.coordinates.addElement(PassThrough(), {c_jammed_x}, {WaveTilesX});
                auto c_jammed_y = graph.mapper.get<JammedWaveTileNumber>(otherLoads[0], 1);
                graph.coordinates.addElement(PassThrough(), {c_jammed_y}, {WaveTilesY});
            }

            if(storeD > 0)
            {
                auto d_jammed_x = graph.mapper.get<JammedWaveTileNumber>(storeD, 0);
                graph.coordinates.addElement(PassThrough(), {WaveTilesX}, {d_jammed_x});
                auto d_jammed_y = graph.mapper.get<JammedWaveTileNumber>(storeD, 1);
                graph.coordinates.addElement(PassThrough(), {WaveTilesY}, {d_jammed_y});
            }

            // add LDS Ops for A and/or B if needed
            addLoadWaveLDSOps(graph, params, context, a, loadA[0], forK, K, waveMult);
            addLoadWaveLDSOps(graph, params, context, b, loadB[0], forK, K, waveMult);

            // add LDS Ops for D if needed
            if(storeD > 0)
            {
                auto d = graph.mapper.get<MacroTile>(storeD);
                addStoreWaveLDSOps(graph, params, context, d, storeD, forWaveTilesX);
            }

            addConnectionsMultiply(graph, waveMult);
        }

        KernelGraph lowerTile(KernelGraph                        graph,
                              std::shared_ptr<CommandParameters> params,
                              std::shared_ptr<Context>           context)
        {
            TIMER(t, "KernelGraph::lowerTile");
            rocRoller::Log::getLogger()->debug("KernelGraph::lowerTile()");

            auto visitor = LowerTileVisitor(params, context);
            auto kgraph  = rewrite(graph, visitor);

            auto contractions = kgraph.control.getNodes<TensorContraction>().to<std::vector>();
            AssertFatal(contractions.size() <= 1,
                        "More than one TensorContraction not supported yet.");

            if(contractions.size() == 1)
            {
                auto tag         = contractions[0];
                auto op          = kgraph.control.getNode<TensorContraction>(tag);
                auto macrotile_a = kgraph.coordinates.getNode<MacroTile>(op.a);
                auto macrotile_b = kgraph.coordinates.getNode<MacroTile>(op.b);
                auto d           = kgraph.mapper.get<MacroTile>(tag);
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
                visit(rocRoller::overloaded{
                          [&](auto op) {},
                          [&](LoadTiled const& load) { addLDSOps(kgraph, context, tag); }},
                      std::get<Operation>(elem));
            }

            return kgraph;
        }

    }
}
