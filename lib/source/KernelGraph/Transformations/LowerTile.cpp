
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

        // These functions are declared in AddLDS.cpp
        void addLDSOps(KernelGraph& graph, std::shared_ptr<Context> context, int tag);
        void addLoadWaveLDSOps(KernelGraph&                       graph,
                               std::shared_ptr<CommandParameters> params,
                               std::shared_ptr<Context>           context,
                               int                                tile,
                               int                                load,
                               int                                forK,
                               int                                K,
                               int                                waveMult);
        void addStoreWaveLDSOps(KernelGraph&                       graph,
                                std::shared_ptr<CommandParameters> params,
                                std::shared_ptr<Context>           context,
                                int                                tile,
                                int                                store,
                                int                                upperLoop);

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

            graph.mapper.connect(
                waveMult, d, Connections::typeArgument<MacroTile>(NaryArgument::DEST));

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
                auto tag            = contractions[0];
                auto op             = kgraph.control.getNode<TensorContraction>(tag);
                auto [a_tag, a_mac] = kgraph.getDimension<MacroTile>(tag, NaryArgument::LHS);
                auto [b_tag, b_mac] = kgraph.getDimension<MacroTile>(tag, NaryArgument::RHS);
                auto [d_tag, d_amc] = kgraph.getDimension<MacroTile>(tag, NaryArgument::DEST);
                if(a_mac.rank == 2 && b_mac.rank == 2 && op.aDims == std::vector<int>{1}
                   && op.bDims == std::vector<int>{0})
                {
                    lowerMatrixMultiply(kgraph, tag, a_tag, b_tag, d_tag, params, context);
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
