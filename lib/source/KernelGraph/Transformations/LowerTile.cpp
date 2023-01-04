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
        void addLDSOps(KernelHypergraph& graph, std::shared_ptr<Context> context, int tag)
        {
            auto user_tag = graph.mapper.get<CoordGraph::User>(tag);
            auto tile_tag = graph.mapper.get<CoordGraph::MacroTile>(tag);
            auto tile     = graph.coordinates.getNode<CoordGraph::MacroTile>(tile_tag);
            auto load     = graph.control.getNode<ControlHypergraph::LoadTiled>(tag);

            if(tile.memoryType == MemoryType::LDS)
            {
                auto lds = graph.coordinates.addElement(CoordGraph::LDS());

                auto storeLDS = graph.control.addElement(
                    ControlHypergraph::StoreLDSTile(load.vtype.dataType));
                auto barrier = graph.control.addElement(ControlHypergraph::Barrier());
                auto loadLDS = graph.control.addElement(ControlHypergraph::LoadLDSTile(load.vtype));

                graph.mapper.connect<CoordGraph::LDS>(storeLDS, lds);
                graph.mapper.connect<CoordGraph::MacroTile>(storeLDS, tile_tag);

                graph.mapper.connect<CoordGraph::LDS>(loadLDS, lds);
                graph.mapper.connect<CoordGraph::MacroTile>(loadLDS, tile_tag);

                auto workgroupSizes = context->kernel()->workgroupSize();
                loadMacroTileFromLDS(graph, loadLDS, lds, tile_tag, workgroupSizes);
                storeMacroTileIntoLDS(graph, storeLDS, lds, tile_tag, workgroupSizes);

                graph.coordinates.addElement(CoordGraph::DataFlow(), {user_tag}, {tile_tag});

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

        void addWaveLDSOps(KernelHypergraph&                  graph,
                           std::shared_ptr<CommandParameters> params,
                           std::shared_ptr<Context>           context,
                           int                                tile,
                           int                                load,
                           int                                forK,
                           int                                K,
                           int                                waveMult)
        {
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::LowerTileVisitor::addWaveLDSOps: LoadTiled({})", load);

            auto macrotile = graph.coordinates.getNode<CoordGraph::MacroTile>(tile);

            if(macrotile.memoryType == MemoryType::WAVE_LDS)
            {
                // change loadTiled to LoadLDSTile under Multiply
                // and update its macrotile's memory type
                auto vtype = graph.control.getNode<ControlHypergraph::LoadTiled>(load).vtype;
                graph.control.setElement(load, ControlHypergraph::LoadLDSTile(vtype));
                macrotile.memoryType = MemoryType::WAVE;
                graph.coordinates.setElement(tile, macrotile);

                auto user = graph.coordinates
                                .getInputNodeIndices(tile, CoordGraph::isEdge<CoordGraph::DataFlow>)
                                .to<std::vector>();
                AssertFatal(user.size() == 1);
                graph.coordinates.deleteElement(
                    user, std::vector<int>{tile}, CoordGraph::isEdge<CoordGraph::DataFlow>);
                auto sdims
                    = graph.coordinates
                          .getOutputNodeIndices(user[0], CoordGraph::isEdge<CoordGraph::Split>)
                          .to<std::vector>();

                auto lds = graph.coordinates.addElement(CoordGraph::LDS());

                // remove workgroups, macrotile numbers and tile edges from sdims
                loadWaveMacroTileFromLDS(graph, macrotile, load, sdims, K, lds);

                // add new loadTiled node to load a macrotile into VGPRs from global memory under the forK loop
                auto load_macrotile_from_global
                    = graph.control.addElement(ControlHypergraph::LoadTiled(vtype));
                graph.control.addElement(
                    ControlHypergraph::Body(), {forK}, {load_macrotile_from_global});
                graph.mapper.connect<CoordGraph::User>(load_macrotile_from_global, user[0]);

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
                    CoordGraph::MacroTile(macrotile.sizes, MemoryType::VGPR, {t_m, t_n}));
                auto internalTileDim
                    = graph.coordinates.getNode<CoordGraph::MacroTile>(internalTile);
                internalTileDim.layoutType = macrotile.layoutType;
                graph.coordinates.setElement(internalTile, internalTileDim);
                graph.mapper.connect<CoordGraph::MacroTile>(load_macrotile_from_global,
                                                            internalTile);

                // user --DataFlow--> internalTile
                graph.coordinates.addElement(CoordGraph::DataFlow(), {user[0]}, {internalTile});

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
                    = graph.control.addElement(ControlHypergraph::StoreLDSTile(vtype.dataType));
                auto barrier = graph.control.addElement(ControlHypergraph::Barrier());
                graph.control.addElement(ControlHypergraph::Sequence(),
                                         {load_macrotile_from_global},
                                         {store_macrotile_into_LDS});
                graph.control.addElement(
                    ControlHypergraph::Sequence(), {store_macrotile_into_LDS}, {barrier});
                graph.control.addElement(ControlHypergraph::Sequence(), {barrier}, {waveMult});
                graph.mapper.connect<CoordGraph::MacroTile>(store_macrotile_into_LDS, internalTile);

                // lower tile StoreLDSTile : store macrotile into LDS
                storeMacroTileIntoLDS(
                    graph, store_macrotile_into_LDS, lds, internalTile, workgroupSizes);

                // LDS --DataFlow--> macrotile
                graph.coordinates.addElement(CoordGraph::DataFlow(), {lds}, {tile});

                graph.mapper.connect<CoordGraph::LDS>(load, lds);
                graph.mapper.connect<CoordGraph::LDS>(load_macrotile_from_global, lds);
                graph.mapper.connect<CoordGraph::LDS>(store_macrotile_into_LDS, lds);
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
            graph.mapper.disconnect<CoordGraph::Workgroup>(loadA[0], a_workgroup_y, 1);
            graph.coordinates.deleteElement(a_workgroup_y);

            // remove passthrough between B row block and x-workgroup
            auto b_tilenum_x   = graph.mapper.get<CoordGraph::MacroTileNumber>(loadB[0], 0);
            auto b_workgroup_x = graph.mapper.get<CoordGraph::Workgroup>(loadB[0], 0);
            graph.coordinates.deleteElement(std::vector<int>{b_tilenum_x},
                                            std::vector<int>{b_workgroup_x},
                                            CoordGraph::isEdge<CoordGraph::PassThrough>);
            graph.mapper.disconnect<CoordGraph::Workgroup>(loadB[0], b_workgroup_x, 0);
            graph.coordinates.deleteElement(b_workgroup_x);

            // A row block is x-workgroup, column block is for loop index
            graph.coordinates.addElement(CoordGraph::PassThrough(), {a_tilenum_y}, {K});

            // B row block is for loop index, column block is y-workgroup
            graph.coordinates.addElement(CoordGraph::PassThrough(), {b_tilenum_x}, {K});

            // TODO : create helper functions to make this lowering modular and readable.
            auto waveMult = graph.control.addElement(ControlHypergraph::Multiply());
            // connections are: 0: lhs (A); 1: rhs (B); 2: dst (D)
            graph.mapper.connect<CoordGraph::MacroTile>(waveMult, d, 2);

            auto [waveA_tag, waveA] = graph.getDimension<CoordGraph::WaveTile>(loadA[0]);
            auto [waveB_tag, waveB] = graph.getDimension<CoordGraph::WaveTile>(loadB[0]);
            uint num_elements       = waveA.sizes[0] * waveB.sizes[1];
            uint wfs                = context->kernel()->wavefront_size();
            uint num_agpr           = num_elements / wfs; // number of output registers per thread

            auto initD = graph.control.addElement(
                ControlHypergraph::Assign{Register::Type::Accumulator, literal(0.f), num_agpr});

            graph.mapper.connect<CoordGraph::MacroTile>(initD, d);

            graph.control.addElement(ControlHypergraph::Sequence(), {initD}, {forK});
            graph.control.addElement(ControlHypergraph::Body(), {forK}, {waveMult});
            graph.control.addElement(ControlHypergraph::Body(), {waveMult}, {loadA[0]});
            graph.control.addElement(ControlHypergraph::Body(), {waveMult}, {loadB[0]});

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

                graph.control.addElement(ControlHypergraph::Body(), {lowerLoop}, {initD});

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
            {
                graph.control.addElement(ControlHypergraph::Body(), {kernel}, {initD});
            }

            // add LDSOps if needed
            addWaveLDSOps(graph, params, context, a, loadA[0], forK, K, waveMult);
            addWaveLDSOps(graph, params, context, b, loadB[0], forK, K, waveMult);

            addConnectionsMultiply(graph, waveMult);
        }

        KernelHypergraph addComputeIndexOperations(KernelHypergraph const& original)
        {
            auto kgraph = original;
            auto kernel = *kgraph.control.roots().begin();

            // MATRIX_A and MATRIX_B loads within a ForLoop
            auto multiplies
                = kgraph.control.getNodes<ControlHypergraph::Multiply>().to<std::vector>();
            AssertFatal(multiplies.size() <= 1, "More than one Multiply not supported yet.");
            std::vector<int> forK;
            if(multiplies.size() == 1)
            {
                forK = kgraph.control.getInputNodeIndices<ControlHypergraph::Body>(multiplies[0])
                           .to<std::vector>();
                AssertFatal(forK.size() == 1, "Didn't find for loop.");
                auto mulLoads = kgraph.control
                                    .findNodes(
                                        multiplies[0],
                                        [&](int tag) -> bool {
                                            return isOperation<ControlHypergraph::LoadTiled>(
                                                       kgraph.control.getElement(tag))
                                                   || isOperation<ControlHypergraph::LoadLDSTile>(
                                                       kgraph.control.getElement(tag));
                                        },
                                        Graph::Direction::Downstream)
                                    .to<std::vector>();
                AssertFatal(mulLoads.size() == 2, "More than one Multiply not supported yet.");
                auto storesLDS = kgraph.control
                                     .findNodes(
                                         forK[0],
                                         [&](int tag) -> bool {
                                             return isOperation<ControlHypergraph::StoreLDSTile>(
                                                 kgraph.control.getElement(tag));
                                         },
                                         Graph::Direction::Downstream)
                                     .to<std::vector>();
                auto loads
                    = kgraph.control
                          .findNodes(
                              forK[0],
                              [&](int tag) -> bool {
                                  if(isOperation<ControlHypergraph::LoadTiled>(
                                         kgraph.control.getElement(tag)))
                                  {
                                      auto parents
                                          = kgraph.control.parentNodes(tag).to<std::vector>();
                                      AssertFatal(parents.size() == 1);
                                      return parents[0] != multiplies[0];
                                  }
                                  return false;
                              },
                              Graph::Direction::Downstream)
                          .to<std::vector>();
                AssertFatal(storesLDS.size() == loads.size(),
                            "Either store LDS or load is missing");
                if(storesLDS.size() == 0)
                    kgraph = addComputeIndexAB(
                        kgraph, forK[0], mulLoads[0], mulLoads[1], -1, -1, -1, -1);
                else if(storesLDS.size() == 1
                        && isOperation<ControlHypergraph::LoadLDSTile>(
                            kgraph.control.getElement(mulLoads[0])))
                    kgraph = addComputeIndexAB(
                        kgraph, forK[0], mulLoads[0], mulLoads[1], loads[0], storesLDS[0], -1, -1);
                else if(storesLDS.size() == 1
                        && isOperation<ControlHypergraph::LoadLDSTile>(
                            kgraph.control.getElement(mulLoads[1])))
                    kgraph = addComputeIndexAB(
                        kgraph, forK[0], mulLoads[0], mulLoads[1], -1, -1, loads[0], storesLDS[0]);
                else if(storesLDS.size() == 2)
                    kgraph = addComputeIndexAB(kgraph,
                                               forK[0],
                                               mulLoads[0],
                                               mulLoads[1],
                                               loads[0],
                                               storesLDS[0],
                                               loads[1],
                                               storesLDS[1]);
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

            // VGPR/LDS loads anywhere
            std::unordered_set<int> reachable_from_forK;
            if(!forK.empty())
                reachable_from_forK
                    = kgraph.control.depthFirstVisit(forK[0]).to<std::unordered_set>();
            auto loadVGPR
                = kgraph.control
                      .findNodes(
                          kernel,
                          [&](int tag) -> bool {
                              if(reachable_from_forK.find(tag) != reachable_from_forK.end())
                                  return false;
                              auto load = kgraph.control.get<ControlHypergraph::LoadTiled>(tag);
                              auto loadLDS
                                  = kgraph.control.get<ControlHypergraph::LoadLDSTile>(tag);
                              if(load || loadLDS)
                              {
                                  auto [tile_tag, tile]
                                      = kgraph.getDimension<CoordGraph::MacroTile>(tag);
                                  if(tile.memoryType == MemoryType::VGPR
                                     || tile.memoryType == MemoryType::LDS)
                                      return true;
                              }
                              return false;
                          },
                          Graph::Direction::Downstream)
                      .to<std::vector>();

            for(auto const tag : loadVGPR)
            {
                kgraph = addComputeIndexVGPR(kgraph, tag, tag, false);
            }

            // MATRIX_ACCUMULATOR stores anywhere
            auto storeAccums
                = kgraph.control
                      .findNodes(
                          kernel,
                          [&](int tag) -> bool {
                              auto store = kgraph.control.get<ControlHypergraph::StoreTiled>(tag);
                              if(store)
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

            // VGPR/LDS stores anywhere
            auto storeVGPR
                = kgraph.control
                      .findNodes(
                          kernel,
                          [&](int tag) -> bool {
                              if(reachable_from_forK.find(tag) != reachable_from_forK.end())
                                  return false;
                              auto store = kgraph.control.get<ControlHypergraph::StoreTiled>(tag);
                              auto storeLDS
                                  = kgraph.control.get<ControlHypergraph::StoreLDSTile>(tag);
                              if(store || storeLDS)
                              {
                                  auto [tile_tag, tile]
                                      = kgraph.getDimension<CoordGraph::MacroTile>(tag);
                                  if(tile.memoryType == MemoryType::VGPR
                                     || tile.memoryType == MemoryType::LDS)
                                      return true;
                              }
                              return false;
                          },
                          Graph::Direction::Downstream)
                      .to<std::vector>();

            for(auto const tag : storeVGPR)
            {
                kgraph = addComputeIndexVGPR(kgraph, tag, tag, true);
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

            // Find WAVE tile load/store operations and add ComputeIndex operations
            kgraph = addComputeIndexOperations(kgraph);

            return kgraph;
        }

    }
}
