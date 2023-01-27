
#include "DataTypes/DataTypes.hpp"
#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        using namespace CoordinateGraph;
        using namespace ControlGraph;
        namespace Expression = rocRoller::Expression;
        using namespace Expression;

        /**
         * @brief Replace operation with a scope.  Does not delete the original operation.
         */
        int replaceWithScope(KernelGraph& graph, int op)
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
                    if(std::holds_alternative<Sequence>(cedge))
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

        /**
         * Helper to make a ComputeIndex node and add connections.
         */
        int makeComputeIndex(KernelGraph&     graph,
                             int              target,
                             int              increment,
                             int              base,
                             int              offset,
                             int              stride,
                             int              buffer,
                             bool             forward,
                             DataType         valueType,
                             std::vector<int> zero       = {},
                             DataType         offsetType = DataType::UInt64,
                             DataType         strideType = DataType::UInt64)
        {
            auto ci = graph.control.addElement(
                ComputeIndex(forward, valueType, offsetType, strideType));
            if(base > 0)
            {
                graph.mapper.connect(
                    ci, base, Connections::ComputeIndex{Connections::ComputeIndexArgument::BASE});
            }
            if(buffer > 0)
            {
                graph.mapper.connect(
                    ci,
                    buffer,
                    Connections::ComputeIndex{Connections::ComputeIndexArgument::BUFFER});
            }
            if(increment > 0)
            {
                graph.mapper.connect(
                    ci,
                    increment,
                    Connections::ComputeIndex{Connections::ComputeIndexArgument::INCREMENT});
            }
            if(offset > 0)
            {
                graph.mapper.connect(
                    ci,
                    offset,
                    Connections::ComputeIndex{Connections::ComputeIndexArgument::OFFSET});
            }
            if(stride > 0)
            {
                graph.mapper.connect(
                    ci,
                    stride,
                    Connections::ComputeIndex{Connections::ComputeIndexArgument::STRIDE});
            }
            if(target > 0)
            {
                graph.mapper.connect(
                    ci,
                    target,
                    Connections::ComputeIndex{Connections::ComputeIndexArgument::TARGET});
            }
            for(int i = 0; i < zero.size(); ++i)
            {
                graph.mapper.connect(
                    ci,
                    zero[i],
                    Connections::ComputeIndex{Connections::ComputeIndexArgument::ZERO, i});
            }
            return ci;
        }

        std::tuple<int, int, int> computeIndexVGPRMATRIXAB(KernelGraph& graph, int load, int sdim)
        {
            AssertFatal(isOperation<LoadTiled>(graph.control.getElement(load)));

            auto user    = graph.mapper.get<User>(load);
            auto mac     = graph.mapper.get<MacroTileNumber>(load, sdim);
            auto i_thr_x = graph.mapper.get<ThreadTileIndex>(load, 0);
            auto i_thr_y = graph.mapper.get<ThreadTileIndex>(load, 1);

            auto dtype = graph.control.get<LoadTiled>(load)->vtype.dataType;

            auto offset_mac = graph.coordinates.addElement(Offset(), {user}, {mac});
            auto stride_mac = graph.coordinates.addElement(Stride(), {user}, {mac});
            auto row_offset = graph.coordinates.addElement(Offset(), {user}, {i_thr_x});
            auto row_stride = graph.coordinates.addElement(Stride(), {user}, {i_thr_x});
            auto col_offset = graph.coordinates.addElement(Offset(), {user}, {i_thr_y});
            auto col_stride = graph.coordinates.addElement(Stride(), {user}, {i_thr_y});
            auto buffer     = graph.coordinates.addElement(Buffer(), {user}, {i_thr_x});

            graph.mapper.connect<Offset>(load, offset_mac, -1);
            graph.mapper.connect<Offset>(load, row_offset, 0);
            graph.mapper.connect<Offset>(load, col_offset, 1);
            graph.mapper.connect<Stride>(load, stride_mac, -1);
            graph.mapper.connect<Stride>(load, row_stride, 0);
            graph.mapper.connect<Stride>(load, col_stride, 1);
            graph.mapper.connect<Buffer>(load, buffer);

            auto ci_mac = makeComputeIndex(graph,
                                           user,
                                           mac,
                                           -1,
                                           offset_mac,
                                           stride_mac,
                                           buffer,
                                           false,
                                           dtype,
                                           {i_thr_x, i_thr_y});
            auto ci_row = makeComputeIndex(graph,
                                           user,
                                           i_thr_x,
                                           mac,
                                           row_offset,
                                           row_stride,
                                           buffer,
                                           false,
                                           dtype,
                                           {mac, i_thr_y});
            auto ci_col = makeComputeIndex(graph,
                                           user,
                                           i_thr_y,
                                           i_thr_x,
                                           col_offset,
                                           col_stride,
                                           buffer,
                                           false,
                                           dtype,
                                           {mac, i_thr_x});

            graph.control.addElement(Sequence(), {ci_mac}, {ci_row});
            graph.control.addElement(Sequence(), {ci_row}, {ci_col});

            auto offset_mac_expr = std::make_shared<Expression::Expression>(
                Expression::DataFlowTag{offset_mac, Register::Type::Vector, DataType::UInt64});
            auto stride_mac_expr = std::make_shared<Expression::Expression>(
                Expression::DataFlowTag{stride_mac, Register::Type::Scalar, DataType::UInt64});

            auto offsetUpdate = graph.control.addElement(
                Assign{Register::Type::Vector, offset_mac_expr + stride_mac_expr});
            graph.mapper.connect<Dimension>(offsetUpdate, offset_mac);

            return {ci_mac, ci_col, offsetUpdate};
        }

        std::tuple<int, int>
            computeIndexVGPR(KernelGraph& graph, int loadstore, int source, bool forward)
        {
            auto i_thr_x = graph.mapper.get<ThreadTileIndex>(loadstore, 0);
            auto i_thr_y = graph.mapper.get<ThreadTileIndex>(loadstore, 1);

            DataType dtype, offsettype = DataType::UInt64;
            {
                auto l  = graph.control.get<LoadTiled>(loadstore);
                auto ll = graph.control.get<LoadLDSTile>(loadstore);
                auto s  = graph.control.get<StoreTiled>(loadstore);
                auto sl = graph.control.get<StoreLDSTile>(loadstore);
                if(l)
                    dtype = l->vtype.dataType;
                if(ll)
                {
                    dtype      = ll->vtype.dataType;
                    offsettype = DataType::UInt32;
                }
                if(s)
                    dtype = s->dataType;
                if(sl)
                {
                    dtype      = sl->dataType;
                    offsettype = DataType::UInt32;
                }
            }

            int row_offset, row_stride, col_offset, col_stride, buffer;
            if(forward)
            {
                row_offset = graph.coordinates.addElement(Offset(), {i_thr_x}, {source});
                row_stride = graph.coordinates.addElement(Stride(), {i_thr_x}, {source});
                col_offset = graph.coordinates.addElement(Offset(), {i_thr_y}, {source});
                col_stride = graph.coordinates.addElement(Stride(), {i_thr_y}, {source});
                buffer     = graph.coordinates.addElement(Buffer(), {i_thr_x}, {source});
            }
            else
            {
                row_offset = graph.coordinates.addElement(Offset(), {source}, {i_thr_x});
                row_stride = graph.coordinates.addElement(Stride(), {source}, {i_thr_x});
                col_offset = graph.coordinates.addElement(Offset(), {source}, {i_thr_y});
                col_stride = graph.coordinates.addElement(Stride(), {source}, {i_thr_y});
                buffer     = graph.coordinates.addElement(Buffer(), {source}, {i_thr_x});
            }

            graph.mapper.connect<Offset>(loadstore, row_offset, 0);
            graph.mapper.connect<Offset>(loadstore, col_offset, 1);
            graph.mapper.connect<Stride>(loadstore, row_stride, 0);
            graph.mapper.connect<Stride>(loadstore, col_stride, 1);
            graph.mapper.connect<Buffer>(loadstore, buffer);

            auto ci_row = makeComputeIndex(graph,
                                           source,
                                           i_thr_x,
                                           -1,
                                           row_offset,
                                           row_stride,
                                           buffer,
                                           forward,
                                           dtype,
                                           {i_thr_y},
                                           offsettype,
                                           offsettype);
            auto ci_col = makeComputeIndex(graph,
                                           source,
                                           i_thr_y,
                                           i_thr_x,
                                           col_offset,
                                           col_stride,
                                           buffer,
                                           forward,
                                           dtype,
                                           {i_thr_x},
                                           offsettype,
                                           offsettype);

            return {ci_row, ci_col};
        }

        /**
         * @brief Add ComputeIndex operations to graph for a MATRIX_A or MATRIX_B LoadLDSTile.
         */
        std::tuple<int, int> computeIndexLDSMATRIXAB(KernelGraph& graph, int load, int sdim)
        {
            AssertFatal(isOperation<LoadLDSTile>(graph.control.getElement(load)));
            auto lds  = graph.mapper.get<LDS>(load);
            auto wave = graph.mapper.get<WaveTileNumber>(load, sdim);
            auto vgpr = graph.mapper.get<VGPR>(load);

            auto dtype = graph.control.get<LoadLDSTile>(load)->vtype.dataType;

            auto offset_wave = graph.coordinates.addElement(Offset(), {lds}, {wave});
            auto stride_wave = graph.coordinates.addElement(Stride(), {lds}, {wave});
            auto offset_vgpr = graph.coordinates.addElement(Offset(), {lds}, {vgpr});
            auto stride_vgpr = graph.coordinates.addElement(Stride(), {lds}, {vgpr});

            graph.mapper.connect<Offset>(load, offset_wave, 0);
            graph.mapper.connect<Offset>(load, offset_vgpr, 1);
            graph.mapper.connect<Stride>(load, stride_wave, 0);
            graph.mapper.connect<Stride>(load, stride_vgpr, 1);

            auto ci_wave = makeComputeIndex(graph,
                                            lds,
                                            wave,
                                            -1,
                                            offset_wave,
                                            stride_wave,
                                            -1,
                                            false,
                                            dtype,
                                            {vgpr},
                                            DataType::UInt32,
                                            DataType::UInt32);
            auto ci_vgpr = makeComputeIndex(graph,
                                            lds,
                                            vgpr,
                                            offset_wave,
                                            offset_vgpr,
                                            stride_vgpr,
                                            -1,
                                            false,
                                            dtype,
                                            {wave},
                                            DataType::UInt32,
                                            DataType::UInt32);

            graph.control.addElement(Sequence(), {ci_wave}, {ci_vgpr});

            return {ci_wave, ci_vgpr};
        }

        /**
         * @brief Add ComputeIndex operations to graph for a MATRIX_A or MATRIX_B LoadTiled.
         */
        std::tuple<int, int, int> computeIndexMATRIXAB(KernelGraph& graph, int load, int sdim)
        {
            auto user = graph.mapper.get<User>(load);
            auto mac  = graph.mapper.get<MacroTileNumber>(load, sdim);
            auto wave = graph.mapper.get<WaveTileNumber>(load, sdim);
            auto vgpr = graph.mapper.get<VGPR>(load);

            auto dtype = graph.control.get<LoadTiled>(load)->vtype.dataType;

            auto offset_mac  = graph.coordinates.addElement(Offset(), {user}, {mac});
            auto stride_mac  = graph.coordinates.addElement(Stride(), {user}, {mac});
            auto offset_wave = graph.coordinates.addElement(Offset(), {user}, {wave});
            auto stride_wave = graph.coordinates.addElement(Stride(), {user}, {wave});
            auto offset_vgpr = graph.coordinates.addElement(Offset(), {user}, {vgpr});
            auto stride_vgpr = graph.coordinates.addElement(Stride(), {user}, {vgpr});
            auto buffer      = graph.coordinates.addElement(Buffer(), {user}, {mac});

            graph.mapper.connect<Offset>(load, offset_mac, -1);
            graph.mapper.connect<Offset>(load, offset_wave, 0);
            graph.mapper.connect<Offset>(load, offset_vgpr, 1);
            graph.mapper.connect<Stride>(load, stride_mac, -1);
            graph.mapper.connect<Stride>(load, stride_wave, 0);
            graph.mapper.connect<Stride>(load, stride_vgpr, 1);
            graph.mapper.connect<Buffer>(load, buffer);

            auto ci_mac = makeComputeIndex(
                graph, user, mac, -1, offset_mac, stride_mac, buffer, false, dtype, {wave, vgpr});
            auto ci_wave = makeComputeIndex(graph,
                                            user,
                                            wave,
                                            offset_mac,
                                            offset_wave,
                                            stride_wave,
                                            buffer,
                                            false,
                                            dtype,
                                            {mac, vgpr});
            auto ci_vgpr = makeComputeIndex(graph,
                                            user,
                                            vgpr,
                                            offset_wave,
                                            offset_vgpr,
                                            stride_vgpr,
                                            buffer,
                                            false,
                                            dtype,
                                            {mac, wave});

            graph.control.addElement(Sequence(), {ci_mac}, {ci_wave});
            graph.control.addElement(Sequence(), {ci_wave}, {ci_vgpr});

            auto offset_mac_expr = std::make_shared<Expression::Expression>(
                Expression::DataFlowTag{offset_mac, Register::Type::Vector, DataType::UInt64});
            auto stride_mac_expr = std::make_shared<Expression::Expression>(
                Expression::DataFlowTag{stride_mac, Register::Type::Scalar, DataType::UInt64});

            auto offsetUpdate = graph.control.addElement(
                Assign{Register::Type::Vector, offset_mac_expr + stride_mac_expr});
            graph.mapper.connect<Dimension>(offsetUpdate, offset_mac);

            return {ci_mac, ci_vgpr, offsetUpdate};
        }

        /**
         * @brief Add ComputeIndex operations to graph for MATRIX_A and MATRIX_B loads.
         */
        void addComputeIndexAB(KernelGraph& graph,
                               int          op,
                               int          scope,
                               int          setCoord,
                               int          mulLoadA,
                               int          mulLoadB,
                               int          loadA,
                               int          storeLDSA,
                               int          loadB,
                               int          storeLDSB)
        {
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::addComputeIndexAB({}, {}, {})", op, mulLoadA, mulLoadB);

            int top = scope;

            // If the loads are sitting under a SetCoordinate node inside the ForLoop,
            // add the SetCoordinate node underneath the scope. All of the ComputeIndex
            // operations should then be under the SetCoordinate node.
            if(setCoord >= 0)
            {
                auto newSetCoord = graph.control.addElement(graph.control.getElement(setCoord));
                graph.control.addElement(Body(), {scope}, {newSetCoord});
                for(auto const& c : graph.mapper.getConnections(setCoord))
                {
                    graph.mapper.connect(newSetCoord, c.coordinate, c.connection);
                }
                top = newSetCoord;
                graph.control.addElement(Sequence(), {newSetCoord}, {op});
            }

            // MATRIX_A; y is summation
            auto nodeA = graph.control.getElement(mulLoadA);
            // LoadTiled A
            if(isOperation<LoadTiled>(nodeA))
            {
                auto [topA, bottomA, updateA] = computeIndexMATRIXAB(graph, mulLoadA, 1);
                graph.control.addElement(Body(), {top}, {topA});
                if(setCoord < 0)
                    graph.control.addElement(Sequence(), {bottomA}, {op});
                graph.control.addElement(ForLoopIncrement(), {op}, {updateA});
            }
            // LoadLDSTile A
            else if(isOperation<LoadLDSTile>(nodeA))
            {
                auto [topA, bottomA] = computeIndexLDSMATRIXAB(graph, mulLoadA, 1);
                graph.control.addElement(Body(), {top}, {topA});
                if(setCoord < 0)
                    graph.control.addElement(Sequence(), {bottomA}, {op});
            }

            // MATRIX_B; x is summation
            auto nodeB = graph.control.getElement(mulLoadB);
            // LoadTiled B
            if(isOperation<LoadTiled>(nodeB))
            {
                auto [topB, bottomB, updateB] = computeIndexMATRIXAB(graph, mulLoadB, 0);
                graph.control.addElement(Body(), {top}, {topB});
                if(setCoord < 0)
                    graph.control.addElement(Sequence(), {bottomB}, {op});
                graph.control.addElement(ForLoopIncrement(), {op}, {updateB});
            }
            // LoadLDSTile B
            else if(isOperation<LoadLDSTile>(nodeB))
            {
                auto [topB, bottomB] = computeIndexLDSMATRIXAB(graph, mulLoadB, 0);
                graph.control.addElement(Body(), {top}, {topB});
                if(setCoord < 0)
                    graph.control.addElement(Sequence(), {bottomB}, {op});
            }

            if(loadA > 0 && storeLDSA > 0)
            {
                // LoadTiled A
                auto [topA, bottomA, updateA] = computeIndexVGPRMATRIXAB(graph, loadA, 1);
                graph.control.addElement(Body(), {top}, {topA});
                if(setCoord < 0)
                    graph.control.addElement(Sequence(), {bottomA}, {op});
                graph.control.addElement(ForLoopIncrement(), {op}, {updateA});

                // StoreLDSTile A
                auto lds                          = graph.mapper.get<LDS>(storeLDSA);
                auto [store_ci_row, store_ci_col] = computeIndexVGPR(graph, storeLDSA, lds, true);
                graph.control.addElement(Body(), {top}, {store_ci_row});
                graph.control.addElement(Sequence(), {store_ci_row}, {store_ci_col});
                if(setCoord < 0)
                    graph.control.addElement(Sequence(), {store_ci_col}, {op});
            }

            if(loadB > 0 && storeLDSB > 0)
            {
                // LoadTiled B
                auto [topB, bottomB, updateB] = computeIndexVGPRMATRIXAB(graph, loadB, 0);
                graph.control.addElement(Body(), {top}, {topB});
                if(setCoord < 0)
                    graph.control.addElement(Sequence(), {bottomB}, {op});
                graph.control.addElement(ForLoopIncrement(), {op}, {updateB});

                // StoreLDSTile B
                auto lds                          = graph.mapper.get<LDS>(storeLDSB);
                auto [store_ci_row, store_ci_col] = computeIndexVGPR(graph, storeLDSB, lds, true);
                graph.control.addElement(Body(), {top}, {store_ci_row});
                graph.control.addElement(Sequence(), {store_ci_row}, {store_ci_col});
                if(setCoord < 0)
                    graph.control.addElement(Sequence(), {store_ci_col}, {op});
            }
        }

        /**
         * @brief Add ComputeIndex operations to graph for a MATRIX_ACCUM load/store.
         */
        void addComputeIndexC(KernelGraph& graph, int op, bool forward)
        {
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::addComputeIndexC({}, {})", op, forward);

            auto scope = replaceWithScope(graph, op);

            auto node   = graph.control.getElement(op);
            auto source = -1;
            if(isOperation<LoadTiled>(node) || isOperation<StoreTiled>(node))
                source = graph.mapper.get<User>(op);
            else if(isOperation<LoadLDSTile>(node) || isOperation<StoreLDSTile>(node))
                source = graph.mapper.get<LDS>(op);

            AssertFatal(source > 0, "User or LDS dimension not found");

            auto vgpr_block = graph.mapper.get<VGPRBlockNumber>(op);
            auto vgpr_index = graph.mapper.get<VGPRBlockIndex>(op);

            DataType dtype, offsettype = DataType::UInt64;
            {
                auto l  = graph.control.get<LoadTiled>(op);
                auto ll = graph.control.get<LoadLDSTile>(op);
                auto s  = graph.control.get<StoreTiled>(op);
                auto sl = graph.control.get<StoreLDSTile>(op);
                if(l)
                    dtype = l->vtype.dataType;
                if(ll)
                {
                    dtype      = ll->vtype.dataType;
                    offsettype = DataType::UInt32;
                }
                if(s)
                    dtype = s->dataType;
                if(sl)
                {
                    dtype      = sl->dataType;
                    offsettype = DataType::UInt32;
                }
            }

            int offset_vgpr_block, stride_vgpr_block, offset_vgpr_index, stride_vgpr_index, buffer;
            if(forward)
            {
                offset_vgpr_block = graph.coordinates.addElement(Offset(), {vgpr_block}, {source});
                stride_vgpr_block = graph.coordinates.addElement(Stride(), {vgpr_block}, {source});
                offset_vgpr_index = graph.coordinates.addElement(Offset(), {vgpr_index}, {source});
                stride_vgpr_index = graph.coordinates.addElement(Stride(), {vgpr_index}, {source});
                buffer            = graph.coordinates.addElement(Buffer(), {vgpr_block}, {source});
            }
            else
            {
                offset_vgpr_block = graph.coordinates.addElement(Offset(), {source}, {vgpr_block});
                stride_vgpr_block = graph.coordinates.addElement(Stride(), {source}, {vgpr_block});
                offset_vgpr_index = graph.coordinates.addElement(Offset(), {source}, {vgpr_index});
                stride_vgpr_index = graph.coordinates.addElement(Stride(), {source}, {vgpr_index});
                buffer            = graph.coordinates.addElement(Buffer(), {source}, {vgpr_block});
            }

            graph.mapper.connect<Offset>(op, offset_vgpr_block, 0);
            graph.mapper.connect<Offset>(op, offset_vgpr_index, 1);
            graph.mapper.connect<Stride>(op, stride_vgpr_block, 0);
            graph.mapper.connect<Stride>(op, stride_vgpr_index, 1);
            graph.mapper.connect<Buffer>(op, buffer);

            auto ci_vgpr_block = makeComputeIndex(graph,
                                                  source,
                                                  vgpr_block,
                                                  -1,
                                                  offset_vgpr_block,
                                                  stride_vgpr_block,
                                                  buffer,
                                                  forward,
                                                  dtype,
                                                  {vgpr_index},
                                                  offsettype,
                                                  offsettype);
            auto ci_vgpr_index = makeComputeIndex(graph,
                                                  source,
                                                  vgpr_index,
                                                  offset_vgpr_block,
                                                  offset_vgpr_index,
                                                  stride_vgpr_index,
                                                  buffer,
                                                  forward,
                                                  dtype,
                                                  {vgpr_block},
                                                  offsettype,
                                                  offsettype);

            graph.control.addElement(Body(), {scope}, {ci_vgpr_block});
            graph.control.addElement(Sequence(), {ci_vgpr_block}, {ci_vgpr_index});
            graph.control.addElement(Sequence(), {ci_vgpr_index}, {op});
        }

        /**
         * @brief Add ComputeIndex operations to graph for loads/stores.
         */
        void addComputeIndexVGPR(KernelGraph& graph, int op, bool forward)
        {
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::addComputeIndexVGPR({}, {})", op, forward);

            auto scope = replaceWithScope(graph, op);

            auto node   = graph.control.getElement(op);
            auto source = -1;
            if(isOperation<LoadTiled>(node) || isOperation<StoreTiled>(node))
                source = graph.mapper.get<User>(op);
            else if(isOperation<LoadLDSTile>(node) || isOperation<StoreLDSTile>(node))
                source = graph.mapper.get<LDS>(op);

            AssertFatal(source > 0, "User or LDS dimension not found");

            auto [ci_row, ci_col] = computeIndexVGPR(graph, op, source, forward);

            graph.control.addElement(Body(), {scope}, {ci_row});
            graph.control.addElement(Sequence(), {ci_row}, {ci_col});
            graph.control.addElement(Sequence(), {ci_col}, {op});
        }

        KernelGraph addComputeIndexOperations(KernelGraph const& original)
        {
            auto kgraph = original;
            auto kernel = *kgraph.control.roots().begin();

            // MATRIX_A and MATRIX_B loads within a ForLoop
            auto multiplies = kgraph.control.getNodes<Multiply>().to<std::vector>();

            std::map<int, int> forKScopes;
            for(auto const& multiply : multiplies)
            {
                // Find the first ForLoop above the Multiply
                auto allForLoops = kgraph.control.findNodes(
                    multiply,
                    [&](int tag) -> bool {
                        return isOperation<ForLoopOp>(kgraph.control.getElement(tag));
                    },
                    Graph::Direction::Upstream);
                auto forK  = *allForLoops.begin();
                int  scope = -1;
                // loads under Multiply
                auto mulLoads
                    = kgraph.control
                          .findNodes(
                              multiply,
                              [&](int tag) -> bool {
                                  return isOperation<LoadTiled>(kgraph.control.getElement(tag))
                                         || isOperation<LoadLDSTile>(
                                             kgraph.control.getElement(tag));
                              },
                              Graph::Direction::Downstream)
                          .to<std::vector>();
                AssertFatal(mulLoads.size() == 2,
                            "Multiply doesn't support more than two operands.");
                // Find all of the nodes inbetween the ForLoop and the Multiply
                auto pathToMultiply = kgraph.control
                                          .path<Graph::Direction::Downstream>(
                                              std::vector<int>{forK}, std::vector<int>{multiply})
                                          .to<std::vector>();

                // Find all of the StoreLDS nodes for A or B between the ForLoop and Multiply
                std::vector<int> storesLDSAB;
                std::copy_if(pathToMultiply.begin(),
                             pathToMultiply.end(),
                             std::back_inserter(storesLDSAB),
                             [&](int tag) -> bool {
                                 auto storeLDS = kgraph.control.get<StoreLDSTile>(tag);
                                 if(storeLDS)
                                 {
                                     auto [tile_tag, tile] = kgraph.getDimension<MacroTile>(tag);
                                     if(tile.layoutType != LayoutType::MATRIX_ACCUMULATOR)
                                         return true;
                                 }
                                 return false;
                             });

                // Find all of the LoadTiled nodes for A or B between the ForLoop and Multiply
                std::vector<int> loadsAB;
                std::copy_if(pathToMultiply.begin(),
                             pathToMultiply.end(),
                             std::back_inserter(loadsAB),
                             [&](int tag) -> bool {
                                 return isOperation<LoadTiled>(kgraph.control.getElement(tag));
                             });

                AssertFatal(storesLDSAB.size() == loadsAB.size(),
                            "Either store LDS or load is missing");

                // Find all of the SetCoordinate nodes between the ForLoop and Multiply
                std::vector<int> setCoords;
                std::copy_if(pathToMultiply.begin(),
                             pathToMultiply.end(),
                             std::back_inserter(setCoords),
                             [&](int tag) -> bool {
                                 return isOperation<SetCoordinate>(kgraph.control.getElement(tag));
                             });
                int setCoord = setCoords.empty() ? -1 : setCoords[0];

                // If a scope doesn't exist yet above the ForLoop, create one
                // and add scope to the forKScopes map.
                if(forKScopes.count(forK) == 0)
                {
                    forKScopes[forK] = replaceWithScope(kgraph, forK);
                }
                scope = forKScopes[forK];

                if(storesLDSAB.size() == 0)
                    addComputeIndexAB(
                        kgraph, forK, scope, setCoord, mulLoads[0], mulLoads[1], -1, -1, -1, -1);
                else if(storesLDSAB.size() == 1
                        && isOperation<LoadLDSTile>(kgraph.control.getElement(mulLoads[0])))
                    addComputeIndexAB(kgraph,
                                      forK,
                                      scope,
                                      setCoord,
                                      mulLoads[0],
                                      mulLoads[1],
                                      loadsAB[0],
                                      storesLDSAB[0],
                                      -1,
                                      -1);
                else if(storesLDSAB.size() == 1
                        && isOperation<LoadLDSTile>(kgraph.control.getElement(mulLoads[1])))
                    addComputeIndexAB(kgraph,
                                      forK,
                                      scope,
                                      setCoord,
                                      mulLoads[0],
                                      mulLoads[1],
                                      -1,
                                      -1,
                                      loadsAB[0],
                                      storesLDSAB[0]);
                else if(storesLDSAB.size() == 2)
                    addComputeIndexAB(kgraph,
                                      forK,
                                      scope,
                                      setCoord,
                                      mulLoads[0],
                                      mulLoads[1],
                                      loadsAB[0],
                                      storesLDSAB[0],
                                      loadsAB[1],
                                      storesLDSAB[1]);
            }

            // MATRIX_ACCUMULATOR loads anywhere
            auto loadAccums = kgraph.control
                                  .findNodes(
                                      kernel,
                                      [&](int tag) -> bool {
                                          auto load = kgraph.control.get<LoadTiled>(tag);
                                          if(load)
                                          {
                                              auto [tile_tag, tile]
                                                  = kgraph.getDimension<MacroTile>(tag);
                                              if(tile.layoutType == LayoutType::MATRIX_ACCUMULATOR)
                                                  return true;
                                          }
                                          return false;
                                      },
                                      Graph::Direction::Downstream)
                                  .to<std::vector>();

            for(auto const tag : loadAccums)
            {
                addComputeIndexC(kgraph, tag, false);
            }

            std::vector<int> allForKs;
            for(auto const& forKScope : forKScopes)
            {
                allForKs.push_back(forKScope.first);
            }

            auto reachable_from_forK
                = kgraph.control.depthFirstVisit(allForKs).to<std::unordered_set>();

            // VGPR/LDS loads anywhere other than under ForK(s)
            auto loadVGPR
                = kgraph.control
                      .findNodes(
                          kernel,
                          [&](int tag) -> bool {
                              if(reachable_from_forK.find(tag) != reachable_from_forK.end())
                                  return false;
                              auto load    = kgraph.control.get<LoadTiled>(tag);
                              auto loadLDS = kgraph.control.get<LoadLDSTile>(tag);
                              if(load || loadLDS)
                              {
                                  auto [tile_tag, tile] = kgraph.getDimension<MacroTile>(tag);
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
                addComputeIndexVGPR(kgraph, tag, false);
            }

            // MATRIX_ACCUMULATOR & WAVE stores anywhere
            auto storeAccums = kgraph.control
                                   .findNodes(
                                       kernel,
                                       [&](int tag) -> bool {
                                           auto store    = kgraph.control.get<StoreTiled>(tag);
                                           auto storeLDS = kgraph.control.get<StoreLDSTile>(tag);
                                           if(store || storeLDS)
                                           {
                                               auto [tile_tag, tile]
                                                   = kgraph.getDimension<MacroTile>(tag);
                                               if(tile.layoutType == LayoutType::MATRIX_ACCUMULATOR
                                                  && tile.memoryType == MemoryType::WAVE)
                                                   return true;
                                           }
                                           return false;
                                       },
                                       Graph::Direction::Downstream)
                                   .to<std::vector>();

            for(auto const tag : storeAccums)
            {
                addComputeIndexC(kgraph, tag, true);
            }

            // VGPR/LDS stores anywhere
            auto storeVGPR
                = kgraph.control
                      .findNodes(
                          kernel,
                          [&](int tag) -> bool {
                              if(reachable_from_forK.find(tag) != reachable_from_forK.end())
                                  return false;
                              auto store    = kgraph.control.get<StoreTiled>(tag);
                              auto storeLDS = kgraph.control.get<StoreLDSTile>(tag);
                              if(store || storeLDS)
                              {
                                  auto [tile_tag, tile] = kgraph.getDimension<MacroTile>(tag);
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
                addComputeIndexVGPR(kgraph, tag, true);
            }

            return kgraph;
        }
    }
}
