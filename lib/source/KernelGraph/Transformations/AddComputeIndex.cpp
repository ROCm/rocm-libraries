
#include "DataTypes/DataTypes.hpp"
#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        using namespace CoordGraph;
        using namespace ControlHypergraph;
        namespace Expression = rocRoller::Expression;
        using namespace Expression;

        /**
         * @brief Replace operation with a scope.  Does not delete the original operation.
         */
        std::pair<int, KernelHypergraph> replaceWithScope(KernelHypergraph const& original, int op)
        {
            auto graph = original;
            auto scope = graph.control.addElement(Scope());

            auto location = original.control.getLocation(op);
            for(auto const& input : location.incoming)
            {
                auto edge = original.control.getElement(input);
                int  parent
                    = *original.control.getNeighbours<Graph::Direction::Upstream>(input).begin();
                graph.control.deleteElement(input);
                graph.control.addElement(edge, {parent}, {scope});
            }
            for(auto const& output : location.outgoing)
            {
                auto edge = original.control.getElement(output);
                if(std::holds_alternative<ControlEdge>(edge))
                {
                    auto cedge = std::get<ControlEdge>(edge);
                    if(std::holds_alternative<Sequence>(cedge))
                    {
                        graph.control.deleteElement(output);
                        int child
                            = *original.control.getNeighbours<Graph::Direction::Downstream>(output)
                                   .begin();
                        graph.control.addElement(edge, {scope}, {child});
                    }
                }
            }

            return {scope, graph};
        }

        std::tuple<int, int, int>
            computeIndexVGPRMATRIXAB(KernelHypergraph& graph, int load, int user, int sdim)
        {
            AssertFatal(isOperation<LoadTiled>(graph.control.getElement(load)));
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

            auto ci_mac = graph.control.addElement(ComputeIndex(
                user, mac, -1, offset_mac, stride_mac, buffer, false, dtype, {i_thr_x, i_thr_y}));
            auto ci_row = graph.control.addElement(ComputeIndex(
                user, i_thr_x, mac, row_offset, row_stride, buffer, false, dtype, {mac, i_thr_y}));
            auto ci_col = graph.control.addElement(ComputeIndex(user,
                                                                i_thr_y,
                                                                i_thr_x,
                                                                col_offset,
                                                                col_stride,
                                                                buffer,
                                                                false,
                                                                dtype,
                                                                {mac, i_thr_x}));

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
            computeIndexVGPR(KernelHypergraph& graph, int loadstore, int source, bool forward)
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

            auto ci_row = graph.control.addElement(ComputeIndex(source,
                                                                i_thr_x,
                                                                -1,
                                                                row_offset,
                                                                row_stride,
                                                                buffer,
                                                                forward,
                                                                dtype,
                                                                {i_thr_y},
                                                                offsettype,
                                                                offsettype));
            auto ci_col = graph.control.addElement(ComputeIndex(source,
                                                                i_thr_y,
                                                                i_thr_x,
                                                                col_offset,
                                                                col_stride,
                                                                buffer,
                                                                forward,
                                                                dtype,
                                                                {i_thr_x},
                                                                offsettype,
                                                                offsettype));

            return {ci_row, ci_col};
        }

        /**
         * @brief Add ComputeIndex operations to graph for a MATRIX_A or MATRIX_B LoadLDSTile.
         */
        std::tuple<int, int> computeIndexLDSMATRIXAB(KernelHypergraph& graph, int load, int sdim)
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

            auto ci_wave = graph.control.addElement(ComputeIndex(lds,
                                                                 wave,
                                                                 -1,
                                                                 offset_wave,
                                                                 stride_wave,
                                                                 -1,
                                                                 false,
                                                                 dtype,
                                                                 {vgpr},
                                                                 DataType::UInt32,
                                                                 DataType::UInt32));
            auto ci_vgpr = graph.control.addElement(ComputeIndex(lds,
                                                                 vgpr,
                                                                 offset_wave,
                                                                 offset_vgpr,
                                                                 stride_vgpr,
                                                                 -1,
                                                                 false,
                                                                 dtype,
                                                                 {wave},
                                                                 DataType::UInt32,
                                                                 DataType::UInt32));

            graph.control.addElement(Sequence(), {ci_wave}, {ci_vgpr});

            return {ci_wave, ci_vgpr};
        }

        /**
         * @brief Add ComputeIndex operations to graph for a MATRIX_A or MATRIX_B LoadTiled.
         */
        std::tuple<int, int, int> computeIndexMATRIXAB(KernelHypergraph& graph, int load, int sdim)
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
            auto ci_mac      = graph.control.addElement(ComputeIndex(
                user, mac, -1, offset_mac, stride_mac, buffer, false, dtype, {wave, vgpr}));
            auto ci_wave     = graph.control.addElement(ComputeIndex(user,
                                                                 wave,
                                                                 offset_mac,
                                                                 offset_wave,
                                                                 stride_wave,
                                                                 buffer,
                                                                 false,
                                                                 dtype,
                                                                 {mac, vgpr}));
            auto ci_vgpr     = graph.control.addElement(ComputeIndex(user,
                                                                 vgpr,
                                                                 offset_wave,
                                                                 offset_vgpr,
                                                                 stride_vgpr,
                                                                 buffer,
                                                                 false,
                                                                 dtype,
                                                                 {mac, wave}));

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
        KernelHypergraph addComputeIndexAB(KernelHypergraph const& original,
                                           int                     op,
                                           int                     mulLoadA,
                                           int                     mulLoadB,
                                           int                     loadA,
                                           int                     storeLDSA,
                                           int                     loadB,
                                           int                     storeLDSB)
        {
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::addComputeIndexAB({}, {}, {})", op, mulLoadA, mulLoadB);

            auto [scope, graph] = replaceWithScope(original, op);

            // MATRIX_A; y is summation
            auto nodeA = original.control.getElement(mulLoadA);
            // LoadTiled A
            if(isOperation<LoadTiled>(nodeA))
            {
                auto [topA, bottomA, updateA] = computeIndexMATRIXAB(graph, mulLoadA, 1);
                graph.control.addElement(Body(), {scope}, {topA});
                graph.control.addElement(Sequence(), {bottomA}, {op});
                graph.control.addElement(ForLoopIncrement(), {op}, {updateA});
            }
            // LoadLDSTile A
            else if(isOperation<LoadLDSTile>(nodeA))
            {
                auto [topA, bottomA] = computeIndexLDSMATRIXAB(graph, mulLoadA, 1);
                graph.control.addElement(Body(), {scope}, {topA});
                graph.control.addElement(Sequence(), {bottomA}, {op});
            }

            // MATRIX_B; x is summation
            auto nodeB = original.control.getElement(mulLoadB);
            // LoadTiled B
            if(isOperation<LoadTiled>(nodeB))
            {
                auto [topB, bottomB, updateB] = computeIndexMATRIXAB(graph, mulLoadB, 0);
                graph.control.addElement(Body(), {scope}, {topB});
                graph.control.addElement(Sequence(), {bottomB}, {op});
                graph.control.addElement(ForLoopIncrement(), {op}, {updateB});
            }
            // LoadLDSTile B
            else if(isOperation<LoadLDSTile>(nodeB))
            {
                auto [topB, bottomB] = computeIndexLDSMATRIXAB(graph, mulLoadB, 0);
                graph.control.addElement(Body(), {scope}, {topB});
                graph.control.addElement(Sequence(), {bottomB}, {op});
            }

            if(loadA > 0 && storeLDSA > 0)
            {
                // LoadTiled A
                auto user                     = graph.mapper.get<User>(loadA);
                auto [topA, bottomA, updateA] = computeIndexVGPRMATRIXAB(graph, loadA, user, 1);
                graph.control.addElement(Body(), {scope}, {topA});
                graph.control.addElement(Sequence(), {bottomA}, {op});
                graph.control.addElement(ForLoopIncrement(), {op}, {updateA});

                // StoreLDSTile A
                auto lds                          = graph.mapper.get<LDS>(storeLDSA);
                auto [store_ci_row, store_ci_col] = computeIndexVGPR(graph, storeLDSA, lds, true);
                graph.control.addElement(Body(), {scope}, {store_ci_row});
                graph.control.addElement(Sequence(), {store_ci_row}, {store_ci_col});
                graph.control.addElement(Sequence(), {store_ci_col}, {op});
            }

            if(loadB > 0 && storeLDSB > 0)
            {
                // LoadTiled B
                auto user                     = graph.mapper.get<User>(loadB);
                auto [topB, bottomB, updateB] = computeIndexVGPRMATRIXAB(graph, loadB, user, 0);
                graph.control.addElement(Body(), {scope}, {topB});
                graph.control.addElement(Sequence(), {bottomB}, {op});
                graph.control.addElement(ForLoopIncrement(), {op}, {updateB});

                // StoreLDSTile B
                auto lds                          = graph.mapper.get<LDS>(storeLDSB);
                auto [store_ci_row, store_ci_col] = computeIndexVGPR(graph, storeLDSB, lds, true);
                graph.control.addElement(Body(), {scope}, {store_ci_row});
                graph.control.addElement(Sequence(), {store_ci_row}, {store_ci_col});
                graph.control.addElement(Sequence(), {store_ci_col}, {op});
            }

            return graph;
        }

        /**
         * @brief Add ComputeIndex operations to graph for a MATRIX_ACCUM load/store.
         */
        KernelHypergraph
            addComputeIndexC(KernelHypergraph const& original, int op, int loadstore, bool forward)
        {
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::addComputeIndexC({}, {}, {})", op, loadstore, forward);

            auto [scope, graph] = replaceWithScope(original, op);

            auto user       = graph.mapper.get<User>(loadstore);
            auto vgpr_block = graph.mapper.get<VGPRBlockNumber>(loadstore);
            auto vgpr_index = graph.mapper.get<VGPRBlockIndex>(loadstore);

            DataType dtype;
            {
                auto l = graph.control.get<LoadTiled>(loadstore);
                auto s = graph.control.get<StoreTiled>(loadstore);
                dtype  = l ? l->vtype.dataType : s->dataType;
            }

            int offset_vgpr_block, stride_vgpr_block, offset_vgpr_index, stride_vgpr_index, buffer;
            if(forward)
            {
                offset_vgpr_block = graph.coordinates.addElement(Offset(), {vgpr_block}, {user});
                stride_vgpr_block = graph.coordinates.addElement(Stride(), {vgpr_block}, {user});
                offset_vgpr_index = graph.coordinates.addElement(Offset(), {vgpr_index}, {user});
                stride_vgpr_index = graph.coordinates.addElement(Stride(), {vgpr_index}, {user});
                buffer            = graph.coordinates.addElement(Buffer(), {vgpr_block}, {user});
            }
            else
            {
                offset_vgpr_block = graph.coordinates.addElement(Offset(), {user}, {vgpr_block});
                stride_vgpr_block = graph.coordinates.addElement(Stride(), {user}, {vgpr_block});
                offset_vgpr_index = graph.coordinates.addElement(Offset(), {user}, {vgpr_index});
                stride_vgpr_index = graph.coordinates.addElement(Stride(), {user}, {vgpr_index});
                buffer            = graph.coordinates.addElement(Buffer(), {user}, {vgpr_block});
            }

            auto ci_vgpr_block = graph.control.addElement(ComputeIndex(user,
                                                                       vgpr_block,
                                                                       -1,
                                                                       offset_vgpr_block,
                                                                       stride_vgpr_block,
                                                                       buffer,
                                                                       forward,
                                                                       dtype,
                                                                       {vgpr_index}));
            auto ci_vgpr_index = graph.control.addElement(ComputeIndex(user,
                                                                       vgpr_index,
                                                                       offset_vgpr_block,
                                                                       offset_vgpr_index,
                                                                       stride_vgpr_index,
                                                                       buffer,
                                                                       forward,
                                                                       dtype,
                                                                       {vgpr_block}));

            graph.control.addElement(Body(), {scope}, {ci_vgpr_block});
            graph.control.addElement(Sequence(), {ci_vgpr_block}, {ci_vgpr_index});
            graph.control.addElement(Sequence(), {ci_vgpr_index}, {op});

            return graph;
        }

        /**
         * @brief Add ComputeIndex operations to graph for loads/stores.
         */
        KernelHypergraph addComputeIndexVGPR(KernelHypergraph const& original,
                                             int                     op,
                                             int                     loadstore,
                                             bool                    forward)
        {
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::addComputeIndexVGPR({}, {}, {})", op, loadstore, forward);

            auto [scope, graph] = replaceWithScope(original, op);

            auto node   = original.control.getElement(loadstore);
            auto source = -1;
            if(isOperation<LoadTiled>(node) || isOperation<StoreTiled>(node))
                source = graph.mapper.get<User>(loadstore);
            else if(isOperation<LoadLDSTile>(node) || isOperation<StoreLDSTile>(node))
                source = graph.mapper.get<LDS>(loadstore);

            AssertFatal(source > 0, "User or LDS dimension not found");

            auto [ci_row, ci_col] = computeIndexVGPR(graph, loadstore, source, forward);

            graph.control.addElement(Body(), {scope}, {ci_row});
            graph.control.addElement(Sequence(), {ci_row}, {ci_col});
            graph.control.addElement(Sequence(), {ci_col}, {op});

            return graph;
        }
    }
}
