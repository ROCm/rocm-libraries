
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

        /**
         * @brief Add ComputeIndex operations to graph for a MATRIX_A or MATRIX_B load.
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
        KernelHypergraph
            addComputeIndexAB(KernelHypergraph const& original, int op, int loadA, int loadB)
        {
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::addComputeIndexAB({}, {}, {})", op, loadA, loadB);

            auto [scope, graph] = replaceWithScope(original, op);

            // MATRIX_A; y is summation
            auto [topA, bottomA, updateA] = computeIndexMATRIXAB(graph, loadA, 1);
            graph.control.addElement(Body(), {scope}, {topA});
            graph.control.addElement(Sequence(), {bottomA}, {op});
            graph.control.addElement(ForLoopIncrement(), {op}, {updateA});

            // MATRIX_B; x is summation
            auto [topB, bottomB, updateB] = computeIndexMATRIXAB(graph, loadB, 0);
            graph.control.addElement(Body(), {scope}, {topB});
            graph.control.addElement(Sequence(), {bottomB}, {op});
            graph.control.addElement(ForLoopIncrement(), {op}, {updateB});

            return graph;
        }

        /**
         * @brief Add ComputeIndex operations to graph for a MATRIX_ACCUM load.
         */
        KernelHypergraph
            addComputeIndexC(KernelHypergraph const& original, int op, int load, bool forward)
        {
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::addComputeIndexC({}, {}, {})", op, load, forward);

            auto [scope, graph] = replaceWithScope(original, op);

            auto user       = graph.mapper.get<User>(load);
            auto vgpr_block = graph.mapper.get<VGPRBlockNumber>(load);
            auto vgpr_index = graph.mapper.get<VGPRBlockIndex>(load);

            DataType dtype;
            {
                auto l = graph.control.get<LoadTiled>(load);
                auto s = graph.control.get<StoreTiled>(load);
                dtype  = l ? l->vtype.dataType : s->dataType;
            }

            int offset_vgpr_block, stride_vgpr_block;
            int offset_vgpr_index, stride_vgpr_index;
            int buffer;
            if(!forward)
            {
                offset_vgpr_block = graph.coordinates.addElement(Offset(), {user}, {vgpr_block});
                stride_vgpr_block = graph.coordinates.addElement(Stride(), {user}, {vgpr_block});
                offset_vgpr_index = graph.coordinates.addElement(Offset(), {user}, {vgpr_index});
                stride_vgpr_index = graph.coordinates.addElement(Stride(), {user}, {vgpr_index});
                buffer            = graph.coordinates.addElement(Buffer(), {user}, {vgpr_block});
            }
            else
            {
                offset_vgpr_block = graph.coordinates.addElement(Offset(), {vgpr_block}, {user});
                stride_vgpr_block = graph.coordinates.addElement(Stride(), {vgpr_block}, {user});
                offset_vgpr_index = graph.coordinates.addElement(Offset(), {vgpr_index}, {user});
                stride_vgpr_index = graph.coordinates.addElement(Stride(), {vgpr_index}, {user});
                buffer            = graph.coordinates.addElement(Buffer(), {vgpr_block}, {user});
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
         * @brief Add ComputeIndex operations to graph for a MATRIX_ACCUM load.
         */
        KernelHypergraph
            addComputeIndexVGPR(KernelHypergraph const& original, int op, int load, bool forward)
        {
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::addComputeIndexVGPR({}, {}, {})", op, load, forward);

            auto [scope, graph] = replaceWithScope(original, op);

            auto user    = graph.mapper.get<User>(load);
            auto i_thr_x = graph.mapper.get<ThreadTileIndex>(load, 0);
            auto i_thr_y = graph.mapper.get<ThreadTileIndex>(load, 1);

            DataType dtype;
            {
                auto l = graph.control.get<LoadTiled>(load);
                auto s = graph.control.get<StoreTiled>(load);
                dtype  = l ? l->vtype.dataType : s->dataType;
            }

            auto row_offset = graph.coordinates.addElement(Offset(), {user}, {i_thr_x});
            auto row_stride = graph.coordinates.addElement(Stride(), {user}, {i_thr_x});
            auto col_offset = graph.coordinates.addElement(Offset(), {user}, {i_thr_y});
            auto col_stride = graph.coordinates.addElement(Stride(), {user}, {i_thr_y});
            auto buffer     = graph.coordinates.addElement(Buffer(), {user}, {i_thr_x});

            auto ci_row = graph.control.addElement(ComputeIndex(
                user, i_thr_x, -1, row_offset, row_stride, buffer, forward, dtype, {i_thr_y}));
            auto ci_col = graph.control.addElement(ComputeIndex(
                user, i_thr_y, i_thr_x, col_offset, col_stride, buffer, forward, dtype, {i_thr_x}));

            graph.control.addElement(Body(), {scope}, {ci_row});
            graph.control.addElement(Sequence(), {ci_row}, {ci_col});
            graph.control.addElement(Sequence(), {ci_col}, {op});

            return graph;
        }
    }
}
