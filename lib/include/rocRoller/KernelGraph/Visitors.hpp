/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include <iostream>
#include <memory>
#include <set>
#include <variant>

#include <rocRoller/Graph/Hypergraph.hpp>
#include <rocRoller/KernelGraph/ControlGraph/ControlGraph.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/CoordinateEdge_fwd.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/CoordinateGraph.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Reindexer.hpp>

#include <rocRoller/Expression_fwd.hpp>

#include <rocRoller/AssemblyKernel.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        using namespace ControlGraph;
        using namespace CoordinateGraph;

        using CTEdge = rocRoller::KernelGraph::CoordinateGraph::Edge;

        // clang-format off
        template <typename T>
        concept CCoordinateEdgeVisitor
            = requires(T & vis, int tag, Tile const& edge)
        {
            { vis.visitCoordinateEdge(tag, edge) }
                 -> std::convertible_to<rocRoller::KernelGraph::CoordinateGraph::Edge>;
        };
        // clang-format on

        /***************************
         * KernelGraphs rewrite utils
         */

        template <typename T>
        inline KernelGraph rewriteDimensions(KernelGraph const& k, T& visitor)
        {
            KernelGraph graph = k;
            if constexpr(CCoordinateEdgeVisitor<T>)
            {
                for(auto const& tag : k.coordinates.getEdges())
                {
                    auto edge = k.coordinates.getEdge(tag);
                    auto visA = [&](auto&& arg) { return visitor.visitCoordinateEdge(tag, arg); };
                    auto visB = [&](auto&& arg) { return std::visit(visA, arg); };

                    auto newEdge = std::visit(visB, edge);
                    graph.coordinates.setElement(tag, newEdge);
                }
            }

            for(auto const& tag : k.coordinates.getNodes())
            {
                auto x = k.coordinates.getNode(tag);
                auto y
                    = std::visit([&](auto&& arg) { return visitor.visitDimension(tag, arg); }, x);
                graph.coordinates.setElement(tag, y);
            }

            for(auto const& tag : k.control.getNodes())
            {
                auto x = k.control.getNode(tag);
                auto y
                    = std::visit([&](auto&& arg) { return visitor.visitOperation(tag, arg); }, x);
                graph.control.setElement(tag, y);
            }

            return graph;
        }

#define MAKE_EDGE_VISITOR(CLS)                           \
    virtual void visitEdge(KernelGraph&       graph,     \
                           KernelGraph const& original,  \
                           GraphReindexer&    reindexer, \
                           int                tag,       \
                           CLS const&         edge)      \
    {                                                    \
        copyEdge(graph, original, reindexer, tag);       \
    }

#define MAKE_OPERATION_VISITOR(CLS)                                          \
    virtual void visitOperation(KernelGraph&       graph,                    \
                                KernelGraph const& original,                 \
                                GraphReindexer&    reindexer,                \
                                int                tag,                      \
                                CLS const&         dst)                      \
    {                                                                        \
        copyOperation(graph, original, reindexer, tag);                      \
        if(m_rewriteCoordinates)                                             \
            reindexExpressions(graph, reindexer.control.at(tag), reindexer); \
    }

        /**
         * The BaseGraphVisitor implements a "copy" re-write visitor.
         *
         * To rewrite, for example, only LoadLinear edges, simply
         * override the `visitEdge` method.
         */
        struct BaseGraphVisitor
        {
            BaseGraphVisitor(ContextPtr       context,
                             Graph::Direction controlGraphOrder  = Graph::Direction::Downstream,
                             bool             rewriteCoordinates = true)
                : m_context(context)
                , m_controlGraphDirection(controlGraphOrder)
                , m_rewriteCoordinates(rewriteCoordinates)
            {
            }

            void copyEdge(KernelGraph&       graph,
                          KernelGraph const& original,
                          GraphReindexer&    reindexer,
                          int                edge)
            {
                auto             location = original.coordinates.getLocation(edge);
                std::vector<int> inputs;
                for(auto const& input : location.incoming)
                {
                    if(reindexer.coordinates.count(input) == 0)
                    {
                        auto newInput
                            = graph.coordinates.addElement(original.coordinates.getElement(input));
                        reindexer.coordinates.emplace(input, newInput);
                    }
                    inputs.push_back(reindexer.coordinates.at(input));
                }

                std::vector<int> outputs;
                for(auto const& output : location.outgoing)
                {
                    if(reindexer.coordinates.count(output) == 0)
                    {
                        auto newOutput
                            = graph.coordinates.addElement(original.coordinates.getElement(output));
                        reindexer.coordinates.emplace(output, newOutput);
                    }
                    outputs.push_back(reindexer.coordinates.at(output));
                }

                auto newEdge = graph.coordinates.addElement(
                    original.coordinates.getElement(edge), inputs, outputs);

                reindexer.coordinates.emplace(edge, newEdge);
            }

            /**
             * @brief Reconnect inputs from original operation to new operation.
             */
            void reconnectOperation(KernelGraph&       graph,
                                    KernelGraph const& original,
                                    GraphReindexer&    reindexer,
                                    int                new_tag,
                                    int                old_tag)
            {
                auto location = original.control.getLocation(old_tag);

                if(m_controlGraphDirection == Graph::Direction::Downstream)
                {
                    for(auto const& input : location.incoming)
                    {
                        int parent
                            = *original.control.getNeighbours<Graph::Direction::Upstream>(input)
                                   .begin();
                        AssertFatal(reindexer.control.count(parent) > 0,
                                    "Missing control input: ",
                                    ShowValue(old_tag),
                                    ShowValue(input),
                                    ShowValue(parent));
                        graph.control.addElement(original.control.getElement(input),
                                                 {reindexer.control.at(parent)},
                                                 {new_tag});
                    }
                }
                else
                {
                    for(auto const& output : location.outgoing)
                    {
                        int child
                            = *original.control.getNeighbours<Graph::Direction::Downstream>(output)
                                   .begin();
                        AssertFatal(reindexer.control.count(child) > 0,
                                    "Missing control output: ",
                                    ShowValue(old_tag),
                                    ShowValue(output),
                                    ShowValue(child));
                        graph.control.addElement(original.control.getElement(output),
                                                 {new_tag},
                                                 {reindexer.control.at(child)});
                    }
                }
            }

            /**
             * @brief Reconnect mappings from original operation to new operation.
             */
            void reconnectMappings(KernelGraph&       graph,
                                   KernelGraph const& original,
                                   GraphReindexer&    reindexer,
                                   int                new_tag,
                                   int                old_tag)
            {
                if(m_rewriteCoordinates)
                {
                    for(auto const& c : original.mapper.getConnections(old_tag))
                    {
                        AssertFatal(reindexer.coordinates.count(c.coordinate) > 0,
                                    "Missing mapped coordinate: ",
                                    ShowValue(old_tag),
                                    ShowValue(c.coordinate));
                        graph.mapper.connect(
                            new_tag, reindexer.coordinates.at(c.coordinate), c.connection);
                    }
                }
                else
                {
                    for(auto const& c : original.mapper.getConnections(old_tag))
                    {
                        graph.mapper.connect(new_tag, c.coordinate, c.connection);
                    }
                }
            }

            /**
             * @brief Copy operation from original graph to new graph.
             *
             * Inputs and mappings are preserved.
             */
            void copyOperation(KernelGraph&       graph,
                               KernelGraph const& original,
                               GraphReindexer&    reindexer,
                               int                tag)
            {
                auto op = graph.control.addElement(original.control.getElement(tag));
                reconnectOperation(graph, original, reindexer, op, tag);
                reconnectMappings(graph, original, reindexer, op, tag);

                reindexer.control.emplace(tag, op);
            }

            Expression::ExpressionPtr wavefrontSize() const
            {
                uint wfs = static_cast<uint>(m_context->kernel()->wavefront_size());
                return Expression::literal(wfs);
            }

            std::array<Expression::ExpressionPtr, 3> workgroupSize() const
            {
                auto const& wgSize = m_context->kernel()->workgroupSize();
                return {
                    Expression::literal(wgSize[0]),
                    Expression::literal(wgSize[1]),
                    Expression::literal(wgSize[2]),
                };
            }

            Expression::ExpressionPtr workgroupCountX() const
            {
                return m_context->kernel()->workgroupCount(0);
            }

            Expression::ExpressionPtr workgroupCountY() const
            {
                return m_context->kernel()->workgroupCount(1);
            }

            Expression::ExpressionPtr workgroupCountZ() const
            {
                return m_context->kernel()->workgroupCount(2);
            }

            Graph::Direction controlGraphDirection() const
            {
                return m_controlGraphDirection;
            }

            bool rewriteCoordinates() const
            {
                return m_rewriteCoordinates;
            }

            MAKE_EDGE_VISITOR(ConstructMacroTile);
            MAKE_EDGE_VISITOR(DataFlow);
            MAKE_EDGE_VISITOR(Offset);
            MAKE_EDGE_VISITOR(Stride);
            MAKE_EDGE_VISITOR(View);
            MAKE_EDGE_VISITOR(DestructMacroTile);
            MAKE_EDGE_VISITOR(Flatten);
            MAKE_EDGE_VISITOR(Forget);
            MAKE_EDGE_VISITOR(Inherit);
            MAKE_EDGE_VISITOR(Join);
            MAKE_EDGE_VISITOR(MakeOutput);
            MAKE_EDGE_VISITOR(PassThrough);
            MAKE_EDGE_VISITOR(Split);
            MAKE_EDGE_VISITOR(Tile);

            MAKE_OPERATION_VISITOR(Assign);
            MAKE_OPERATION_VISITOR(Barrier);
            MAKE_OPERATION_VISITOR(ComputeIndex);
            MAKE_OPERATION_VISITOR(ConditionalOp);
            MAKE_OPERATION_VISITOR(AssertOp);
            MAKE_OPERATION_VISITOR(Deallocate);
            MAKE_OPERATION_VISITOR(DoWhileOp);
            MAKE_OPERATION_VISITOR(Exchange);
            MAKE_OPERATION_VISITOR(ForLoopOp);
            MAKE_OPERATION_VISITOR(Kernel);
            MAKE_OPERATION_VISITOR(LoadLDSTile);
            MAKE_OPERATION_VISITOR(LoadLinear);
            MAKE_OPERATION_VISITOR(LoadTiled);
            MAKE_OPERATION_VISITOR(LoadVGPR);
            MAKE_OPERATION_VISITOR(LoadSGPR);
            MAKE_OPERATION_VISITOR(Multiply);
            MAKE_OPERATION_VISITOR(NOP);
            MAKE_OPERATION_VISITOR(Block);
            MAKE_OPERATION_VISITOR(Scope);
            MAKE_OPERATION_VISITOR(SetCoordinate);
            MAKE_OPERATION_VISITOR(StoreLDSTile);
            MAKE_OPERATION_VISITOR(LoadTileDirect2LDS);
            MAKE_OPERATION_VISITOR(StoreLinear);
            MAKE_OPERATION_VISITOR(StoreTiled);
            MAKE_OPERATION_VISITOR(StoreVGPR);
            MAKE_OPERATION_VISITOR(StoreSGPR);
            MAKE_OPERATION_VISITOR(TensorContraction);
            MAKE_OPERATION_VISITOR(UnrollOp);
            MAKE_OPERATION_VISITOR(WaitZero);
            MAKE_OPERATION_VISITOR(SeedPRNG);

            virtual void visitEdge(KernelGraph&                   graph,
                                   KernelGraph const&             original,
                                   GraphReindexer&                reindexer,
                                   int                            tag,
                                   CoordinateTransformEdge const& edge)
            {
                std::visit([&](auto&& arg) { visitEdge(graph, original, reindexer, tag, arg); },
                           edge);
            }

            virtual void visitEdge(KernelGraph&        graph,
                                   KernelGraph const&  original,
                                   GraphReindexer&     reindexer,
                                   int                 tag,
                                   DataFlowEdge const& edge)
            {
                std::visit([&](auto&& arg) { visitEdge(graph, original, reindexer, tag, arg); },
                           edge);
            }

            virtual void visitRoot(KernelGraph&       graph,
                                   KernelGraph const& original,
                                   GraphReindexer&    reindexer,
                                   int                tag)
            {
                copyOperation(graph, original, reindexer, tag);
            }

        protected:
            ContextPtr       m_context;
            Graph::Direction m_controlGraphDirection;
            bool             m_rewriteCoordinates;
        };
#undef MAKE_OPERATION_VISITOR
#undef MAKE_EDGE_VISITOR

        /**
         * Apply rewrite visitor to kernel graph.
         *
         * Edges are visited in topological order.
         *
         * Internal convenience routine.
         */
        template <typename T>
        inline KernelGraph rewrite(KernelGraph const& original, T& visitor)
        {
            KernelGraph    graph;
            GraphReindexer reindexer;

            graph.addConstraints(original.getConstraints());

            if(visitor.rewriteCoordinates())
            {
                // add coordinate roots
                for(auto const& index : original.coordinates.roots())
                {
                    reindexer.coordinates.emplace(
                        index,
                        graph.coordinates.addElement(original.coordinates.getElement(index)));
                }

                for(auto const& index : original.coordinates.topologicalSort())
                {
                    auto element = original.coordinates.getElement(index);
                    if(std::holds_alternative<CTEdge>(element))
                    {
                        auto edge = std::get<CTEdge>(element);
                        std::visit(
                            [&](auto&& arg) {
                                visitor.visitEdge(graph, original, reindexer, index, arg);
                            },
                            edge);
                    }
                }
            }
            else
            {
                graph.coordinates = original.coordinates;
            }

            // add control flow roots
            int kernel = *original.control.roots().begin();

            if(visitor.controlGraphDirection() == Graph::Direction::Downstream)
            {
                visitor.visitRoot(graph, original, reindexer, kernel);
            }

            for(auto const& index : visitor.controlGraphDirection() == Graph::Direction::Downstream
                                        ? original.control.topologicalSort()
                                        : original.control.reverseTopologicalSort())
            {
                auto element = original.control.getElement(index);
                if(std::holds_alternative<Operation>(element))
                {
                    auto node = std::get<Operation>(element);
                    if(std::holds_alternative<Kernel>(node))
                        continue;
                    std::visit(
                        [&](auto&& arg) {
                            visitor.visitOperation(graph, original, reindexer, index, arg);
                        },
                        node);
                }
            }

            if(visitor.controlGraphDirection() == Graph::Direction::Upstream)
            {
                visitor.visitRoot(graph, original, reindexer, kernel);
            }

            return graph;
        }
    }
}
