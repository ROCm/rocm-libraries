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

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/LowerLinear.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        namespace Expression = rocRoller::Expression;
        using namespace ControlGraph;
        using namespace CoordinateGraph;
        using namespace Expression;

        struct LoopDistributeVisitor : public BaseGraphVisitor
        {
            LoopDistributeVisitor(ContextPtr context, ExpressionPtr loopSize)
                : BaseGraphVisitor(context)
                , m_loopSize(loopSize)
                , m_loopStride(Expression::literal(1))
            {
            }

            using BaseGraphVisitor::visitEdge;
            using BaseGraphVisitor::visitOperation;

            int loopIndexDim(KernelGraph& graph)
            {
                if(m_loopDimTag < 0)
                {
                    m_loopDimTag = graph.coordinates.addElement(Linear(m_loopSize, m_loopStride));
                }

                return m_loopDimTag;
            }

            int getLoop(int localTag, KernelGraph& graph)
            {
                if(m_loopDims.count(localTag))
                    return m_loopDims.at(localTag);

                auto loopIndex = loopIndexDim(graph);
                auto loop      = graph.coordinates.addElement(ForLoop(m_loopSize, m_loopStride));
                graph.coordinates.addElement(DataFlow(), {loopIndex}, {loop});

                m_loopDims.emplace(localTag, loop);

                return loop;
            }

            void addLoopDst(KernelGraph&                   graph,
                            KernelGraph const&             original,
                            GraphReindexer&                reindexer,
                            int                            tag,
                            CoordinateTransformEdge const& edge)
            {
                auto new_tag = reindexer.coordinates.at(tag);

                auto outgoing_nodes
                    = graph.coordinates.getNeighbours<Graph::Direction::Downstream>(new_tag)
                          .to<std::vector>();
                auto dstTag = -1;
                for(auto const& out : outgoing_nodes)
                {
                    auto odim = graph.coordinates.getNode(out);
                    if(isDimension<Workgroup>(odim))
                    {
                        dstTag = out;
                        break;
                    }
                }

                AssertFatal(dstTag > 0, "addLoopDst : Workgroup dimension not found");

                auto loop = getLoop(dstTag, graph);
                outgoing_nodes.insert(outgoing_nodes.begin(), loop);

                auto incoming_nodes
                    = graph.coordinates.getNeighbours<Graph::Direction::Upstream>(new_tag)
                          .to<std::vector>();

                graph.coordinates.deleteElement(new_tag);
                graph.coordinates.addElement(new_tag, edge, incoming_nodes, outgoing_nodes);
            }

            void visitEdge(KernelGraph&       graph,
                           KernelGraph const& original,
                           GraphReindexer&    reindexer,
                           int                tag,
                           Tile const&        edge) override
            {
                copyEdge(graph, original, reindexer, tag);
                addLoopDst(graph, original, reindexer, tag, edge);
            }

            void visitEdge(KernelGraph&       graph,
                           KernelGraph const& original,
                           GraphReindexer&    reindexer,
                           int                tag,
                           Inherit const&     edge) override
            {
                copyEdge(graph, original, reindexer, tag);
                addLoopDst(graph, original, reindexer, tag, edge);
            }

            void addLoopSrc(KernelGraph&                   graph,
                            KernelGraph const&             original,
                            GraphReindexer&                reindexer,
                            int                            tag,
                            CoordinateTransformEdge const& edge)
            {
                auto new_tag = reindexer.coordinates.at(tag);
                auto incoming_nodes
                    = graph.coordinates.getNeighbours<Graph::Direction::Upstream>(new_tag)
                          .to<std::vector>();
                auto srcTag = -1;
                for(auto const& in : incoming_nodes)
                {
                    auto idim = graph.coordinates.getNode(in);
                    if(isDimension<Workgroup>(idim))
                    {
                        srcTag = in;
                        break;
                    }
                }

                AssertFatal(srcTag > 0, "addLoopSrc : Workgroup dimension not found");

                auto loop = getLoop(srcTag, graph);
                incoming_nodes.insert(incoming_nodes.begin(), loop);

                auto outgoing_nodes
                    = graph.coordinates.getNeighbours<Graph::Direction::Downstream>(new_tag)
                          .to<std::vector>();
                graph.coordinates.deleteElement(new_tag);
                graph.coordinates.addElement(new_tag, edge, incoming_nodes, outgoing_nodes);
            }

            void visitEdge(KernelGraph&       graph,
                           KernelGraph const& original,
                           GraphReindexer&    reindexer,
                           int                tag,
                           Flatten const&     edge) override
            {
                copyEdge(graph, original, reindexer, tag);

                auto incoming_nodes
                    = graph.coordinates.getNeighbours<Graph::Direction::Upstream>(tag)
                          .to<std::vector>();
                if(incoming_nodes.size() > 1)
                    addLoopSrc(graph, original, reindexer, tag, edge);
            }

            void visitEdge(KernelGraph&       graph,
                           KernelGraph const& original,
                           GraphReindexer&    reindexer,
                           int                tag,
                           Forget const&      edge) override
            {
                copyEdge(graph, original, reindexer, tag);
                addLoopSrc(graph, original, reindexer, tag, edge);
            }

            void visitOperation(KernelGraph&       graph,
                                KernelGraph const& original,
                                GraphReindexer&    reindexer,
                                int                tag,
                                LoadVGPR const&    op) override
            {
                copyOperation(graph, original, reindexer, tag);

                auto new_tag = reindexer.control.at(tag);
                auto incoming_edge_tag
                    = graph.control.getNeighbours<Graph::Direction::Upstream>(new_tag)
                          .to<std::vector>();
                AssertFatal(incoming_edge_tag.size() == 1, "one parent node is expected");
                auto incoming_edge = graph.control.getElement(incoming_edge_tag[0]);

                graph.control.deleteElement(incoming_edge_tag[0]);
                graph.control.addElement(incoming_edge_tag[0],
                                         incoming_edge,
                                         std::vector<int>{m_loopOp},
                                         std::vector<int>{new_tag});
            }

            void visitRoot(KernelGraph&       graph,
                           KernelGraph const& original,
                           GraphReindexer&    reindexer,
                           int                tag) override
            {
                BaseGraphVisitor::visitRoot(graph, original, reindexer, tag);

                auto iterTag = loopIndexDim(graph);
                auto new_tag = reindexer.control.at(tag);

                // create loopOp and attach with Kernel
                auto loopVarExp = std::make_shared<Expression::Expression>(
                    DataFlowTag{iterTag, Register::Type::Scalar, DataType::Int32});
                m_loopOp = graph.control.addElement(ForLoopOp{loopVarExp < m_loopSize, ""});
                graph.control.addElement(Body(), {new_tag}, {m_loopOp});

                // create initOp and attach with the loopOp
                auto zero   = Expression::literal(0);
                auto initOp = graph.control.addElement(Assign{Register::Type::Scalar, zero});
                graph.control.addElement(Initialize(), {m_loopOp}, {initOp});

                // create incOp and attach with the loopOp
                auto incOp = graph.control.addElement(
                    Assign{Register::Type::Scalar, loopVarExp + m_loopStride});
                graph.control.addElement(ForLoopIncrement(), {m_loopOp}, {incOp});

                graph.mapper.connect<Dimension>(m_loopOp, iterTag);
                graph.mapper.connect(initOp, iterTag, NaryArgument::DEST);
                graph.mapper.connect(incOp, iterTag, NaryArgument::DEST);
            }

        private:
            int m_loopDimTag = -1;

            ExpressionPtr m_loopSize, m_loopStride;
            int           m_loopOp;

            std::unordered_map<int, int> m_loopDims;
        };

        KernelGraph LowerLinearLoop::apply(KernelGraph const& k)
        {
            TIMER(t, "KernelGraph::lowerLinearLoop");
            auto visitor = LoopDistributeVisitor(m_context, m_loopSize);
            return rewrite(k, visitor);
        }
    }
}
