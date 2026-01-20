/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2026 AMD ROCm(TM) Software
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

#include <rocRoller/KernelGraph/Transforms/CleanArguments.hpp>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>

#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>

#include <rocRoller/Operations/Command.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        namespace Expression = rocRoller::Expression;
        using namespace ControlGraph;
        using namespace CoordinateGraph;
        using namespace Expression;

        /**
         * Visitor for cleaning all of the Dimensions within a graph.
         */
        struct CleanArgumentsVisitor
        {
            CleanArgumentsVisitor(KernelGraph const& graph, ContextPtr context, CommandPtr command)
                : m_graph(graph)
                , m_kernel(context->kernel())
                , m_command(command)
                , m_context(context)
            {
            }

            ExpressionPtr cleanExpr(ExpressionPtr expr)
            {
                return FastArithmetic(m_context)(expr);
            }

            template <CCoordinateTransformEdge T>
            rocRoller::KernelGraph::CoordinateGraph::Edge visitCoordinateEdge(int      tag,
                                                                              T const& edge)
            {
                auto divideBySize = [&](int dimTag) {
                    using ET  = Expression::EvaluationTime;
                    auto dim  = m_graph.coordinates.getNode(dimTag);
                    auto size = getSize(dim);
                    if(size && !Expression::evaluationTimes(size)[ET::Translate])
                    {
                        auto resultType = resultVariableType(size);
                        if(resultType == DataType::Int32 || resultType == DataType::Int64
                           || resultType == DataType::UInt32 || resultType == DataType::UInt64)
                            enableDivideBy(size, m_context);
                    }
                };
                if constexpr(std::same_as<Tile, T>)
                {
                    auto loc = m_graph.coordinates.getLocation(tag);
                    for(int i = 1; i < loc.outgoing.size(); i++)
                        divideBySize(loc.outgoing[i]);
                }

                if constexpr(std::same_as<Flatten, T>)
                {
                    auto loc = m_graph.coordinates.getLocation(tag);
                    for(int i = 1; i < loc.incoming.size(); i++)
                        divideBySize(loc.incoming[i]);
                }
                return edge;
            }

            template <CDataFlowEdge T>
            rocRoller::KernelGraph::CoordinateGraph::Edge visitCoordinateEdge(int      tag,
                                                                              T const& edge)
            {
                return edge;
            }

            template <CCoordinateTransformEdge Dim, Graph::Direction Dir>
            bool hasConnectedDimension(int tag)
            {
                auto pred
                    = [this](int tag) { return m_graph.coordinates.get<Dim>(tag).has_value(); };

                return !m_graph.coordinates.getNeighbours<Dir>(tag).filter(pred).empty();
            }

            template <CDimension T>
            Dimension visitDimension(int tag, T const& dim)
            {
                auto d = dim;

                d.size   = cleanExpr(dim.size);
                d.stride = cleanExpr(dim.stride);
                d.offset = cleanExpr(dim.offset);

                if constexpr(std::same_as<User, T>)
                {
                    if(!m_kernel->hasArgument(dim.argumentName))
                    {
                        auto arg = findArgumentByName(m_command, dim.argumentName);
                        AssertFatal(arg, ShowValue(dim.argumentName));

                        m_kernel->addCommandArgument(arg);
                    }
                }

                return d;
            }

            Operation visitOperation(int tag, Assign const& op)
            {
                auto cleanOp       = op;
                cleanOp.expression = cleanExpr(op.expression);
                return cleanOp;
            }

            Operation visitOperation(int tag, ConditionalOp const& op)
            {
                auto cleanOp      = op;
                cleanOp.condition = cleanExpr(op.condition);
                return cleanOp;
            }

            Operation visitOperation(int tag, AssertOp const& op)
            {
                auto cleanOp      = op;
                cleanOp.condition = cleanExpr(op.condition);
                return cleanOp;
            }

            Operation visitOperation(int tag, ForLoopOp const& op)
            {
                auto cleanOp      = op;
                cleanOp.condition = cleanExpr(op.condition);
                return cleanOp;
            }

            template <COperation T>
            Operation visitOperation(int tag, T const& op)
            {
                return op;
            }

        private:
            KernelGraph const& m_graph;
            AssemblyKernelPtr  m_kernel;
            CommandPtr         m_command;
            ContextPtr         m_context;
        };
        static_assert(CCoordinateEdgeVisitor<CleanArgumentsVisitor>);

        /**
         * Rewrite HyperGraph to make sure no more CommandArgument
         * values are present within the graph.
         */
        KernelGraph CleanArguments::apply(KernelGraph const& k)
        {
            rocRoller::Log::getLogger()->debug("KernelGraph::cleanArguments()");
            auto visitor = CleanArgumentsVisitor(k, m_context, m_command);
            return rewriteDimensions(k, visitor);
        }

    }
}
