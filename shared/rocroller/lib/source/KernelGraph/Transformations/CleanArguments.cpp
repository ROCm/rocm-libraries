// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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
