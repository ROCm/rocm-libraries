#include <rocRoller/KernelGraph/Transforms/CleanArguments.hpp>

#include <rocRoller/AssemblyKernel.hpp>
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
         * Clean Graph
         *
         * Replaces all CommandArguments found within the graph with the appropriate
         * AssemblyKernelArgument.
         */

        /**
         * Removes all CommandArgruments found within an expression with the appropriate
         * AssemblyKernel Argument. This is used by CleanArgumentsVisitor.
         */
        struct CleanExpressionVisitor
        {
            CleanExpressionVisitor(AssemblyKernelPtr kernel)
                : m_kernel(kernel)
            {
            }

            template <CUnary Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                Expr cpy = expr;
                if(expr.arg)
                {
                    cpy.arg = call(expr.arg);
                }
                return std::make_shared<Expression::Expression>(cpy);
            }

            template <CBinary Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                Expr cpy = expr;
                if(expr.lhs)
                {
                    cpy.lhs = call(expr.lhs);
                }
                if(expr.rhs)
                {
                    cpy.rhs = call(expr.rhs);
                }
                return std::make_shared<Expression::Expression>(cpy);
            }

            template <CTernary Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                Expr cpy = expr;
                if(expr.lhs)
                {
                    cpy.lhs = call(expr.lhs);
                }
                if(expr.r1hs)
                {
                    cpy.r1hs = call(expr.r1hs);
                }
                if(expr.r2hs)
                {
                    cpy.r2hs = call(expr.r2hs);
                }
                return std::make_shared<Expression::Expression>(cpy);
            }

            // Finds the AssemblyKernelArgument with the same name as the provided
            // CommandArgument.
            ExpressionPtr operator()(CommandArgumentPtr const& expr) const
            {
                auto argument = m_kernel->findArgument(expr->name());
                return std::make_shared<Expression::Expression>(
                    std::make_shared<AssemblyKernelArgument>(argument));
            }

            template <CValue Value>
            ExpressionPtr operator()(Value const& expr) const
            {
                return std::make_shared<Expression::Expression>(expr);
            }

            ExpressionPtr call(ExpressionPtr expr) const
            {
                if(!expr)
                    return expr;

                return std::visit(*this, *expr);
            }

        private:
            AssemblyKernelPtr m_kernel;
        };

        /**
         * Removes all CommandArgruments found within an expression with the appropriate
         * AssemblyKernel Argument.
         */
        ExpressionPtr cleanArguments(ExpressionPtr expr, AssemblyKernelPtr kernel)
        {
            auto visitor = CleanExpressionVisitor(kernel);
            return visitor.call(expr);
        }

        /**
         * Visitor for cleaning all of the Dimensions within a graph.
         *
         * Calls cleanArguments on all of the expressions stored
         * within a Dimension.
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
                        if(resultType == DataType::Int32 || resultType == DataType::Int64)
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
                auto d   = dim;
                d.size   = cleanExpr(dim.size);
                d.stride = cleanExpr(dim.stride);
                d.offset = cleanExpr(dim.offset);

                if constexpr(std::same_as<User, T>)
                {
                    if(!m_kernel->hasArgument(dim.argumentName))
                    {
                        auto args  = m_command->getArguments();
                        auto myArg = std::find_if(
                            args.begin(), args.end(), [&dim](CommandArgumentPtr arg) {
                                return dim.argumentName == arg->name();
                            });

                        AssertFatal(myArg != args.end(), ShowValue(dim.argumentName));

                        m_kernel->addCommandArgument(*myArg);
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
            TIMER(t, "KernelGraph::cleanArguments");
            rocRoller::Log::getLogger()->debug("KernelGraph::cleanArguments()");
            auto visitor = CleanArgumentsVisitor(k, m_context, m_command);
            return rewriteDimensions(k, visitor);
        }

    }
}
