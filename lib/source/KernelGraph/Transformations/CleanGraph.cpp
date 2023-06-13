#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/CleanArguments.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>

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
            CleanExpressionVisitor(std::shared_ptr<AssemblyKernel> kernel)
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
            std::shared_ptr<AssemblyKernel> m_kernel;
        };

        /**
         * Removes all CommandArgruments found within an expression with the appropriate
         * AssemblyKernel Argument.
         */
        ExpressionPtr cleanArguments(ExpressionPtr expr, std::shared_ptr<AssemblyKernel> kernel)
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
            CleanArgumentsVisitor(std::shared_ptr<AssemblyKernel> kernel)
                : m_cleanArguments(kernel)
            {
            }

            template <CDimension T>
            Dimension visitDimension(int tag, T const& dim)
            {
                auto d   = dim;
                d.size   = m_cleanArguments.call(dim.size);
                d.stride = m_cleanArguments.call(dim.stride);
                return d;
            }

            Operation visitOperation(ForLoopOp const& op)
            {
                auto forOp      = op;
                forOp.condition = m_cleanArguments.call(op.condition);
                return forOp;
            }

            template <COperation T>
            Operation visitOperation(T const& op)
            {
                return op;
            }

        private:
            CleanExpressionVisitor m_cleanArguments;
        };

        /**
         * Rewrite HyperGraph to make sure no more CommandArgument
         * values are present within the graph.
         */
        KernelGraph CleanArguments::apply(KernelGraph const& k)
        {
            TIMER(t, "KernelGraph::cleanArguments");
            rocRoller::Log::getLogger()->debug("KernelGraph::cleanArguments()");
            auto visitor = CleanArgumentsVisitor(m_kernel);
            return rewriteDimensions(k, visitor);
        }

    }
}
