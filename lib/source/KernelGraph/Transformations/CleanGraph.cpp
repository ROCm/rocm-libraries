#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        using CoordinateTransform::MacroTile;

        using namespace CoordinateTransform;
        namespace Expression = rocRoller::Expression;
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
            std::shared_ptr<AssemblyKernel> m_kernel;

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
                    cpy.arg = (*this)(expr.arg);
                }
                return std::make_shared<Expression::Expression>(cpy);
            }

            template <CBinary Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                Expr cpy = expr;
                if(expr.lhs)
                {
                    cpy.lhs = (*this)(expr.lhs);
                }
                if(expr.rhs)
                {
                    cpy.rhs = (*this)(expr.rhs);
                }
                return std::make_shared<Expression::Expression>(cpy);
            }

            template <CTernary Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                Expr cpy = expr;
                if(expr.lhs)
                {
                    cpy.lhs = (*this)(expr.lhs);
                }
                if(expr.r1hs)
                {
                    cpy.r1hs = (*this)(expr.r1hs);
                }
                if(expr.r2hs)
                {
                    cpy.r2hs = (*this)(expr.r2hs);
                }
                return std::make_shared<Expression::Expression>(cpy);
            }

            // Finds the AssemblyKernelArgument with the same name as the provided
            // CommandArgument.
            ExpressionPtr operator()(std::shared_ptr<CommandArgument> const& expr) const
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

            ExpressionPtr operator()(ExpressionPtr expr) const
            {
                if(!expr)
                    return expr;

                return std::visit(*this, *expr);
            }
        };

        /**
         * Removes all CommandArgruments found within an expression with the appropriate
         * AssemblyKernel Argument.
         */
        ExpressionPtr cleanArguments(ExpressionPtr expr, std::shared_ptr<AssemblyKernel> kernel)
        {
            auto visitor = CleanExpressionVisitor(kernel);
            return visitor(expr);
        }

        /**
         * Visitor for cleaning all of the Dimensions within a graph.
         *
         * Calls cleanArguments on all of the expressions stored
         * within a Dimension.
         */
        // delete this when new graph arch complete
        struct CleanArgumentsVisitor
        {
            CleanArgumentsVisitor(std::shared_ptr<AssemblyKernel> kernel)
                : m_clean_arguments(kernel)
            {
            }

            template <typename T>
            Dimension visitDimension(T const& dim)
            {
                auto d   = dim;
                d.size   = m_clean_arguments(dim.size);
                d.stride = m_clean_arguments(dim.stride);
                return d;
            }

            ControlGraph::Operation visitOperation(ControlGraph::ForLoopOp const& op)
            {
                auto forOp      = op;
                forOp.condition = m_clean_arguments(op.condition);
                return forOp;
            }

            template <typename T>
            ControlGraph::Operation visitOperation(T const& op)
            {
                return op;
            }

        private:
            CleanExpressionVisitor m_clean_arguments;
        };

        // rename this when new graph arch complete, tidy up namespaces
        struct CleanArgumentsVisitor2
        {
            CleanArgumentsVisitor2(std::shared_ptr<AssemblyKernel> kernel)
                : m_clean_arguments(kernel)
            {
            }

            template <typename T>
            CoordGraph::Dimension visitDimension(int tag, T const& dim)
            {
                auto d   = dim;
                d.size   = m_clean_arguments(dim.size);
                d.stride = m_clean_arguments(dim.stride);
                return d;
            }

            ControlHypergraph::Operation visitOperation(ControlHypergraph::ForLoopOp const& op)
            {
                auto forOp      = op;
                forOp.condition = m_clean_arguments(op.condition);
                return forOp;
            }

            template <typename T>
            ControlHypergraph::Operation visitOperation(T const& op)
            {
                return op;
            }

        private:
            CleanExpressionVisitor m_clean_arguments;
        };

        /**
         * Rewrite HyperGraph to make sure no more CommandArgument
         * values are present within the graph.
         */
        // delete this when new graph arch complete
        KernelGraph cleanArguments(KernelGraph k, std::shared_ptr<AssemblyKernel> kernel)
        {
            TIMER(t, "KernelGraph::cleanArguments");
            rocRoller::Log::getLogger()->debug("KernelGraph::cleanArguments()");
            auto visitor = CleanArgumentsVisitor(kernel);
            return rewriteDimensions(k, visitor);
        }

        KernelHypergraph cleanArguments(KernelHypergraph k, std::shared_ptr<AssemblyKernel> kernel)
        {
            TIMER(t, "KernelGraph::cleanArguments");
            rocRoller::Log::getLogger()->debug("KernelGraph::cleanArguments()");
            auto visitor = CleanArgumentsVisitor2(kernel);
            return rewriteDimensions(k, visitor);
        }

    }
}
