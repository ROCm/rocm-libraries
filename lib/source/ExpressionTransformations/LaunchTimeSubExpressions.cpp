#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/Expression.hpp>

template <typename T>
constexpr auto cast_to_unsigned(T val)
{
    return static_cast<typename std::make_unsigned<T>::type>(val);
}

namespace rocRoller
{
    namespace Expression
    {
        /**
         * Launch-time subexpressions
         *
         * Attempt to replace complex operations found within an expression with
         * pre-calculated kernel arguments.
         *
         * Challenge: By the time we see most expressions, most launch-time known
         * values have already been converted to kernel arguments and are seen as
         * KernelExecute time.  For now this is solved by applying this optimization
         * as soon as the expression is made.  We could in theory work backward from
         * the kernel argument to the command argument.
         * TODO: Apply this to every expression by working backward from the kernel argument.
         *
         * Challenge: Applying this to the same expression twice will add two kernel arguments.  This is inefficient.
         * We identify expressions that already have kernel arguments and reuse them.
         */

        struct LaunchTimeExpressionVisitor
        {
            LaunchTimeExpressionVisitor(std::shared_ptr<Context> cxt)
                : m_context(cxt)
            {
            }

            template <typename T>
            ExpressionPtr launchEval(T const& expr)
            {
                ExpressionPtr exPtr  = std::make_shared<Expression>(expr);
                auto          kernel = m_context->kernel();
                for(auto const& arg : kernel->arguments())
                {
                    if(identical(arg.expression, exPtr))
                    {
                        return std::make_shared<Expression>(
                            std::make_shared<AssemblyKernelArgument>(arg));
                    }
                }

                // If no existing expressions matches:
                std::string argName = concatenate("LAUNCH_", kernel->arguments().size());
                auto        resType = resultType(expr);
                kernel->addArgument({argName, resType.second, DataDirection::ReadOnly, exPtr});

                return std::make_shared<Expression>(
                    std::make_shared<AssemblyKernelArgument>(kernel->findArgument(argName)));
            }

            template <CTernary Expr>
            ExpressionPtr operator()(Expr const& expr)
            {
                if(evaluationTimes(expr)[EvaluationTime::KernelLaunch])
                {
                    return launchEval(expr);
                }
                else
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
                    return std::make_shared<Expression>(cpy);
                }
            }

            template <CBinary Expr>
            ExpressionPtr operator()(Expr const& expr)
            {
                if(evaluationTimes(expr)[EvaluationTime::KernelLaunch])
                {
                    return launchEval(expr);
                }
                else
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
                    return std::make_shared<Expression>(cpy);
                }
            }

            template <CUnary Expr>
            ExpressionPtr operator()(Expr const& expr)
            {
                if(evaluationTimes(expr)[EvaluationTime::KernelLaunch])
                {
                    return launchEval(expr);
                }
                else
                {
                    Expr cpy = expr;
                    if(expr.arg)
                    {
                        cpy.arg = call(expr.arg);
                    }
                    return std::make_shared<Expression>(cpy);
                }
            }

            template <CValue Value>
            ExpressionPtr operator()(Value const& expr)
            {
                return std::make_shared<Expression>(expr);
            }

            ExpressionPtr call(ExpressionPtr expr)
            {
                if(!expr)
                    return expr;

                return std::visit(*this, *expr);
            }

        private:
            std::shared_ptr<Context> m_context;
        };

        ExpressionPtr launchTimeSubExpressions(ExpressionPtr expr, std::shared_ptr<Context> cxt)
        {
            auto visitor = LaunchTimeExpressionVisitor(cxt);
            return visitor.call(expr);
        }

    }
}
