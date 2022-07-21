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
         * TODO: Identify expressions that already have kernel arguments and reuse them.
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
                auto k = m_context->kernel();

                auto argName = concatenate("LAUNCH_", k->arguments().size());
                auto resType = resultType(expr);
                auto exPtr   = std::make_shared<Expression>(expr);
                k->addArgument({argName, resType.second, DataDirection::ReadOnly, exPtr});

                return std::make_shared<Expression>(
                    std::make_shared<AssemblyKernelArgument>(k->findArgument(argName)));
            }

            template <CTernary Expr>
            ExpressionPtr operator()(Expr const& expr)
            {
                if(evaluationTimes(expr)[EvaluationTime::KernelLaunch])
                    return launchEval(expr);
                else
                    return std::make_shared<Expression>(
                        Expr({(*this)(expr.lhs), (*this)(expr.r1hs), (*this)(expr.r2hs)}));
            }

            template <CBinary Expr>
            ExpressionPtr operator()(Expr const& expr)
            {
                if(evaluationTimes(expr)[EvaluationTime::KernelLaunch])
                    return launchEval(expr);
                else
                    return std::make_shared<Expression>(
                        Expr({(*this)(expr.lhs), (*this)(expr.rhs)}));
            }

            template <CUnary Expr>
            ExpressionPtr operator()(Expr const& expr)
            {
                if(evaluationTimes(expr)[EvaluationTime::KernelLaunch])
                    return launchEval(expr);
                else
                    return std::make_shared<Expression>(Expr({(*this)(expr.arg)}));
            }

            ExpressionPtr operator()(MatrixMultiply const& expr)
            {
                return std::make_shared<Expression>(MatrixMultiply(expr));
            }

            template <CValue Value>
            ExpressionPtr operator()(Value const& expr)
            {
                return std::make_shared<Expression>(expr);
            }

            ExpressionPtr operator()(ExpressionPtr expr)
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
            return visitor(expr);
        }

    }
}
