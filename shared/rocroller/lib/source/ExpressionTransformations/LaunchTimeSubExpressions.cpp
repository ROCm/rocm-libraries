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

#include <rocRoller/ExpressionTransformations.hpp>

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
         */

        struct LaunchTimeExpressionVisitor
        {
            LaunchTimeExpressionVisitor(ContextPtr ctx, bool allowNewArgs)
                : m_context(ctx)
                , m_allowNewArgs(allowNewArgs)
                , m_minComplexity(ctx->kernelOptions().minLaunchTimeExpressionComplexity)
            {
            }

            /**
             * Return an argument for `expr`, if one exists.
             */
            ExpressionPtr existingLaunchEval(ExpressionPtr expr)
            {
                return m_context->kernel()->findArgumentForExpression(expr);
            }

            ExpressionPtr addLaunchEval(ExpressionPtr expr)
            {
                auto kernel  = m_context->kernel();
                auto varType = resultVariableType(expr);

                auto argName = kernel->uniqueArgName(argumentName(expr));

                Log::debug("LTSE: Adding arg {} complexity {}, varType {}, expr {}",
                           argName,
                           complexity(expr),
                           toString(varType),
                           toString(expr));

                return kernel->addArgument(
                    {.name = argName, .variableType = varType, .expression = expr});
            }

            ExpressionPtr maybeLaunchEval(ExpressionPtr expr, bool ignoreComplexity)
            {
                auto evalTimes = evaluationTimes(expr);

                if(evalTimes[EvaluationTime::Translate] || !evalTimes[EvaluationTime::KernelLaunch])
                {
                    return nullptr;
                }

                {
                    auto arg = existingLaunchEval(expr);
                    if(arg)
                        return arg;
                }

                if(!m_allowNewArgs)
                    return nullptr;

                LaunchTimeExpressionVisitor sub(m_context, false);
                auto                        ex2    = sub.call(expr);
                auto                        myComp = complexity(ex2);

                if(ignoreComplexity || !evalTimes[EvaluationTime::KernelExecute]
                   || complexity(expr) >= m_minComplexity)
                    return addLaunchEval(expr);

                return nullptr;
            }

            template <CExpression T>
            ExpressionPtr maybeLaunchEval(T const& expr, bool ignoreComplexity = false)
            {
                return maybeLaunchEval(std::make_shared<Expression>(expr), ignoreComplexity);
            }

            ExpressionPtr operator()(ScaledMatrixMultiply const& expr)
            {
                {
                    auto launchResult = maybeLaunchEval(expr);
                    if(launchResult)
                        return launchResult;
                }

                {
                    ScaledMatrixMultiply cpy = expr;
                    cpy.matA                 = call(expr.matA);
                    cpy.matB                 = call(expr.matB);
                    cpy.matC                 = call(expr.matC);
                    cpy.scaleA               = call(expr.scaleA);
                    cpy.scaleB               = call(expr.scaleB);

                    return std::make_shared<Expression>(cpy);
                }
            }

            template <CTernary Expr>
            ExpressionPtr operator()(Expr const& expr)
            {
                {
                    auto launchResult = maybeLaunchEval(expr);
                    if(launchResult)
                        return launchResult;
                }

                {
                    Expr cpy = expr;
                    cpy.lhs  = call(expr.lhs);
                    cpy.r1hs = call(expr.r1hs);
                    cpy.r2hs = call(expr.r2hs);

                    return std::make_shared<Expression>(cpy);
                }
            }

            template <CBinary Expr>
            ExpressionPtr operator()(Expr const& expr)
            {
                {
                    auto launchResult = maybeLaunchEval(expr);
                    if(launchResult)
                        return launchResult;
                }

                {
                    Expr cpy = expr;
                    cpy.lhs  = call(expr.lhs);
                    cpy.rhs  = call(expr.rhs);

                    return std::make_shared<Expression>(cpy);
                }
            }

            template <CUnary Expr>
            ExpressionPtr operator()(Expr const& expr)
            {

                {
                    auto launchResult = maybeLaunchEval(expr);
                    if(launchResult)
                        return launchResult;
                }

                {
                    Expr cpy = expr;
                    cpy.arg  = call(expr.arg);

                    return std::make_shared<Expression>(cpy);
                }
            }

            ExpressionPtr operator()(CommandArgumentPtr const& expr)
            {
                // For a Value, if we still have a CommandArgument, we need to
                // convert it to a KernelArgument, regardless of complexity.

                auto launchResult = maybeLaunchEval(expr, true);
                if(launchResult)
                    return launchResult;

                return std::make_shared<Expression>(expr);
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
            ContextPtr m_context;
            int        m_minComplexity;
            bool       m_allowNewArgs;
        };

        ExpressionPtr launchTimeSubExpressions(ExpressionPtr expr, ContextPtr ctx)
        {
            auto startStr = toString(expr);

            if(ctx->kernel()->startedCodeGeneration())
            {
                auto visitor = LaunchTimeExpressionVisitor(ctx, false);

                // Since code gen has started, we can't add any new kernel arguments.  So our
                // goal is to find the largest subexpressions that exist as kernel arguments.

                // The following might exist:
                // kernArg0: comArg0
                // kernArg1: comArg0 / comArg1
                // kernArg2: kernArg0 / kernArg1

                // If our expression is:
                // initial: ((kernArg0 / comArg1) + (comArg0 / kernArg1))
                // restore: ((comArg0 / comArg1) + (comArg0 / (comArg0 / comArg1))
                // pass  1: (kernArg1 + (kernArg0 / kernArg1))
                // pass  2: (kernArg1 + kernArg2)

                // Maybe this can be done more efficiently?

                auto v0 = restoreCommandArguments(expr);
                auto v1 = visitor.call(v0);
                auto v2 = visitor.call(v1);

                Log::trace("launchTimeSubExpressions: {} -> {} -> {} -> {}",
                           toString(expr),
                           toString(v0),
                           toString(v1),
                           toString(v2));

                AssertFatal(resultVariableType(expr) == resultVariableType(v2),
                            ShowValue(expr),
                            ShowValue(v2));

                return v2;
            }
            else
            {
                auto v0 = restoreCommandArguments(expr);

                // Challenge: Find subexpressions that are *actually* nontrivial to calculate.
                // Example:
                // kernArg0: comArg0 * comArg1 + comArg0 / comArg2

                // If our expression is:
                // 4 + comArg0 * comArg1 + comArg0 / comArg2
                // Then the initial pass could gauge the complexity of the whole expression
                // including the part that is already a kernel argument, and it will add a new
                // argument for this whole expression.

                // If we first substitute existing args without adding any new ones:
                // 4 + kernArg0
                // Then the complexity will be low and we won't add a new arg just to save this
                // single addition.

                // This should restore any existing subexpressions that are more complex than
                // just a single value.
                auto visitor1 = LaunchTimeExpressionVisitor(ctx, false);
                auto v1       = visitor1.call(v0);

                // That allows this call to have an accurate picture of complexity.
                auto visitor2 = LaunchTimeExpressionVisitor(ctx, true);
                auto v2       = visitor2.call(v1);

                Log::trace("launchTimeSubExpressions: {} -> {} -> {} -> {}",
                           toString(expr),
                           toString(v0),
                           toString(v1),
                           toString(v2));

                AssertFatal(resultVariableType(expr) == resultVariableType(v2),
                            ShowValue(expr),
                            ShowValue(v2));

                return v2;
            }
        }

    }
}
