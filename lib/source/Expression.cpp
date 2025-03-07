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

#include <variant>

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/InstructionValues/Register_fwd.hpp>
#include <rocRoller/Utilities/Generator.hpp>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/Utilities/Timer.hpp>

namespace rocRoller
{
    namespace Expression
    {
        ExpressionPtr fromKernelArgument(AssemblyKernelArgument const& arg)
        {
            return std::make_shared<Expression>(std::make_shared<AssemblyKernelArgument>(arg));
        }

        /*
         * identical
         */

        struct ExpressionIdenticalVisitor
        {
            bool operator()(ScaledMatrixMultiply const& a, ScaledMatrixMultiply const& b)
            {
                bool matA   = false;
                bool matB   = false;
                bool matC   = false;
                bool scaleA = false;
                bool scaleB = false;

                matA = call(a.matA, b.matA);
                if(a.matA == nullptr && b.matA == nullptr)
                {
                    matA = true;
                }

                matB = call(a.matB, b.matB);
                if(a.matB == nullptr && b.matB == nullptr)
                {
                    matB = true;
                }

                matC = call(a.matC, b.matC);
                if(a.matC == nullptr && b.matC == nullptr)
                {
                    matC = true;
                }

                scaleA = call(a.scaleA, b.scaleA);
                if(a.scaleA == nullptr && b.scaleA == nullptr)
                {
                    scaleA = true;
                }

                scaleB = call(a.scaleB, b.scaleB);
                if(a.scaleB == nullptr && b.scaleB == nullptr)
                {
                    scaleB = true;
                }

                return matA && matB && matC && scaleA && scaleB;
            }

            template <CTernary T>
            bool operator()(T const& a, T const& b)
            {
                bool lhs  = false;
                bool r1hs = false;
                bool r2hs = false;

                lhs = call(a.lhs, b.lhs);
                if(a.lhs == nullptr && b.lhs == nullptr)
                {
                    lhs = true;
                }

                r1hs = call(a.r1hs, b.r1hs);
                if(a.r1hs == nullptr && b.r1hs == nullptr)
                {
                    r1hs = true;
                }

                r2hs = call(a.r2hs, b.r2hs);

                if(a.r2hs == nullptr && b.r2hs == nullptr)
                {
                    r2hs = true;
                }
                return lhs && r1hs && r2hs;
            }

            template <CBinary T>
            bool operator()(T const& a, T const& b)
            {
                bool lhs = false;
                bool rhs = false;

                lhs = call(a.lhs, b.lhs);
                if(a.lhs == nullptr && b.lhs == nullptr)
                {
                    lhs = true;
                }

                rhs = call(a.rhs, b.rhs);
                if(a.rhs == nullptr && b.rhs == nullptr)
                {
                    rhs = true;
                }

                return lhs && rhs;
            }

            template <CUnary T>
            bool operator()(T const& a, T const& b)
            {
                if(a.arg == nullptr && b.arg == nullptr)
                {
                    return true;
                }
                return call(a.arg, b.arg);
            }

            constexpr bool operator()(CommandArgumentValue const& a, CommandArgumentValue const& b)
            {
                return a == b;
            }

            bool operator()(CommandArgumentPtr const& a, CommandArgumentPtr const& b)
            {
                // Need to be careful not to invoke the overloaded operators, we want to compare
                // the pointers directly.
                // a->expression && b->expression -> logical and of both expressions
                if(a.get() == b.get())
                    return true;

                if(a == nullptr || b == nullptr)
                    return false;

                return (*a) == (*b);
            }

            bool operator()(AssemblyKernelArgumentPtr const& a, AssemblyKernelArgumentPtr const& b)
            {
                if(a->name == b->name)
                    return true;

                if((a->expression != nullptr) && (b->expression != nullptr))
                    return call(a->expression, b->expression);

                return false;
            }

            bool operator()(Register::ValuePtr const& a, Register::ValuePtr const& b)
            {
                return a->sameAs(b);
            }

            constexpr bool operator()(DataFlowTag const& a, DataFlowTag const& b)
            {
                return a == b;
            }

            bool operator()(WaveTilePtr const& a, WaveTilePtr const& b)
            {
                return a == b;
            }

            // a & b are different operator/value classes
            template <class T, class U>
            requires(!std::same_as<T, U>) constexpr bool operator()(T const& a, U const& b)
            {
                return false;
            }

            bool call(ExpressionPtr const& a, ExpressionPtr const& b)
            {
                if(a == nullptr)
                {
                    return b == nullptr;
                }
                else if(b == nullptr)
                {
                    return false;
                }
                return std::visit(*this, *a, *b);
            }

            bool call(Expression const& a, Expression const& b)
            {
                return std::visit(*this, a, b);
            }
        };

        bool identical(ExpressionPtr const& a, ExpressionPtr const& b)
        {
            auto visitor = ExpressionIdenticalVisitor();
            return visitor.call(a, b);
        }

        bool identical(Expression const& a, Expression const& b)
        {
            auto visitor = ExpressionIdenticalVisitor();
            return visitor.call(a, b);
        }

        struct ExpressionEquivalentVisitor
        {
            ExpressionEquivalentVisitor(AlgebraicProperties properties)
                : m_properties(properties)
            {
            }

            bool operator()(ScaledMatrixMultiply const& a, ScaledMatrixMultiply const& b)
            {
                bool matA   = false;
                bool matB   = false;
                bool matC   = false;
                bool scaleA = false;
                bool scaleB = false;

                matA = call(a.matA, b.matA);
                if(a.matA == nullptr && b.matA == nullptr)
                {
                    matA = true;
                }

                matB = call(a.matB, b.matB);
                if(a.matB == nullptr && b.matB == nullptr)
                {
                    matB = true;
                }

                matC = call(a.matC, b.matC);
                if(a.matC == nullptr && b.matC == nullptr)
                {
                    matC = true;
                }

                scaleA = call(a.scaleA, b.scaleA);
                if(a.scaleA == nullptr && b.scaleA == nullptr)
                {
                    scaleA = true;
                }

                scaleB = call(a.scaleB, b.scaleB);
                if(a.scaleB == nullptr && b.scaleB == nullptr)
                {
                    scaleB = true;
                }

                return matA && matB && matC && scaleA && scaleB;
            }

            template <CTernary T>
            bool operator()(T const& a, T const& b)
            {
                bool lhs  = false;
                bool r1hs = false;
                bool r2hs = false;

                lhs = call(a.lhs, b.lhs);
                if(a.lhs == nullptr && b.lhs == nullptr)
                {
                    lhs = true;
                }

                r1hs = call(a.r1hs, b.r1hs);
                if(a.r1hs == nullptr && b.r1hs == nullptr)
                {
                    r1hs = true;
                }

                r2hs = call(a.r2hs, b.r2hs);

                if(a.r2hs == nullptr && b.r2hs == nullptr)
                {
                    r2hs = true;
                }
                return lhs && r1hs && r2hs;
            }

            template <CBinary T>
            bool operator()(T const& a, T const& b)
            {
                bool lhs = false;
                bool rhs = false;

                lhs = call(a.lhs, b.lhs);
                if(a.lhs == nullptr && b.lhs == nullptr)
                {
                    lhs = true;
                }

                rhs = call(a.rhs, b.rhs);
                if(a.rhs == nullptr && b.rhs == nullptr)
                {
                    rhs = true;
                }

                bool result = lhs && rhs;

                // Test if equivalent if expression is commutative
                if(!result && CCommutativeBinary<T> && m_properties[AlgebraicProperty::Commutative])
                {
                    lhs = call(a.lhs, b.rhs);
                    if(a.lhs == nullptr && b.rhs == nullptr)
                    {
                        lhs = true;
                    }

                    rhs = call(a.rhs, b.lhs);
                    if(a.rhs == nullptr && b.lhs == nullptr)
                    {
                        rhs = true;
                    }

                    result = lhs && rhs;
                }

                return result;
            }

            template <CUnary T>
            bool operator()(T const& a, T const& b)
            {
                if(a.arg == nullptr && b.arg == nullptr)
                {
                    return true;
                }
                return call(a.arg, b.arg);
            }

            constexpr bool operator()(CommandArgumentValue const& a, CommandArgumentValue const& b)
            {
                return a == b;
            }

            bool operator()(CommandArgumentPtr const& a, CommandArgumentPtr const& b)
            {
                return (*a) == (*b);
            }

            bool operator()(AssemblyKernelArgumentPtr const& a, AssemblyKernelArgumentPtr const& b)
            {
                if(a->name == b->name)
                    return true;

                if((a->expression != nullptr) && (b->expression != nullptr))
                    return call(a->expression, b->expression);

                return false;
            }

            bool operator()(Register::ValuePtr const& a, Register::ValuePtr const& b)
            {
                return a->sameAs(b);
            }

            constexpr bool operator()(DataFlowTag const& a, DataFlowTag const& b)
            {
                return a == b;
            }

            bool operator()(WaveTilePtr const& a, WaveTilePtr const& b)
            {
                return a == b;
            }

            // a & b are different operator/value classes
            template <class T, class U>
            requires(!std::same_as<T, U>) bool operator()(T const& a, U const& b)
            {
                return false;
            }

            bool call(ExpressionPtr const& a, ExpressionPtr const& b)
            {
                if(a == nullptr)
                {
                    return b == nullptr;
                }
                else if(b == nullptr)
                {
                    return false;
                }
                return std::visit(*this, *a, *b);
            }

        private:
            AlgebraicProperties const m_properties;
        };

        bool equivalent(ExpressionPtr const& a,
                        ExpressionPtr const& b,
                        AlgebraicProperties  properties)
        {
            auto visitor = ExpressionEquivalentVisitor(properties);
            return visitor.call(a, b);
        }

        /*
         * comments
         */

        struct ExpressionSetCommentVisitor
        {
            std::string comment;
            bool        throwIfNotSupported = true;

            template <typename Expr>
            requires(CUnary<Expr> || CBinary<Expr> || CTernary<Expr>) void operator()(Expr& expr)
            {
                expr.comment = std::move(comment);
            }

            void operator()(auto& expr)
            {
                if(throwIfNotSupported)
                    Throw<FatalError>("Cannot set a comment for a base expression.");
            }

            void call(Expression& expr)
            {
                return std::visit(*this, expr);
            }
        };

        void setComment(Expression& expr, std::string comment)
        {
            auto visitor = ExpressionSetCommentVisitor{std::move(comment)};
            return visitor.call(expr);
        }

        void setComment(ExpressionPtr& expr, std::string comment)
        {
            if(expr)
            {
                setComment(*expr, std::move(comment));
            }
            else
            {
                Throw<FatalError>("Cannot set the comment for a null expression pointer.");
            }
        }

        void copyComment(ExpressionPtr const& dst, ExpressionPtr const& src)
        {
            if(!dst || !src)
                return;
            copyComment(*dst, *src);
        }

        void copyComment(Expression& dst, ExpressionPtr const& src)
        {
            if(!src)
                return;
            copyComment(dst, *src);
        }

        void copyComment(ExpressionPtr const& dst, Expression const& src)
        {
            if(!dst)
                return;
            copyComment(*dst, src);
        }

        void copyComment(Expression& dst, Expression const& src)
        {

            if(&src == &dst)
                return;

            auto comment = getComment(src);
            if(comment.empty())
                return;

            comment = getComment(dst) + std::move(comment);

            ExpressionSetCommentVisitor vis{std::move(comment), false};
            vis.call(dst);
        }

        struct ExpressionGetCommentVisitor
        {
            bool includeRegisterComments = true;

            template <typename Expr>
            requires(CUnary<Expr> || CBinary<Expr> || CTernary<Expr>) std::string
                operator()(Expr const& expr) const
            {
                return expr.comment;
            }

            std::string operator()(Register::ValuePtr const& expr) const
            {
                if(includeRegisterComments && expr)
                    return expr->name();

                return "";
            }

            std::string operator()(auto const& expr) const
            {
                return "";
            }

            std::string call(Expression const& expr) const
            {
                return std::visit(*this, expr);
            }
        };

        std::string getComment(Expression const& expr, bool includeRegisterComments)
        {
            auto visitor = ExpressionGetCommentVisitor{includeRegisterComments};
            return visitor.call(expr);
        }

        std::string getComment(ExpressionPtr const& expr, bool includeRegisterComments)
        {
            if(!expr)
            {
                return "";
            }
            return getComment(*expr, includeRegisterComments);
        }

        std::string getComment(ExpressionPtr const& expr)
        {
            return getComment(expr, true);
        }

        std::string getComment(Expression const& expr)
        {
            return getComment(expr, true);
        }

        void appendComment(Expression& expr, std::string comment)
        {
            setComment(expr, getComment(expr) + comment);
        }

        void appendComment(ExpressionPtr& expr, std::string comment)
        {
            setComment(expr, getComment(expr) + comment);
        }

        /*
         * stream operators
         */

        std::ostream& operator<<(std::ostream& stream, ResultType const& obj)
        {
            return stream << toString(obj);
        }

        std::string toString(ResultType const& obj)
        {
            return concatenate("{", obj.regType, ", ", obj.varType, "}");
        }

        std::ostream& operator<<(std::ostream& stream, ExpressionPtr const& expr)
        {
            return stream << toString(expr);
        }

        std::ostream& operator<<(std::ostream& stream, Expression const& expr)
        {
            return stream << toString(expr);
        }

        std::ostream& operator<<(std::ostream& stream, std::vector<ExpressionPtr> const& exprs)
        {
            auto iter = exprs.begin();
            stream << "[";
            if(iter != exprs.end())
                stream << *iter;
            iter++;

            for(; iter != exprs.end(); iter++)
                stream << ", " << *iter;

            stream << "]";

            return stream;
        }

        struct ExpressionComplexityVisitor
        {

            template <CUnary Expr>
            int operator()(Expr const& expr) const
            {
                return Expr::Complexity + call(expr.arg);
            }

            template <CBinary Expr>
            int operator()(Expr const& expr) const
            {
                return Expr::Complexity + call(expr.lhs) + call(expr.rhs);
            }

            template <CTernary Expr>
            int operator()(Expr const& expr) const
            {
                return Expr::Complexity + call(expr.lhs) + call(expr.r1hs) + call(expr.r2hs);
            }

            int operator()(ScaledMatrixMultiply const& expr) const
            {
                return ScaledMatrixMultiply::Complexity + call(expr.matA) + call(expr.matB)
                       + call(expr.matC) + call(expr.scaleA) + call(expr.scaleB);
            }

            template <CValue Value>
            constexpr int operator()(Value const& expr) const
            {
                return 0;
            }

            int call(ExpressionPtr expr) const
            {
                if(!expr)
                    return 0;

                return call(*expr);
            }

            int call(Expression const& expr) const
            {
                return std::visit(*this, expr);
            }

        private:
        };

        int complexity(ExpressionPtr expr)
        {
            return ExpressionComplexityVisitor().call(expr);
        }

        int complexity(Expression const& expr)
        {
            return ExpressionComplexityVisitor().call(expr);
        }

        template <CExpression T>
        struct ContainsVisitor
        {
            bool operator()(T const& expr)
            {
                return true;
            }

            template <CUnary U>
            requires(!std::same_as<T, U>) bool operator()(U const& expr)
            {
                return call(expr.arg);
            }

            template <CBinary U>
            requires(!std::same_as<T, U>) bool operator()(U const& expr)
            {
                return call(expr.lhs) || call(expr.rhs);
            }

            template <CTernary U>
            requires(!std::same_as<T, U>) bool operator()(U const& expr)
            {
                return call(expr.lhs) || call(expr.r1hs) || call(expr.r2hs);
            }

            template <std::same_as<ScaledMatrixMultiply> U>
            requires(!std::same_as<T, U>) bool operator()(U const& expr)
            {
                return call(expr.matA) || call(expr.matB) || call(expr.matC) || call(expr.scaleA)
                       || call(expr.scaleB);
            }

            template <CValue U>
            requires(!std::same_as<T, U>) bool operator()(U const& expr)
            {
                return false;
            }

            bool call(Expression const& expr)
            {
                return std::visit(*this, expr);
            }

            bool call(ExpressionPtr const& expr)
            {
                if(!expr)
                    return false;

                return call(*expr);
            }
        };

        template <CExpression T>
        __attribute__((noinline)) bool contains(Expression const& expr)
        {
            ContainsVisitor<T> v;
            return v.call(expr);
        }

        template <CExpression T>
        __attribute__((noinline)) bool contains(ExpressionPtr expr)
        {
            AssertFatal(expr != nullptr);

            return contains<T>(*expr);
        }

        /**
         * Force instantiation of contains() for every type of expression, so
         * that it can be implemented in the .cpp file.
         */
        struct ContainsInstantiateVisitor
        {
            ExpressionPtr expr;

            template <CExpression T>
            bool operator()(T const& exprType)
            {
                return contains<T>(expr);
            }
        };

        bool containsType(ExpressionPtr exprType, ExpressionPtr expr)
        {
            AssertFatal(exprType != nullptr);

            ContainsInstantiateVisitor v{expr};

            return std::visit(v, *exprType);
        }

        struct ContainsSubExpressionVisitor
        {
            ContainsSubExpressionVisitor(Expression const& expr)
                : subExpr(expr)
            {
            }

            ContainsSubExpressionVisitor(ExpressionPtr const& exprPtr)
                : subExpr(*exprPtr)
            {
            }

            template <CUnary U>
            bool operator()(U const& expr) const
            {
                return identical(expr, subExpr) || call(expr.arg);
            }

            template <CBinary U>
            bool operator()(U const& expr) const
            {
                return identical(expr, subExpr) || call(expr.lhs) || call(expr.rhs);
            }

            template <CTernary U>
            bool operator()(U const& expr) const
            {
                return identical(expr, subExpr) || call(expr.lhs) || call(expr.r1hs)
                       || call(expr.r2hs);
            }

            template <CValue U>
            bool operator()(U const& expr) const
            {
                return identical(expr, subExpr);
            }

            bool operator()(ScaledMatrixMultiply const& expr) const
            {
                return identical(expr, subExpr) || call(expr.matA) || call(expr.matB)
                       || call(expr.matC) || call(expr.scaleA) || call(expr.scaleB);
            }

            bool call(Expression const& expr) const
            {
                return std::visit(*this, expr);
            }

            bool call(ExpressionPtr const& expr) const
            {
                if(!expr)
                    return false;

                return call(*expr);
            }

            Expression const& subExpr;
        };

        bool containsSubExpression(ExpressionPtr const& expr, ExpressionPtr const& subExpr)
        {
            auto visitor = ContainsSubExpressionVisitor(subExpr);
            return visitor.call(expr);
        }

        bool containsSubExpression(Expression const& expr, Expression const& subExpr)
        {
            auto visitor = ContainsSubExpressionVisitor(subExpr);
            return visitor.call(expr);
        }
    }
}
