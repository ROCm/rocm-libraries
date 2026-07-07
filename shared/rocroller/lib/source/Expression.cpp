// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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
        /*
         * identical
         */

        struct ExpressionIdenticalVisitor
        {
            bool operator()(ScaledMatrixMultiply const& a, ScaledMatrixMultiply const& b)
            {
                if(a.accumulationPrecision != b.accumulationPrecision)
                    return false;

                if(not call(a.matA, b.matA))
                    return false;

                if(not call(a.matB, b.matB))
                    return false;

                if(not call(a.matC, b.matC))
                    return false;

                if(not call(a.scaleA, b.scaleA))
                    return false;

                if(not call(a.scaleB, b.scaleB))
                    return false;

                return true;
            }

            bool operator()(MatrixMultiply const& a, MatrixMultiply const& b)
            {
                if(a.accumulationPrecision != b.accumulationPrecision)
                    return false;

                if(not call(a.lhs, b.lhs))
                    return false;

                if(not call(a.r1hs, b.r1hs))
                    return false;

                if(not call(a.r2hs, b.r2hs))
                    return false;

                return true;
            }

            template <CNary Expr>
            bool operator()(Expr const& a, Expr const& b)
            {
                if(a.operands.size() != b.operands.size())
                {
                    return false;
                }

                for(size_t i = 0; i < a.operands.size(); ++i)
                {
                    auto const& operandA = a.operands.at(i);
                    auto const& operandB = b.operands.at(i);
                    if(not call(operandA, operandB))
                        return false;
                }
                return true;
            }

            template <CTernary T>
            bool operator()(T const& a, T const& b)
            {
                return call(a.lhs, b.lhs) && call(a.r1hs, b.r1hs) && call(a.r2hs, b.r2hs);
            }

            template <CBinary T>
            bool operator()(T const& a, T const& b)
            {
                return call(a.lhs, b.lhs) && call(a.rhs, b.rhs);
            }

            template <CUnary T>
            bool operator()(T const& a, T const& b)
            {
                return call(a.arg, b.arg);
            }

            constexpr bool operator()(CommandArgumentValue const& a, CommandArgumentValue const& b)
            {
                return a == b;
            }

            bool operator()(BitfieldCombine const& a, BitfieldCombine const& b)
            {
                if(a.srcOffset != b.srcOffset)
                    return false;

                if(a.dstOffset != b.dstOffset)
                    return false;

                if(a.width != b.width)
                    return false;

                if(a.srcIsZero != b.srcIsZero)
                    return false;

                if(a.dstIsZero != b.dstIsZero)
                    return false;

                if(not call(a.lhs, b.lhs))
                    return false;

                if(not call(a.rhs, b.rhs))
                    return false;

                return true;
            }

            bool operator()(BitFieldExtract const& a, BitFieldExtract const& b)
            {
                if(a.outputDataType != b.outputDataType)
                    return false;

                if(a.offset != b.offset)
                    return false;

                if(a.width != b.width)
                    return false;

                if(not call(a.arg, b.arg))
                    return false;

                return true;
            }

            bool operator()(Convert const& a, Convert const& b)
            {
                return (a.destinationType == b.destinationType) && call(a.arg, b.arg);
            }

            bool operator()(Reinterpret const& a, Reinterpret const& b)
            {
                bool destinationType = (a.destinationType == b.destinationType);
                return destinationType && call(a.arg, b.arg);
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
                if(a->getName() == b->getName())
                    return true;

                if((a->getExpression() != nullptr) && (b->getExpression() != nullptr))
                    return call(a->getExpression(), b->getExpression());

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

            constexpr bool operator()(PositionalArgument const& a, PositionalArgument const& b)
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
                if(a.accumulationPrecision != b.accumulationPrecision)
                    return false;

                if(not call(a.matA, b.matA))
                    return false;

                if(not call(a.matB, b.matB))
                    return false;

                if(not call(a.matC, b.matC))
                    return false;

                if(not call(a.scaleA, b.scaleA))
                    return false;

                if(not call(a.scaleB, b.scaleB))
                    return false;

                return true;
            }

            bool operator()(MatrixMultiply const& a, MatrixMultiply const& b)
            {
                if(a.accumulationPrecision != b.accumulationPrecision)
                    return false;

                if(not call(a.lhs, b.lhs))
                    return false;

                if(not call(a.r1hs, b.r1hs))
                    return false;

                if(not call(a.r2hs, b.r2hs))
                    return false;

                return true;
            }

            template <CNary Expr>
            bool operator()(Expr const& a, Expr const& b)
            {
                if(a.operands.size() != b.operands.size())
                {
                    return false;
                }

                for(size_t i = 0; i < a.operands.size(); ++i)
                {
                    auto const& operandA = a.operands.at(i);
                    auto const& operandB = b.operands.at(i);
                    if(not call(operandA, operandB))
                        return false;
                }
                return true;
            }

            template <CTernary T>
            bool operator()(T const& a, T const& b)
            {
                return call(a.lhs, b.lhs) && call(a.r1hs, b.r1hs) && call(a.r2hs, b.r2hs);
            }

            template <CBinary T>
            bool operator()(T const& a, T const& b)
            {
                bool result = call(a.lhs, b.lhs) && call(a.rhs, b.rhs);

                // Test if equivalent if expression is commutative
                if(!result && CCommutativeBinary<T> && m_properties[AlgebraicProperty::Commutative])
                {
                    result = call(a.lhs, b.rhs) && call(a.rhs, b.lhs);
                }

                return result;
            }

            template <CUnary T>
            bool operator()(T const& a, T const& b)
            {
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

            bool operator()(BitfieldCombine const& a, BitfieldCombine const& b)
            {
                if(a.srcOffset != b.srcOffset)
                    return false;

                if(a.dstOffset != b.dstOffset)
                    return false;

                if(a.width != b.width)
                    return false;

                if(a.srcIsZero != b.srcIsZero)
                    return false;

                if(a.dstIsZero != b.dstIsZero)
                    return false;

                if(not call(a.lhs, b.lhs))
                    return false;

                if(not call(a.rhs, b.rhs))
                    return false;

                return true;
            }

            bool operator()(BitFieldExtract const& a, BitFieldExtract const& b)
            {
                if(a.outputDataType != b.outputDataType)
                    return false;

                if(a.offset != b.offset)
                    return false;

                if(a.width != b.width)
                    return false;

                if(not call(a.arg, b.arg))
                    return false;

                return true;
            }

            bool operator()(Convert const& a, Convert const& b)
            {
                if(a.destinationType != b.destinationType)
                    return false;

                if(not call(a.arg, b.arg))
                    return false;

                return true;
            }

            bool operator()(Reinterpret const& a, Reinterpret const& b)
            {
                if(a.destinationType != b.destinationType)
                    return false;

                if(not call(a.arg, b.arg))
                    return false;

                return true;
            }

            bool operator()(AssemblyKernelArgumentPtr const& a, AssemblyKernelArgumentPtr const& b)
            {
                if(a->getName() == b->getName())
                    return true;

                if((a->getExpression() != nullptr) && (b->getExpression() != nullptr))
                    return call(a->getExpression(), b->getExpression());

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

            constexpr bool operator()(PositionalArgument const& a, PositionalArgument const& b)
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
            requires(CUnary<Expr> || CBinary<Expr> || CTernary<Expr> || CNary<Expr>) void
                operator()(Expr& expr)
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
            requires(CUnary<Expr> || CBinary<Expr> || CTernary<Expr> || CNary<Expr>) std::string
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
            return concatenate("{", obj.regType, ", ", obj.varType, ", ", obj.valueCount, "}");
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

            template <CNary Expr>
            int operator()(Expr const& expr) const
            {
                auto complexity = Expr::Complexity;
                for(auto const& operand : expr.operands)
                {
                    complexity = complexity + call(operand);
                }
                return complexity;
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
    }
}
