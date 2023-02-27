
#include <variant>

#include "DataTypes/DataTypes.hpp"
#include "InstructionValues/Register_fwd.hpp"
#include "Utilities/Generator.hpp"

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
         * to string
         */

        struct ExpressionToStringVisitor
        {
            template <CTernary Expr>
            std::string operator()(Expr const& expr) const
            {
                return concatenate(ExpressionInfo<Expr>::name(),
                                   "(",
                                   call(expr.lhs),
                                   ", ",
                                   call(expr.r1hs),
                                   ", ",
                                   call(expr.r2hs),
                                   ")");
            }

            template <CBinary Expr>
            std::string operator()(Expr const& expr) const
            {
                return concatenate(
                    ExpressionInfo<Expr>::name(), "(", call(expr.lhs), ", ", call(expr.rhs), ")");
            }
            template <CUnary Expr>
            std::string operator()(Expr const& expr) const
            {
                return concatenate(ExpressionInfo<Expr>::name(), "(", call(expr.arg), ")");
            }
            std::string operator()(std::shared_ptr<Register::Value> const& expr) const
            {
                // This allows an unallocated register value to be rendered into a string which
                // improves debugging by allowing the string representation of that expression
                // to be put into the source file as a comment.
                // Trying to generate the code for the expression will throw an exception.

                std::string tostr = "UNALLOCATED";
                if(expr->canUseAsOperand())
                    tostr = expr->toString();

                return tostr + ":" + TypeAbbrev(expr->variableType());
            }
            std::string operator()(std::shared_ptr<CommandArgument> const& expr) const
            {
                if(expr)
                    return concatenate("CommandArgument(", expr->name(), ")");
                else
                    return "CommandArgument(nullptr)";
            }

            std::string operator()(CommandArgumentValue const& expr) const
            {
                return std::visit(
                    [](auto const& val) { return concatenate(val) + typeid(val).name(); }, expr);
            }

            std::string operator()(AssemblyKernelArgumentPtr const& expr) const
            {
                return expr->name;
            }

            std::string operator()(WaveTilePtr const& expr) const
            {
                return "WaveTile";
            }

            std::string operator()(DataFlowTag const& expr) const
            {
                return concatenate("DataFlowTag(", expr.tag, ")");
            }

            std::string call(Expression const& expr) const
            {
                std::string comment = getComment(expr);
                if(comment.length() > 0)
                {
                    return concatenate("{", comment, ": ", std::visit(*this, expr), "}");
                }

                return std::visit(*this, expr);
            }

            std::string call(ExpressionPtr expr) const
            {
                if(!expr)
                    return "nullptr";

                return call(*expr);
            }
        };

        std::string toString(Expression const& expr)
        {
            auto visitor = ExpressionToStringVisitor();
            return visitor.call(expr);
        }

        std::string toString(ExpressionPtr const& expr)
        {
            auto visitor = ExpressionToStringVisitor();
            return visitor.call(expr);
        }

        /*
         * result type
         */

        struct ExpressionResultTypeVisitor
        {
            template <typename T>
            requires(CBinary<T>&& CArithmetic<T>) ResultType operator()(T const& expr) const
            {
                auto lhsVal = call(expr.lhs);
                auto rhsVal = call(expr.rhs);

                auto regType = Register::PromoteType(lhsVal.regType, rhsVal.regType);

                auto varType = VariableType::Promote(lhsVal.varType, rhsVal.varType);

                // TODO: Delete once FastDivision uses only libdivide.
                if constexpr(std::same_as<MultiplyHigh, T>)
                    varType = DataType::Int32;

                return {regType, varType};
            }

            template <typename T>
            requires(CTernary<T>&& CArithmetic<T>) ResultType operator()(T const& expr) const
            {
                auto lhsVal  = call(expr.lhs);
                auto r1hsVal = call(expr.r1hs);
                auto r2hsVal = call(expr.r2hs);

                auto regType = Register::PromoteType(lhsVal.regType, r1hsVal.regType);
                regType      = Register::PromoteType(regType, r2hsVal.regType);

                auto varType = VariableType::Promote(lhsVal.varType, r1hsVal.varType);
                varType      = VariableType::Promote(varType, r2hsVal.varType);

                return {regType, varType};
            }

            template <typename T>
            requires(CUnary<T>&& CArithmetic<T>) ResultType operator()(T const& expr) const
            {
                auto argVal = call(expr.arg);
                return argVal;
            }

            template <DataType DATATYPE>
            ResultType operator()(Convert<DATATYPE> const& expr) const
            {
                auto argVal = call(expr.arg);
                return {argVal.regType, DATATYPE};
            }

            template <typename T>
            requires(CBinary<T>&& CComparison<T>) ResultType operator()(T const& expr) const
            {
                auto lhsVal = call(expr.lhs);
                auto rhsVal = call(expr.rhs);
                return comparison(lhsVal, rhsVal);
            }

            ResultType comparison(ResultType const& lhsVal, ResultType const& rhsVal) const
            {
                // Can't compare between two different types on the GPU.
                AssertFatal(lhsVal.regType == Register::Type::Literal
                                || rhsVal.regType == Register::Type::Literal
                                || lhsVal.varType == rhsVal.varType,
                            ShowValue(lhsVal.varType),
                            ShowValue(rhsVal.varType));

                auto inputRegType = Register::PromoteType(lhsVal.regType, rhsVal.regType);
                auto inputVarType = VariableType::Promote(lhsVal.varType, rhsVal.varType);

                switch(inputRegType)
                {
                case Register::Type::Literal:
                    return {Register::Type::Literal, DataType::Bool};
                case Register::Type::Scalar:
                    if(inputVarType == DataType::Int32 || inputVarType == DataType::UInt32
                       || inputVarType == DataType::Bool)
                        return {Register::Type::Special, DataType::Bool};
                    else
                        return {Register::Type::Scalar, DataType::Bool32};
                case Register::Type::Vector:
                    return {Register::Type::Scalar, DataType::Bool32};

                default:
                    break;
                }
                Throw<FatalError>("Invalid register types: ",
                                  ShowValue(lhsVal.regType),
                                  ShowValue(rhsVal.regType));
            }

            ResultType operator()(CommandArgumentPtr const& expr) const
            {
                AssertFatal(expr != nullptr, "Null subexpression!");
                return {Register::Type::Literal, expr->variableType()};
            }

            ResultType operator()(AssemblyKernelArgumentPtr const& expr) const
            {
                AssertFatal(expr != nullptr, "Null subexpression!");
                return {Register::Type::Scalar, expr->variableType};
            }

            ResultType operator()(CommandArgumentValue const& expr) const
            {
                return {Register::Type::Literal, variableType(expr)};
            }

            ResultType operator()(Register::ValuePtr const& expr) const
            {
                AssertFatal(expr != nullptr, "Null subexpression!");
                return {expr->regType(), expr->variableType()};
            }

            ResultType operator()(DataFlowTag const& expr) const
            {
                return {expr.regType, expr.varType};
            }

            ResultType operator()(WaveTilePtr const& expr) const
            {
                return call(expr->vgpr);
            }

            ResultType call(Expression const& expr) const
            {
                return std::visit(*this, expr);
            }

            ResultType call(ExpressionPtr const& expr) const
            {
                return call(*expr);
            }
        };

        VariableType resultVariableType(ExpressionPtr const& expr)
        {
            ExpressionResultTypeVisitor v;
            return v.call(expr).varType;
        }

        Register::Type resultRegisterType(ExpressionPtr const& expr)
        {
            ExpressionResultTypeVisitor v;
            return v.call(expr).regType;
        }

        ResultType resultType(ExpressionPtr const& expr)
        {
            ExpressionResultTypeVisitor v;
            return v.call(expr);
        }

        ResultType resultType(Expression const& expr)
        {
            ExpressionResultTypeVisitor v;
            return v.call(expr);
        }

        /*
         * identical
         */

        struct ExpressionIdenticalVisitor
        {
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

            bool operator()(CommandArgumentValue const& a, CommandArgumentValue const& b)
            {
                return a == b;
            }

            bool operator()(CommandArgumentPtr const& a, CommandArgumentPtr const& b)
            {
                return (*a) == (*b);
            }

            bool operator()(AssemblyKernelArgumentPtr const& a, AssemblyKernelArgumentPtr const& b)
            {
                return (*a) == (*b);
            }

            bool operator()(Register::ValuePtr const& a, Register::ValuePtr const& b)
            {
                return a->toString() == b->toString();
            }

            bool operator()(DataFlowTag const& a, DataFlowTag const& b)
            {
                return a == b;
            }

            bool operator()(WaveTilePtr const& a, WaveTilePtr const& b)
            {
                return a == b;
            }

            // a & b are different operator/value classes
            template <class T, class U>
            bool operator()(T const& a, U const& b)
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
        };

        bool identical(ExpressionPtr const& a, ExpressionPtr const& b)
        {
            auto visitor = ExpressionIdenticalVisitor();
            return visitor.call(a, b);
        }

        /*
         * comments
         */

        struct ExpressionSetCommentVisitor
        {
            std::string comment;

            ExpressionSetCommentVisitor(std::string input)
                : comment(input)
            {
            }

            template <typename Expr>
            requires(CUnary<Expr> || CBinary<Expr> || CTernary<Expr>) void operator()(Expr& expr)
            {
                expr.comment = comment;
            }

            void operator()(auto& expr)
            {
                Throw<FatalError>("Cannot set a comment for a base expression.");
            }

            void call(Expression& expr)
            {
                return std::visit(*this, expr);
            }
        };

        void setComment(Expression& expr, std::string comment)
        {
            auto visitor = ExpressionSetCommentVisitor(comment);
            return visitor.call(expr);
        }

        void setComment(ExpressionPtr& expr, std::string comment)
        {
            if(expr)
            {
                setComment(*expr, comment);
            }
            else
            {
                Throw<FatalError>("Cannot set the comment for a null expression pointer.");
            }
        }

        struct ExpressionGetCommentVisitor
        {
            template <typename Expr>
            requires(CUnary<Expr> || CBinary<Expr> || CTernary<Expr>) std::string
                operator()(Expr const& expr) const
            {
                return expr.comment;
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

        std::string getComment(Expression const& expr)
        {
            auto visitor = ExpressionGetCommentVisitor();
            return visitor.call(expr);
        }

        std::string getComment(ExpressionPtr const& expr)
        {
            if(!expr)
            {
                return "";
            }
            return getComment(*expr);
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
            return stream << "{" << obj.regType << ", " << obj.varType << "}";
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
    }
}
