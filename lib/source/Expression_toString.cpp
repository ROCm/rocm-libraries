
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
        std::string toString(EvaluationTime t)
        {
            switch(t)
            {
            case EvaluationTime::Translate:
                return "Translate";
            case EvaluationTime::KernelLaunch:
                return "KernelLaunch";
            case EvaluationTime::KernelExecute:
                return "KernelExecute";
            case EvaluationTime::Count:
            default:
                break;
            }
            Throw<FatalError>("Invalid EvaluationTime");
        }

        std::string toString(AlgebraicProperty t)
        {
            switch(t)
            {
            case AlgebraicProperty::Commutative:
                return "Commutative";
            case AlgebraicProperty::Associative:
                return "Associative";
            case AlgebraicProperty::Count:
            default:
                break;
            }
            Throw<FatalError>("Invalid EvaluationTime");
        }

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
            std::string operator()(Register::ValuePtr const& expr) const
            {
                // This allows an unallocated register value to be rendered into a string which
                // improves debugging by allowing the string representation of that expression
                // to be put into the source file as a comment.
                // Trying to generate the code for the expression will throw an exception.

                std::string tostr = "UNALLOCATED";
                if(expr->canUseAsOperand())
                    tostr = expr->toString();

                // The call() function appends the result type, so add ":" to separate the
                // value from the type.
                return tostr + ":";
            }
            std::string operator()(CommandArgumentPtr const& expr) const
            {
                if(expr)
                    return concatenate("CommandArgument(", expr->name(), ")");
                else
                    return "CommandArgument(nullptr)";
            }

            std::string operator()(CommandArgumentValue const& expr) const
            {
                return std::visit([](auto const& val) { return concatenate(val, ":"); }, expr);
            }

            std::string operator()(AssemblyKernelArgumentPtr const& expr) const
            {
                // The call() function appends the result type, so add ":" to separate the
                // value from the type.
                return expr->name + ":";
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
                auto functionalPart = std::visit(*this, expr);
                auto vt             = resultVariableType(expr);
                functionalPart += TypeAbbrev(vt);

                std::string comment = getComment(expr);
                if(comment.length() > 0)
                {
                    return concatenate("{", comment, ": ", functionalPart, "}");
                }

                return functionalPart;
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
    }
}
