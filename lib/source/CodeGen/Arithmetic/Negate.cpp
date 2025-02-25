#include <rocRoller/CodeGen/Arithmetic/Negate.hpp>
#include <rocRoller/Utilities/Component.hpp>

namespace rocRoller
{
    // Register supported components
    RegisterComponent(NegateGenerator);

    template <>
    std::shared_ptr<UnaryArithmeticGenerator<Expression::Negate>> GetGenerator<Expression::Negate>(
        Register::ValuePtr dst, Register::ValuePtr arg, Expression::Negate const&)
    {
        return Component::Get<UnaryArithmeticGenerator<Expression::Negate>>(
            getContextFromValues(dst, arg), dst->regType(), dst->variableType().dataType);
    }

    Generator<Instruction> NegateGenerator::generate(Register::ValuePtr dest,
                                                     Register::ValuePtr arg,
                                                     Expression::Negate const&)
    {
        AssertFatal(arg != nullptr);
        AssertFatal(dest != nullptr);

        auto zero = Register::Value::Literal(0);
        zero->setVariableType(dest->variableType());
        co_yield generateOp<Expression::Subtract>(dest, zero, arg);
    }

}
