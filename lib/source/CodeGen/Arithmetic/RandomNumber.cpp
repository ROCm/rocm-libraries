#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>
#include <rocRoller/CodeGen/Arithmetic/RandomNumber.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/Utilities/Component.hpp>

namespace rocRoller
{
    RegisterComponent(RandomNumberGenerator);

    template <>
    std::shared_ptr<UnaryArithmeticGenerator<Expression::RandomNumber>>
        GetGenerator<Expression::RandomNumber>(Register::ValuePtr dst,
                                               Register::ValuePtr arg,
                                               Expression::RandomNumber const&)
    {
        return Component::Get<UnaryArithmeticGenerator<Expression::RandomNumber>>(
            getContextFromValues(dst, arg), arg->regType(), arg->variableType().dataType);
    }

    Generator<Instruction> RandomNumberGenerator::generate(Register::ValuePtr dest,
                                                           Register::ValuePtr seed,
                                                           Expression::RandomNumber const&)
    {
        AssertFatal(false,
                    "Random number expression should be replaced with equivalent expressions "
                    "during transformation");
        co_return;
    }
}
