#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>
#include <rocRoller/CodeGen/Arithmetic/RandomNumber.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/Utilities/Component.hpp>

namespace rocRoller
{
    RegisterComponent(RandomNumberGenerator);

    template <>
    std::shared_ptr<UnaryArithmeticGenerator<Expression::RandomNumber>>
        GetGenerator<Expression::RandomNumber>(Register::ValuePtr dst, Register::ValuePtr arg)
    {
        return Component::Get<UnaryArithmeticGenerator<Expression::RandomNumber>>(
            getContextFromValues(dst, arg), arg->regType(), arg->variableType().dataType);
    }

    Generator<Instruction> RandomNumberGenerator::generate(Register::ValuePtr dest,
                                                           Register::ValuePtr seed)
    {
        AssertFatal(seed != nullptr);
        AssertFatal(dest != nullptr);
        co_yield_(Instruction("v_prng_b32", {dest}, {seed}, {}, "Generate a random number"));
    }
}
