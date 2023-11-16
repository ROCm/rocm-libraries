#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>
#include <rocRoller/CodeGen/Arithmetic/Conditional.hpp>

namespace rocRoller
{
    RegisterComponent(ConditionalGenerator);

    template <>
    std::shared_ptr<TernaryArithmeticGenerator<Expression::Conditional>>
        GetGenerator<Expression::Conditional>(Register::ValuePtr dst,
                                              Register::ValuePtr lhs,
                                              Register::ValuePtr r1hs,
                                              Register::ValuePtr r2hs)
    {
        return Component::Get<TernaryArithmeticGenerator<Expression::Conditional>>(
            getContextFromValues(dst, lhs, r1hs, r2hs),
            dst->regType(),
            dst->variableType().dataType);
    }

    Generator<Instruction> ConditionalGenerator::generate(Register::ValuePtr dest,
                                                          Register::ValuePtr cond,
                                                          Register::ValuePtr r1hs,
                                                          Register::ValuePtr r2hs)
    {
        AssertFatal(cond != nullptr);
        AssertFatal(r1hs != nullptr);
        AssertFatal(r2hs != nullptr);
        AssertFatal(dest->valueCount() == 1,
                    "Non-1 value count not supported",
                    ShowValue(dest->valueCount()));

        if(dest->regType() == Register::Type::Scalar)
        {
            Register::ValuePtr left, right;

            // Swap sides depending if we use SCC or !SCC
            if(!cond->isSCC())
            {
                co_yield(
                    Instruction::Lock(Scheduling::Dependency::SCC, "Start of Conditional(SCC)"));
                co_yield generateOp<Expression::Equal>(
                    m_context->getSCC(), cond, Register::Value::Literal(0));
                left  = r2hs;
                right = r1hs;
            }
            else
            {
                left  = r1hs;
                right = r2hs;
            }

            auto const elementSize = dest->variableType().getElementSize();
            if(elementSize == 8)
            {
                co_yield_(Instruction("s_cselect_b64", {dest}, {left, right}, {}, ""));
            }
            else if(elementSize == 4)
            {
                co_yield_(Instruction("s_cselect_b32", {dest}, {left, right}, {}, ""));
            }
            else
            {
                AssertFatal(false, "Unsupported scalar size ", ShowValue(elementSize));
            }

            if(!cond->isSCC())
            {
                co_yield(Instruction::Unlock("End of Conditional(SCC)"));
            }
        }
        else
        {
            AssertFatal(false, "Non-SGPRs are not supported");
        }
    }
}
