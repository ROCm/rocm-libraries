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
                left  = std::move(r2hs);
                right = std::move(r1hs);
            }
            else
            {
                left  = std::move(r1hs);
                right = std::move(r2hs);
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
            AssertFatal(cond->isVCC() || cond->regType() == Register::Type::Scalar,
                        ShowValue(cond->regType()));

            auto const elementSize = dest->variableType().getElementSize();
            if(elementSize == 8)
            {
                co_yield_(Instruction("v_cndmask_b32_e64",
                                      {dest->subset({0})},
                                      {r2hs->subset({0}), r1hs->subset({0}), cond},
                                      {},
                                      ""));
                co_yield_(Instruction("v_cndmask_b32_e64",
                                      {dest->subset({1})},
                                      {r2hs->subset({1}), r1hs->subset({1}), cond},
                                      {},
                                      ""));
            }
            else if(elementSize == 4)
            {
                co_yield_(Instruction("v_cndmask_b32_e64", {dest}, {r2hs, r1hs, cond}, {}, ""));
            }
            else
            {
                AssertFatal(false, "Unsupported size ", ShowValue(elementSize));
            }
        }
    }
}
