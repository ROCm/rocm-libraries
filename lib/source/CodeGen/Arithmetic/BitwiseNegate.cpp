#include <rocRoller/CodeGen/Arithmetic/BitwiseNegate.hpp>
#include <rocRoller/Utilities/Component.hpp>

namespace rocRoller
{
    // Register supported components
    RegisterComponent(BitwiseNegateGenerator);

    template <>
    std::shared_ptr<UnaryArithmeticGenerator<Expression::BitwiseNegate>>
        GetGenerator<Expression::BitwiseNegate>(Register::ValuePtr dst, Register::ValuePtr arg)
    {
        return Component::Get<UnaryArithmeticGenerator<Expression::BitwiseNegate>>(
            getContextFromValues(dst, arg), dst->regType(), dst->variableType().dataType);
    }

    Generator<Instruction> BitwiseNegateGenerator::generate(Register::ValuePtr dest,
                                                            Register::ValuePtr arg)
    {
        AssertFatal(arg != nullptr);
        AssertFatal(dest != nullptr);

        auto elementSize = std::max(DataTypeInfo::Get(dest->variableType()).elementSize,
                                    DataTypeInfo::Get(arg->variableType()).elementSize);

        if(dest->regType() == Register::Type::Scalar)
        {
            if(elementSize <= 4)
                co_yield_(Instruction("s_not_b32", {dest}, {arg}, {}, ""));
            else if(elementSize == 8)
                co_yield_(Instruction("s_not_b64", {dest}, {arg}, {}, ""));
            else
                Throw<FatalError>("Unsupported element size for bitwiseNegate operation:: ",
                                  ShowValue(elementSize * 8));
        }
        else if(dest->regType() == Register::Type::Vector)
        {
            if(elementSize <= 4)
            {
                co_yield_(Instruction("v_not_b32", {dest}, {arg}, {}, ""));
            }
            else if(elementSize == 8)
            {
                co_yield_(
                    Instruction("v_not_b32", {dest->subset({0})}, {arg->subset({0})}, {}, ""));
                co_yield_(
                    Instruction("v_not_b32", {dest->subset({1})}, {arg->subset({1})}, {}, ""));
            }
            else
            {
                Throw<FatalError>("Unsupported element size for bitwiseNegate operation:: ",
                                  ShowValue(elementSize * 8));
            }
        }
        else
        {
            Throw<FatalError>("Unsupported register type for bitwiseNegate operation: ",
                              ShowValue(dest->regType()));
        }
    }
}
