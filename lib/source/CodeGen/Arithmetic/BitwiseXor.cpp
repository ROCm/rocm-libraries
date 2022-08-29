#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>
#include <rocRoller/CodeGen/Arithmetic/BitwiseXor.hpp>
#include <rocRoller/Utilities/Component.hpp>

namespace rocRoller
{
    RegisterComponent(BitwiseXorGenerator);

    template <>
    std::shared_ptr<BinaryArithmeticGenerator<Expression::BitwiseXor>>
        GetGenerator<Expression::BitwiseXor>(Register::ValuePtr dst,
                                             Register::ValuePtr lhs,
                                             Register::ValuePtr rhs)
    {
        return Component::Get<BinaryArithmeticGenerator<Expression::BitwiseXor>>(
            getContextFromValues(dst, lhs, rhs), dst->regType(), dst->variableType().dataType);
    }

    Generator<Instruction> BitwiseXorGenerator::generate(std::shared_ptr<Register::Value> dest,
                                                         std::shared_ptr<Register::Value> lhs,
                                                         std::shared_ptr<Register::Value> rhs)
    {
        AssertFatal(lhs != nullptr);
        AssertFatal(rhs != nullptr);

        auto elementSize = std::max({DataTypeInfo::Get(dest->variableType()).elementSize,
                                     DataTypeInfo::Get(lhs->variableType()).elementSize,
                                     DataTypeInfo::Get(rhs->variableType()).elementSize});

        if(dest->regType() == Register::Type::Scalar)
        {
            if(elementSize <= 4)
            {
                co_yield_(Instruction("s_xor_b32", {dest}, {lhs, rhs}, {}, ""));
            }
            else if(elementSize == 8)
            {
                co_yield_(Instruction("s_xor_b64", {dest}, {lhs, rhs}, {}, ""));
            }
            else
            {
                Throw<FatalError>("Unsupported element size for bitwiseXor operation:: ",
                                  ShowValue(elementSize * 8));
            }
        }
        else if(dest->regType() == Register::Type::Vector)
        {
            if(elementSize <= 4)
            {
                co_yield_(Instruction("v_xor_b32", {dest}, {lhs, rhs}, {}, ""));
            }
            else if(elementSize == 8)
            {
                co_yield_(Instruction("v_xor_b32",
                                      {dest->subset({0})},
                                      {lhs->subset({0}), rhs->subset({0})},
                                      {},
                                      ""));
                co_yield_(Instruction("v_xor_b32",
                                      {dest->subset({1})},
                                      {lhs->subset({1}), rhs->subset({1})},
                                      {},
                                      ""));
            }
            else
            {
                Throw<FatalError>("Unsupported element size for bitwiseXor operation:: ",
                                  ShowValue(elementSize * 8));
            }
        }
        else
        {
            Throw<FatalError>("Unsupported register type for bitwiseXor operation: ",
                              ShowValue(dest->regType()));
        }
    }
}
