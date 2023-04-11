#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>
#include <rocRoller/CodeGen/Arithmetic/ShiftL.hpp>
#include <rocRoller/Utilities/Component.hpp>

namespace rocRoller
{
    RegisterComponent(ShiftLGenerator);

    template <>
    std::shared_ptr<BinaryArithmeticGenerator<Expression::ShiftL>> GetGenerator<Expression::ShiftL>(
        Register::ValuePtr dst, Register::ValuePtr lhs, Register::ValuePtr rhs)
    {
        return Component::Get<BinaryArithmeticGenerator<Expression::ShiftL>>(
            getContextFromValues(dst, lhs, rhs), dst->regType(), dst->variableType().dataType);
    }

    Generator<Instruction> ShiftLGenerator::generate(std::shared_ptr<Register::Value> dest,
                                                     std::shared_ptr<Register::Value> value,
                                                     std::shared_ptr<Register::Value> shiftAmount)
    {
        co_yield describeOpArgs("dest", dest, "value", value, "shiftAmount", shiftAmount);

        AssertFatal(dest != nullptr);
        AssertFatal(value != nullptr);
        AssertFatal(shiftAmount != nullptr);

        auto toShift = shiftAmount->regType() == Register::Type::Literal ? shiftAmount
                                                                         : shiftAmount->subset({0});

        auto resultSize = dest->variableType().getElementSize();
        auto inputSize  = value->variableType().getElementSize();

        auto resultDWords = CeilDivide(resultSize, 4ul);
        auto inputDWords  = CeilDivide(inputSize, 4ul);

        // We only want to do this conversion if there are more DWords in the output than the
        // input.  For example if we are using the shift to convert between Halfx2 and Half, both
        // use only 1 DWord.

        AssertFatal(resultDWords >= inputDWords,
                    ShowValue(dest->variableType()),
                    ShowValue(value->variableType()));

        if(resultDWords > inputDWords)
        {
            AssertFatal(!dest->intersects(shiftAmount),
                        "Destination intersecting with shift amount not yet supported.");
            co_yield generateConvertOp(dest->variableType().getArithmeticType(), dest, value);
            value = dest;
        }

        if(dest->regType() == Register::Type::Scalar)
        {
            if(resultSize <= 4)
            {
                co_yield_(Instruction("s_lshl_b32", {dest}, {value, toShift}, {}, ""));
            }
            else if(resultSize == 8)
            {
                co_yield_(Instruction("s_lshl_b64", {dest}, {value, toShift}, {}, ""));
            }
            else
            {
                Throw<FatalError>("Unsupported element size for shift left operation:: ",
                                  ShowValue(resultSize * 8));
            }
        }
        else if(dest->regType() == Register::Type::Vector)
        {
            if(resultSize <= 4)
            {
                co_yield_(Instruction("v_lshlrev_b32", {dest}, {toShift, value}, {}, ""));
            }
            else if(resultSize == 8)
            {
                co_yield_(Instruction("v_lshlrev_b64", {dest}, {toShift, value}, {}, ""));
            }
            else
            {
                Throw<FatalError>("Unsupported element size for shift left operation:: ",
                                  ShowValue(resultSize * 8));
            }
        }
        else
        {
            Throw<FatalError>("Unsupported register type for shift left operation: ",
                              ShowValue(dest->regType()));
        }
    }
}
