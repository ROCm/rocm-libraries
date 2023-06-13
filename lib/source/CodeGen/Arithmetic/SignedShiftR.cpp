#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>
#include <rocRoller/CodeGen/Arithmetic/SignedShiftR.hpp>
#include <rocRoller/Utilities/Component.hpp>

namespace rocRoller
{
    RegisterComponent(SignedShiftRGenerator);

    template <>
    std::shared_ptr<BinaryArithmeticGenerator<Expression::SignedShiftR>>
        GetGenerator<Expression::SignedShiftR>(Register::ValuePtr dst,
                                               Register::ValuePtr lhs,
                                               Register::ValuePtr rhs)
    {
        return Component::Get<BinaryArithmeticGenerator<Expression::SignedShiftR>>(
            getContextFromValues(dst, lhs, rhs), dst->regType(), dst->variableType().dataType);
    }

    Generator<Instruction> SignedShiftRGenerator::generate(Register::ValuePtr dest,
                                                           Register::ValuePtr value,
                                                           Register::ValuePtr shiftAmount)
    {
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
            if(resultSize == 4)
            {
                co_yield_(Instruction("s_ashr_i32", {dest}, {value, toShift}, {}, ""));
            }
            else if(resultSize == 8)
            {
                co_yield_(Instruction("s_ashr_i64", {dest}, {value, toShift}, {}, ""));
            }
            else
            {
                Throw<FatalError>("Unsupported datatype for arithmetic shift right operation: ",
                                  ShowValue(dest));
            }
        }
        else if(dest->regType() == Register::Type::Vector)
        {
            if(resultSize == 4)
            {
                co_yield_(Instruction("v_ashrrev_i32", {dest}, {toShift, value}, {}, ""));
            }
            else if(resultSize == 8)
            {
                co_yield_(Instruction("v_ashrrev_i64", {dest}, {toShift, value}, {}, ""));
            }
            else
            {
                Throw<FatalError>("Unsupported datatype for arithmetic shift right operation: ",
                                  ShowValue(dest));
            }
        }
        else
        {
            Throw<FatalError>("Unsupported register type for arithmetic shift right operation: ",
                              ShowValue(dest));
        }
    }
}
