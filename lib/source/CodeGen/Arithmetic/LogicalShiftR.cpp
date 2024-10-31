#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>
#include <rocRoller/CodeGen/Arithmetic/LogicalShiftR.hpp>
#include <rocRoller/Utilities/Component.hpp>

namespace rocRoller
{
    RegisterComponent(LogicalShiftRGenerator);

    template <>
    std::shared_ptr<BinaryArithmeticGenerator<Expression::LogicalShiftR>>
        GetGenerator<Expression::LogicalShiftR>(Register::ValuePtr dst,
                                                Register::ValuePtr lhs,
                                                Register::ValuePtr rhs,
                                                Expression::LogicalShiftR const&)
    {
        return Component::Get<BinaryArithmeticGenerator<Expression::LogicalShiftR>>(
            getContextFromValues(dst, lhs, rhs), dst->regType(), dst->variableType().dataType);
    }

    Generator<Instruction> LogicalShiftRGenerator::generate(Register::ValuePtr dest,
                                                            Register::ValuePtr value,
                                                            Register::ValuePtr shiftAmount,
                                                            Expression::LogicalShiftR const&)
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
        if(resultDWords > inputDWords)
        {
            AssertFatal(!dest->intersects(shiftAmount),
                        "Destination intersecting with shift amount not yet supported.");

            auto uintInput = value->subset(iota<int>(0, value->registerCount()).to<std::vector>());
            uintInput->setVariableType(getIntegerType(false, inputSize));

            auto destTypeUnsigned = getIntegerType(false, resultSize);
            co_yield generateConvertOp(destTypeUnsigned, dest, uintInput);

            value = dest;
        }

        if(dest->regType() == Register::Type::Scalar)
        {
            if(resultSize <= 4)
            {
                co_yield_(Instruction("s_lshr_b32", {dest}, {value, toShift}, {}, ""));
            }
            else if(resultSize == 8)
            {
                co_yield_(Instruction("s_lshr_b64", {dest}, {value, toShift}, {}, ""));
            }
            else
            {
                Throw<FatalError>("Unsupported element size for shift right operation:: ",
                                  ShowValue(resultSize * 8));
            }
        }
        else if(dest->regType() == Register::Type::Vector)
        {
            if(resultSize <= 4)
            {
                co_yield_(Instruction("v_lshrrev_b32", {dest}, {toShift, value}, {}, ""));
            }
            else if(resultSize == 8)
            {
                co_yield_(Instruction("v_lshrrev_b64", {dest}, {toShift, value}, {}, ""));
            }
            else
            {
                Throw<FatalError>("Unsupported element size for shift right operation:: ",
                                  ShowValue(resultSize * 8));
            }
        }
        else
        {
            Throw<FatalError>("Unsupported register type for shift right operation: ",
                              ShowValue(dest->regType()));
        }
    }
}
