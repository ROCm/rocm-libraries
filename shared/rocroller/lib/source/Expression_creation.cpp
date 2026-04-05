// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <variant>

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/InstructionValues/Register_fwd.hpp>
#include <rocRoller/Utilities/Generator.hpp>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/Utilities/Timer.hpp>

namespace rocRoller
{
    namespace Expression
    {
        ExpressionPtr fromKernelArgument(AssemblyKernelArgument const& arg)
        {
            return std::make_shared<Expression>(std::make_shared<AssemblyKernelArgument>(arg));
        }

        ExpressionPtr multiplyHigh(ExpressionPtr a, ExpressionPtr b)
        {
            AssertFatal(!isRaw32Literal(a) and !isRaw32Literal(b),
                        "Raw32 is a bit type and cannot be used in multiplyHigh operation: ",
                        ShowValue(a),
                        ", ",
                        ShowValue(b));
            return std::make_shared<Expression>(MultiplyHigh{a, b});
        }

        ExpressionPtr multiplyAdd(ExpressionPtr a, ExpressionPtr b, ExpressionPtr c)
        {
            AssertFatal(!isRaw32Literal(a) and !isRaw32Literal(b) and !isRaw32Literal(c),
                        "Raw32 is a bit type and cannot be used in multiplyAdd operation: ",
                        ShowValue(a),
                        ", ",
                        ShowValue(b),
                        ", ",
                        ShowValue(c));
            return std::make_shared<Expression>(MultiplyAdd{a, b, c});
        }

        ExpressionPtr addShiftL(ExpressionPtr a, ExpressionPtr b, ExpressionPtr c)
        {
            AssertFatal(!isRaw32Literal(a) and !isRaw32Literal(b) and !isRaw32Literal(c),
                        "Raw32 is a bit type and cannot be used in addShiftL operation: ",
                        ShowValue(a),
                        ", ",
                        ShowValue(b),
                        ", ",
                        ShowValue(c));
            return std::make_shared<Expression>(AddShiftL{a, b, c});
        }

        ExpressionPtr shiftLAdd(ExpressionPtr a, ExpressionPtr b, ExpressionPtr c)
        {
            AssertFatal(!isRaw32Literal(b) and !isRaw32Literal(c),
                        "Raw32 is a bit type and cannot be used in shiftLAdd operation: ",
                        ShowValue(b),
                        ", ",
                        ShowValue(c));
            return std::make_shared<Expression>(ShiftLAdd{a, b, c});
        }

        ExpressionPtr conditional(ExpressionPtr a, ExpressionPtr b, ExpressionPtr c)
        {
            return std::make_shared<Expression>(Conditional{a, b, c});
        }

        ExpressionPtr magicMultiple(ExpressionPtr a)
        {
            AssertFatal(!isRaw32Literal(a),
                        "Raw32 is a bit type and cannot be used in magicMultiple operation: ",
                        ShowValue(a));
            return std::make_shared<Expression>(MagicMultiple{a});
        }

        ExpressionPtr magicShifts(ExpressionPtr a)
        {
            AssertFatal(!isRaw32Literal(a),
                        "Raw32 is a bit type and cannot be used in magicShifts operation: ",
                        ShowValue(a));
            return std::make_shared<Expression>(MagicShifts{a});
        }

        ExpressionPtr magicShiftAndSign(ExpressionPtr a)
        {
            AssertFatal(!isRaw32Literal(a),
                        "Raw32 is a bit type and cannot be used in magicShiftAndSign operation: ",
                        ShowValue(a));
            return std::make_shared<Expression>(MagicShiftAndSign{a});
        }

        ExpressionPtr bfe(ExpressionPtr a, uint32_t offset, uint32_t width)
        {
            return std::make_shared<Expression>(
                BitFieldExtract{{.arg{a}}, DataType::None, offset, width});
        }

        ExpressionPtr bfe(DataType dt, ExpressionPtr a, uint32_t offset, uint32_t width)
        {
            return std::make_shared<Expression>(BitFieldExtract{{.arg{a}}, dt, offset, width});
        }

        ExpressionPtr bfc(ExpressionPtr       src,
                          ExpressionPtr       dst,
                          uint32_t            srcOffset,
                          uint32_t            dstOffset,
                          uint32_t            width,
                          std::optional<bool> srcIsZero,
                          std::optional<bool> dstIsZero)
        {
            return std::make_shared<Expression>(
                BitfieldCombine{{src, dst}, srcOffset, dstOffset, width, srcIsZero, dstIsZero});
        }

        ExpressionPtr concat(const std::vector<ExpressionPtr>& ops, VariableType v)
        {
            return std::make_shared<Expression>(Concatenate{{ops}, v});
        }

        ExpressionPtr dataFlowTag(int tag, Register::Type t, VariableType v)
        {
            return std::make_shared<Expression>(DataFlowTag{tag, t, v});
        }

        ExpressionPtr positionalArgument(int slot, Register::Type t, VariableType v)
        {
            return std::make_shared<Expression>(PositionalArgument{slot, t, v});
        }
    }
}
