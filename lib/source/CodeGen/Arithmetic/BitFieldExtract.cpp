/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>
#include <rocRoller/CodeGen/Arithmetic/BitFieldExtract.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/Utilities/Component.hpp>

namespace rocRoller
{
    RegisterComponentTemplateSpec(BitFieldExtractGenerator, DataType::Half);
    RegisterComponentTemplateSpec(BitFieldExtractGenerator, DataType::BFloat16);
    RegisterComponentTemplateSpec(BitFieldExtractGenerator, DataType::FP8);
    RegisterComponentTemplateSpec(BitFieldExtractGenerator, DataType::BF8);
    RegisterComponentTemplateSpec(BitFieldExtractGenerator, DataType::FP6);
    RegisterComponentTemplateSpec(BitFieldExtractGenerator, DataType::BF6);
    RegisterComponentTemplateSpec(BitFieldExtractGenerator, DataType::FP4);
    RegisterComponentTemplateSpec(BitFieldExtractGenerator, DataType::Int8);
    RegisterComponentTemplateSpec(BitFieldExtractGenerator, DataType::Int16);
    RegisterComponentTemplateSpec(BitFieldExtractGenerator, DataType::Int32);
    RegisterComponentTemplateSpec(BitFieldExtractGenerator, DataType::Int64);
    RegisterComponentTemplateSpec(BitFieldExtractGenerator, DataType::Raw32);
    RegisterComponentTemplateSpec(BitFieldExtractGenerator, DataType::UInt8);
    RegisterComponentTemplateSpec(BitFieldExtractGenerator, DataType::UInt16);
    RegisterComponentTemplateSpec(BitFieldExtractGenerator, DataType::UInt32);
    RegisterComponentTemplateSpec(BitFieldExtractGenerator, DataType::UInt64);

    template <>
    std::shared_ptr<UnaryArithmeticGenerator<Expression::BitFieldExtract>> GetGenerator(
        Register::ValuePtr dst, Register::ValuePtr arg, Expression::BitFieldExtract const&)
    {
        return Component::Get<UnaryArithmeticGenerator<Expression::BitFieldExtract>>(
            getContextFromValues(dst, arg), dst->regType(), dst->variableType().dataType);
    }

    // Template to hold the actual generate implementation
    // Makes partial specialization of the template feasible
    template <bool IS_INTEGRAL, bool DO_SIGNED>
    class BFEGeneratorBase
    {
    public:
        // Method to generate instructions
        static Generator<Instruction> generate(Register::ValuePtr                 dst,
                                               Register::ValuePtr                 arg,
                                               Expression::BitFieldExtract const& expr);
    };

    template <DataType DATATYPE>
    Generator<Instruction> BitFieldExtractGenerator<DATATYPE>::generate(
        Register::ValuePtr dst, Register::ValuePtr arg, Expression::BitFieldExtract const& expr)
    {
        if(Expression::getComment(expr) != "")
        {
            co_yield Instruction::Comment(Expression::getComment(expr));
        }
        else
        {
            co_yield Instruction::Comment(concatenate("BitFieldExtract<",
                                                      static_cast<int>(expr.offset),
                                                      ",",
                                                      static_cast<int>(expr.width),
                                                      ">(",
                                                      arg->description(),
                                                      ")"));
        }
        co_yield BFEGeneratorBase<EnumTypeInfo<DATATYPE>::IsIntegral,
                                  EnumTypeInfo<DATATYPE>::IsIntegral
                                      && EnumTypeInfo<DATATYPE>::IsSigned>::generate(dst,
                                                                                     arg,
                                                                                     expr);
    }

    template <bool DO_SIGNED>
    class BFEGeneratorBase<true, DO_SIGNED>
    {
    public:
        // Method to generate instructions
        static Generator<Instruction> generate(Register::ValuePtr                 dst,
                                               Register::ValuePtr                 arg,
                                               Expression::BitFieldExtract const& expr)
        {
            const auto& dstType = dst->variableType();
            const auto& srcType = arg->variableType();
            const auto& dstInfo = DataTypeInfo::Get(dstType);
            const auto& srcInfo = DataTypeInfo::Get(srcType);

            AssertFatal(arg != nullptr);
            AssertFatal(dstType.dataType == expr.outputDataType);
            AssertFatal(expr.offset + expr.width <= srcInfo.elementBits);
            AssertFatal(expr.width <= dstInfo.elementBits);

            constexpr auto op_v32 = DO_SIGNED ? "v_bfe_i32" : "v_bfe_u32";
            constexpr auto op_s32 = DO_SIGNED ? "s_bfe_i32" : "s_bfe_u32";
            constexpr auto op_s64 = DO_SIGNED ? "s_bfe_i64" : "s_bfe_u64";
            using ShiftRType      = std::
                conditional_t<DO_SIGNED, Expression::ArithmeticShiftR, Expression::LogicalShiftR>;

            // determine which registers to extract from, and where to extract them to
            auto& from   = arg;
            auto& to     = dst;
            auto  offset = expr.offset;
            if(from->registerCount() > to->registerCount())
            {
                // determine which registers contain the bits being extracted
                const auto lo_reg = expr.offset >> 5;
                const auto hi_reg = (expr.offset + expr.width - 1) >> 5;
                // only take from the registers that are actually of interest
                std::vector<int> ind(hi_reg - lo_reg + 1);
                std::iota(ind.begin(), ind.end(), lo_reg);
                from = arg->subset(ind);
                // adjust the offset to the registers actually being used
                offset = expr.offset - lo_reg * 32;
            }

            if(to->registerCount() > from->registerCount())
            {
                // determine where the results should be written
                std::vector<int> ind(from->registerCount());
                // use the highest registers if this is signed, that way we can
                // do sign extension / fill the upper registers with zeros
                // with the appropriate shift right
                std::iota(ind.begin(), ind.end(), dst->registerCount() - from->registerCount());
                to = dst->subset(ind);
            }
            AssertFatal(from->registerCount() == to->registerCount(),
                        "BFE does not support dst and arg of different sizes.");

            if(dst->regType() == Register::Type::Vector)
            {
                if(from->registerCount() == 1)
                {
                    co_yield_(Instruction(op_v32,
                                          {to},
                                          {from,
                                           Register::Value::Literal(offset),
                                           Register::Value::Literal(expr.width)},
                                          {},
                                          ""));
                }
                else if(from->registerCount() == 2)
                {
                    // no 64 bit BFE instruction available for vector regs
                    // arg << (64 - width - offset) >>> (64 - width) gives the same result
                    co_yield generateOp<Expression::ShiftL>(
                        to, from, Register::Value::Literal(64 - expr.width - offset));
                    co_yield generateOp<ShiftRType>(
                        to, to, Register::Value::Literal(64 - expr.width));
                }
                else
                {
                    Throw<FatalError>("Unsupported element size for BitFieldExtract operation: ",
                                      ShowValue(dstInfo.elementBits));
                }
            }
            else if(dst->regType() == Register::Type::Scalar)
            {
                if(from->registerCount() == 1)
                {
                    co_yield_(
                        Instruction(op_s32,
                                    {to},
                                    {from, Register::Value::Literal((expr.width << 16) | offset)},
                                    {},
                                    concatenate(ShowValue(offset), ", ", ShowValue(expr.width))));
                }
                else if(from->registerCount() == 2)
                {
                    co_yield_(
                        Instruction(op_s64,
                                    {to},
                                    {from, Register::Value::Literal((expr.width << 16) | offset)},
                                    {},
                                    concatenate(ShowValue(offset), ", ", ShowValue(expr.width))));
                }
                else
                {
                    Throw<FatalError>("Unsupported element size for BitFieldExtract operation: ",
                                      ShowValue(dstInfo.elementBits));
                }
            }
            if(dst->registerCount() > to->registerCount())
            {
                // if we extracted to an area smaller than the actual destination, extend the result
                co_yield generateOp<ShiftRType>(
                    dst,
                    dst,
                    Register::Value::Literal(32 * (dst->registerCount() - to->registerCount())));
            }
            if(dstInfo.elementBits < dst->registerCount() * 32)
            {
                // Ensure we only sign-extend into the bits we're actually expecting for this datatype
                co_yield generateOp<Expression::BitwiseAnd>(
                    dst, dst, Register::Value::Literal((1u << dstInfo.elementBits) - 1));
            }
        }
    };

    template <>
    Generator<Instruction> BFEGeneratorBase<false, false>::generate(
        Register::ValuePtr dst, Register::ValuePtr arg, Expression::BitFieldExtract const& expr)
    {
        const auto& dstType = dst->variableType();
        const auto& srcType = arg->variableType();
        const auto& dstInfo = DataTypeInfo::Get(dstType);
        const auto& srcInfo = DataTypeInfo::Get(srcType);

        const int dwidth = dstInfo.elementBits;

        AssertFatal(arg != nullptr);
        AssertFatal(dstType.dataType == expr.outputDataType);
        AssertFatal(srcInfo.segmentVariableType.dataType == dstType.dataType,
                    "Can only extract ",
                    dstType.dataType,
                    " from ",
                    dstType.dataType,
                    " or packed ",
                    dstType.dataType);
        AssertFatal(expr.width == dstInfo.elementBits && expr.offset % dstInfo.elementBits == 0);
        AssertFatal(dst->valueCount() <= srcInfo.packing);

        if(dst->valueCount() > 1)
        {
            for(int i = 0; i < dst->valueCount(); i++)
            {
                co_yield generate(dst->element({i}),
                                  arg,
                                  Expression::BitFieldExtract{
                                      {}, expr.outputDataType, expr.offset + dwidth * i, dwidth});
            }
            co_return;
        }
        if(dst->regType() == Register::Type::Vector)
        {
            co_yield_(Instruction(
                "v_bfe_u32",
                {dst},
                {arg, Register::Value::Literal(expr.offset), Register::Value::Literal(expr.width)},
                {},
                ""));
        }
        else if(dst->regType() == Register::Type::Scalar)
        {
            co_yield_(
                Instruction("s_bfe_u32",
                            {dst},
                            {arg, Register::Value::Literal((expr.width << 16) | expr.offset)},
                            {},
                            concatenate(ShowValue(expr.offset), ", ", ShowValue(expr.width))));
        }
    }
}
