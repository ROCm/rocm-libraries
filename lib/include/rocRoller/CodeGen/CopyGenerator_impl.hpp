/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2021-2025 AMD ROCm(TM) Software
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

#pragma once

#include <rocRoller/CodeGen/CopyGenerator.hpp>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>
#include <rocRoller/CodeGen/Arithmetic/Utility.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/InstructionValues/Register.hpp>
#include <rocRoller/Utilities/Error.hpp>

namespace rocRoller
{
    inline CopyGenerator::CopyGenerator(ContextPtr context)
        : m_context(context)
    {
    }

    inline CopyGenerator::~CopyGenerator() = default;

    inline Generator<Instruction> CopyGenerator::conditionalCopy(Register::ValuePtr dest,
                                                                 Register::ValuePtr src,
                                                                 std::string        comment) const
    {
        // Check for invalid register types
        AssertFatal(src->regType() == Register::Type::Scalar
                        || src->regType() == Register::Type::Literal,
                    "Invalid source register type");
        AssertFatal(dest->regType() == Register::Type::Scalar, "Invalid destination register type");

        if(src->regType() == Register::Type::Literal && dest->regType() == Register::Type::Scalar)
        {
            co_yield_(Instruction("s_cmov_b32", {dest}, {src}, {}, comment));
        }
        // Scalar -> Scalar
        else if(dest->regType() == Register::Type::Scalar
                && src->regType() == Register::Type::Scalar)
        {
            for(size_t i = 0; i < src->registerCount(); ++i)
            {
                co_yield_(Instruction(
                    "s_cmov_b32", {dest->subset({i})}, {src->subset({i})}, {}, comment));
            }
        }
        // Catch unhandled copy cases
        else
        {
            Throw<FatalError>("Unhandled conditional copy case: ",
                              Register::toString(src->regType()),
                              " to ",
                              Register::toString(dest->regType()));
        }
    }

    inline Generator<Instruction> CopyGenerator::copy(Register::ValuePtr dest,
                                                      Register::ValuePtr src,
                                                      std::string        comment) const
    {
        auto context = m_context.lock();
        // Check for invalid copies
        if(dest->regType() == Register::Type::Scalar
           && src->regType() == Register::Type::Accumulator)
        {
            const auto& arch = context->targetArchitecture();
            AssertFatal(arch.HasCapability(GPUCapability::HasAccCD),
                        concatenate("Architecture",
                                    arch.target().toString(),
                                    "does not use Accumulator registers."));

            Throw<FatalError>("Can not copy accumulator register into scalar register");
        }

        if(src->sameRegistersAs(dest))
        {
            if(Settings::Get(Settings::LogLvl) >= LogLevel::Debug)
                co_yield Instruction::Comment("Omitting copy to same register: " + dest->toString()
                                              + ", " + src->toString());
            if(!comment.empty())
                co_yield Instruction::Comment(comment);
            co_return;
        }

        if(src->regType() == Register::Type::Literal && dest->regType() == Register::Type::Vector)
        {
            if(dest->variableType().getElementSize() == 4)
            {
                for(size_t k = 0; k < dest->registerCount(); ++k)
                {
                    co_yield_(Instruction("v_mov_b32", {dest->subset({k})}, {src}, {}, comment));
                }
            }
            else if(dest->variableType().getElementSize() == 8
                    && context->targetArchitecture().HasCapability(GPUCapability::v_mov_b64))
            {
                co_yield_(Instruction("v_mov_b64", {dest->element({0})}, {src}, {}, comment));
            }
            else
            {
                for(size_t k = 0; k < dest->registerCount() / 2; ++k)
                {
                    Register::ValuePtr left, right;
                    Arithmetic::get2LiteralDwords(left, right, src);
                    co_yield_(
                        Instruction("v_mov_b32", {dest->subset({2 * k})}, {left}, {}, comment));
                    co_yield_(Instruction(
                        "v_mov_b32", {dest->subset({2 * k + 1})}, {right}, {}, comment));
                }
            }
        }
        else if(src->regType() == Register::Type::Literal
                && dest->regType() == Register::Type::Scalar)
        {
            for(size_t k = 0; k < dest->valueCount(); ++k)
            {
                if(dest->variableType().getElementSize() == 4)
                {
                    co_yield_(Instruction("s_mov_b32", {dest->element({k})}, {src}, {}, comment));
                }
                else if(dest->variableType().getElementSize() == 8)
                {
                    co_yield_(Instruction("s_mov_b64", {dest->element({k})}, {src}, {}, comment));
                }
                else
                {
                    Throw<FatalError>("Unsupported copy scalar datasize");
                }
            }
        }
        else if(src->regType() == Register::Type::Literal
                && dest->regType() == Register::Type::Accumulator)
        {
            const auto& arch = context->targetArchitecture();
            AssertFatal(arch.HasCapability(GPUCapability::HasAccCD),
                        concatenate("Architecture",
                                    arch.target().toString(),
                                    "does not use Accumulator registers."));
            if(dest->variableType().getElementSize() == 4)
            {
                for(size_t k = 0; k < dest->registerCount(); ++k)
                {
                    co_yield_(
                        Instruction("v_accvgpr_write", {dest->subset({k})}, {src}, {}, comment));
                }
            }
            else
            {
                for(size_t k = 0; k < dest->registerCount() / 2; ++k)
                {
                    Register::ValuePtr left, right;
                    Arithmetic::get2LiteralDwords(left, right, src);
                    co_yield_(Instruction(
                        "v_accvgpr_write", {dest->subset({2 * k})}, {left}, {}, comment));
                    co_yield_(Instruction(
                        "v_accvgpr_write", {dest->subset({2 * k + 1})}, {right}, {}, comment));
                }
            }
        }
        else if(dest->regType() == Register::Type::M0
                && (src->regType() == Register::Type::Literal
                    || src->regType() == Register::Type::Scalar))
        {
            co_yield_(Instruction("s_mov_b32", {dest}, {src}, {}, comment));
        }
        else if(dest->regType() == Register::Type::Vector)
        {
            AssertFatal(src->registerCount() == dest->registerCount(),
                        src->registerCount(),
                        " ",
                        src->description(),
                        " to ",
                        dest->registerCount(),
                        " ",
                        dest->description());

            // Scalar/Vector -> Vector
            if(src->regType() == Register::Type::Scalar || src->regType() == Register::Type::Vector)
            {
                if(dest->variableType().getElementSize() == 8
                   && context->targetArchitecture().HasCapability(GPUCapability::v_mov_b64))
                {
                    for(size_t i = 0; i < src->registerCount() / 2; ++i)
                    {
                        co_yield_(Instruction(
                            "v_mov_b64", {dest->element({i})}, {src->element({i})}, {}, comment));
                    }
                }
                else
                {
                    for(size_t i = 0; i < src->registerCount(); ++i)
                    {
                        co_yield_(Instruction(
                            "v_mov_b32", {dest->subset({i})}, {src->subset({i})}, {}, comment));
                    }
                }
            }
            // ACCVGPR -> Vector
            else if(src->regType() == Register::Type::Accumulator)
            {
                const auto& arch = context->targetArchitecture();
                AssertFatal(arch.HasCapability(GPUCapability::HasAccCD),
                            concatenate("Architecture",
                                        arch.target().toString(),
                                        "does not use Accumulator registers."));
                for(size_t i = 0; i < src->registerCount(); ++i)
                {
                    co_yield_(Instruction(
                        "v_accvgpr_read", {dest->subset({i})}, {src->subset({i})}, {}, comment));
                }
            }
            // SCC -> Vector
            else
            {
                AssertFatal(dest->registerCount() == 1, "Untested register count");
                co_yield_(Instruction("v_mov_b32", {dest->subset({0})}, {src}, {}, comment));
            }
        }
        // Scalar -> Scalar
        else if(dest->regType() == Register::Type::Scalar
                && (src->regType() == Register::Type::Scalar || src->isSCC()))
        {
            if(src->isSCC())
            {
                co_yield_(Instruction("s_mov_b32", {dest->subset({0})}, {src}, {}, comment));
            }
            else
            {
                if(dest->variableType().getElementSize() == 8)
                {
                    for(size_t i = 0; i < src->registerCount() / 2; ++i)
                    {
                        co_yield_(Instruction(
                            "s_mov_b64", {dest->element({i})}, {src->element({i})}, {}, comment));
                    }
                }
                else
                {
                    for(size_t i = 0; i < src->registerCount(); ++i)
                    {
                        co_yield_(Instruction(
                            "s_mov_b32", {dest->subset({i})}, {src->subset({i})}, {}, comment));
                    }
                }
            }
        }
        else if(src->regType() != Register::Type::Scalar
                && dest->regType() == Register::Type::Accumulator)
        {
            // Vector -> ACCVGPR
            if(src->regType() == Register::Type::Vector)
            {
                const auto& arch = context->targetArchitecture();
                AssertFatal(arch.HasCapability(GPUCapability::HasAccCD),
                            concatenate("Architecture",
                                        arch.target().toString(),
                                        "does not use Accumulator registers."));
                for(size_t i = 0; i < src->registerCount(); ++i)
                {
                    co_yield_(Instruction(
                        "v_accvgpr_write", {dest->subset({i})}, {src->subset({i})}, {}, comment));
                }
            }
            // ACCVGPR -> ACCVGPR
            else
            {
                for(size_t i = 0; i < src->registerCount(); ++i)
                {
                    co_yield_(Instruction(
                        "v_accvgpr_mov_b32", {dest->subset({i})}, {src->subset({i})}, {}, comment));
                }
            }
        }
        // Scalar -> VCC
        else if((dest->isVCC()) && src->regType() == Register::Type::Scalar
                && ((src->registerCount() == 2 && context->kernel()->wavefront_size() == 64)
                    || (src->registerCount() == 1 && context->kernel()->wavefront_size() == 32)))
        {
            if(context->kernel()->wavefront_size() == 64)
            {
                co_yield_(Instruction("s_mov_b64", {dest}, {src}, {}, comment));
            }
            else
            {
                co_yield_(Instruction("s_mov_b32", {dest}, {src}, {}, comment));
            }
        }
        // Vector -> Scalar
        else if(dest->regType() == Register::Type::Scalar
                && src->regType() == Register::Type::Vector)
        {

            co_yield_(Instruction("v_readfirstlane_b32", {dest}, {src}, {}, comment));
        }
        // Catch unhandled copy cases
        else
        {
            Throw<FatalError>("Unhandled copy case: ",
                              Register::toString(src->regType()),
                              ": ",
                              src->toString(),
                              " to ",
                              Register::toString(dest->regType()),
                              ": ",
                              dest->toString());
        }
    }

    inline Generator<Instruction> CopyGenerator::fill(Register::ValuePtr dest,
                                                      Register::ValuePtr src,
                                                      std::string        comment) const
    {
        // Check for invalid register types
        AssertFatal(src->regType() == Register::Type::Literal, "Invalid source register type");
        AssertFatal(dest->regType() == Register::Type::Scalar
                        || dest->regType() == Register::Type::Vector
                        || dest->regType() == Register::Type::Accumulator,
                    "Invalid destination register type");

        for(size_t i = 0; i < dest->valueCount(); ++i)
            co_yield copy(dest->element({i}), src, comment);
    }

    inline Generator<Instruction> CopyGenerator::ensureType(Register::ValuePtr& dest,
                                                            Register::ValuePtr  src,
                                                            Register::Type      t) const
    {
        co_yield ensureType(dest, src, EnumBitset<Register::Type>{t});
    }

    inline Generator<Instruction> CopyGenerator::ensureType(Register::ValuePtr&        dest,
                                                            Register::ValuePtr         src,
                                                            EnumBitset<Register::Type> types) const
    {
        if(types[src->regType()])
        {
            dest = src;
            co_return;
        }

        // If null or incompatible dest, create compatible dest
        if(dest == nullptr || !types[dest->regType()])
        {
            const auto newType = [&] {
                // Pick least expensive type
                if(types[Register::Type::Scalar])
                    return Register::Type::Scalar;
                if(types[Register::Type::Vector])
                    return Register::Type::Vector;
                if(types[Register::Type::Accumulator])
                    return Register::Type::Accumulator;
                Throw<FatalError>(
                    "Cannot ensure valid register type: no concrete register types provided.",
                    ShowValue(src),
                    ShowValue(types));
            }();

            // Special cases for literals
            auto valueCount = src->valueCount() > 0 ? src->valueCount() : 1;
            auto context    = src->context() != nullptr ? src->context() : m_context.lock();

            // Allocation ctor mutates contiguousChunkWidth, so it is not directly copyable
            auto contiguousChunkWidth
                = src->allocation() != nullptr
                      ? std::max(src->allocation()->options().contiguousChunkWidth,
                                 src->allocation()->options().alignment)
                      : Register::VALUE_CONTIGUOUS;

            dest = Register::Value::Placeholder(context,
                                                newType,
                                                src->variableType(),
                                                valueCount,
                                                {.contiguousChunkWidth = contiguousChunkWidth});
        }

        co_yield copy(dest, src);
    }

    inline Generator<Instruction> CopyGenerator::pack(Register::ValuePtr              dest,
                                                      std::vector<Register::ValuePtr> values,
                                                      std::string                     comment) const
    {
        if(values.size() == 2
           && std::all_of(values.begin(),
                          values.end(),
                          [&](auto v) {
                              return v && v->regType() == Register::Type::Vector
                                     && v->variableType().getElementSize() == 2;
                          })
           && dest->regType() == Register::Type::Vector
           && dest->variableType().getElementSize() == 4)
        {
            co_yield packHalf(dest, values[0], values[1]);
        }
        else
        {
            AssertFatal(!values.empty());
            AssertFatal(std::all_of(values.begin(),
                                    values.end(),
                                    [&](auto v) {
                                        return v->variableType().getElementSize()
                                               == values[0]->variableType().getElementSize();
                                    }),
                        "Requires homogenous element sizes");
            AssertFatal(dest
                            && dest->variableType().getElementSize()
                                   == values.size() * values[0]->variableType().getElementSize(),
                        "Values do not perfectly fit");
            const auto shift_amount = values[0]->variableType().getElementSize();
            if(!dest->canUseAsOperand())
            {
                co_yield dest->allocate();
            }
            co_yield generate(dest, Expression::literal(0), m_context.lock());
            for(int i = values.size() - 1; i > 0; --i)
            {
                co_yield generateOp<Expression::BitwiseOr>(dest, dest, values[i]);
                co_yield generateOp<Expression::ShiftL>(
                    dest, dest, Register::Value::Literal(shift_amount * 8));
            }
            co_yield generateOp<Expression::BitwiseOr>(dest, dest, values[0]);
        }
    }

    inline Generator<Instruction> CopyGenerator::packHalf(Register::ValuePtr dest,
                                                          Register::ValuePtr loVal,
                                                          Register::ValuePtr hiVal,
                                                          std::string        comment) const
    {
        co_yield_(Instruction("v_pack_b32_f16", {dest}, {loVal, hiVal}, {}, ""));
    }
}
