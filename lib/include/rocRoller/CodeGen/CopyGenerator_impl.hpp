/**
 * @brief
 * @copyright Copyright 2021 Advanced Micro Devices, Inc.
 */

#pragma once

#include "CopyGenerator.hpp"

#include "../AssemblyKernel.hpp"
#include "../Context.hpp"
#include "../InstructionValues/Register.hpp"
#include "../Utilities/Error.hpp"
#include "Arithmetic/Utility.hpp"

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
            Throw<FatalError>("Unhandled copy case: ",
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
           && (src->regType() == Register::Type::Vector
               || src->regType() == Register::Type::Accumulator))
        {
            Throw<FatalError>("Can not copy vector register into scalar register");
        }

        if(src->sameAs(dest))
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
                for(size_t i = 0; i < src->registerCount(); ++i)
                {
                    co_yield_(Instruction(
                        "v_mov_b32", {dest->subset({i})}, {src->subset({i})}, {}, comment));
                }
            }
            // ACCVGPR -> Vector
            else if(src->regType() == Register::Type::Accumulator)
            {
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
                for(size_t i = 0; i < src->registerCount(); ++i)
                {
                    co_yield_(Instruction(
                        "s_mov_b32", {dest->subset({i})}, {src->subset({i})}, {}, comment));
                }
            }
        }
        else if(src->regType() != Register::Type::Scalar
                && dest->regType() == Register::Type::Accumulator)
        {
            // Vector -> ACCVGPR
            if(src->regType() == Register::Type::Vector)
            {
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

    inline Generator<Instruction> CopyGenerator::pack(Register::ValuePtr dest,
                                                      Register::ValuePtr loVal,
                                                      Register::ValuePtr hiVal,
                                                      std::string        comment) const
    {
        AssertFatal((dest && dest->regType() == Register::Type::Vector
                     && dest->variableType().dataType == DataType::Halfx2)
                        && (loVal && loVal->regType() == Register::Type::Vector
                            && loVal->variableType().dataType == DataType::Half)
                        && (hiVal && hiVal->regType() == Register::Type::Vector
                            && hiVal->variableType().dataType == DataType::Half),
                    "pack arguments must be vector registers of type Half");

        co_yield_(Instruction("v_pack_B32_F16", {dest}, {loVal, hiVal}, {}, ""));
    }
}
