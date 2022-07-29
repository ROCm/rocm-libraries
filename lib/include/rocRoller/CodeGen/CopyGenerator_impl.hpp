/**
 * @brief
 * @copyright Copyright 2021 Advanced Micro Devices, Inc.
 */

#pragma once

#include "CopyGenerator.hpp"

#include "../AssemblyKernel.hpp"
#include "../Context.hpp"
#include "../InstructionValues/Register.hpp"
#include "../InstructionValues/RegisterUtils.hpp"
#include "../Utilities/Error.hpp"

namespace rocRoller
{
    inline CopyGenerator::CopyGenerator(std::shared_ptr<Context> context)
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

        co_yield Register::AllocateIfNeeded(dest);
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
            throw FatalError("Unhandled copy case: ",
                             Register::ToString(src->regType()),
                             " to ",
                             Register::ToString(dest->regType()));
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
            throw FatalError("Can not copy vector register into scalar register");
        }

        co_yield Register::AllocateIfNeeded(dest);
        if(src->regType() == Register::Type::Literal && dest->regType() == Register::Type::Vector)
        {
            // TODO this assumes 32bit
            for(size_t k = 0; k < dest->registerCount(); ++k)
            {
                co_yield_(Instruction("v_mov_b32", {dest->subset({k})}, {src}, {}, comment));
            }
        }
        else if(src->regType() == Register::Type::Literal
                && dest->regType() == Register::Type::Scalar)
        {
            // TODO this assumes 32bit
            for(size_t k = 0; k < dest->registerCount(); ++k)
            {
                co_yield_(Instruction("s_mov_b32", {dest->subset({k})}, {src}, {}, comment));
            }
        }
        else if(src->regType() == Register::Type::Literal
                && dest->regType() == Register::Type::Accumulator)
        {
            for(size_t k = 0; k < dest->valueCount(); ++k)
            {
                co_yield_(Instruction("v_accvgpr_write", {dest->element({k})}, {src}, {}, comment));
            }
        }
        else if(dest->regType() == Register::Type::Vector)
        {
            AssertFatal(src->registerCount() == dest->registerCount(),
                        src->registerCount(),
                        " ",
                        src->toString(),
                        " to ",
                        dest->registerCount(),
                        " ",
                        dest->toString());

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
            else
            {
                for(size_t i = 0; i < src->registerCount(); ++i)
                {
                    co_yield_(Instruction(
                        "v_accvgpr_read", {dest->subset({i})}, {src->subset({i})}, {}, comment));
                }
            }
        }
        // Scalar -> Scalar
        else if(dest->regType() == Register::Type::Scalar
                && (src->regType() == Register::Type::Scalar || src->isSCC()))
        {
            if(src->registerCount() == 0)
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
            throw FatalError("Unhandled copy case: ",
                             Register::ToString(src->regType()),
                             ": ",
                             src->toString(),
                             " to ",
                             Register::ToString(dest->regType()),
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

        co_yield Register::AllocateIfNeeded(dest);
        for(size_t i = 0; i < dest->valueCount(); ++i)
            co_yield copy(dest->element({i}), src, comment);
    }

    inline Generator<Instruction> CopyGenerator::ensureType(Register::ValuePtr& dest,
                                                            Register::ValuePtr  src,
                                                            Register::Type      t) const
    {
        if(src->regType() == t)
        {
            dest = src;
            co_return;
        }

        if(!(dest != nullptr && dest->regType() == t
             && (dest->variableType() == src->variableType())
             && (dest->valueCount() == src->valueCount())))
        {
            dest = Register::Value::Placeholder(
                src->context(), t, src->variableType(), src->valueCount());
        }
        co_yield copy(dest, src);
    }
}
