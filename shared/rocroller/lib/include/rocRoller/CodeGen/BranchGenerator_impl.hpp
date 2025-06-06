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

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/BranchGenerator.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/InstructionValues/Register.hpp>
#include <rocRoller/Utilities/Error.hpp>

namespace rocRoller
{
    inline BranchGenerator::BranchGenerator(ContextPtr context)
        : m_context(context)
    {
    }

    inline BranchGenerator::~BranchGenerator() = default;

    inline Generator<Instruction> BranchGenerator::branch(Register::ValuePtr destLabel,
                                                          std::string        comment)
    {
        AssertFatal(destLabel->regType() == Register::Type::Label,
                    "Branch target must be a label.");

        auto ctx = m_context.lock();
        if(ctx->kernelOptions().alwaysWaitBeforeBranch)
        {
            co_yield Instruction::Wait(
                WaitCount::Zero(ctx->targetArchitecture(), "DEBUG: Wait before Branch"));
        }
        else
        {
            co_yield Instruction::Wait(
                WaitCount::Max(ctx->targetArchitecture(), "Keep queues within max waitcnt limit"));
        }

        co_yield_(Instruction("s_branch", {}, {destLabel}, {}, comment));
    }

    inline Generator<Instruction> BranchGenerator::branchConditional(Register::ValuePtr destLabel,
                                                                     Register::ValuePtr condition,
                                                                     bool               zero,
                                                                     std::string        comment)
    {
        auto context = m_context.lock();
        AssertFatal(destLabel->regType() == Register::Type::Label,
                    "Branch target must be a label.");

        AssertFatal(condition->isSCC() || condition->isVCC()
                        || (condition->regType() == Register::Type::Scalar
                            && ((condition->registerCount() == 2
                                 && context->kernel()->wavefront_size() == 64)
                                || (condition->registerCount() == 1
                                    && context->kernel()->wavefront_size() == 32))),
                    "Condition must be vcc, scc, or scalar. If it's a scalar, it must be size 1 "
                    "for wavefront=32 and size 2 for wavefront=64.",
                    ShowValue(condition->isSCC()),
                    ShowValue(condition->isVCC()),
                    ShowValue(condition->regType()),
                    ShowValue(condition->registerCount()),
                    ShowValue(context->kernel()->wavefront_size()));

        if(condition->regType() == Register::Type::Scalar)
        {
            co_yield context->copier()->copy(context->getVCC(), condition);
        }
        std::string conditionType;
        std::string conditionLocation;
        if(condition->isVCC() || condition->regType() == Register::Type::Scalar)
        {
            conditionType     = zero ? "z" : "nz";
            conditionLocation = "vcc";
        }
        else
        {
            conditionType     = zero ? "0" : "1";
            conditionLocation = "scc";
        }
        if(context->kernelOptions().alwaysWaitBeforeBranch)
        {
            co_yield Instruction::Wait(
                WaitCount::Zero(context->targetArchitecture(), "DEBUG: Wait before Branch"));
        }
        else
        {
            co_yield Instruction::Wait(WaitCount::Max(context->targetArchitecture(),
                                                      "Keep queues within max waitcnt limit"));
        }
        co_yield_(Instruction(concatenate("s_cbranch_", conditionLocation, conditionType),
                              {},
                              {destLabel},
                              {},
                              comment));
    }

    inline Generator<Instruction> BranchGenerator::branchIfZero(Register::ValuePtr destLabel,
                                                                Register::ValuePtr condition,
                                                                std::string        comment)
    {
        co_yield branchConditional(destLabel, condition, true, comment);
    }

    inline Generator<Instruction> BranchGenerator::branchIfNonZero(Register::ValuePtr destLabel,
                                                                   Register::ValuePtr condition,
                                                                   std::string        comment)
    {
        co_yield branchConditional(destLabel, condition, false, comment);
    }

    inline Register::ValuePtr BranchGenerator::resultRegister(Expression::ExpressionPtr expr)
    {
        auto context = m_context.lock();

        // Resolve DataFlowTags and evaluate exprs with translate time source operands.
        expr = dataFlowTagPropagation(expr, context);

        auto [conditionRegisterType, conditionVariableType] = Expression::resultType(expr);
        return conditionVariableType == DataType::Bool ? context->getSCC() : context->getVCC();
    }
}
