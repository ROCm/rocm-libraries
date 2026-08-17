// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <rocRoller/CodeGen/BranchGenerator.hpp>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/InstructionValues/Register.hpp>
#include <rocRoller/KernelOptions_detail.hpp>
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
        if(ctx->kernelOptions()->alwaysWaitBeforeBranch)
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

        AssertFatal(condition->isSCC() || condition->isVCC() || condition->isEXECZ()
                        || (condition->regType() == Register::Type::Scalar
                            && ((condition->registerCount() == 2
                                 && context->kernel()->wavefront_size() == 64)
                                || (condition->registerCount() == 1
                                    && context->kernel()->wavefront_size() == 32))),
                    "Condition must be vcc, scc, execz, or scalar. If it's a scalar, it must be "
                    "size 1 for wavefront=32 and size 2 for wavefront=64.",
                    ShowValue(condition->isSCC()),
                    ShowValue(condition->isVCC()),
                    ShowValue(condition->isEXECZ()),
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
        // s_cbranch_execz: If EXECZ is 1 (set), then jump to `destLabel`.
        // s_cbranch_execnz: If EXECZ is 0 (not set), then jump to `destLabel`.
        else if(condition->isEXECZ())
        {
            conditionType     = zero ? "nz" : "z";
            conditionLocation = "exec";
        }
        else
        {
            conditionType     = zero ? "0" : "1";
            conditionLocation = "scc";
        }
        if(context->kernelOptions()->alwaysWaitBeforeBranch)
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

        auto [conditionRegisterType, conditionVariableType, conditionValueCount]
            = Expression::resultType(expr);
        return conditionVariableType == DataType::Bool ? context->getSCC() : context->getVCC();
    }
}
