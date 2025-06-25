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

#include <rocRoller/AssemblyKernel.hpp>

#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>
#include <rocRoller/DataTypes/DataTypes_Utils.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/KernelOptions_detail.hpp>

namespace rocRoller
{

    Generator<Instruction> AssemblyKernel::allocateInitialRegisters()
    {
        auto ctx = m_context.lock();

        VariableType rawPointer{DataType::Raw32, PointerType::PointerGlobal};
        m_argumentPointer
            = std::make_shared<Register::Value>(ctx, Register::Type::Scalar, rawPointer, 1);
        m_argumentPointer->setName("Kernel argument pointer");
        co_yield m_argumentPointer->allocate();

        if(ctx->targetArchitecture().HasCapability(GPUCapability::WorkgroupIdxViaTTMP))
        {
            m_workgroupIndex[0] = ctx->getTTMP9();
            m_workgroupIndex[0]->setName("Workgroup Index X");
        }
        else
        {
            m_workgroupIndex[0]
                = Register::Value::Placeholder(ctx, Register::Type::Scalar, DataType::UInt32, 1);
            m_workgroupIndex[0]->setName("Workgroup Index X");
            co_yield m_workgroupIndex[0]->allocate();
        }

        if(m_kernelDimensions > 1)
        {
            m_workgroupIndex[1]
                = Register::Value::Placeholder(ctx, Register::Type::Scalar, DataType::UInt32, 1);
            m_workgroupIndex[1]->setName("Workgroup Index Y");
            co_yield m_workgroupIndex[1]->allocate();
        }
        if(m_kernelDimensions > 2)
        {
            m_workgroupIndex[2]
                = Register::Value::Placeholder(ctx, Register::Type::Scalar, DataType::UInt32, 1);
            m_workgroupIndex[2]->setName("Workgroup Index Z");
            co_yield m_workgroupIndex[2]->allocate();
        }

        bool packedWorkitem
            = m_kernelDimensions > 1
              && ctx->targetArchitecture().HasCapability(GPUCapability::PackedWorkitemIDs);

        if(packedWorkitem)
        {
            m_packedWorkitemIndex
                = Register::Value::Placeholder(ctx, Register::Type::Vector, DataType::Raw32, 1);
            m_packedWorkitemIndex->setName("Packed Workitem Index");
            co_yield m_packedWorkitemIndex->allocate();
        }

        m_workitemIndex[0]
            = Register::Value::Placeholder(ctx, Register::Type::Vector, DataType::UInt32, 1);
        m_workitemIndex[0]->setName("Workitem Index X");
        co_yield m_workitemIndex[0]->allocate();

        if(m_kernelDimensions > 1)
        {
            m_workitemIndex[1]
                = Register::Value::Placeholder(ctx, Register::Type::Vector, DataType::UInt32, 1);
            m_workitemIndex[1]->setName("Workitem Index Y");
            co_yield m_workitemIndex[1]->allocate();
        }

        if(m_kernelDimensions > 2)
        {
            m_workitemIndex[2]
                = Register::Value::Placeholder(ctx, Register::Type::Vector, DataType::UInt32, 1);
            m_workitemIndex[2]->setName("Workitem Index Z");
            co_yield m_workitemIndex[2]->allocate();
        }
    }

    Generator<Instruction> AssemblyKernel::preamble()
    {
        m_startedCodeGeneration = true;
        auto archName           = m_context.lock()->targetArchitecture().target().toString();

        co_yield Instruction::Directive(".amdgcn_target \"amdgcn-amd-amdhsa--" + archName + "\"");
        co_yield Instruction::Directive(".set .amdgcn.next_free_vgpr, 0");
        co_yield Instruction::Directive(".set .amdgcn.next_free_sgpr, 0");
        co_yield Instruction::Directive(".text");
        co_yield Instruction::Directive(concatenate(".globl ", kernelName()));
        co_yield Instruction::Directive(".p2align 8");
        co_yield Instruction::Directive(concatenate(".type ", kernelName(), ",@function"));

        co_yield allocateInitialRegisters();

        co_yield Instruction::Label(m_kernelStartLabel);
    }

    Generator<Instruction> AssemblyKernel::prolog()
    {
        auto ctx = m_context.lock();

        co_yield Instruction::Comment(ctx->kernelOptions()->toString());

        if(ctx->kernelOptions()->preloadKernelArguments)
            co_yield ctx->argLoader()->loadAllArguments();

        if(ctx->targetArchitecture().HasCapability(GPUCapability::WorkgroupIdxViaTTMP))
        {
            if(m_kernelDimensions > 1)
            {
                co_yield generateOp(
                    m_workgroupIndex[1],
                    ctx->getTTMP7(),
                    Expression::BitFieldExtract{
                        {nullptr, "Extract 16 bit Y coordinate"}, DataType::UInt32, 0, 16});
            }

            if(m_kernelDimensions > 2)
            {
                co_yield generateOp(
                    m_workgroupIndex[2],
                    ctx->getTTMP7(),
                    Expression::BitFieldExtract{
                        {nullptr, "Extract 16 bit Z coordinate"}, DataType::UInt32, 16, 16});
            }
        }

        if(m_packedWorkitemIndex)
        {
            co_yield generateOp(
                m_workitemIndex[0],
                m_packedWorkitemIndex,
                Expression::BitFieldExtract{
                    {nullptr, "Extract 10 bit X coordinate"}, DataType::UInt32, 0, 10});

            if(m_kernelDimensions > 1)
            {
                co_yield generateOp(
                    m_workitemIndex[1],
                    m_packedWorkitemIndex,
                    Expression::BitFieldExtract{
                        {nullptr, "Extract 10 bit Y coordinate"}, DataType::UInt32, 10, 10});
            }

            if(m_kernelDimensions > 2)
            {
                co_yield generateOp(
                    m_workitemIndex[2],
                    m_packedWorkitemIndex,
                    Expression::BitFieldExtract{
                        {nullptr, "Extract 10 bit Z coordinate"}, DataType::UInt32, 20, 10});
            }

            // No more need for packed value.
            m_packedWorkitemIndex.reset();
            if(ctx->kernelOptions()->preloadKernelArguments)
                m_argumentPointer.reset();
            else
                m_argumentPointer->setReadOnly();
        }

        for(auto& reg : m_workgroupIndex)
        {
            if(reg)
            {
                co_yield Instruction::Comment(
                    fmt::format("Marking {} as read-only", reg->description()));
                reg->setReadOnly();
            }
        }
        for(auto& reg : m_workitemIndex)
        {
            if(reg)
            {
                co_yield Instruction::Comment(
                    fmt::format("Marking {} as read-only", reg->description()));
                reg->setReadOnly();
            }
        }
    }

    Generator<Instruction> AssemblyKernel::postamble() const
    {
        co_yield Instruction("s_endpgm", {}, {}, {}, concatenate("End of ", kernelName()));

        co_yield Instruction::Label(m_kernelEndLabel);
        co_yield Instruction::Directive(concatenate(".size ",
                                                    kernelName(),
                                                    ", ",
                                                    m_kernelEndLabel->toString(),
                                                    "-",
                                                    m_kernelStartLabel->toString()));

        co_yield Instruction::Directive(".rodata");
        co_yield Instruction::Directive(".p2align 6");
        co_yield Instruction::Directive(concatenate(".amdhsa_kernel ", kernelName()));
        co_yield Instruction::Comment("Resource limits");

        auto const& kernelOpts = m_context.lock()->kernelOptions();
        int nextFreeVGPR = kernelOpts->setNextFreeVGPRToMax ? kernelOpts->maxVGPRs : total_vgprs();

        co_yield Instruction::Directive(concatenate("  .amdhsa_next_free_vgpr ", nextFreeVGPR));
        co_yield Instruction::Directive("  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr");

        if(dynamicSharedMemBytes() != nullptr)
            co_yield Instruction::Directive(concatenate("  .amdhsa_group_segment_fixed_size ",
                                                        group_segment_fixed_size(),
                                                        " // lds bytes"));

        if(m_context.lock()->targetArchitecture().HasCapability(GPUCapability::HasAccumOffset))
        {
            co_yield Instruction::Directive(concatenate("  .amdhsa_accum_offset ", accum_offset()));
        }

        if(m_wavefrontSize == 32)
            co_yield Instruction::Directive(".amdhsa_wavefront_size32 1");

        co_yield Instruction::Comment("Initial kernel state");
        co_yield Instruction::Directive("  .amdhsa_user_sgpr_kernarg_segment_ptr 1");
        co_yield Instruction::Directive(".amdhsa_system_sgpr_workgroup_id_x 1");
        co_yield Instruction::Directive(
            concatenate(".amdhsa_system_sgpr_workgroup_id_y ", m_kernelDimensions > 1));
        co_yield Instruction::Directive(
            concatenate(".amdhsa_system_sgpr_workgroup_id_z ", m_kernelDimensions > 2));

        co_yield Instruction::Directive(".amdhsa_system_sgpr_workgroup_info 0");

        // 0: 1D, 1: 2D, 2: 3D.
        co_yield Instruction::Directive(
            concatenate(".amdhsa_system_vgpr_workitem_id ", m_kernelDimensions - 1));
        co_yield Instruction::Directive(".end_amdhsa_kernel");
    }

    std::string AssemblyKernel::uniqueArgName(std::string const& base) const
    {
        int         idx = m_argumentNames.size();
        std::string rv;
        do
        {
            rv = concatenate(base, "_", idx);
            idx++;
        } while(m_argumentNames.contains(rv));

        return std::move(rv);
    }

    Expression::ExpressionPtr
        AssemblyKernel::findArgumentForExpression(Expression::ExpressionPtr exp,
                                                  ptrdiff_t&                idx) const
    {
        idx                     = -1;
        auto simplified         = simplify(exp);
        auto restored           = restoreCommandArguments(exp);
        auto restoredSimplified = simplify(restored);

        auto match = [exp, simplified, restored, restoredSimplified](auto const& arg) {
            auto equivalentToAny = [exp, simplified, restored, restoredSimplified](
                                       Expression::ExpressionPtr const& anExpression) {
                return equivalent(anExpression, exp) || equivalent(anExpression, simplified)
                       || equivalent(anExpression, restored)
                       || equivalent(anExpression, restoredSimplified);
            };

            if(equivalentToAny(arg.expression))
                return true;

            auto simpleArg = simplify(arg.expression);
            if(equivalentToAny(simpleArg))
                return true;

            auto restoredArg = restoreCommandArguments(arg.expression);

            if(equivalentToAny(restoredArg))
                return true;

            auto restoredSimplifiedArg = simplify(restoredArg);
            if(equivalentToAny(restoredSimplifiedArg))
                return true;

            return false;
        };
        auto iter = std::find_if(m_arguments.begin(), m_arguments.end(), match);

        if(iter != m_arguments.end())
        {
            idx = iter - m_arguments.begin();
            return Expression::fromKernelArgument(*iter);
        }
        return nullptr;
    }
    Expression::ExpressionPtr
        AssemblyKernel::findArgumentForExpression(Expression::ExpressionPtr exp) const
    {
        ptrdiff_t idx;
        return findArgumentForExpression(exp, idx);
    }

    Expression::ExpressionPtr AssemblyKernel::addArgument(AssemblyKernelArgument arg)
    {
        AssertFatal(m_argumentNames.find(arg.name) == m_argumentNames.end(),
                    "Error: Two arguments with the same name: " + arg.name);

        if(arg.expression)
            AssertFatal(resultVariableType(arg.expression) == arg.variableType,
                        ShowValue(resultVariableType(arg.expression)),
                        ShowValue(arg.variableType),
                        ShowValue(arg));

        if(arg.expression && m_context.lock()->kernelOptions()->deduplicateArguments)
        {
            ptrdiff_t idx;
            auto      existingArg = findArgumentForExpression(arg.expression, idx);
            if(existingArg)
            {
                m_argumentNames[arg.name] = idx;
                return existingArg;
            }
        }

        auto typeInfo = DataTypeInfo::Get(arg.variableType);
        if(isScaleType(typeInfo.variableType.dataType))
        {
            auto packedVarType = typeInfo.packedVariableType();
            AssertFatal(packedVarType, "Scale types must have a packed variable type.");
            typeInfo = DataTypeInfo::Get(packedVarType.value());
        }
        if(arg.offset == -1)
        {
            arg.offset = RoundUpToMultiple<int>(m_argumentSize, typeInfo.alignment);
        }
        if(arg.size == -1)
        {
            arg.size = CeilDivide(typeInfo.elementBits, 8u);
        }
        m_argumentSize = std::max(m_argumentSize, arg.offset + arg.size);

        m_argumentNames[arg.name] = m_arguments.size();
        m_arguments.push_back(std::move(arg));

        return Expression::fromKernelArgument(m_arguments.back());
    }

}
