// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/AssemblyKernel.hpp>

#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>
#include <rocRoller/CodeGen/BranchGenerator.hpp>
#include <rocRoller/DataTypes/DataTypes_Utils.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/InstructionValues/LabelAllocator.hpp>
#include <rocRoller/KernelOptions_detail.hpp>

namespace rocRoller
{

    Generator<Instruction> AssemblyKernel::allocateInitialRegisters()
    {
        auto ctx = m_context.lock();

        auto argLoader = ctx->argLoader();

        if(argLoader->anyManuallyLoadedArguments())
        {
            VariableType rawPointer{DataType::Raw32, PointerType::PointerGlobal};
            m_argumentPointer
                = std::make_shared<Register::Value>(ctx, Register::Type::Scalar, rawPointer, 1);
            m_argumentPointer->setName("Kernel argument pointer");
            co_yield m_argumentPointer->allocate();
        }
        else
        {
            co_yield Instruction::Comment("No kernel argument pointer needed");
        }

        co_yield ctx->argLoader()->allocatePreloadedRegisters(m_preloadedRegOffset,
                                                              m_numPreloadedRegs);

        m_workgroupIndex[0]
            = Register::Value::Placeholder(ctx, Register::Type::Scalar, DataType::UInt32, 1);
        m_workgroupIndex[0]->setName("Workgroup Index X");
        co_yield m_workgroupIndex[0]->allocate();

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

        Register::AllocationOptions allocOptions = {};

        if(ctx->targetArchitecture().HasCapability(GPUCapability::HasVGPRIndexing))
            allocOptions.forceReservedRegion = true;

        bool packedWorkitem
            = m_kernelDimensions > 1
              && ctx->targetArchitecture().HasCapability(GPUCapability::PackedWorkitemIDs);

        if(packedWorkitem)
        {
            m_packedWorkitemIndex = Register::Value::Placeholder(
                ctx, Register::Type::Vector, DataType::Raw32, 1, allocOptions);
            m_packedWorkitemIndex->setName("Packed Workitem Index");
            co_yield m_packedWorkitemIndex->allocate();
        }

        m_workitemIndex[0] = Register::Value::Placeholder(
            ctx, Register::Type::Vector, DataType::UInt32, 1, allocOptions);
        m_workitemIndex[0]->setName("Workitem Index X");
        co_yield m_workitemIndex[0]->allocate();

        if(m_kernelDimensions > 1)
        {
            m_workitemIndex[1] = Register::Value::Placeholder(
                ctx, Register::Type::Vector, DataType::UInt32, 1, allocOptions);
            m_workitemIndex[1]->setName("Workitem Index Y");
            co_yield m_workitemIndex[1]->allocate();
        }

        if(m_kernelDimensions > 2)
        {
            m_workitemIndex[2] = Register::Value::Placeholder(
                ctx, Register::Type::Vector, DataType::UInt32, 1, allocOptions);
            m_workitemIndex[2]->setName("Workitem Index Z");
            co_yield m_workitemIndex[2]->allocate();
        }
    }

    Generator<Instruction> AssemblyKernel::preamble()
    {
        m_startedCodeGeneration = true;
        auto archName = m_context.lock()->targetArchitecture().target().toAssemblerString();

        co_yield Instruction::Comment("Kernel Arguments:");
        for(auto const& arg : m_arguments)
            co_yield Instruction::Comment(arg.toString());

        co_yield Instruction::Directive(".amdgcn_target \"amdgcn-amd-amdhsa--" + archName + "\"");
        co_yield Instruction::Directive(".set .amdgcn.next_free_vgpr, 0");
        co_yield Instruction::Directive(".set .amdgcn.next_free_sgpr, 0");
        co_yield Instruction::Directive(".text");
        co_yield Instruction::Directive(concatenate(".globl ", kernelName()));
        co_yield Instruction::Directive(".p2align 8");
        co_yield Instruction::Directive(concatenate(".type ", kernelName(), ",@function"));

        co_yield allocateInitialRegisters();

        co_yield Instruction::Label(m_kernelStartLabel);

        if(m_numPreloadedRegs > 0)
        {
            co_yield Instruction::Comment("Pad to make room for arg loading preamble.");
            for(int i = 0; i < 64; i++)
                co_yield Instruction::Nop();

            auto afterPaddingLabel = m_kernelStartLabel->getLabel() + "_exec_begin";
            co_yield Instruction::Label(
                afterPaddingLabel,
                "Kernel execution starts here. Use this label instead of the "
                "kernel name when setting a breakpoint.");
        }
    }

    Generator<Instruction> AssemblyKernel::prolog()
    {
        auto ctx = m_context.lock();

        co_yield Instruction::Comment(ctx->kernelOptions()->toString());

        if(!ctx->kernelOptions()->lazyLoadKernelArguments)
            co_yield ctx->argLoader()->eagerLoadArguments();

        co_yield ctx->argLoader()->splitOutArgumentRegisters();

        if(ctx->targetArchitecture().HasCapability(GPUCapability::WorkgroupIdxViaTTMP))
        {
            co_yield Instruction(
                "s_mov_b32", {m_workgroupIndex[0]}, {ctx->getTTMP9()}, {}, "Load workgroup ID X");

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

            /*
             * Compute (X,Y,Z) position of WG in the grid when workgroup clusters are used.
             * If workgroup clusters are enabled:
             *   For each dimension (X, Y, Z):
             *     Extract cluster ID and number of workgroups from ttmp6
             *     Compute:
             *        m_workgroupIndex[dim] = clusterID_dim * numWgInCluster_dim + wgIdxInCluster_dim
             */
            if(ctx->targetArchitecture().HasCapability(GPUCapability::HasWorkgroupClusters))
            {
                auto notInCluster = ctx->labelAllocator()->label("NotInCluster");

                // Read ib_sts2 register
                // TODO fix this by adding special hwreg support, maybe a special s_getreg variant?
                auto temp = Register::Value::Placeholder(
                    ctx, Register::Type::Scalar, DataType::UInt32, 1);
                co_yield Instruction(
                    "s_getreg_b32", {temp}, {}, {"hwreg(HW_REG_IB_STS2, 6, 4)"}, "Read ib_sts2");
                // Check if clusters are enabled with a branch if zero
                co_yield ctx->brancher()->branchIfZero(
                    notInCluster,
                    temp,
                    "Not in a workgroup cluster, skip cluster-specific workgroupIndex "
                    "calculations");

                auto wgTemp = Register::Value::Placeholder(
                    ctx, Register::Type::Scalar, DataType::UInt32, 1);
                wgTemp->setName("wgTemp for wgidx calc");

                auto nwgTemp = Register::Value::Placeholder(
                    ctx, Register::Type::Scalar, DataType::UInt32, 1);
                nwgTemp->setName("nwgTemp for wgidx calc");

                // X dimension
                co_yield generateOp<Expression::BitwiseAnd>(
                    wgTemp, ctx->getTTMP6(), Register::Value::Literal(0xf));

                co_yield generateOp(nwgTemp,
                                    ctx->getTTMP6(),
                                    Expression::BitFieldExtract{
                                        {nullptr, "Extract 4 bit nwg_x"}, DataType::UInt32, 12, 4});

                co_yield generateOp<Expression::Add>(nwgTemp, nwgTemp, Register::Value::Literal(1));

                co_yield generateOp<Expression::MultiplyAdd>(
                    m_workgroupIndex[0], m_workgroupIndex[0], nwgTemp, wgTemp);

                // Y dimension
                if(m_kernelDimensions > 1)
                {
                    co_yield generateOp(
                        wgTemp,
                        ctx->getTTMP6(),
                        Expression::BitFieldExtract{
                            {nullptr, "Extract 4 bit wg_y"}, DataType::UInt32, 4, 4});

                    co_yield generateOp(
                        nwgTemp,
                        ctx->getTTMP6(),
                        Expression::BitFieldExtract{
                            {nullptr, "Extract 4 bit nwg_y"}, DataType::UInt32, 16, 4});

                    co_yield generateOp<Expression::Add>(
                        nwgTemp, nwgTemp, Register::Value::Literal(1));

                    co_yield generateOp<Expression::MultiplyAdd>(
                        m_workgroupIndex[1], m_workgroupIndex[1], nwgTemp, wgTemp);
                }

                // Z dimension
                if(m_kernelDimensions > 2)
                {
                    co_yield generateOp(
                        wgTemp,
                        ctx->getTTMP6(),
                        Expression::BitFieldExtract{
                            {nullptr, "Extract 4 bit wg_z"}, DataType::UInt32, 8, 4});

                    co_yield generateOp(
                        nwgTemp,
                        ctx->getTTMP6(),
                        Expression::BitFieldExtract{
                            {nullptr, "Extract 4 bit nwg_z"}, DataType::UInt32, 20, 4});

                    co_yield generateOp<Expression::Add>(
                        nwgTemp, nwgTemp, Register::Value::Literal(1));

                    co_yield generateOp<Expression::MultiplyAdd>(
                        m_workgroupIndex[2], m_workgroupIndex[2], nwgTemp, wgTemp);
                }
                co_yield_(Instruction::Label(notInCluster));
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

            for(auto& workitem : m_workitemIndex)
            {
                if(workitem)
                    workitem->setReadOnly();
            }
        }

        if(ctx->kernelOptions()->lazyLoadKernelArguments && m_argumentPointer)
        {
            m_argumentPointer->setReadOnly();
        }
        else
        {
            // We're done loading kernel arguments so we don't need the pointer anymore.
            m_argumentPointer.reset();
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
        if(m_context.lock()->argLoader()->anyManuallyLoadedArguments())
            co_yield Instruction::Directive("  .amdhsa_user_sgpr_kernarg_segment_ptr 1");

        if(m_numPreloadedRegs > 0)
        {
            co_yield Instruction::Directive(
                concatenate("  .amdhsa_user_sgpr_kernarg_preload_length ", m_numPreloadedRegs));
            co_yield Instruction::Directive(
                concatenate("  .amdhsa_user_sgpr_kernarg_preload_offset ", m_preloadedRegOffset));
        }

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

        auto match = [this, exp, simplified, restored, restoredSimplified](auto const& arg) {
            auto equivalentToAny = [exp, simplified, restored, restoredSimplified](
                                       Expression::ExpressionPtr const& anExpression) {
                return equivalent(anExpression, exp) || equivalent(anExpression, simplified)
                       || equivalent(anExpression, restored)
                       || equivalent(anExpression, restoredSimplified);
            };

            if(equivalentToAny(arg.getExpression()))
                return true;

            if(arg.getSimplifiedExpr() && equivalentToAny(arg.getSimplifiedExpr()))
                return true;

            if(arg.getRestoredExpr() && equivalentToAny(arg.getRestoredExpr()))
                return true;

            if(arg.getSimplifiedRestoredExpr() && equivalentToAny(arg.getSimplifiedRestoredExpr()))
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
        auto const  argName       = arg.getName();
        auto const& argExpression = arg.getExpression();

        AssertFatal(m_argumentNames.find(argName) == m_argumentNames.end(),
                    "Error: Two arguments with the same name: " + argName);

        if(argExpression)
            AssertFatal(resultVariableType(argExpression) == arg.getVariableType(),
                        ShowValue(resultVariableType(argExpression)),
                        ShowValue(arg.getVariableType()),
                        ShowValue(arg));

        if(argExpression && m_context.lock()->kernelOptions()->deduplicateArguments)
        {
            ptrdiff_t idx;
            auto      existingArg = findArgumentForExpression(argExpression, idx);
            if(existingArg)
            {
                m_argumentNames[argName] = idx;
                return existingArg;
            }
        }

        auto typeInfo = DataTypeInfo::Get(arg.getVariableType());
        if(isScaleType(typeInfo.variableType.dataType))
        {
            auto packedVarType = typeInfo.packedVariableType();
            AssertFatal(packedVarType, "Scale types must have a packed variable type.");
            typeInfo = DataTypeInfo::Get(packedVarType.value());
        }
        if(arg.getOffset() == -1)
        {
            arg.setOffset(RoundUpToMultiple<int>(m_argumentSize, typeInfo.alignment));
        }
        if(arg.getSize() == -1)
        {
            arg.setSize(CeilDivide(typeInfo.elementBits, 8u));
        }

        m_argumentSize           = std::max(m_argumentSize, arg.getOffset() + arg.getSize());
        m_argumentNames[argName] = m_arguments.size();
        m_arguments.push_back(std::move(arg));

        return Expression::fromKernelArgument(m_arguments.back());
    }

}
