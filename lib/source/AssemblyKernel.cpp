

#include <rocRoller/AssemblyKernel.hpp>

#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/ExpressionTransformations.hpp>

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

        co_yield Instruction::Comment(ctx->kernelOptions().toString());

        if(ctx->kernelOptions().preloadKernelArguments)
            co_yield ctx->argLoader()->loadAllArguments();

        if(m_packedWorkitemIndex)
        {
            auto tenBits = Register::Value::Literal(0x3FF);
            auto ten     = Register::Value::Literal(10);
            auto twenty  = Register::Value::Literal(20);

            co_yield_(Instruction("v_and_b32",
                                  {m_workitemIndex[0]},
                                  {tenBits, m_packedWorkitemIndex},
                                  {},
                                  "Mask off 10 bit X coordinate"));

            if(m_kernelDimensions > 1)
            {
                co_yield_(Instruction("v_lshrrev_b32",
                                      {m_workitemIndex[1]},
                                      {ten, m_packedWorkitemIndex},
                                      {},
                                      "Shift 10 bit Y Coordinate"));
                co_yield_(Instruction("v_and_b32",
                                      {m_workitemIndex[1]},
                                      {tenBits, m_workitemIndex[1]},
                                      {},
                                      "Mask off 10 bit Y coordinate"));
            }

            if(m_kernelDimensions > 2)
            {
                co_yield_(Instruction("v_lshrrev_b32",
                                      {m_workitemIndex[2]},
                                      {twenty, m_packedWorkitemIndex},
                                      {},
                                      "Shift 10 bit Z Coordinate"));
                co_yield_(Instruction("v_and_b32",
                                      {m_workitemIndex[2]},
                                      {tenBits, m_workitemIndex[2]},
                                      {},
                                      "Mask off 10 bit Z coordinate"));
            }

            // No more need for packed value.
            m_packedWorkitemIndex.reset();
            if(ctx->kernelOptions().preloadKernelArguments)
                m_argumentPointer.reset();
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
        int nextFreeVGPR = kernelOpts.setNextFreeVGPRToMax ? kernelOpts.maxVGPRs : total_vgprs();

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
        idx             = -1;
        auto simplified = simplify(exp);

        auto match = [exp, simplified](auto const& arg) {
            if(equivalent(arg.expression, exp) || equivalent(arg.expression, simplified))
                return true;

            auto simpleArg = simplify(arg.expression);
            if(equivalent(simpleArg, exp) || equivalent(simpleArg, simplified))
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

        auto const& typeInfo = DataTypeInfo::Get(arg.variableType);

        if(arg.expression && m_context.lock()->kernelOptions().deduplicateArguments)
        {
            ptrdiff_t idx;
            auto      existingArg = findArgumentForExpression(arg.expression, idx);
            if(existingArg)
            {
                m_argumentNames[arg.name] = idx;
                return existingArg;
            }
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
