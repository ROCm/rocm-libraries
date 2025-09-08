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

/**
 */

#pragma once

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/Utilities/Utils.hpp>

#include <rocRoller/Utilities/Logging.hpp>

namespace rocRoller
{

    inline AssemblyKernel::AssemblyKernel(ContextPtr context, std::string const& kernelName)
        : m_context(context)
    {
        AssertFatal(context);
        m_wavefrontSize
            = context->targetArchitecture().GetCapability(GPUCapability::DefaultWavefrontSize);

        setKernelName(kernelName);
    }

    inline AssemblyKernel::AssemblyKernel() noexcept                          = default;
    inline AssemblyKernel::AssemblyKernel(AssemblyKernel const& rhs) noexcept = default;
    inline AssemblyKernel::AssemblyKernel(AssemblyKernel&& rhs) noexcept      = default;

    inline AssemblyKernel::~AssemblyKernel() = default;

    inline ContextPtr AssemblyKernel::context() const
    {
        return m_context.lock();
    }

    inline CommandPtr AssemblyKernel::command() const
    {
        return m_command;
    }

    inline AssemblyKernel& AssemblyKernel::operator=(AssemblyKernel const& rhs) = default;
    inline AssemblyKernel& AssemblyKernel::operator=(AssemblyKernel&& rhs) = default;

    inline std::string AssemblyKernel::kernelName() const
    {
        return m_kernelName;
    }

    inline void AssemblyKernel::setKernelName(std::string const& name)
    {
        m_kernelName = name;

        m_kernelStartLabel = Register::Value::Label(name);
        m_kernelEndLabel   = Register::Value::Label(concatenate(".L", kernelName(), "_end"));
    }

    inline int AssemblyKernel::kernelDimensions() const
    {
        return m_kernelDimensions;
    }

    inline void AssemblyKernel::setKernelDimensions(int dims)
    {
        m_kernelDimensions = dims;
    }

    inline void AssemblyKernel::setWavefrontSize(int wavefrontSize)
    {
        m_wavefrontSize = wavefrontSize;
    }

    inline Register::ValuePtr AssemblyKernel::argumentPointer() const
    {
        return m_argumentPointer;
    }

    inline void AssemblyKernel::clearIndexRegisters()
    {
        Log::debug("Clearing index registers!");
        m_argumentPointer.reset();
        m_packedWorkitemIndex.reset();
        for(auto& r : m_workgroupIndex)
            r.reset();
        for(auto& r : m_workitemIndex)
            r.reset();
    }

    inline bool AssemblyKernel::startedCodeGeneration() const
    {
        return m_startedCodeGeneration;
    }

    inline void AssemblyKernel::startCodeGeneration()
    {
        m_startedCodeGeneration = true;
    }

    inline Register::ValuePtr AssemblyKernel::kernelStartLabel() const
    {
        return m_kernelStartLabel;
    }

    inline Register::ValuePtr AssemblyKernel::kernelEndLabel() const
    {
        return m_kernelEndLabel;
    }

    inline Generator<Instruction> AssemblyKernel::amdgpu_metadata()
    {
        std::string rv = "\n.amdgpu_metadata\n";
        rv += amdgpu_metadata_yaml();
        rv += "\n.end_amdgpu_metadata\n";
        co_yield Instruction::Directive(rv);
    }

    inline int AssemblyKernel::kernarg_segment_size() const
    {
        if(m_arguments.empty())
            return 0;

        auto const& lastArg = m_arguments.back();
        return lastArg.offset + lastArg.size;
    }

    inline int AssemblyKernel::group_segment_fixed_size() const
    {
        return m_context.lock()->ldsAllocator()->maxUsed();
    }

    inline int AssemblyKernel::private_segment_fixed_size() const
    {
        return 0;
    }

    inline int AssemblyKernel::kernarg_segment_align() const
    {
        size_t rv = 8;
        for(auto const& arg : m_arguments)
            rv = std::max(rv, DataTypeInfo::Get(arg.variableType).alignment);

        return rv;
    }

    inline int AssemblyKernel::wavefront_size() const
    {
        return m_wavefrontSize;
    }

    inline int AssemblyKernel::sgpr_count() const
    {
        return m_context.lock()->allocator(Register::Type::Scalar)->useCount();
    }

    inline int AssemblyKernel::vgpr_count() const
    {
        return m_context.lock()->allocator(Register::Type::Vector)->useCount();
    }

    inline int AssemblyKernel::agpr_count() const
    {
        return m_context.lock()->allocator(Register::Type::Accumulator)->useCount();
    }

    inline int AssemblyKernel::accum_offset() const
    {
        return RoundUpToMultiple(vgpr_count(), 4);
    }

    inline int AssemblyKernel::total_vgprs() const
    {
        if(m_context.lock()->targetArchitecture().HasCapability(
               rocRoller::GPUCapability::ArchAccUnifiedRegs))
            return accum_offset() + agpr_count();
        return std::max(vgpr_count(), agpr_count());
    }

    inline int AssemblyKernel::max_flat_workgroup_size() const
    {
        return m_workgroupSize[0] * m_workgroupSize[1] * m_workgroupSize[2];
    }

    inline std::vector<AssemblyKernelArgument> const& AssemblyKernel::arguments() const
    {
        return m_arguments;
    }

    inline size_t AssemblyKernel::argumentSize() const
    {
        return m_argumentSize;
    }

    inline bool AssemblyKernel::hasArgument(std::string const& name) const
    {
        return m_argumentNames.contains(name);
    }

    inline std::vector<AssemblyKernelArgument> AssemblyKernel::resetArguments()
    {
        m_argumentNames.clear();
        m_argumentSize = 0;
        return std::exchange(m_arguments, {});
    }

    inline void AssemblyKernel::addCommandArguments(std::vector<CommandArgumentPtr> args)
    {
        for(auto arg : args)
        {
            addCommandArgument(arg);
        }
    }

    inline Expression::ExpressionPtr AssemblyKernel::addCommandArgument(CommandArgumentPtr arg)
    {
        return addArgument({arg->name(),
                            arg->variableType(),
                            arg->direction(),
                            std::make_shared<Expression::Expression>(arg)});
    }

    inline AssemblyKernelArgument const& AssemblyKernel::findArgument(std::string const& name) const
    {
        auto iter = m_argumentNames.find(name);
        AssertFatal(iter != m_argumentNames.end(), "Could not find argument with name " + name);

        return m_arguments.at(iter->second);
    }

    inline std::array<unsigned int, 3> const& AssemblyKernel::workgroupSize() const
    {
        return m_workgroupSize;
    }

    inline Expression::ExpressionPtr AssemblyKernel::workgroupCount(size_t index)
    {
        if(m_workgroupCount.at(index) == nullptr)
        {
            // TODO: Workgroup count should be computed at launch time:
            // Expression::ExpressionPtr exPtr
            //     = m_workitemCount[index] / Expression::literal(m_workgroupSize[index]);
            auto expr = Expression::literal(0u);

            std::string argName = concatenate("LAUNCH_WORKGROUPCOUNT_", index);
            auto        resType = Expression::resultType(expr);
            m_workgroupCount[index]
                = addArgument({argName, resType.varType, DataDirection::ReadOnly, expr});
        }
        return m_workgroupCount[index];
    }

    inline std::array<Expression::ExpressionPtr, 3> const& AssemblyKernel::workitemCount() const
    {
        return m_workitemCount;
    }

    inline Expression::ExpressionPtr const& AssemblyKernel::dynamicSharedMemBytes() const
    {
        return m_dynamicSharedMemBytes;
    }

    inline void AssemblyKernel::setWorkgroupSize(std::array<unsigned int, 3> const& val)
    {
        for(auto const& v : val)
        {
            AssertFatal(v != 0);
        }
        m_workgroupSize = val;
    }

    inline void
        AssemblyKernel::setWorkitemCount(std::array<Expression::ExpressionPtr, 3> const& val)
    {
        m_workitemCount = val;
    }

    inline void AssemblyKernel::setDynamicSharedMemBytes(Expression::ExpressionPtr const& val)
    {
        m_dynamicSharedMemBytes = val;
    }

    inline std::array<Register::ValuePtr, 3> const& AssemblyKernel::workgroupIndex() const
    {
        return m_workgroupIndex;
    }

    inline std::array<Register::ValuePtr, 3> const& AssemblyKernel::workitemIndex() const
    {
        return m_workitemIndex;
    }

    inline void AssemblyKernel::setKernelGraphMeta(KernelGraph::KernelGraphPtr graph)
    {
        m_kernelGraph = graph;
    }

    inline void AssemblyKernel::setCommandMeta(CommandPtr command)
    {
        m_command = command;
    }

    inline std::shared_ptr<KernelGraph::KernelGraph> AssemblyKernel::kernel_graph() const
    {
        return m_kernelGraph;
    }
}
