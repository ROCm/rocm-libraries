/**
 */

#pragma once

#include "ArgumentLoader.hpp"

#include "../AssemblyKernel.hpp"
#include "../CodeGen/MemoryInstructions.hpp"
#include "../InstructionValues/Register.hpp"
#include "../Utilities/Utils.hpp"

namespace rocRoller
{
    inline ArgumentLoader::ArgumentLoader(std::shared_ptr<AssemblyKernel> kernel)
        : m_kernel(kernel)
        , m_context(kernel->context())
    {
    }

    inline Register::ValuePtr ArgumentLoader::argumentPointer() const
    {
        return m_kernel->argumentPointer();
    }

    template <typename Iter, typename End>
    inline int PickInstructionWidthBytes(int offset, int endOffset, Iter beginReg, End endReg)
    {
        if(offset >= endOffset)
        {
            return 0;
        }

        AssertFatal(beginReg < endReg, "Inconsistent register bounds");
        AssertFatal(offset % 4 == 0, "Must be 4-byte aligned.");

        static const std::vector<int> widths{16, 8, 4, 2, 1};

        for(auto width : widths)
        {
            auto widthBytes = width * 4;

            if((offset % widthBytes == 0) && (offset + widthBytes) <= endOffset
               && (*beginReg % width == 0) && IsContiguousRange(beginReg, beginReg + width))
            {
                return widthBytes;
            }
        }

        Throw<FatalError>("Should be impossible to get here!");
    }

    inline Generator<Instruction>
        ArgumentLoader::loadRange(int offset, int endOffset, Register::ValuePtr& value) const
    {
        AssertFatal(offset >= 0 && endOffset >= 0, "Negative offset");
        AssertFatal(offset < endOffset, "Attempt to load 0 bytes.");

        int sizeBytes      = endOffset - offset;
        int totalRegisters = CeilDivide(sizeBytes, 4);

        co_yield Instruction::Comment(
            concatenate("Kernel arguments: Loading ", sizeBytes, " bytes"));

        if(value == nullptr || value->registerCount() < totalRegisters)
        {
            value = Register::Value::Placeholder(
                m_context.lock(), Register::Type::Scalar, DataType::Raw32, totalRegisters);
        }

        if(value->allocationState() != Register::AllocationState::Allocated)
            value->allocateNow();

        int widthBytes;

        int regIdx = 0;

        auto regIndices = Generated(value->registerIndices());
        auto regIter    = regIndices.begin();

        auto argPtr = argumentPointer();

        while((widthBytes = PickInstructionWidthBytes(offset, endOffset, regIter, regIndices.end()))
              > 0)
        {
            auto widthDwords = widthBytes / 4;

            auto regRange  = Generated(iota(regIdx, regIdx + widthDwords));
            auto regSubset = value->subset(regRange);

            auto offsetLiteral = Register::Value::Literal(offset);

            auto comment = concatenate("Load ", offset, "..", endOffset);

            co_yield m_context.lock()->mem()->loadScalar(
                regSubset, argPtr, offsetLiteral, widthBytes);

            offset += widthBytes;
            regIter += widthDwords;
            regIdx += widthDwords;
        }
    }

    inline Generator<Instruction> ArgumentLoader::loadAllArguments()
    {
        if(m_kernel->arguments().empty())
        {
            co_yield_(Instruction::Comment("No kernel arguments"));
            co_return;
        }

        int beginOffset = std::numeric_limits<int>::max();
        int endOffset   = 0;

        for(auto const& arg : m_kernel->arguments())
        {
            beginOffset = std::min<int>(beginOffset, arg.offset);
            endOffset   = std::max<int>(endOffset, arg.offset + arg.size);
        }

        Register::ValuePtr allArgs;

        co_yield loadRange(beginOffset, endOffset, allArgs);

        for(auto const& arg : m_kernel->arguments())
        {
            // TODO Fix argument alignment
            // Note use of element size instead of arg size.
            // Previously the argument size may have been aligned,
            // which may have resulted in arg.size > getElementSize.
            auto beginReg = arg.offset / 4;
            auto endReg   = (arg.offset + static_cast<int>(arg.variableType.getElementSize())) / 4;
            auto regRange = Generated(iota(beginReg, endReg));

            auto subReg = allArgs->subset(regRange);
            subReg->setName(arg.name);
            subReg->setVariableType(arg.variableType);
            m_loadedValues[arg.name] = subReg;
        }
    }

    inline Generator<Instruction> ArgumentLoader::loadArgument(std::string const& argName)
    {
        auto const& arg = m_kernel->findArgument(argName);

        co_yield loadArgument(arg);
    }

    inline Generator<Instruction> ArgumentLoader::loadArgument(AssemblyKernelArgument const& arg)
    {
        Register::ValuePtr value;
        co_yield Instruction::Comment(concatenate("Loading arg ", arg.name));
        co_yield loadRange(arg.offset, arg.offset + arg.size, value);

        AssertFatal(value);
        value->setName(arg.name);

        value->setVariableType(arg.variableType);

        m_loadedValues[arg.name] = value;
    }

    inline void ArgumentLoader::releaseArgument(std::string const& argName)
    {
        m_loadedValues.erase(argName);
    }

    inline void ArgumentLoader::releaseAllArguments()
    {
        m_loadedValues.clear();
    }

    inline Generator<Instruction> ArgumentLoader::getValue(std::string const&  argName,
                                                           Register::ValuePtr& value)
    {
        auto iter = m_loadedValues.find(argName);
        if(iter == m_loadedValues.end())
        {
            co_yield loadArgument(argName);
            iter = m_loadedValues.find(argName);
        }

        value = iter->second;
    }
}
