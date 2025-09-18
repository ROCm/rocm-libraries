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

#include <rocRoller/CodeGen/ArgumentLoader.hpp>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/InstructionValues/Register.hpp>
#include <rocRoller/Utilities/Settings.hpp>
#include <rocRoller/Utilities/Utils.hpp>

namespace rocRoller
{
    ArgumentLoader::ArgumentLoader(AssemblyKernelPtr kernel)
        : m_kernel(kernel)
        , m_context(kernel->context())
    {
    }

    Register::ValuePtr ArgumentLoader::argumentPointer() const
    {
        return m_kernel->argumentPointer();
    }

    Generator<Instruction>
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
            value = Register::Value::Placeholder(m_context.lock(),
                                                 Register::Type::Scalar,
                                                 DataType::Raw32,
                                                 totalRegisters,
                                                 Register::AllocationOptions::FullyContiguous());
        }

        // This is still needed even with deferred `subset()` since we generate
        // different instructions depending on the alignment of the registers.
        // The initial allocation of the arguments is also not something that
        // can be deferred in order to alleviate register pressure.
        if(value->allocationState() != Register::AllocationState::Allocated)
            value->allocateNow();

        int widthBytes;

        int regIdx = 0;

        auto regIndices = value->registerIndices().to<std::vector>();
        auto regIter    = regIndices.begin();

        auto argPtr = argumentPointer();
        AssertFatal(argPtr != nullptr, ShowValue(offset), ShowValue(endOffset));

        while((widthBytes = PickInstructionWidthBytes(offset, endOffset, regIter, regIndices.end()))
              > 0)
        {
            auto widthDwords = widthBytes / 4;

            auto regRange  = Generated(iota(regIdx, regIdx + widthDwords));
            auto regSubset = value->subset(regRange);

            co_yield m_context.lock()->mem()->loadScalar(regSubset, argPtr, offset, widthBytes);

            offset += widthBytes;
            regIter += widthDwords;
            regIdx += widthDwords;
        }
    }

    Generator<Instruction> ArgumentLoader::loadAllArguments()
    {
        auto const& args    = m_kernel->arguments();
        std::string comment = "Loading Kernel Arguments: \n";
        for(auto const& arg : args)
            comment += arg.toString() + "\n";
        co_yield Instruction::Comment(comment);

        if(args.empty())
        {
            co_yield Instruction::Comment("No kernel arguments");
            co_return;
        }

        auto argPtr = argumentPointer();
        // TODO: coalesce loads when possible
        for(auto const& arg : args)
        {
            auto numRegisters = std::max(1u, arg.size / (Register::bitsPerRegister / 8));
            auto r            = Register::Value::Placeholder(m_context.lock(),
                                                  Register::Type::Scalar,
                                                  DataType::Raw32,
                                                  numRegisters,
                                                  Register::AllocationOptions::FullyContiguous());
            r->allocateNow();
            r->setName(arg.name);
            r->setVariableType(arg.variableType);
            m_loadedValues[arg.name] = r;
            co_yield m_context.lock()->mem()->loadScalar(r, argPtr, arg.offset, arg.size);
        }
    }

    Generator<Instruction> ArgumentLoader::loadArgument(std::string const& argName)
    {
        auto const& arg = m_kernel->findArgument(argName);

        co_yield loadArgument(arg);
    }

    Generator<Instruction> ArgumentLoader::loadArgument(AssemblyKernelArgument const& arg)
    {
        Register::ValuePtr value;
        co_yield Instruction::Comment(concatenate("Loading arg ", arg.name));
        co_yield loadRange(arg.offset, arg.offset + arg.size, value);

        AssertFatal(value);
        value->setName(arg.name);

        value->setVariableType(arg.variableType);

        m_loadedValues[arg.name] = value;
    }

    void ArgumentLoader::releaseArgument(std::string const& argName)
    {
        m_loadedValues.erase(argName);
    }

    void ArgumentLoader::releaseAllArguments()
    {
        m_loadedValues.clear();
    }

    Generator<Instruction> ArgumentLoader::getValue(std::string const&  argName,
                                                    Register::ValuePtr& value)
    {
        auto realName = m_kernel->findArgument(argName).name;

        if(Settings::Get(Settings::AuditControlTracers))
        {
            auto inst = Instruction::Comment("Get arg " + realName);
            inst.setReferencedArg(realName);
            co_yield inst;
        }

        auto iter = m_loadedValues.find(realName);
        if(iter == m_loadedValues.end())
        {
            for(auto const& pair : m_loadedValues)
            {
                Log::debug("Loaded {}", pair.first);
            }
            Log::debug("Loading {} ({})", realName, argName);
            co_yield loadArgument(realName);
            iter = m_loadedValues.find(realName);
        }

        if(value == nullptr)
        {
            value = iter->second;
        }
        else
        {
            co_yield m_context.lock()->copier()->copy(value, iter->second, "ArgLoader");
        }
    }
}
