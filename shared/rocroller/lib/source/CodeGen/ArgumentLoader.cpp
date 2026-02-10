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
// #include <rocRoller/GPUArchitecture/GPUArchitectureLibrary.hpp>
#include <rocRoller/InstructionValues/Register.hpp>
#include <rocRoller/Utilities/Settings.hpp>
#include <rocRoller/Utilities/Utils.hpp>

// #define debug critical
// #define Debug Critical

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

    Generator<Instruction> ArgumentLoader::allocatePreloadedRegisters(int& offset, int& count)
    {
        offset   = 0;
        count    = 0;
        auto ctx = m_context.lock();

        for(auto const& arg : m_kernel->arguments())
        {
            if(arg.preloaded)
                count += std::max(1, arg.size / 4);
            else
            {
                m_manuallyLoadedOffset = arg.offset;
                break;
            }
        }

        if(count > 0)
        {
            m_preloadedBlock
                = Register::Value::Placeholder(ctx,
                                               Register::Type::Scalar,
                                               DataType::Raw32,
                                               count,
                                               Register::AllocationOptions::FullyContiguous());

            m_preloadedBlock->setName("Preloaded argument block");

            co_yield m_preloadedBlock->allocate();
        }
    }

    Generator<Instruction> ArgumentLoader::splitOutArgumentRegisters()
    {

        if(anyPreloadedArguments())
        {
            Log::debug("Splitting out preloaded args:");
            co_yield splitOutArgs(m_preloadedBlock, 0);
            m_preloadedBlock.reset();
        }

        if(anyManuallyLoadedArguments())
        {
            Log::debug("Splitting out manually loaded args:");
            co_yield splitOutArgs(m_manuallyLoadedBlock, m_manuallyLoadedOffset);
            m_manuallyLoadedBlock.reset();
        }
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
            value->setName(
                fmt::format("Manually loaded argument block for range {}-{}", offset, endOffset));
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

    void ArgumentLoader::decidePreloadedKernargs(std::vector<AssemblyKernelArgument>& args)
    {
        auto ctx = m_context.lock();

        auto archPreloadedSGPRs
            = ctx->targetArchitecture().GetCapability(GPUCapability::MaxPreloadedKernargs);
        auto optionPreloadedSGPRs = ctx->kernelOptions()->systemPreloadedKernelArguments;

        auto getArgSGPRs
            = [](AssemblyKernelArgument const& arg) { return std::max(arg.size / 4, 1); };

        auto totalArgSGPRs = [&]() {
            auto view = args | std::views::transform(getArgSGPRs);
            return std::reduce(view.begin(), view.end());
        }();

        // Start with arch defined amount
        auto maxPreloadedSGPRs = archPreloadedSGPRs;

        // If we need to manually load any args, we need 2 SGPRs for the pointer.
        if(totalArgSGPRs > maxPreloadedSGPRs)
            maxPreloadedSGPRs -= 2;

        // Apply the kernel option limit if it was specified.
        if(optionPreloadedSGPRs > 0)
            maxPreloadedSGPRs = std::min(maxPreloadedSGPRs, optionPreloadedSGPRs);

        int totalAssignedSGPRs = 0;

        // Take the first args first, but if one doesn't fit, try to find a later one.
        for(auto& arg : args)
        {
            auto argSGPRs = getArgSGPRs(arg);

            if(totalAssignedSGPRs + argSGPRs <= maxPreloadedSGPRs)
            {
                arg.preloaded = true;
                totalAssignedSGPRs += argSGPRs;
            }
            else
            {
                arg.preloaded = false;
            }
        }

        auto isPreloaded = [](AssemblyKernelArgument const& arg) { return arg.preloaded; };

        auto firstNonPreloaded = std::stable_partition(args.begin(), args.end(), isPreloaded);

        auto largestArgFirst
            = [&totalArgSGPRs](AssemblyKernelArgument const& a, AssemblyKernelArgument const& b) {
                  return a.size > b.size;
              };

        std::stable_sort(args.begin(), firstNonPreloaded, largestArgFirst);
        std::stable_sort(firstNonPreloaded, args.end(), largestArgFirst);

        for(auto const& arg : args)
        {
            Log::debug("Argument: {} ({}) ({})", arg.name, arg.size, arg.preloaded);
        }

        int  offset              = 0;
        bool startedNonPreloaded = false;

        for(auto& arg : args)
        {
            if(!arg.preloaded && !startedNonPreloaded)
            {
                startedNonPreloaded = true;
                // AssertFatal(offset / 4 == (firstNonPreloaded - args.begin()),
                //             "Offset mismatch",
                //             ShowValue(offset),
                //             ShowValue(firstNonPreloaded - args.begin()));
                arg.offset = RoundUpToMultiple(offset, arg.size);
            }
            else
            {
                AssertFatal(offset % arg.size == 0,
                            "No bubbles allowed in either segment!",
                            ShowValue(offset),
                            ShowValue(arg.size),
                            ShowValue(startedNonPreloaded)
                        );
                arg.offset = offset;
            }

            Log::debug("Arg: {} size {} offset {}", arg.name, arg.size, arg.offset);

            offset = arg.offset + arg.size;
        }
    }

    Generator<Instruction> ArgumentLoader::splitOutArgs(Register::ValuePtr rawRegs, int beginOffset)
    {
        AssertFatal(rawRegs->variableType() == DataType::Raw32, ShowValue(rawRegs->variableType()));
        AssertFatal(rawRegs->allocationState() == Register::AllocationState::Allocated,
                    ShowValue(rawRegs->allocationState()));
        auto const& args = m_kernel->arguments();

        int endOffset = beginOffset + (rawRegs->registerCount() * 4);

        std::vector<std::vector<int>> indices;
        // indices.reserve(args.size());
        std::vector<int> argIndices;

        for(int i = 0; i < args.size(); i++)
        {
            auto const& arg    = args[i];
            auto        argEnd = arg.offset + arg.size;
            if(arg.offset >= beginOffset && argEnd <= endOffset)
            {
                AssertFatal(!m_loadedValues.contains(arg.name),
                            ShowValue(arg),
                            ShowValue(beginOffset),
                            ShowValue(endOffset));

                auto beginReg = (arg.offset - beginOffset) / 4;
                auto endReg   = beginReg + (arg.size / 4);
                auto range    = iota(beginReg, endReg);

                indices.emplace_back(range.begin(), range.end());
                argIndices.push_back(i);
            }
        }

        auto launchTimeOnly = m_kernel->launchTimeOnlyArguments();

        if(Log::getLogger()->should_log(LogLevel::Debug))
        {
            Log::debug("Splitting regs:");
            for(int i = 0; i < indices.size(); i++)
            {
                auto        argIdx = argIndices[i];
                std::string ltOnly
                    = launchTimeOnly.contains(args.at(argIdx).name) ? " (LT only)" : "";
                Log::debug("{} ({}){}: {}",
                           argIdx,
                           args.at(argIdx).name,
                           ltOnly,
                           fmt::join(indices[i], ", "));
            }
        }

        auto valueRegs = rawRegs->split(indices);

        for(int valIdx = 0; valIdx < valueRegs.size(); valIdx++)
        {
            int         argIdx = argIndices[valIdx];
            auto const& arg    = args.at(argIdx);

            auto subReg = valueRegs[valIdx];
            subReg->setName(arg.name);
            subReg->setVariableType(arg.variableType);
            subReg->allocation()->setOptions(Register::AllocationOptions::FullyContiguous());

            co_yield Instruction::Comment(subReg->description());

            // if(!launchTimeOnly.contains(arg.name))
            m_loadedValues[arg.name] = subReg;
        }
    }

    bool ArgumentLoader::anyPreloadedArguments() const
    {
        if(!m_anyPreloadedArguments.has_value())
            populateAnyArgumentsFlags();
        return m_anyPreloadedArguments.value();
    }

    bool ArgumentLoader::anyManuallyLoadedArguments() const
    {
        if(!m_anyManuallyLoadedArguments.has_value())
            populateAnyArgumentsFlags();

        return m_anyManuallyLoadedArguments.value();
    }

    void ArgumentLoader::populateAnyArgumentsFlags() const
    {
        m_anyPreloadedArguments      = false;
        m_anyManuallyLoadedArguments = false;

        for(auto const& arg : m_kernel->arguments())
        {
            if(arg.preloaded)
                m_anyPreloadedArguments = true;
            else
                m_anyManuallyLoadedArguments = true;
        }
    }

#if 1
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

        int beginOffset = std::numeric_limits<int>::max();
        int endOffset   = 0;

        for(auto const& arg : args)
        {
            if(!m_loadedValues.contains(arg.name) && !arg.preloaded)
            {
                beginOffset = std::min<int>(beginOffset, arg.offset);
                endOffset   = std::max<int>(endOffset, arg.offset + arg.size);
            }
        }

        // beginOffset = std::max(beginOffset, m_manuallyLoadedOffset);

        // Register::ValuePtr allArgs;

        if(beginOffset >= endOffset)
            co_return;

        co_yield loadRange(beginOffset, endOffset, m_manuallyLoadedBlock);

        // Log::critical("Splitting out manually loaded args:");
        // co_yield splitOutArgs(allArgs, beginOffset);
    }

#else
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

        auto        argPtr         = argumentPointer();
        auto const& launchTimeOnly = m_kernel->launchTimeOnlyArguments();

        // TODO: coalesce loads when possible
        for(auto const& arg : args)
        {
            // Skip loading arguments that are only used at launch time
            // (for expression evaluation) and not during kernel execution
            if(launchTimeOnly.contains(arg.name))
            {
                Log::debug("Skipping load of launch-time-only arg {}", arg.name);
                co_yield Instruction::Comment(
                    concatenate("Skipping load of launch-time-only arg ", arg.name));
                continue;
            }

            Log::debug("Loading argument {}", arg.name);
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
#endif

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
