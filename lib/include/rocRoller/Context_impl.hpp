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

#pragma once

#include <rocRoller/Context.hpp>

#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitecture.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitectureLibrary.hpp>
#include <rocRoller/ScheduledInstructions.hpp>
#include <rocRoller/Scheduling/Scheduling.hpp>

namespace rocRoller
{
    inline Scheduling::InstructionStatus Context::peek(Instruction const& inst)
    {
        return m_observer->peek(inst);
    }

    inline void Context::schedule(Instruction& inst)
    {
        auto status = m_observer->peek(inst);
        inst.setPeekedStatus(std::move(status));
        m_observer->modify(inst);
        m_observer->observe(inst);

        m_instructions->schedule(inst);
    }

    template <std::ranges::input_range T>
    inline void Context::schedule(T const& insts)
    {
        for(Instruction const& inst : insts)
        {
            scheduleCopy(inst);
        }
    }

    template <std::ranges::input_range T>
    inline void Context::schedule(T&& insts)
    {
        for(Instruction const& inst : insts)
        {
            scheduleCopy(inst);
        }
    }

    inline std::shared_ptr<Register::Allocator> Context::allocator(Register::Type registerType)
    {
        return m_allocators.at(static_cast<int>(registerType));
    }

    inline LabelAllocatorPtr Context::labelAllocator() const
    {
        return m_labelAllocator;
    }

    inline std::shared_ptr<LDSAllocator> Context::ldsAllocator() const
    {
        return m_ldsAllocator;
    }

    inline GPUArchitecture const& Context::targetArchitecture() const
    {
        return m_targetArch;
    }

    inline int Context::hipDeviceIndex() const
    {
        return m_hipDeviceIdx;
    }

    inline std::shared_ptr<Scheduling::IObserver> Context::observer() const
    {
        return m_observer;
    }

    inline AssemblyKernelPtr Context::kernel() const
    {
        return m_kernel;
    }

    inline std::shared_ptr<ArgumentLoader> Context::argLoader() const
    {
        return m_argLoader;
    }

    inline std::shared_ptr<ScheduledInstructions> Context::instructions() const
    {
        return m_instructions;
    }

    inline std::shared_ptr<MemoryInstructions> Context::mem() const
    {
        return m_mem;
    }

    inline std::shared_ptr<CopyGenerator> Context::copier() const
    {
        return m_copier;
    }

    inline std::shared_ptr<BranchGenerator> Context::brancher() const
    {
        return m_brancher;
    }

    inline std::shared_ptr<CrashKernelGenerator> Context::crasher() const
    {
        return m_crasher;
    }

    inline std::shared_ptr<RandomGenerator> Context::random() const
    {
        return m_random;
    }

    inline KernelOptions const& Context::kernelOptions()
    {
        return m_kernelOptions;
    }

    inline std::string Context::assemblyFileName() const
    {
        return m_assemblyFileName;
    }

    inline RegTagManPtr Context::registerTagManager() const
    {
        return m_registerTagMan;
    }
}
