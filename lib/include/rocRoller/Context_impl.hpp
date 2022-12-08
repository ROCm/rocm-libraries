
#pragma once

#include "Context.hpp"

#include "CodeGen/Instruction.hpp"
#include "GPUArchitecture/GPUArchitecture.hpp"
#include "GPUArchitecture/GPUArchitectureLibrary.hpp"
#include "ScheduledInstructions.hpp"
#include "Scheduling/Scheduling.hpp"

namespace rocRoller
{
    inline Scheduling::InstructionStatus Context::peek(Instruction const& inst)
    {
        return m_observer->peek(inst);
    }

    inline void Context::schedule(Instruction& inst)
    {
        m_observer->modify(inst);
        m_observer->observe(inst);

        m_instructions->schedule(inst);
    }

    template <std::ranges::input_range T>
    inline void Context::schedule(T const& insts)
    {
        for(Instruction const& inst : insts)
        {
            Instruction copy = inst;
            schedule(copy);
        }
    }

    template <std::ranges::input_range T>
    inline void Context::schedule(T&& insts)
    {
        for(Instruction inst : insts)
        {
            schedule(inst);
        }
    }

    /**
     * @brief Get the Register Hazard Map.
     *
     * This tracks the state of registers that may have associated wait state hazards
     * caused by previous instructions.
     *
     * @return std::shared_ptr<RegisterHazardMap>
     */
    inline std::shared_ptr<RegisterHazardMap> Context::getRegisterHazardMap() const
    {
        return m_regMap;
    }

    inline std::shared_ptr<Register::Allocator> Context::allocator(Register::Type registerType)
    {
        return m_allocators.at(static_cast<int>(registerType));
    }

    inline std::shared_ptr<LabelAllocator> Context::labelAllocator() const
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

    inline std::shared_ptr<AssemblyKernel> Context::kernel() const
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

    inline std::shared_ptr<RandomGenerator> Context::random() const
    {
        return m_random;
    }

    inline KernelOptions& Context::kernelOptions()
    {
        return m_kernelOptions;
    }

    inline void Context::setKernelOptions(KernelOptions input)
    {
        m_kernelOptions = input;
    }

    inline std::string Context::assemblyFileName() const
    {
        return m_assemblyFileName;
    }

    inline std::shared_ptr<RegisterTagManager> Context::registerTagManager() const
    {
        return m_registerTagMan;
    }
}
