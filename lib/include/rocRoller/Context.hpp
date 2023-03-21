
#pragma once

#include <fstream>
#include <iostream>
#include <memory>
#include <ranges>
#include <string>
#include <unordered_map>
#include <vector>

#include "Context_fwd.hpp"

#include "AssemblyKernel_fwd.hpp"
#include "CodeGen/ArgumentLoader_fwd.hpp"
#include "CodeGen/BranchGenerator_fwd.hpp"
#include "CodeGen/CopyGenerator_fwd.hpp"
#include "CodeGen/Instruction_fwd.hpp"
#include "CodeGen/MemoryInstructions_fwd.hpp"
#include "GPUArchitecture/GPUArchitectureTarget_fwd.hpp"
#include "InstructionValues/LDSAllocator_fwd.hpp"
#include "InstructionValues/LabelAllocator_fwd.hpp"
#include "InstructionValues/RegisterAllocator_fwd.hpp"
#include "InstructionValues/Register_fwd.hpp"
#include "KernelGraph/RegisterTagManager_fwd.hpp"
#include "KernelGraph/ScopeManager_fwd.hpp"
#include "ScheduledInstructions_fwd.hpp"
#include "Scheduling/Scheduling_fwd.hpp"
#include "Utilities/Random_fwd.hpp"

#include "DataTypes/DataTypes.hpp"
#include "KernelOptions.hpp"

#include "GPUArchitecture/GPUArchitecture.hpp"
#include "Scheduling/WaitStateHazardCounter.hpp"

namespace rocRoller
{
    using RegisterHazardMap = std::unordered_map<Register::RegisterId,
                                                 std::vector<Scheduling::WaitStateHazardCounter>,
                                                 Register::RegisterIdHash>;

    class Context : public std::enable_shared_from_this<Context>
    {
    public:
        Context();
        ~Context();

        static ContextPtr ForDefaultHipDevice(std::string const&   kernelName,
                                              KernelOptions const& kernelOpts = {});

        static ContextPtr ForHipDevice(int                  deviceIdx,
                                       std::string const&   kernelName,
                                       KernelOptions const& kernelOpts = {});

        static ContextPtr ForTarget(GPUArchitectureTarget const& arch,
                                    std::string const&           kernelName,
                                    KernelOptions const&         kernelOpts = {});

        static ContextPtr ForTarget(GPUArchitecture const& arch,
                                    std::string const&     kernelName,
                                    KernelOptions const&   kernelOpts = {});

        Scheduling::InstructionStatus peek(Instruction const& inst);

        void schedule(Instruction& inst);
        template <std::ranges::input_range T>
        void schedule(T const& insts);
        template <std::ranges::input_range T>
        void schedule(T&& insts);

        std::shared_ptr<Register::Allocator> allocator(Register::Type registerType);

        Register::ValuePtr getVCC();
        Register::ValuePtr getVCC_LO();
        Register::ValuePtr getVCC_HI();
        Register::ValuePtr getSCC();
        Register::ValuePtr getExec();

        std::shared_ptr<Scheduling::IObserver> observer() const;

        std::shared_ptr<AssemblyKernel>        kernel() const;
        std::shared_ptr<ArgumentLoader>        argLoader() const;
        std::shared_ptr<ScheduledInstructions> instructions() const;
        std::shared_ptr<MemoryInstructions>    mem() const;
        std::shared_ptr<CopyGenerator>         copier() const;
        std::shared_ptr<BranchGenerator>       brancher() const;
        KernelOptions&                         kernelOptions();

        std::shared_ptr<RandomGenerator> random() const;
        void                             setRandomSeed(int seed);

        std::string assemblyFileName() const;

        std::shared_ptr<LabelAllocator> labelAllocator() const;
        std::shared_ptr<LDSAllocator>   ldsAllocator() const;
        RegTagManPtr                    registerTagManager() const;
        GPUArchitecture const&          targetArchitecture() const;
        int                             hipDeviceIndex() const;

        std::shared_ptr<RegisterHazardMap> getRegisterHazardMap() const;

        /**
         * @brief Get register scope manager.
         */
        std::shared_ptr<KernelGraph::ScopeManager> getScopeManager() const;

        /**
         * @brief Set register scope manager.
         */
        void setScopeManager(std::shared_ptr<KernelGraph::ScopeManager>);

    private:
        static ContextPtr Create(int                    deviceIndex,
                                 GPUArchitecture const& arch,
                                 std::string const&     kernelName,
                                 KernelOptions const&   kernelOpts);

        // If we are generating code for a real Hip device, refers to its
        // device index.
        int             m_hipDeviceIdx = -1;
        GPUArchitecture m_targetArch;
        RegTagManPtr    m_registerTagMan;
        std::array<std::shared_ptr<Register::Allocator>, static_cast<size_t>(Register::Type::Count)>
            m_allocators;

        std::shared_ptr<Scheduling::IObserver>     m_observer;
        std::shared_ptr<AssemblyKernel>            m_kernel;
        std::shared_ptr<ArgumentLoader>            m_argLoader;
        std::shared_ptr<ScheduledInstructions>     m_instructions;
        std::shared_ptr<MemoryInstructions>        m_mem;
        std::shared_ptr<LabelAllocator>            m_labelAllocator;
        std::shared_ptr<LDSAllocator>              m_ldsAllocator;
        std::shared_ptr<CopyGenerator>             m_copier;
        std::shared_ptr<BranchGenerator>           m_brancher;
        std::shared_ptr<RandomGenerator>           m_random;
        std::shared_ptr<KernelGraph::ScopeManager> m_scope;

        std::string   m_assemblyFileName;
        KernelOptions m_kernelOptions;

        // Represents registers and their associated wait state hazards
        std::shared_ptr<RegisterHazardMap> m_regMap;
    };

    std::ostream& operator<<(std::ostream&, ContextPtr const&);
}

#include "Context_impl.hpp"
