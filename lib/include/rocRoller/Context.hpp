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

#include <fstream>
#include <iostream>
#include <memory>
#include <ranges>
#include <string>
#include <unordered_map>
#include <vector>

#include <rocRoller/AssemblyKernel_fwd.hpp>
#include <rocRoller/CodeGen/ArgumentLoader_fwd.hpp>
#include <rocRoller/CodeGen/BranchGenerator_fwd.hpp>
#include <rocRoller/CodeGen/CopyGenerator_fwd.hpp>
#include <rocRoller/CodeGen/CrashKernelGenerator_fwd.hpp>
#include <rocRoller/CodeGen/Instruction_fwd.hpp>
#include <rocRoller/CodeGen/MemoryInstructions_fwd.hpp>
#include <rocRoller/Context_fwd.hpp>
#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/Expression_fwd.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitecture.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitectureTarget_fwd.hpp>
#include <rocRoller/InstructionValues/LDSAllocator_fwd.hpp>
#include <rocRoller/InstructionValues/LabelAllocator_fwd.hpp>
#include <rocRoller/InstructionValues/RegisterAllocator_fwd.hpp>
#include <rocRoller/InstructionValues/Register_fwd.hpp>
#include <rocRoller/KernelGraph/RegisterTagManager_fwd.hpp>
#include <rocRoller/KernelGraph/ScopeManager_fwd.hpp>
#include <rocRoller/KernelOptions.hpp>
#include <rocRoller/ScheduledInstructions_fwd.hpp>
#include <rocRoller/Scheduling/Scheduling_fwd.hpp>
#include <rocRoller/Utilities/Random_fwd.hpp>

class ContextFixture;

namespace rocRoller
{
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

        Register::ValuePtr getM0();
        Register::ValuePtr getVCC();
        Register::ValuePtr getVCC_LO();
        Register::ValuePtr getVCC_HI();
        Register::ValuePtr getSCC();
        Register::ValuePtr getExec();

        std::shared_ptr<Scheduling::IObserver> observer() const;

        AssemblyKernelPtr                      kernel() const;
        std::shared_ptr<ArgumentLoader>        argLoader() const;
        std::shared_ptr<ScheduledInstructions> instructions() const;
        std::shared_ptr<MemoryInstructions>    mem() const;
        std::shared_ptr<CopyGenerator>         copier() const;
        std::shared_ptr<BranchGenerator>       brancher() const;
        std::shared_ptr<CrashKernelGenerator>  crasher() const;
        KernelOptions const&                   kernelOptions();

        std::shared_ptr<RandomGenerator> random() const;
        void                             setRandomSeed(int seed);

        std::string assemblyFileName() const;

        LabelAllocatorPtr             labelAllocator() const;
        std::shared_ptr<LDSAllocator> ldsAllocator() const;
        RegTagManPtr                  registerTagManager() const;
        GPUArchitecture const&        targetArchitecture() const;
        int                           hipDeviceIndex() const;

        void setKernel(AssemblyKernelPtr);

        /**
         * @brief Returns an expression representing how much scratch space is required (in bytes)
         *
         * @return Expression::ExpressionPtr
         */
        Expression::ExpressionPtr getScratchAmount() const;

        /**
         * @brief Allocate more scratch space
         *
         * @param size Number of bytes requested
         */
        void allocateScratch(Expression::ExpressionPtr size);

        /**
         * @brief Get register scope manager.
         */
        std::shared_ptr<KernelGraph::ScopeManager> getScopeManager() const;

        /**
         * @brief Set register scope manager.
         */
        void setScopeManager(std::shared_ptr<KernelGraph::ScopeManager>);

        friend class ::ContextFixture;

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
        AssemblyKernelPtr                          m_kernel;
        std::shared_ptr<ArgumentLoader>            m_argLoader;
        std::shared_ptr<ScheduledInstructions>     m_instructions;
        std::shared_ptr<MemoryInstructions>        m_mem;
        LabelAllocatorPtr                          m_labelAllocator;
        std::shared_ptr<LDSAllocator>              m_ldsAllocator;
        Expression::ExpressionPtr                  m_scratchAllocator;
        std::shared_ptr<CopyGenerator>             m_copier;
        std::shared_ptr<BranchGenerator>           m_brancher;
        std::shared_ptr<CrashKernelGenerator>      m_crasher;
        std::shared_ptr<RandomGenerator>           m_random;
        std::shared_ptr<KernelGraph::ScopeManager> m_scope;

        std::string   m_assemblyFileName;
        KernelOptions m_kernelOptions;
    };

    std::ostream& operator<<(std::ostream&, ContextPtr const&);
}

#include <rocRoller/Context_impl.hpp>
