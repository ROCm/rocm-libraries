// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/Context.hpp>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/BranchGenerator.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/CodeGen/CrashKernelGenerator.hpp>
#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitecture.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitectureLibrary.hpp>
#include <rocRoller/GPUArchitecture/GPUCapability.hpp>
#include <rocRoller/InstructionValues/LabelAllocator.hpp>
#include <rocRoller/InstructionValues/Register.hpp>
#include <rocRoller/KernelGraph/RegisterTagManager.hpp>
#include <rocRoller/Scheduling/Observers/ObserverCreation.hpp>
#include <rocRoller/Utilities/Random.hpp>
#include <rocRoller/Utilities/Settings.hpp>

namespace rocRoller
{
    Context::Context()
    {
        // Initialize scratch sizes for each policy with zero
        for(size_t i = 0; i < static_cast<size_t>(Operations::ScratchPolicy::Count); ++i)
        {
            m_scratchSizes[i] = Expression::literal(0u);
        }
    }

    ContextPtr Context::ForDefaultHipDevice(std::string const&   kernelName,
                                            KernelOptions const& kernelOpts)
    {
        int         idx  = -1;
        auto const& arch = GPUArchitectureLibrary::getInstance()->GetDefaultHipDeviceArch(idx);

        return Create(idx, arch, kernelName, kernelOpts);
    }

    ContextPtr Context::ForHipDevice(int                  deviceIdx,
                                     std::string const&   kernelName,
                                     KernelOptions const& kernelOpts)
    {
        auto const& arch = GPUArchitectureLibrary::getInstance()->GetHipDeviceArch(deviceIdx);

        return Create(deviceIdx, arch, kernelName, kernelOpts);
    }

    ContextPtr Context::ForTarget(GPUArchitectureTarget const& target,
                                  std::string const&           kernelName,
                                  KernelOptions const&         kernelOpts)
    {
        auto const arch = GPUArchitectureLibrary::getInstance()->GetArch(target);
        return ForTarget(arch, kernelName, kernelOpts);
    }

    ContextPtr Context::ForTarget(GPUArchitecture const& arch,
                                  std::string const&     kernelName,
                                  KernelOptions const&   kernelOpts)
    {
        return Create(-1, arch, kernelName, kernelOpts);
    }

    void Context::setRandomSeed(int seed)
    {
        m_random = std::make_shared<RandomGenerator>(seed);
    }

    ContextPtr Context::Create(int                    deviceIdx,
                               GPUArchitecture const& arch,
                               std::string const&     kernelName,
                               KernelOptions const&   kernelOpts)
    {
        auto rv = std::make_shared<Context>();

        rv->m_kernelOptions = kernelOpts;

        rv->m_hipDeviceIdx = deviceIdx;
        rv->m_targetArch   = arch;

        for(size_t i = 0; i < rv->m_allocators.size(); i++)
        {
            auto regType = static_cast<Register::Type>(i);
            switch(regType)
            {
            case Register::Type::Accumulator:
            {
                auto const maxACCVGPRs = rv->m_targetArch.HasCapability(GPUCapability::HasAccCD)
                                             ? kernelOpts->maxACCVGPRs
                                             : 0;
                rv->m_allocators[i] = std::make_shared<Register::Allocator>(regType, maxACCVGPRs);
                break;
            }
            case Register::Type::Vector:
                rv->m_allocators[i]
                    = std::make_shared<Register::Allocator>(regType, kernelOpts->maxVGPRs);
                break;
            case Register::Type::Scalar:
                rv->m_allocators[i]
                    = std::make_shared<Register::Allocator>(regType, kernelOpts->maxSGPRs);
                break;
            default:
                // We do not allocate other types of register representation
                rv->m_allocators[i] = std::make_shared<Register::Allocator>(regType, 0);
                break;
            }
        }

        rv->m_kernel       = std::make_shared<AssemblyKernel>(rv, kernelName);
        rv->m_argLoader    = std::make_shared<ArgumentLoader>(rv->m_kernel);
        rv->m_instructions = std::make_shared<ScheduledInstructions>(rv);
        rv->m_mem          = std::make_shared<MemoryInstructions>(rv);
        rv->m_copier       = std::make_shared<CopyGenerator>(rv);
        rv->m_brancher     = std::make_shared<BranchGenerator>(rv);
        rv->m_crasher      = std::make_shared<CrashKernelGenerator>(rv);
        // Using a seed of zero by default ensures consistent results.
        // Set ROCROLLER_RANDOM_SEED to specify a different seed.
        rv->m_random = std::make_shared<RandomGenerator>(0);

        rv->m_labelAllocator = std::make_shared<LabelAllocator>(kernelName);

        rv->m_assemblyFileName = Settings::getInstance()->get(Settings::AssemblyFile);
        if(rv->m_assemblyFileName.empty())
        {
            rv->m_assemblyFileName = kernelName.size() ? kernelName : "a";
            rv->m_assemblyFileName += "_" + arch.target().toString();
            std::replace(rv->m_assemblyFileName.begin(), rv->m_assemblyFileName.end(), ':', '-');
            rv->m_assemblyFileName += ".s";
        }
        rv->m_ldsAllocator = std::make_shared<LDSAllocator>(
            rv->targetArchitecture().GetCapability(GPUCapability::MaxLdsSize));

        rv->m_observer = Scheduling::createObserver(rv);

        rv->m_registerTagMan = std::make_shared<RegisterTagManager>(rv);
        return rv;
    }

    Context::~Context() = default;

    Register::ValuePtr Context::getM0()
    {
        return std::make_shared<Register::Value>(
            shared_from_this(), Register::Type::M0, DataType::UInt32, 1);
    }

    Register::ValuePtr Context::getVCC()
    {
        auto wavefrontSize = kernel()->wavefront_size();
        if(wavefrontSize == 32)
        {
            return std::make_shared<Register::Value>(
                shared_from_this(), Register::Type::VCC_LO, DataType::Bool32, 1);
        }
        else if(wavefrontSize == 64)
        {
            return std::make_shared<Register::Value>(
                shared_from_this(), Register::Type::VCC, DataType::Bool64, 1);
        }
        else
        {
            Throw<FatalError>("getVCC() is only implemented for wave32 or wave64");
        }
    }

    Register::ValuePtr Context::getVCC_LO()
    {
        return std::make_shared<Register::Value>(
            shared_from_this(), Register::Type::VCC_LO, DataType::Bool32, 1);
    }

    Register::ValuePtr Context::getVCC_HI()
    {
        return std::make_shared<Register::Value>(
            shared_from_this(), Register::Type::VCC_HI, DataType::Bool32, 1);
    }

    Register::ValuePtr Context::getSCC()
    {
        return std::make_shared<Register::Value>(
            shared_from_this(), Register::Type::SCC, DataType::Bool, 1);
    }

    Register::ValuePtr Context::getEXECZ()
    {
        return std::make_shared<Register::Value>(
            shared_from_this(), Register::Type::EXECZ, DataType::Bool, 1);
    }

    Register::ValuePtr Context::getEXEC()
    {
        auto wavefrontSize = kernel()->wavefront_size();
        if(wavefrontSize == 32)
        {
            return std::make_shared<Register::Value>(
                shared_from_this(), Register::Type::EXEC_LO, DataType::Bool32, 1);
        }
        else if(wavefrontSize == 64)
        {
            return std::make_shared<Register::Value>(
                shared_from_this(), Register::Type::EXEC, DataType::Bool64, 1);
        }
        else
        {
            Throw<FatalError>("getEXEC() is only implemented for wave32 or wave64");
        }
    }

    Register::ValuePtr Context::getTTMP7()
    {
        return std::make_shared<Register::Value>(
            shared_from_this(), Register::Type::TTMP7, DataType::UInt32, 1);
    }

    Register::ValuePtr Context::getTTMP9()
    {
        return std::make_shared<Register::Value>(
            shared_from_this(), Register::Type::TTMP9, DataType::UInt32, 1);
    }

    Register::ValuePtr Context::getSpecial(Register::Type t)
    {
        using Register::Type;

        switch(t)
        {
        case Type::M0:
            return getM0();
        case Type::SCC:
            return getSCC();
        case Type::VCC:
            return getVCC();
        case Type::VCC_LO:
            return getVCC_LO();
        case Type::VCC_HI:
            return getVCC_HI();
        case Type::EXECZ:
            return getEXECZ();
        case Type::EXEC:
        case Type::EXEC_LO:
        case Type::EXEC_HI:
            return getEXEC();
        case Type::TTMP7:
            return getTTMP7();
        case Type::TTMP9:
            return getTTMP9();

        default:
            break;
        }

        Throw<FatalError>("Register must be Special");
    }

    std::ostream& operator<<(std::ostream& stream, ContextPtr const& ctx)
    {
        return stream << "Context " << ctx.get();
    }

    std::shared_ptr<KernelGraph::ScopeManager> Context::getScopeManager() const
    {
        return m_scope;
    }

    void Context::setScopeManager(std::shared_ptr<KernelGraph::ScopeManager> scope)
    {
        m_scope = scope;
    }

    void Context::setKernel(AssemblyKernelPtr assemblyKernel)
    {
        m_kernel = assemblyKernel;
    }

    Expression::ExpressionPtr Context::allocateScratch(Operations::ScratchPolicy policy,
                                                       Expression::ExpressionPtr size)
    {
        auto idx            = static_cast<size_t>(policy);
        auto currentOffset  = m_scratchSizes[idx];
        m_scratchSizes[idx] = simplify(m_scratchSizes[idx] + size);
        return currentOffset;
    }

    Expression::ExpressionPtr Context::getScratchAmount(Operations::ScratchPolicy policy) const
    {
        return m_scratchSizes[static_cast<size_t>(policy)];
    }

    void Context::scheduleCopy(Instruction const& inst)
    {
        auto copy = inst;
        schedule(copy);
    }

}
