// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <concepts>
#include <string>
#include <vector>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitecture.hpp>
#include <rocRoller/GPUArchitecture/GPUInstructionInfo.hpp>
#include <rocRoller/KernelOptions_detail.hpp>
#include <rocRoller/Scheduling/LDSModel.hpp>
#include <rocRoller/Scheduling/Observers/FunctionalUnit/MEMObserver.hpp>
#include <rocRoller/Utilities/Utils.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        VMEMObserver::VMEMObserver(ContextPtr ctx)
            : MEMObserver(ctx,
                          "VMEM",
                          MEMObserver::getWeights(ctx).vmemCycles,
                          MEMObserver::getWeights(ctx).vmemQueueSize)
        {
        }

        bool VMEMObserver::isMEMInstruction(Instruction const& inst) const
        {
            return GPUInstructionInfo::isVMEM(inst.getOpCode());
        }

        int VMEMObserver::getWait(Instruction const& inst) const
        {
            return inst.getWaitCount().vmcnt();
        }

        DSMEMObserver::DSMEMObserver(ContextPtr ctx)
            : MEMObserver(ctx,
                          "DSMEM",
                          MEMObserver::getWeights(ctx).dsmemCycles,
                          MEMObserver::getWeights(ctx).dsmemQueueSize)
        {
        }

        bool DSMEMObserver::runtimeRequired(ContextPtr const& ctx)
        {
            return ctx->kernelOptions()->dsObserver == DSObserverType::DSMEMObserver;
        }

        bool DSMEMObserver::isMEMInstruction(Instruction const& inst) const
        {
            return GPUInstructionInfo::isLDS(inst.getOpCode());
        }

        int DSMEMObserver::getWait(Instruction const& inst) const
        {
            return inst.getWaitCount().dscnt();
        }

        WeightlessDSMemObserver::WeightlessDSMemObserver(ContextPtr ctx)
            : m_context(ctx)
        {
        }

        bool WeightlessDSMemObserver::runtimeRequired(ContextPtr const& ctx)
        {
            return ctx->kernelOptions()->dsObserver == DSObserverType::WeightlessDSMemObserver;
        }

        InstructionStatus WeightlessDSMemObserver::peek(Instruction const& inst) const
        {
            if(!m_scheduler.has_value())
            {
                // Observers get created before workgroupSize is set in m_context
                auto context = m_context.lock();
                AssertFatal(context != nullptr);

                const auto& gpuArch = context->targetArchitecture().target();
                m_scheduler.emplace(gpuArch.gfx, product(context->kernel()->workgroupSize()) / 64);
            }

            InstructionStatus status;

            if(GPUInstructionInfo::isLDS(inst.getOpCode()))
            {
                const auto ldsInfo = LDSModel::getLdsInfoFromOpcodeIfSupported(inst.getOpCode());
                if(ldsInfo.has_value())
                {
                    const auto [direction, dwords] = *ldsInfo;

                    auto ctx = m_context.lock();
                    AssertFatal(ctx != nullptr);

                    if(inst.getModelledAddresses().has_value())
                    {
                        std::vector<size_t> addresses = inst.getModelledAddresses().value();
                        auto [stallCycles, additionalCycles]
                            = m_scheduler.value().predictCycles({{direction}, dwords, addresses});

                        status.stallCycles      = stallCycles / 4;
                        status.additionalCycles = additionalCycles / 4;
                    }
                    else
                    {
                        Log::warn("Missing modelled addresses for {}",
                                  inst.toString(LogLevel::Terse));
                    }
                }
            }

            const auto waitcnt = inst.getWaitCount().dscnt();
            if(waitcnt >= 0)
            {
                auto ctx = m_context.lock();
                AssertFatal(ctx != nullptr);
                auto stallCycles   = m_scheduler.value().predictWaitcntStallCycles(waitcnt);
                status.stallCycles = stallCycles / 4;
            }

            return status;
        }

        void WeightlessDSMemObserver::modify(Instruction& inst) const
        {
            if(GPUInstructionInfo::isLDS(inst.getOpCode()))
            {
                const auto status = peek(inst);
                inst.addComment(fmt::format("WeightlessDSMemObserver {}: stall {}, additional {}",
                                            m_scheduler.value().getProgramCycle(),
                                            status.stallCycles,
                                            status.additionalCycles));
            }
        }

        void WeightlessDSMemObserver::observe(Instruction const& inst)
        {
            m_scheduler.value().incrementProgramCycleBy(inst.totalCycles() * 4);
            m_scheduler.value().updateQueues();

            if(GPUInstructionInfo::isLDS(inst.getOpCode()))
            {
                auto ldsInfo = LDSModel::getLdsInfoFromOpcodeIfSupported(inst.getOpCode());
                if(ldsInfo.has_value() && inst.getModelledAddresses().has_value())
                {
                    auto [direction, dwords] = *ldsInfo;

                    std::vector<size_t> addresses = inst.getModelledAddresses().value();
                    m_scheduler.value().scheduleInstruction({{direction}, dwords, addresses});
                }
            }
        }
    }
}
