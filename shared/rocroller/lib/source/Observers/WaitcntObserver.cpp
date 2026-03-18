// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/GPUArchitecture/GPUInstructionInfo.hpp>
#include <rocRoller/Scheduling/Observers/WaitcntObserver.hpp>

#include <rocRoller/KernelOptions_detail.hpp>
#include <rocRoller/Utilities/Settings.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        WaitcntState::WaitcntState() = default;

        WaitcntState::WaitcntState(WaitQueueMap<bool> const&             needsWaitZero,
                                   WaitQueueMap<GPUWaitQueueType> const& typeInQueue,
                                   WaitCntQueues const& instruction_queues_with_alloc)
            : m_needsWaitZero(needsWaitZero)
            , m_typeInQueue(typeInQueue)
        {
            // Here we're iterating through all of the Register::ValuePtrs and
            // converting them to RegisterIDs
            for(auto& queue : instruction_queues_with_alloc)
            {
                if(m_instructionQueues.find(queue.first) == m_instructionQueues.end())
                {
                    m_instructionQueues[queue.first] = {};
                }
                for(auto& dsts : queue.second)
                {
                    m_instructionQueues[queue.first].emplace_back(
                        std::vector<Register::RegisterId>{});
                    for(auto& dst : dsts)
                    {
                        if(dst)
                        {
                            for(auto& regid : dst->getRegisterIds())
                            {
                                m_instructionQueues[queue.first]
                                                   [m_instructionQueues[queue.first].size() - 1]
                                                       .emplace_back(regid);
                            }
                        }
                    }
                }
            }
        }

        void WaitcntState::assertSafeToBranchTo(const WaitcntState& labelState,
                                                std::string const&  label) const
        {
            if(*this == labelState)
                return;

            bool fail = false;

            // In Debug mode, defer throwing the exception until we have
            // captured a more complete error message.
            bool longErrMsg = Settings::Get(Settings::LogLvl) >= LogLevel::Debug;

            std::string msg
                = "Branching to label '" + label + "' with a different waitcnt state.\n";

            // If queues do not have needsWaitZero set, and none of the instructions
            // contain a destination, it is still safe to branch, even if the
            // queues do not match exactly.
            for(auto const& [queue, instructions] : m_instructionQueues)
            {
                if(m_needsWaitZero.at(queue) || labelState.m_needsWaitZero.at(queue))
                {
                    fail = true;
                    msg += concatenate(" Wait zero: ",
                                       ShowValue(m_needsWaitZero.at(queue)),
                                       ShowValue(labelState.m_needsWaitZero.at(queue)),
                                       ShowValue(queue),
                                       "\n");

                    if(!longErrMsg)
                        AssertFatal(!fail, msg);
                }

                for(auto const& instruction : instructions)
                {
                    if(!instruction.empty())
                    {
                        fail = true;
                        msg += concatenate(" Extra register at label: ",
                                           ShowValue(instruction),
                                           ShowValue(queue),
                                           "\n");

                        if(!longErrMsg)
                            AssertFatal(!fail, msg);
                    }
                }

                for(auto const& instruction : labelState.m_instructionQueues.at(queue))
                {
                    if(!instruction.empty())
                    {
                        fail = true;
                        msg += concatenate(" Extra register at branch: ",
                                           ShowValue(instruction),
                                           ShowValue(queue),
                                           "\n");

                        if(!longErrMsg)
                            AssertFatal(!fail, msg);
                    }
                }
            }
            AssertFatal(!fail, msg);
        }

        WaitcntObserver::WaitcntObserver() = default;

        WaitcntObserver::WaitcntObserver(ContextPtr context)
            : m_context(context)
        {
            m_includeExplanation
                = Settings::getInstance()->get(Settings::LogLvl) >= LogLevel::Verbose;
            m_displayState = Settings::getInstance()->get(Settings::LogLvl) >= LogLevel::Debug;

            {
                auto const& architecture = context->targetArchitecture();

                auto hasBarrier       = architecture.HasCapability(GPUCapability::s_barrier);
                auto hasBarrierSignal = architecture.HasCapability(GPUCapability::s_barrier_signal);

                AssertFatal(hasBarrier || hasBarrierSignal,
                            "Either s_barrier or s_barrier_signal must be supported.",
                            ShowValue(architecture.target()));

                m_barrierOpcode = hasBarrierSignal ? "s_barrier_signal" : "s_barrier";
            }

            for(int i = 0; i < static_cast<int>(GPUWaitQueue::Count); i++)
            {
                GPUWaitQueue waitQueue         = static_cast<GPUWaitQueue>(i);
                m_instructionQueues[waitQueue] = {};
                m_needsWaitZero[waitQueue]     = false;
                m_typeInQueue[waitQueue]       = GPUWaitQueueType::None;
            }

            AssertFatal(m_instructionQueues.size() == static_cast<uint8_t>(GPUWaitQueue::Count),
                        ShowValue(m_instructionQueues.size()),
                        ShowValue(static_cast<uint32_t>(GPUWaitQueue::Count)));
            AssertFatal(m_needsWaitZero.size() == static_cast<uint8_t>(GPUWaitQueue::Count),
                        ShowValue(m_needsWaitZero.size()),
                        ShowValue(static_cast<uint32_t>(GPUWaitQueue::Count)));
            AssertFatal(m_typeInQueue.size() == static_cast<uint8_t>(GPUWaitQueue::Count),
                        ShowValue(m_typeInQueue.size()),
                        ShowValue(static_cast<uint32_t>(GPUWaitQueue::Count)));
        };

        void WaitcntObserver::observe(Instruction const& inst)
        {
            auto               context      = m_context.lock();
            auto const&        architecture = context->targetArchitecture();
            GPUInstructionInfo info         = architecture.GetInstructionInfo(inst.getOpCode());

            if(context->kernelOptions()->assertWaitCntState)
            {
                if(info.isBranch())
                {
                    AssertFatal(inst.getSrcs()[0],
                                "Branch without a label\n",
                                ShowValue(inst.toString(LogLevel::Debug)));
                    addBranchState(inst.getSrcs()[0]->toString());
                }
                else if(inst.isLabel())
                {
                    addLabelState(inst.getLabel());
                }
            }

            auto instWaitQueues = info.getWaitQueues();

            WaitCount waiting = inst.getWaitCount();

            if(std::find(
                   instWaitQueues.begin(), instWaitQueues.end(), GPUWaitQueueType::FinalInstruction)
               != instWaitQueues.end())
            {
                waiting = WaitCount::Zero(context->targetArchitecture());

                if(context->kernelOptions()->assertWaitCntState)
                {
                    assertLabelConsistency();
                }
            }

            for(int i = 0; i < static_cast<int>(GPUWaitQueue::Count); i++)
            {
                applyWaitToQueue(waiting.getCount(static_cast<GPUWaitQueue>(i)),
                                 static_cast<GPUWaitQueue>(i));
            }

            for(GPUWaitQueueType queueType : instWaitQueues)
            {
                GPUWaitQueue waitQueue = fromWaitQueueType(queueType);
                if(queueType != GPUWaitQueueType::None
                   && m_instructionQueues.find(waitQueue) != m_instructionQueues.end())
                {
                    int instWaitCnt = info.getWaitCount();
                    if(instWaitCnt >= 0)
                    {
                        if(instWaitCnt == 0)
                        {
                            m_needsWaitZero[waitQueue] = true;
                            instWaitCnt                = 1;
                        }
                        else if(m_typeInQueue[waitQueue] != GPUWaitQueueType::None
                                && m_typeInQueue[waitQueue] != queueType)
                        {
                            m_needsWaitZero[waitQueue] = true;
                        }
                        for(int i = 0; i < instWaitCnt; i++)
                        {
                            WaitQueueRegisters queueRegisters;
                            append(queueRegisters, inst.getAllDsts());
                            // track LDS access to avoid write-after-read races.
                            auto isLDSReg = [](Register::ValuePtr const reg) -> bool {
                                return reg->regType() == Register::Type::LocalData;
                            };
                            append(queueRegisters, filter(isLDSReg, inst.getAllSrcs()));

                            m_instructionQueues[waitQueue].push_back(std::move(queueRegisters));
                        }
                        m_typeInQueue[waitQueue] = queueType;
                    }
                }
            }
        }

        std::string WaitcntObserver::getWaitQueueState() const
        {
            std::stringstream retval;
            for(int i = 0; i < static_cast<int>(GPUWaitQueue::Count); i++)
            {
                GPUWaitQueue waitQueue = static_cast<GPUWaitQueue>(i);

                // Only include state information for wait queues in a non-default state.
                if(m_needsWaitZero.at(waitQueue)
                   || m_typeInQueue.at(waitQueue) != GPUWaitQueueType::None
                   || m_instructionQueues.at(waitQueue).size() > 0)
                {
                    if(retval.rdbuf()->in_avail() == 0)
                    {
                        retval << "\nWait Queue State:";
                    }
                    retval << "\n--Queue: " << waitQueue;
                    retval << "\n----Needs Wait Zero: "
                           << (m_needsWaitZero.at(waitQueue) ? "True" : "False");
                    retval << "\n----Type In Queue  : " << m_typeInQueue.at(waitQueue);
                    retval << "\n----Registers      : ";

                    for(int queue_i = 0; queue_i < m_instructionQueues.at(waitQueue).size();
                        queue_i++)
                    {
                        retval << "\n------Dst: {";
                        for(auto& reg : m_instructionQueues.at(waitQueue)[queue_i])
                        {
                            if(reg)
                            {
                                retval << reg->toString() << ", ";
                            }
                        }
                        retval << "}";
                    }
                }
            }
            return retval.str();
        }

        WaitCount WaitcntObserver::computeZeroBarrierWaitCount(Instruction const& inst,
                                                               std::string*       explanation) const
        {
            auto context = m_context.lock();

            WaitCount rv;

            if(context->kernelOptions()->alwaysWaitZeroBeforeBarrier)
            {
                const auto& architecture = context->targetArchitecture();
                if(inst.getOpCode() == m_barrierOpcode)
                {
                    if(explanation != nullptr)
                    {
                        *explanation += "WaitCnt Needed: alwaysWaitZeroBeforeBarrier is set.\n";
                    }
                    rv.combine(WaitCount::Zero(architecture));
                }
            }
            return rv;
        }

        WaitCount WaitcntObserver::computeSyncQueueWaitCount(Instruction const& inst,
                                                             std::string*       explanation) const
        {
            auto context = m_context.lock();

            WaitCount rv;

            auto queuesToSync = inst.getWaitCount().queuesToSync();

            if(queuesToSync.any())
            {
                const auto& architecture = context->targetArchitecture();

                for(int i = 0; i < static_cast<int>(GPUWaitQueueType::Count); i++)
                {
                    GPUWaitQueueType queueType = static_cast<GPUWaitQueueType>(i);
                    GPUWaitQueue     queue     = fromWaitQueueType(queueType);

                    if(queuesToSync[queueType])
                    {
                        if(!m_instructionQueues.at(queue).empty()
                           && (m_typeInQueue.at(queue) == queueType || m_needsWaitZero.at(queue)))
                        {
                            rv.combine(WaitCount(architecture, queue, 0));

                            if(explanation)
                            {
                                *explanation += fmt::format("Wait for Queue {} {}: empty: {}, "
                                                            "needsWaitZero: {}, typeInQueue: {}",
                                                            toString(queueType),
                                                            queuesToSync[queueType],
                                                            m_instructionQueues.at(queue).empty(),
                                                            m_needsWaitZero.at(queue),
                                                            toString(m_typeInQueue.at(queue)));
                            }
                        }
                    }
                }
            }

            return rv;
        }

        WaitCount WaitcntObserver::computeRegisterWaitCount(Instruction const& inst,
                                                            std::string*       explanation) const
        {
            auto        context      = m_context.lock();
            const auto& architecture = context->targetArchitecture();

            WaitCount retval;

            // No wait required before LDS reads as the wait happens before LDS barriers
            if(GPUInstructionInfo::isLDSRead(inst.getOpCode()))
                return retval;

            if(inst.getOpCode().size() > 0 && inst.hasRegisters())
            {
                for(int i = 0; i < static_cast<int>(GPUWaitQueue::Count); i++)
                {
                    GPUWaitQueue waitQueue = static_cast<GPUWaitQueue>(i);
                    for(int queue_i = m_instructionQueues.at(waitQueue).size() - 1; queue_i >= 0;
                        queue_i--)
                    {
                        if(inst.isAfterWriteDependency(m_instructionQueues.at(waitQueue)[queue_i]))
                        {
                            if(m_needsWaitZero.at(waitQueue))
                            {
                                retval.combine(WaitCount(architecture, waitQueue, 0));
                                if(explanation != nullptr)
                                {
                                    *explanation += "WaitCnt Needed: Intersects with registers in '"
                                                    + toString(waitQueue)
                                                    + "', which needs a wait zero.\n";
                                }
                            }
                            else
                            {
                                int waitval
                                    = m_instructionQueues.at(waitQueue).size() - (queue_i + 1);
                                retval.combine(WaitCount(architecture, waitQueue, waitval));
                                if(explanation != nullptr)
                                {
                                    *explanation += "WaitCnt Needed: Intersects with registers in '"
                                                    + toString(waitQueue) + "', at "
                                                    + std::to_string(queue_i)
                                                    + " and the queue size is "
                                                    + std::to_string(
                                                        m_instructionQueues.at(waitQueue).size())
                                                    + ", so a waitcnt of " + std::to_string(waitval)
                                                    + " is required.\n";
                                }
                            }
                            break;
                        }
                    }
                }
            }

            return retval;
        }

        WaitCount WaitcntObserver::computeWaitCount(Instruction const& inst,
                                                    std::string*       explanation) const
        {
            auto        context      = m_context.lock();
            const auto& architecture = context->targetArchitecture();

            WaitCount retval = computeZeroBarrierWaitCount(inst, explanation);
            retval.combine(computeSyncQueueWaitCount(inst, explanation));
            retval.combine(computeRegisterWaitCount(inst, explanation));

            return retval.getAsSaturatedWaitCount(architecture);
        }
    }
}
