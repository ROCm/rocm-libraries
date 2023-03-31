
#include <rocRoller/Scheduling/Observers/WaitcntObserver.hpp>

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
            // Here we're iterating through all of the std::shared_ptr<Register::Value>s and
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

        bool WaitcntState::safeToBranchTo(const WaitcntState& labelState) const
        {
            if(*this == labelState)
                return true;

            // If queues do not have needsWaitZero set, and none of the instructions
            // contain a destination, it is still safe to branch, even if the
            // queues do not match exactly.
            for(auto const& [queue, instructions] : m_instructionQueues)
            {
                if(m_needsWaitZero.at(queue) || labelState.m_needsWaitZero.at(queue))
                    return false;

                for(auto const& instruction : instructions)
                {
                    if(!instruction.empty())
                        return false;
                }

                for(auto const& instruction : labelState.m_instructionQueues.at(queue))
                {
                    if(!instruction.empty())
                        return false;
                }
            }

            return true;
        }

        WaitcntObserver::WaitcntObserver() = default;

        WaitcntObserver::WaitcntObserver(std::shared_ptr<Context> context)
            : m_context(context)
        {
            m_includeExplanation
                = Settings::getInstance()->get(Settings::LogLvl) >= LogLevel::Verbose;
            m_displayState = Settings::getInstance()->get(Settings::LogLvl) >= LogLevel::Debug;

            for(uint8_t i = 0; i < static_cast<uint8_t>(GPUWaitQueue::Count); i++)
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

            if(context->kernelOptions().assertWaitCntState)
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

                if(context->kernelOptions().assertWaitCntState)
                {
                    assertLabelConsistency();
                }
            }

            for(uint8_t i = 0; i < static_cast<uint8_t>(GPUWaitQueue::Count); i++)
            {
                applyWaitToQueue(waiting.getCount(static_cast<GPUWaitQueue>(i)),
                                 static_cast<GPUWaitQueue>(i));
            }

            for(GPUWaitQueueType queueType : instWaitQueues)
            {
                GPUWaitQueue waitQueue(queueType);
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
                            m_instructionQueues[waitQueue].push_back(inst.getDsts());
                        }
                        m_typeInQueue[waitQueue] = queueType;
                    }
                }
            }
        }

        std::string WaitcntObserver::getWaitQueueState() const
        {
            std::stringstream retval;
            for(uint8_t i = 0; i < static_cast<uint8_t>(GPUWaitQueue::Count); i++)
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
                    retval << "\n--Queue: " << waitQueue.ToString();
                    retval << "\n----Needs Wait Zero: "
                           << (m_needsWaitZero.at(waitQueue) ? "True" : "False");
                    retval << "\n----Type In Queue  : " << m_typeInQueue.at(waitQueue).ToString();
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

    }
}
