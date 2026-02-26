// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <sstream>

#include <rocRoller/Context.hpp>
#include <rocRoller/GPUArchitecture/GPUInstructionInfo.hpp>
#include <rocRoller/Scheduling/Observers/WaitcntObserver.hpp>
#include <rocRoller/Scheduling/Scheduling.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        constexpr inline bool WaitcntObserver::required(GPUArchitectureTarget const& target)
        {
            return true;
        }

        inline InstructionStatus WaitcntObserver::peek(Instruction const& inst) const
        {
            auto rv = InstructionStatus::Wait(computeWaitCount(inst));

            // The new length of each queue is:
            // - The current length of the queue
            // - With the new WaitCount applied.
            // - Plus the contribution from this instruction

            // Get current length of queue
            for(auto const& pair : m_instructionQueues)
            {
                if(pair.second.size() > 0)
                {
                    auto wqType = m_typeInQueue.at(pair.first);
                    auto idx    = static_cast<size_t>(wqType);

                    AssertFatal(idx < rv.waitLengths.size(),
                                ShowValue(static_cast<size_t>(wqType)),
                                ShowValue(rv.waitLengths.size()),
                                ShowValue(pair.second.size()));

                    rv.waitLengths.at(idx) = pair.second.size();
                }
            }

            // Apply the waitcount from this instruction.
            for(int i = 0; i < static_cast<int>(GPUWaitQueueType::Count); i++)
            {
                auto wqType = static_cast<GPUWaitQueueType>(i);
                auto wq     = fromWaitQueueType(wqType);

                auto count = rv.waitCount.getCount(wq);

                if(count >= 0)
                    rv.waitLengths.at(wqType) = std::min(rv.waitLengths.at(wqType), count);
            }

            // Add contribution from this instruction
            GPUInstructionInfo info
                = m_context.lock()->targetArchitecture().GetInstructionInfo(inst.getOpCode());
            auto whichQueues = info.getWaitQueues();
            for(auto qt : whichQueues)
            {
                auto idx = static_cast<size_t>(qt);
                AssertFatal(
                    idx < rv.waitLengths.size(), ShowValue(qt), ShowValue(rv.waitLengths.size()));
                auto waitCount = info.getWaitCount();
                rv.waitLengths.at(qt) += waitCount == 0 ? 1 : waitCount;
            }

            return rv;
        };

        inline void WaitcntObserver::modify(Instruction& inst) const
        {
            auto        context      = m_context.lock();
            auto const& architecture = context->targetArchitecture();

            // Handle if manually specified waitcnts are over the sat limits.
            inst.addWaitCount(inst.getWaitCount().getAsSaturatedWaitCount(architecture));

            std::string  explanation;
            std::string* pExplanation = nullptr;
            if(m_includeExplanation)
                pExplanation = &explanation;

            inst.addWaitCount(computeWaitCount(inst, pExplanation));
            if(m_includeExplanation)
                inst.addComment(explanation);

            if(m_displayState && !inst.isCommentOnly())
            {
                inst.addComment(getWaitQueueState());
            }
        }

        inline void WaitcntObserver::applyWaitToQueue(int waitCnt, GPUWaitQueue queue)
        {
            if(waitCnt >= 0 && m_instructionQueues[queue].size() > (size_t)waitCnt)
            {
                if(!(m_needsWaitZero[queue]
                     && waitCnt
                            > 0)) //Do not partially clear the queue if a waitcnt zero is needed.
                {
                    m_instructionQueues[queue].erase(m_instructionQueues[queue].begin(),
                                                     m_instructionQueues[queue].begin()
                                                         + m_instructionQueues[queue].size()
                                                         - waitCnt);
                }
                if(m_instructionQueues[queue].size() == 0)
                {
                    m_needsWaitZero[queue] = false;
                    m_typeInQueue[queue]   = GPUWaitQueueType::None;
                }
            }
        }

        inline void WaitcntObserver::addLabelState(std::string const& label)
        {
            m_labelStates[label]
                = WaitcntState(m_needsWaitZero, m_typeInQueue, m_instructionQueues);
        }

        inline void WaitcntObserver::addBranchState(std::string const& label)
        {
            if(m_branchStates.find(label) == m_branchStates.end())
            {
                m_branchStates[label] = {};
            }

            m_branchStates[label].emplace_back(
                WaitcntState(m_needsWaitZero, m_typeInQueue, m_instructionQueues));
        }

        inline void WaitcntObserver::assertLabelConsistency()
        {
            for(auto label_state : m_labelStates)
            {
                if(m_branchStates.find(label_state.first) != m_branchStates.end())
                {
                    for(auto branch_state : m_branchStates[label_state.first])
                    {
                        label_state.second.assertSafeToBranchTo(branch_state, label_state.first);
                    }
                }
            }
        }

    };

}
