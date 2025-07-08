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

                    AssertFatal(wqType < rv.waitLengths.size(),
                                ShowValue(static_cast<size_t>(wqType)),
                                ShowValue(rv.waitLengths.size()),
                                ShowValue(pair.second.size()));

                    rv.waitLengths.at(wqType) = pair.second.size();
                }
            }

            // Apply the waitcount from this instruction.
            for(int i = 0; i < GPUWaitQueueType::Count; i++)
            {
                auto wqType = GPUWaitQueueType(i);
                auto wq     = GPUWaitQueue(wqType);

                auto count = rv.waitCount.getCount(wq);

                if(count >= 0)
                    rv.waitLengths.at(i) = std::min(rv.waitLengths.at(i), count);
            }

            // Add contribution from this instruction
            GPUInstructionInfo info
                = m_context.lock()->targetArchitecture().GetInstructionInfo(inst.getOpCode());
            auto whichQueues = info.getWaitQueues();
            for(auto q : whichQueues)
            {
                AssertFatal(q < rv.waitLengths.size(),
                            ShowValue(static_cast<size_t>(q)),
                            ShowValue(rv.waitLengths.size()));
                auto waitCount = info.getWaitCount();
                rv.waitLengths.at(q) += waitCount == 0 ? 1 : waitCount;
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

        inline bool WaitcntObserver::isDirect2LDS(Instruction const& inst)
        {
            if(inst.getOpCode().rfind("buffer_load_", 0) == 0)
            {
                for(auto const& mod : inst.getModifiers())
                {
                    if(mod.rfind("lds", 0) == 0)
                    {
                        return true;
                    }
                }
            }
            return false;
        }

        inline void WaitcntObserver::observeWaitDirect2LDS(Instruction const& inst)
        {
            if(isDirect2LDS(inst) && !m_needsWaitDirect2LDS)
                m_needsWaitDirect2LDS = true;
            if((inst.getOpCode().rfind("s_barrier", 0) == 0) && m_needsWaitDirect2LDS)
                m_needsWaitDirect2LDS = false;
        }
    };

}
