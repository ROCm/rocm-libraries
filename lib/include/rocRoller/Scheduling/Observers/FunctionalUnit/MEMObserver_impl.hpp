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

#include <rocRoller/Scheduling/Observers/FunctionalUnit/MEMObserver.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        template <typename Derived>
        inline Scheduling::Weights MEMObserver<Derived>::getWeights(ContextPtr ctx)
        {
            return dynamic_cast<Scheduling::LinearWeightedCost*>(
                       Component::Get<Scheduling::Cost>(Scheduling::CostFunction::LinearWeighted,
                                                        ctx)
                           .get())
                ->getWeights();
        }

        template <typename Derived>
        size_t MEMObserver<Derived>::queueLen() const
        {
            return m_completeButNotWaited.size() + m_incomplete.size();
        };

        template <typename Derived>
        MEMObserver<Derived>::Info MEMObserver<Derived>::queuePop()
        {
            AssertFatal(queueLen() > 0);
            if(!m_completeButNotWaited.empty())
            {
                auto rv = m_completeButNotWaited.front();
                m_completeButNotWaited.pop_front();
                return rv;
            }

            auto rv = m_incomplete.front();
            m_incomplete.pop_front();
            return rv;
        }

        template <typename Derived>
        void MEMObserver<Derived>::queueShift()
        {
            AssertFatal(!m_incomplete.empty());
            m_programCycle = std::max(m_programCycle, m_incomplete.front().expectedCompleteCycle);

            m_completeButNotWaited.push_back(std::move(m_incomplete.front()));
            m_incomplete.pop_front();
        }

        template <typename Derived>
        MEMObserver<Derived>::MEMObserver(ContextPtr         ctx,
                                          std::string const& commentTag,
                                          int                cyclesPerInst,
                                          int                queueAllotment)
            : m_context(ctx)
            , m_commentTag(commentTag)
            , m_cyclesPerInst(cyclesPerInst)
            , m_queueAllotment(queueAllotment)
        {
        }

        template <typename Derived>
        InstructionStatus MEMObserver<Derived>::peek(Instruction const& inst) const
        {
            InstructionStatus rv;

            if(isMEMInstruction(inst) && m_incomplete.size() >= m_queueAllotment)
            {
                auto complete  = m_incomplete.front().expectedCompleteCycle;
                rv.stallCycles = std::max(complete - m_programCycle, 0);
            }

            return rv;
        }

        template <typename Derived>
        void MEMObserver<Derived>::modify(Instruction& inst) const
        {
            auto status = peek(inst);
            if(status.stallCycles > 0)
                inst.addComment(fmt::format("{}: Expected stall of {}. CBNW: {}, Inc: {}",
                                            m_commentTag,
                                            status.stallCycles,
                                            m_completeButNotWaited.size(),
                                            m_incomplete.size()));
        }

        template <typename Derived>
        void MEMObserver<Derived>::observe(Instruction const& inst)
        {
            auto wait = getWait(inst);
            if(wait >= 0)
            {
                while(queueLen() > wait)
                {
                    auto info      = queuePop();
                    m_programCycle = std::max(m_programCycle, info.expectedCompleteCycle);
                }
            }

            int instCycles = inst.numExecutedInstructions();

            if(isMEMInstruction(inst))
            {
                while(m_incomplete.size() >= m_queueAllotment)
                    queueShift();

                m_programCycle += instCycles;

                m_incomplete.push_back({.issuedCycle           = m_programCycle,
                                        .expectedCompleteCycle = m_programCycle + m_cyclesPerInst});

                const_cast<Instruction&>(inst).addComment(
                    fmt::format("VMEM: Expected complete at {} (current {})",
                                m_incomplete.back().expectedCompleteCycle,
                                m_programCycle));
            }
            else
            {
                m_programCycle += instCycles + inst.peekedStatus().stallCycles;

                while(!m_incomplete.empty()
                      && m_programCycle >= m_incomplete.front().expectedCompleteCycle)
                    queueShift();
            }
        }
    }
}
