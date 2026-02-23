// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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
            const auto*       derived = static_cast<const Derived*>(this);
            if(derived->isMEMInstruction(inst) && m_incomplete.size() >= m_queueAllotment)
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
            const auto* derived = static_cast<const Derived*>(this);
            auto        wait    = derived->getWait(inst);
            if(wait >= 0)
            {
                while(queueLen() > wait)
                {
                    auto info      = queuePop();
                    m_programCycle = std::max(m_programCycle, info.expectedCompleteCycle);
                }
            }

            int instCycles = inst.numExecutedInstructions();

            if(derived->isMEMInstruction(inst))
            {
                while(m_incomplete.size() >= m_queueAllotment)
                    queueShift();

                m_programCycle += instCycles;

                m_incomplete.push_back({.issuedCycle           = m_programCycle,
                                        .expectedCompleteCycle = m_programCycle + m_cyclesPerInst});

                static_assert(CIsAnyOf<Derived, VMEMObserver, DSMEMObserver>,
                              "Update the comment below if adding new memory observer");

                if constexpr(std::is_same_v<Derived, VMEMObserver>)
                    const_cast<Instruction&>(inst).addComment(
                        fmt::format("VMEM: Expected complete at {} (current {})",
                                    m_incomplete.back().expectedCompleteCycle,
                                    m_programCycle));
                else
                    const_cast<Instruction&>(inst).addComment(
                        fmt::format("DSMEM: Expected complete at {} (current {})",
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
