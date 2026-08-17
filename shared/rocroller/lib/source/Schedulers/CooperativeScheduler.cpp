// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/Scheduling/CooperativeScheduler.hpp>
#include <rocRoller/Scheduling/Costs/MinNopsCost.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        static_assert(Component::Component<CooperativeScheduler>);

        CooperativeScheduler::CooperativeScheduler(ContextPtr ctx, CostFunction cmp)
            : Scheduler{ctx}
        {
            m_cost = Component::Get<Scheduling::Cost>(cmp, m_ctx);
        }

        bool CooperativeScheduler::Match(Argument arg)
        {
            return std::get<0>(arg) == SchedulerProcedure::Cooperative;
        }

        std::shared_ptr<Scheduler> CooperativeScheduler::Build(Argument arg)
        {
            if(!Match(arg))
                return nullptr;

            return std::make_shared<CooperativeScheduler>(std::get<2>(arg), std::get<1>(arg));
        }

        std::string CooperativeScheduler::name() const
        {
            return Name;
        }

        bool CooperativeScheduler::supportsAddingStreams() const
        {
            return true;
        }

        Generator<Instruction>
            CooperativeScheduler::operator()(std::vector<Generator<Instruction>>& seqs)
        {
            std::vector<Generator<Instruction>::iterator> iterators;

            if(seqs.empty())
                co_return;

            size_t numSeqs = 0;

            size_t idx = 0;

            while(true)
            {
                co_yield handleNewNodes(seqs, iterators);
                numSeqs = seqs.size();

                float currentCost = 0;
                bool  lockedOut   = false;

                if(iterators[idx] != seqs[idx].end())
                {
                    auto const& inst = *iterators[idx];

                    lockedOut   = !m_lockstate.isSchedulable(inst, idx);
                    currentCost = (*m_cost)(inst);
                }

                if(iterators[idx] == seqs[idx].end() || lockedOut || currentCost > 0)
                {
                    size_t origIdx    = idx;
                    float  minCost    = std::numeric_limits<float>::max();
                    int    minCostIdx = -1;
                    if(iterators[idx] != seqs[idx].end() && !lockedOut)
                    {
                        minCost    = currentCost;
                        minCostIdx = idx;
                    }

                    idx = (idx + 1) % numSeqs;

                    while(idx != origIdx)
                    {
                        if(iterators[idx] != seqs[idx].end())
                        {
                            auto const& inst = *iterators[idx];
                            lockedOut        = !m_lockstate.isSchedulable(inst, idx);
                            currentCost      = (*m_cost)(inst);
                            if(!lockedOut && currentCost < minCost)
                            {
                                minCost    = currentCost;
                                minCostIdx = idx;
                            }
                        }

                        if(minCost == 0)
                            break;

                        idx = (idx + 1) % numSeqs;
                    }

                    if(minCostIdx == -1)
                        break;

                    idx = minCostIdx;
                }
                co_yield yieldFromStream(iterators[idx], idx);
            }
        }
    }
}
