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

#include <rocRoller/Scheduling/CooperativeScheduler.hpp>
#include <rocRoller/Scheduling/Costs/MinNopsCost.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        RegisterComponent(CooperativeScheduler);
        static_assert(Component::Component<CooperativeScheduler>);

        inline CooperativeScheduler::CooperativeScheduler(ContextPtr ctx, CostFunction cmp)
            : Scheduler{ctx}
        {
            m_cost = Component::Get<Scheduling::Cost>(cmp, m_ctx);
        }

        inline bool CooperativeScheduler::Match(Argument arg)
        {
            return std::get<0>(arg) == SchedulerProcedure::Cooperative;
        }

        inline std::shared_ptr<Scheduler> CooperativeScheduler::Build(Argument arg)
        {
            if(!Match(arg))
                return nullptr;

            return std::make_shared<CooperativeScheduler>(std::get<2>(arg), std::get<1>(arg));
        }

        inline std::string CooperativeScheduler::name() const
        {
            return Name;
        }

        bool CooperativeScheduler::supportsAddingStreams() const
        {
            return true;
        }

        inline Generator<Instruction>
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
