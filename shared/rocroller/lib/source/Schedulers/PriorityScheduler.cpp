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

#include <rocRoller/Scheduling/Costs/MinNopsCost.hpp>
#include <rocRoller/Scheduling/PriorityScheduler.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        RegisterComponent(PriorityScheduler);
        static_assert(Component::Component<PriorityScheduler>);

        inline PriorityScheduler::PriorityScheduler(ContextPtr ctx, CostFunction cmp)
            : Scheduler{ctx}
        {
            m_cost = Component::Get<Scheduling::Cost>(cmp, m_ctx);
        }

        inline bool PriorityScheduler::Match(Argument arg)
        {
            return std::get<0>(arg) == SchedulerProcedure::Priority;
        }

        inline std::shared_ptr<Scheduler> PriorityScheduler::Build(Argument arg)
        {
            if(!Match(arg))
                return nullptr;

            return std::make_shared<PriorityScheduler>(std::get<2>(arg), std::get<1>(arg));
        }

        inline std::string PriorityScheduler::name() const
        {
            return Name;
        }

        bool PriorityScheduler::supportsAddingStreams() const
        {
            return true;
        }

        inline Generator<Instruction>
            PriorityScheduler::operator()(std::vector<Generator<Instruction>>& seqs)
        {
            std::vector<Generator<Instruction>::iterator> iterators;

            if(seqs.empty())
                co_return;

            size_t numSeqs = 0;

            int minCostIdx = -1;
            do
            {
                co_yield handleNewNodes(seqs, iterators);
                numSeqs = seqs.size();

                float minCost = std::numeric_limits<float>::max();
                minCostIdx    = -1;

                for(size_t idx = 0; idx < numSeqs; idx++)
                {
                    if(iterators[idx] == seqs[idx].end())
                        continue;

                    auto const& instr = *iterators[idx];

                    if(!m_lockstate.isSchedulable(instr, idx))
                        continue;

                    float myCost = (*m_cost)(instr);

                    if(myCost < minCost)
                    {
                        minCost    = myCost;
                        minCostIdx = idx;
                    }

                    if(minCost == 0)
                        break;
                }

                if(minCostIdx >= 0)
                {
                    co_yield yieldFromStream(iterators[minCostIdx], minCostIdx);
                }

            } while(minCostIdx >= 0);
        }
    }
}
