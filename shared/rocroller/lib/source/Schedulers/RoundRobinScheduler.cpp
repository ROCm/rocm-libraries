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

#include <rocRoller/Scheduling/RoundRobinScheduler.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        RegisterComponent(RoundRobinScheduler);
        static_assert(Component::Component<RoundRobinScheduler>);

        inline RoundRobinScheduler::RoundRobinScheduler(ContextPtr ctx)
            : Scheduler{ctx}
        {
        }

        inline bool RoundRobinScheduler::Match(Argument arg)
        {
            return std::get<0>(arg) == SchedulerProcedure::RoundRobin;
        }

        inline std::shared_ptr<Scheduler> RoundRobinScheduler::Build(Argument arg)
        {
            if(!Match(arg))
                return nullptr;

            return std::make_shared<RoundRobinScheduler>(std::get<2>(arg));
        }

        inline std::string RoundRobinScheduler::name() const
        {
            return Name;
        }

        inline Generator<Instruction>
            RoundRobinScheduler::operator()(std::vector<Generator<Instruction>>& seqs)
        {
            std::vector<Generator<Instruction>::iterator> iterators;

            if(seqs.empty())
                co_return;

            size_t n         = seqs.size();
            bool   yield_seq = true;

            iterators.reserve(n);
            for(auto& seq : seqs)
            {
                iterators.emplace_back(seq.begin());
            }

            while(yield_seq)
            {
                yield_seq = false;

                for(size_t i = 0; i < n; ++i)
                {
                    if(iterators[i] != seqs[i].end())
                    {
                        if(!m_lockstate.isSchedulable(*iterators[i], i))
                            continue;

                        yield_seq = true;
                        co_yield yieldFromStream(iterators[i], i);
                    }
                }
            }
        }
    }
}
