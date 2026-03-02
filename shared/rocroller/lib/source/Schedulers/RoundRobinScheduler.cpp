// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/Scheduling/RoundRobinScheduler.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        static_assert(Component::Component<RoundRobinScheduler>);

        RoundRobinScheduler::RoundRobinScheduler(ContextPtr ctx)
            : Scheduler{ctx}
        {
        }

        bool RoundRobinScheduler::Match(Argument arg)
        {
            return std::get<0>(arg) == SchedulerProcedure::RoundRobin;
        }

        std::shared_ptr<Scheduler> RoundRobinScheduler::Build(Argument arg)
        {
            if(!Match(arg))
                return nullptr;

            return std::make_shared<RoundRobinScheduler>(std::get<2>(arg));
        }

        std::string RoundRobinScheduler::name() const
        {
            return Name;
        }

        Generator<Instruction>
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
