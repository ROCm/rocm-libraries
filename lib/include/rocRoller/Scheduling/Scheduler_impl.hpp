
#pragma once

#include "Scheduler.hpp"

namespace rocRoller
{
    namespace Scheduling
    {
        static_assert(Component::ComponentBase<Scheduler>);
        static_assert(Component::Component<SequentialScheduler>);
        static_assert(Component::Component<RoundRobinScheduler>);

        inline SequentialScheduler::SequentialScheduler(std::shared_ptr<Context> ctx)
            : m_ctx{ctx}
        {
        }

        inline bool SequentialScheduler::Match(Argument arg)
        {
            return std::get<0>(arg) == SchedulerProcedure::Sequential;
        }

        inline std::shared_ptr<Scheduler> SequentialScheduler::Build(Argument arg)
        {
            if(!Match(arg))
                return nullptr;

            return std::make_shared<SequentialScheduler>(std::get<1>(arg));
        }

        inline std::string SequentialScheduler::name()
        {
            return Name;
        }

        inline Generator<Instruction>
            SequentialScheduler::operator()(std::vector<Generator<Instruction>>& seqs)
        {
            for(auto& seq : seqs)
            {
                co_yield seq;
            }
        }

        inline RoundRobinScheduler::RoundRobinScheduler(std::shared_ptr<Context> ctx)
            : m_ctx{ctx}
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

            return std::make_shared<RoundRobinScheduler>(std::get<1>(arg));
        }

        inline std::string RoundRobinScheduler::name()
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

                for(int i = 0; i < n; ++i)
                {
                    if(iterators[i] != seqs[i].end())
                    {
                        co_yield *(iterators[i]);
                        ++iterators[i];
                        yield_seq = true;
                    }
                }
            }
        }
    }
}
