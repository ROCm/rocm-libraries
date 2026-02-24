// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/Scheduling/SequentialScheduler.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        static_assert(Component::Component<SequentialScheduler>);

        SequentialScheduler::SequentialScheduler(ContextPtr ctx)
            : Scheduler{ctx}
        {
        }

        bool SequentialScheduler::Match(Argument arg)
        {
            return std::get<0>(arg) == SchedulerProcedure::Sequential;
        }

        std::shared_ptr<Scheduler> SequentialScheduler::Build(Argument arg)
        {
            if(!Match(arg))
                return nullptr;

            return std::make_shared<SequentialScheduler>(std::get<2>(arg));
        }

        std::string SequentialScheduler::name() const
        {
            return Name;
        }

        bool SequentialScheduler::supportsAddingStreams() const
        {
            return true;
        }

        Generator<Instruction>
            SequentialScheduler::operator()(std::vector<Generator<Instruction>>& seqs)
        {
            bool yieldedAny = false;

            // a vector of instruction streams
            std::vector<Generator<Instruction>::iterator> iterators;
            co_yield handleNewNodes(seqs, iterators);

            do
            {
                yieldedAny = false;

                for(size_t i = 0; i < seqs.size(); i++)
                {
                    while(iterators[i] != seqs[i].end())
                    {
                        auto const& instr = *iterators[i];
                        if(!m_lockstate.isSchedulable(instr, i))
                            break;
                        for(auto const& inst : yieldFromStream(iterators[i], i))
                            co_yield inst;
                        yieldedAny = true;
                    }

                    if(seqs.size() != iterators.size())
                    {
                        co_yield handleNewNodes(seqs, iterators);
                        yieldedAny = true;
                    }
                }

            } while(yieldedAny);
        }
    }
}
