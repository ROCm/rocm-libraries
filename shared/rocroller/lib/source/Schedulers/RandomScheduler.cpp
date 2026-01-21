// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/Scheduling/RandomScheduler.hpp>
#include <rocRoller/Utilities/Random.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        static_assert(Component::Component<RandomScheduler>);

        RandomScheduler::RandomScheduler(ContextPtr ctx)
            : Scheduler{ctx}
        {
        }

        bool RandomScheduler::Match(Argument arg)
        {
            return std::get<0>(arg) == SchedulerProcedure::Random;
        }

        std::shared_ptr<Scheduler> RandomScheduler::Build(Argument arg)
        {
            if(!Match(arg))
                return nullptr;

            return std::make_shared<RandomScheduler>(std::get<2>(arg));
        }

        std::string RandomScheduler::name() const
        {
            return Name;
        }

        bool RandomScheduler::supportsAddingStreams() const
        {
            return true;
        }

        Generator<Instruction>
            RandomScheduler::operator()(std::vector<Generator<Instruction>>& seqs)
        {
            if(seqs.empty())
                co_return;

            auto random = m_ctx.lock()->random();
            auto random_from
                = [&random](std::vector<size_t> const& vec) -> std::tuple<size_t, size_t> {
                AssertFatal(!vec.empty());

                auto max    = vec.size() - 1;
                auto i_rand = random->next<size_t>(0, max);
                return {vec.at(i_rand), i_rand};
            };

            std::vector<Generator<Instruction>::iterator> iterators;

            co_yield handleNewNodes(seqs, iterators);

            std::vector<size_t> validIterIndexes(iterators.size());
            std::iota(validIterIndexes.begin(), validIterIndexes.end(), 0);

            while(!validIterIndexes.empty())
            {
                auto [idx, rand] = random_from(validIterIndexes);

                auto validIdx = [&](size_t idx) -> bool {
                    if(iterators[idx] == seqs[idx].end())
                        return true;

                    auto const& inst = *iterators[idx];

                    if(!m_lockstate.isSchedulable(inst, idx))
                        return false;

                    auto status = m_ctx.lock()->peek(inst);

                    if(status.outOfRegisters.count() != 0)
                        return false;

                    return true;
                };

                // Try to find a stream that doesn't cause an out of register
                // error, or violate locking rules.
                if(!validIdx(idx))
                {
                    auto iterIndexesToSearch = validIterIndexes;
                    iterIndexesToSearch.erase(iterIndexesToSearch.begin() + rand);

                    do
                    {
                        auto [myIdx, myRand] = random_from(iterIndexesToSearch);
                        idx                  = myIdx;

                        iterIndexesToSearch.erase(iterIndexesToSearch.begin() + myRand);
                    } while(!validIdx(idx));
                }

                if(iterators[idx] != seqs[idx].end())
                {
                    co_yield yieldFromStream(iterators[idx], idx);
                }
                else
                {
                    validIterIndexes.erase(
                        std::remove(validIterIndexes.begin(), validIterIndexes.end(), idx),
                        validIterIndexes.end());
                }

                size_t n = iterators.size();
                co_yield handleNewNodes(seqs, iterators);
                for(size_t i = n; i < iterators.size(); i++)
                {
                    validIterIndexes.push_back(i);
                }
            }
        }
    }
}
