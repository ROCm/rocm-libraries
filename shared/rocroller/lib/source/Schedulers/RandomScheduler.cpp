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

#include <rocRoller/Scheduling/RandomScheduler.hpp>
#include <rocRoller/Utilities/Random.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        RegisterComponent(RandomScheduler);
        static_assert(Component::Component<RandomScheduler>);

        inline RandomScheduler::RandomScheduler(ContextPtr ctx)
            : Scheduler{ctx}
        {
        }

        inline bool RandomScheduler::Match(Argument arg)
        {
            return std::get<0>(arg) == SchedulerProcedure::Random;
        }

        inline std::shared_ptr<Scheduler> RandomScheduler::Build(Argument arg)
        {
            if(!Match(arg))
                return nullptr;

            return std::make_shared<RandomScheduler>(std::get<2>(arg));
        }

        inline std::string RandomScheduler::name() const
        {
            return Name;
        }

        bool RandomScheduler::supportsAddingStreams() const
        {
            return true;
        }

        inline Generator<Instruction>
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
