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

            auto                                          random = m_ctx.lock()->random();
            std::vector<Generator<Instruction>::iterator> iterators;

            co_yield handleNewNodes(seqs, iterators);

            std::vector<size_t> validIterIndexes(iterators.size());
            std::iota(validIterIndexes.begin(), validIterIndexes.end(), 0);

            while(!validIterIndexes.empty())
            {
                size_t idx;

                // Try to find a stream that doesn't cause an out of register error.
                std::vector<size_t> iterIndexesToSearch = validIterIndexes;
                while(!iterIndexesToSearch.empty())
                {
                    size_t i_rand = random->next<size_t>(0, iterIndexesToSearch.size() - 1);
                    idx           = iterIndexesToSearch[i_rand];

                    if(iterators[idx] != seqs[idx].end())
                    {
                        auto status = m_ctx.lock()->peek(*iterators[idx]);
                        if(status.outOfRegisters.count() == 0)
                        {
                            break;
                        }
                    }
                    else
                    {
                        break;
                    }
                    iterIndexesToSearch.erase(iterIndexesToSearch.begin() + i_rand);
                }

                if(iterators[idx] != seqs[idx].end())
                {
                    co_yield yieldFromStream(iterators[idx]);
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

            m_lockstate.isValid(false);
        }
    }
}
