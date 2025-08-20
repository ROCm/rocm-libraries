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

#include <rocRoller/Scheduling/SequentialScheduler.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        RegisterComponent(SequentialScheduler);

        static_assert(Component::Component<SequentialScheduler>);

        inline SequentialScheduler::SequentialScheduler(ContextPtr ctx)
            : Scheduler{ctx}
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

            return std::make_shared<SequentialScheduler>(std::get<2>(arg));
        }

        inline std::string SequentialScheduler::name() const
        {
            return Name;
        }

        bool SequentialScheduler::supportsAddingStreams() const
        {
            return true;
        }

        inline Generator<Instruction>
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
