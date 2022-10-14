
#pragma once

#include "Scheduler.hpp"

namespace rocRoller
{
    namespace Scheduling
    {
        static_assert(Component::ComponentBase<Scheduler>);
        static_assert(Component::Component<SequentialScheduler>);
        static_assert(Component::Component<RoundRobinScheduler>);

        template <typename Begin, typename End>
        Generator<Instruction> consumeComments(Begin& begin, End const& end)
        {
            while(begin != end && begin->isCommentOnly())
            {
                co_yield *begin;
                ++begin;
            }
        }

        inline LockState::LockState()
            : m_dependency(Dependency::None)
            , m_lockdepth(0)
        {
        }

        inline LockState::LockState(Dependency dependency)
            : m_dependency(dependency)
        {
            AssertFatal(m_dependency != Scheduling::Dependency::Count
                            && m_dependency != Scheduling::Dependency::Unlock,
                        "Can not instantiate LockState with Count or Unlock dependency");

            m_lockdepth = 1;
        }

        inline void LockState::add(Instruction const& instruction)
        {
            int inst_lockvalue = instruction.getLockValue();

            // Instruction does not lock or unlock, do nothing
            if(inst_lockvalue == 0)
            {
                return;
            }

            // Instruction can only lock (1) or unlock (-1)
            if(inst_lockvalue != -1 && inst_lockvalue != 1)
            {
                Throw<FatalError>("Invalid instruction lockstate: ", ShowValue(inst_lockvalue));
            }

            // Instruction trying to unlock when there is no lock
            if(m_lockdepth == 0 && inst_lockvalue == -1)
            {
                Throw<FatalError>("Trying to unlock when not locked");
            }

            // Instruction initializes the lockstate
            if(m_lockdepth == 0)
            {
                m_dependency = instruction.getDependency();
            }

            m_lockdepth += inst_lockvalue;

            // Instruction releases lock
            if(m_lockdepth == 0)
            {
                m_dependency = Scheduling::Dependency::None;
            }
        }

        inline bool LockState::isLocked() const
        {
            return m_lockdepth > 0;
        }

        // Will grow into a function that accepts args and checks the lock is in a valid state against those args
        inline void LockState::isValid(bool locked) const
        {
            AssertFatal(isLocked() == locked, "Lock in invalid state");
        }

        inline Dependency LockState::getDependency() const
        {
            return m_dependency;
        }

        inline int LockState::getLockDepth() const
        {
            return m_lockdepth;
        }

        inline LockState Scheduler::getLockState() const
        {
            return m_lockstate;
        }

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
                for(auto& inst : seq)
                {
                    m_lockstate.add(inst);
                    co_yield inst;
                }
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

                for(size_t i = 0; i < n; ++i)
                {
                    if(iterators[i] != seqs[i].end())
                    {
                        do
                        {
                            AssertFatal(iterators[i] != seqs[i].end(),
                                        "End of instruction stream reached without unlocking");
                            yield_seq = true;
                            m_lockstate.add(*(iterators[i]));
                            co_yield *(iterators[i]);
                            ++iterators[i];
                        } while(m_lockstate.isLocked());
                    }
                }
            }

            m_lockstate.isValid(false);
        }
    }
}
