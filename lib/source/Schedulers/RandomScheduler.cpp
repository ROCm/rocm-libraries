
#include <rocRoller/Scheduling/RandomScheduler.hpp>
#include <rocRoller/Utilities/Random.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        RegisterComponent(RandomScheduler);
        static_assert(Component::Component<RandomScheduler>);

        inline RandomScheduler::RandomScheduler(std::shared_ptr<Context> ctx)
            : m_ctx{ctx}
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

            return std::make_shared<RandomScheduler>(std::get<1>(arg));
        }

        inline std::string RandomScheduler::name()
        {
            return Name;
        }

        inline Generator<Instruction>
            RandomScheduler::operator()(std::vector<Generator<Instruction>>& seqs)
        {
            auto                                          random = m_ctx.lock()->random();
            std::vector<Generator<Instruction>::iterator> iterators;

            if(seqs.empty())
                co_return;

            size_t n = seqs.size();

            iterators.reserve(n);
            for(auto& seq : seqs)
            {
                iterators.emplace_back(seq.begin());
            }

            while(n > 0)
            {
                size_t idx = random->next<size_t>(0, n - 1);

                if(iterators[idx] != seqs[idx].end())
                {
                    do
                    {
                        AssertFatal(iterators[idx] != seqs[idx].end(),
                                    "End of instruction stream reached without unlocking");
                        m_lockstate.add(*(iterators[idx]));
                        co_yield *(iterators[idx]);
                        ++iterators[idx];
                        co_yield consumeComments(iterators[idx], seqs[idx].end());
                    } while(m_lockstate.isLocked());
                }
                else
                {
                    iterators.erase(iterators.begin() + idx);
                    seqs.erase(seqs.begin() + idx);
                    n--;
                }
            }

            m_lockstate.isValid(false);
        }
    }
}
