
#include <rocRoller/Scheduling/SequentialScheduler.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        RegisterComponent(SequentialScheduler);

        static_assert(Component::Component<SequentialScheduler>);

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
    }
}
