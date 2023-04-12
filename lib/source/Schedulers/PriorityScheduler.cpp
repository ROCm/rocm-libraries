
#include <rocRoller/Scheduling/Costs/MinNopsCost.hpp>
#include <rocRoller/Scheduling/PriorityScheduler.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        RegisterComponent(PriorityScheduler);
        static_assert(Component::Component<PriorityScheduler>);

        inline PriorityScheduler::PriorityScheduler(std::shared_ptr<Context> ctx, CostFunction cmp)
            : Scheduler{ctx}
        {
            m_cost = Component::Get<Scheduling::Cost>(cmp, m_ctx);
        }

        inline bool PriorityScheduler::Match(Argument arg)
        {
            return std::get<0>(arg) == SchedulerProcedure::Priority;
        }

        inline std::shared_ptr<Scheduler> PriorityScheduler::Build(Argument arg)
        {
            if(!Match(arg))
                return nullptr;

            return std::make_shared<PriorityScheduler>(std::get<2>(arg), std::get<1>(arg));
        }

        inline std::string PriorityScheduler::name() const
        {
            return Name;
        }

        bool PriorityScheduler::supportsAddingStreams() const
        {
            return true;
        }

        inline Generator<Instruction>
            PriorityScheduler::operator()(std::vector<Generator<Instruction>>& seqs)
        {
            std::vector<Generator<Instruction>::iterator> iterators;

            if(seqs.empty())
                co_return;

            size_t numSeqs = 0;

            int minCostIdx = -1;
            do
            {
                co_yield handleNewNodes(seqs, iterators);
                numSeqs = seqs.size();

                float minCost = std::numeric_limits<float>::max();
                minCostIdx    = -1;

                for(size_t idx = 0; idx < numSeqs; idx++)
                {
                    if(iterators[idx] == seqs[idx].end())
                        continue;

                    float myCost = (*m_cost)(iterators[idx]);

                    if(myCost < minCost)
                    {
                        minCost    = myCost;
                        minCostIdx = idx;
                    }

                    if(minCost == 0)
                        break;
                }

                if(minCostIdx >= 0)
                {
                    co_yield yieldFromStream(iterators[minCostIdx]);
                }

            } while(minCostIdx >= 0);

            m_lockstate.isValid(false);
        }
    }
}
