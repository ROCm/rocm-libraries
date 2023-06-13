
#include <rocRoller/Scheduling/CooperativeScheduler.hpp>
#include <rocRoller/Scheduling/Costs/MinNopsCost.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        RegisterComponent(CooperativeScheduler);
        static_assert(Component::Component<CooperativeScheduler>);

        inline CooperativeScheduler::CooperativeScheduler(ContextPtr ctx, CostFunction cmp)
            : Scheduler{ctx}
        {
            m_cost = Component::Get<Scheduling::Cost>(cmp, m_ctx);
        }

        inline bool CooperativeScheduler::Match(Argument arg)
        {
            return std::get<0>(arg) == SchedulerProcedure::Cooperative;
        }

        inline std::shared_ptr<Scheduler> CooperativeScheduler::Build(Argument arg)
        {
            if(!Match(arg))
                return nullptr;

            return std::make_shared<CooperativeScheduler>(std::get<2>(arg), std::get<1>(arg));
        }

        inline std::string CooperativeScheduler::name() const
        {
            return Name;
        }

        bool CooperativeScheduler::supportsAddingStreams() const
        {
            return true;
        }

        inline Generator<Instruction>
            CooperativeScheduler::operator()(std::vector<Generator<Instruction>>& seqs)
        {
            std::vector<Generator<Instruction>::iterator> iterators;

            if(seqs.empty())
                co_return;

            size_t numSeqs = 0;

            size_t idx = 0;
            float  currentCost;

            while(true)
            {
                co_yield handleNewNodes(seqs, iterators);
                numSeqs = seqs.size();

                if(iterators[idx] != seqs[idx].end())
                {
                    currentCost = (*m_cost)(iterators[idx]);
                }
                if(iterators[idx] == seqs[idx].end() || currentCost > 0)
                {
                    size_t origIdx    = idx;
                    float  minCost    = std::numeric_limits<float>::max();
                    int    minCostIdx = -1;
                    if(iterators[idx] != seqs[idx].end())
                    {
                        minCost    = currentCost;
                        minCostIdx = idx;
                    }

                    idx = (idx + 1) % numSeqs;

                    while(idx != origIdx)
                    {
                        if(iterators[idx] != seqs[idx].end())
                        {
                            currentCost = (*m_cost)(iterators[idx]);
                            if(currentCost < minCost)
                            {
                                minCost    = currentCost;
                                minCostIdx = idx;
                            }
                        }

                        if(minCost == 0)
                            break;

                        idx = (idx + 1) % numSeqs;
                    }

                    if(minCostIdx == -1)
                        break;

                    idx = minCostIdx;
                }
                co_yield yieldFromStream(iterators[idx]);
            }

            m_lockstate.isValid(false);
        }
    }
}
