#include <rocRoller/Context.hpp>
#include <rocRoller/Scheduling/Costs/Cost.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        RegisterComponentBase(Cost);

        std::string toString(CostProcedure proc)
        {
            switch(proc)
            {
            case CostProcedure::None:
                return "None";
            case CostProcedure::Uniform:
                return "Uniform";
            case CostProcedure::MinNops:
                return "MinNops";
            case CostProcedure::WaitCntNop:
                return "WaitCntNop";
            case CostProcedure::Count:
                return "Count";
            }

            Throw<FatalError>("Invalid Cost Procedure: ", ShowValue(static_cast<int>(proc)));
        }

        std::ostream& operator<<(std::ostream& stream, CostProcedure proc)
        {
            return stream << toString(proc);
        }

        Cost::Result Cost::operator()(std::vector<Generator<Instruction>::iterator>& iters) const
        {
            Cost::Result retval;
            auto         context = m_ctx.lock();
            size_t       n       = iters.size();

            for(size_t i = 0; i < n; ++i)
            {
                if(iters[i] != std::default_sentinel_t{})
                {
                    float                  peek = cost(context->peek(*(iters[i])));
                    std::tuple<int, float> new_val{i, peek};

                    retval.insert(
                        std::upper_bound(
                            retval.begin(),
                            retval.end(),
                            new_val,
                            [](std::tuple<int, float> lhs, std::tuple<int, float> rhs) -> bool {
                                return (std::get<1>(lhs) < std::get<1>(rhs))
                                       || (std::get<1>(lhs) == std::get<1>(rhs)
                                           && std::get<0>(lhs) < std::get<0>(rhs));
                            }),
                        new_val);
                }
            }
            return retval;
        }

        float Cost::operator()(Generator<Instruction>::iterator& iter) const
        {
            return cost(m_ctx.lock()->peek(*iter));
        }
    }
}
