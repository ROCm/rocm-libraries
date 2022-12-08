
#include <rocRoller/Scheduling/Costs/MinNopsCost.hpp>
#include <rocRoller/Utilities/Random.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        RegisterComponent(MinNopsCost);
        static_assert(Component::Component<MinNopsCost>);

        inline MinNopsCost::MinNopsCost(std::shared_ptr<Context> ctx)
            : Cost{ctx}
        {
        }

        inline bool MinNopsCost::Match(Argument arg)
        {
            return std::get<0>(arg) == CostProcedure::MinNops;
        }

        inline std::shared_ptr<Cost> MinNopsCost::Build(Argument arg)
        {
            if(!Match(arg))
                return nullptr;

            return std::make_shared<MinNopsCost>(std::get<1>(arg));
        }

        inline std::string MinNopsCost::name()
        {
            return Name;
        }

        inline float MinNopsCost::cost(const InstructionStatus& inst) const
        {
            return static_cast<float>(inst.nops);
        }
    }
}
