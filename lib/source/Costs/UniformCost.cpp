
#include <rocRoller/Scheduling/Costs/UniformCost.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        RegisterComponent(UniformCost);
        static_assert(Component::Component<UniformCost>);

        inline UniformCost::UniformCost(std::shared_ptr<Context> ctx)
            : Cost{ctx}
        {
        }

        inline bool UniformCost::Match(Argument arg)
        {
            return std::get<0>(arg) == CostFunction::Uniform;
        }

        inline std::shared_ptr<Cost> UniformCost::Build(Argument arg)
        {
            if(!Match(arg))
                return nullptr;

            return std::make_shared<UniformCost>(std::get<1>(arg));
        }

        inline std::string UniformCost::name() const
        {
            return Name;
        }

        inline float UniformCost::cost(Instruction const&       inst,
                                       InstructionStatus const& status) const
        {
            return 0.0;
        }
    }
}
