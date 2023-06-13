
#include <rocRoller/Scheduling/Costs/NoneCost.hpp>
#include <rocRoller/Utilities/Random.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        RegisterComponent(NoneCost);
        static_assert(Component::Component<NoneCost>);

        inline NoneCost::NoneCost(ContextPtr ctx)
            : Cost{ctx}
        {
            AssertFatal(false, "Cannot use None cost.");
        }

        inline bool NoneCost::Match(Argument arg)
        {
            return std::get<0>(arg) == CostFunction::None;
        }

        inline std::shared_ptr<Cost> NoneCost::Build(Argument arg)
        {
            Throw<FatalError>("Cannot use None cost.");
        }

        inline std::string NoneCost::name() const
        {
            return Name;
        }

        inline float NoneCost::cost(Instruction const& inst, InstructionStatus const& status) const
        {
            Throw<FatalError>("Cannot use None cost.");
        }
    }
}
