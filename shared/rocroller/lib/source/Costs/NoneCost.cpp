// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/Scheduling/Costs/NoneCost.hpp>
#include <rocRoller/Utilities/Random.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        static_assert(Component::Component<NoneCost>);

        NoneCost::NoneCost(ContextPtr ctx)
            : Cost{ctx}
        {
            Throw<FatalError>("Cannot use None cost.");
        }

        bool NoneCost::Match(Argument arg)
        {
            return std::get<0>(arg) == CostFunction::None;
        }

        std::shared_ptr<Cost> NoneCost::Build(Argument arg)
        {
            Throw<FatalError>("Cannot use None cost.");
        }

        std::string NoneCost::name() const
        {
            return Name;
        }

        float NoneCost::cost(Instruction const& inst, InstructionStatus const& status) const
        {
            Throw<FatalError>("Cannot use None cost.");
        }
    }
}
