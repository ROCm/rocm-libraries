// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/Scheduling/Costs/UniformCost.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        static_assert(Component::Component<UniformCost>);

        UniformCost::UniformCost(ContextPtr ctx)
            : Cost{ctx}
        {
        }

        bool UniformCost::Match(Argument arg)
        {
            return std::get<0>(arg) == CostFunction::Uniform;
        }

        std::shared_ptr<Cost> UniformCost::Build(Argument arg)
        {
            if(!Match(arg))
                return nullptr;

            return std::make_shared<UniformCost>(std::get<1>(arg));
        }

        std::string UniformCost::name() const
        {
            return Name;
        }

        float UniformCost::cost(Instruction const& inst, InstructionStatus const& status) const
        {
            return 0.0;
        }
    }
}
