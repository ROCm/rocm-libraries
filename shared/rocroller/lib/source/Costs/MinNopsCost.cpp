// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/Scheduling/Costs/MinNopsCost.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        static_assert(Component::Component<MinNopsCost>);

        MinNopsCost::MinNopsCost(ContextPtr ctx)
            : Cost{ctx}
        {
        }

        bool MinNopsCost::Match(Argument arg)
        {
            return std::get<0>(arg) == CostFunction::MinNops;
        }

        std::shared_ptr<Cost> MinNopsCost::Build(Argument arg)
        {
            if(!Match(arg))
                return nullptr;

            return std::make_shared<MinNopsCost>(std::get<1>(arg));
        }

        std::string MinNopsCost::name() const
        {
            return Name;
        }

        float MinNopsCost::cost(Instruction const& inst, InstructionStatus const& status) const
        {
            return static_cast<float>(status.nops);
        }
    }
}
