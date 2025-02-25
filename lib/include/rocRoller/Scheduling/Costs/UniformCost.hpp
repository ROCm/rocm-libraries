
#pragma once

#include <concepts>
#include <string>
#include <vector>

#include <rocRoller/Scheduling/Costs/Cost.hpp>
#include <rocRoller/Scheduling/Costs/UniformCost_fwd.hpp>

namespace rocRoller
{
    namespace Scheduling
    {

        /**
         * UniformCost: Gives zero cost to all instructions.
         */
        class UniformCost : public Cost
        {
        public:
            UniformCost(ContextPtr);

            using Base = Cost;

            static const std::string Basename;
            static const std::string Name;

            /**
             * Returns true if `CostFunction` is Uniform
             */
            static bool Match(Argument arg);

            /**
             * Return shared pointer of `UniformCost` built from context
             */
            static std::shared_ptr<Cost> Build(Argument arg);

            /**
             * Return Name of `UniformCost`, used for debugging purposes currently
             */
            std::string name() const override;

            /**
             * Call operator orders the instructions.
             */
            float cost(Instruction const& inst, InstructionStatus const& status) const override;
        };
    }
}
