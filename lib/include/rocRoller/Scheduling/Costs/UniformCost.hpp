
#pragma once

#include <concepts>
#include <string>
#include <vector>

#include "Cost.hpp"
#include "UniformCost_fwd.hpp"

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
            UniformCost(std::shared_ptr<Context>);

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
            float cost(const InstructionStatus& inst) const override;
        };
    }
}
