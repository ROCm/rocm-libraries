
#pragma once

#include <concepts>
#include <string>
#include <vector>

#include "Cost.hpp"
#include "MinNopsCost_fwd.hpp"

namespace rocRoller
{
    namespace Scheduling
    {

        /**
         * MinNopsCost: Orders the instructions based on the number of Nops.
         */
        class MinNopsCost : public Cost
        {
        public:
            MinNopsCost(std::shared_ptr<Context>);

            using Base = Cost;

            static const std::string Basename;
            static const std::string Name;

            /**
             * Returns true if `CostProcedure` is MinNops
             */
            static bool Match(Argument arg);

            /**
             * Return shared pointer of `MinNopsCost` built from context
             */
            static std::shared_ptr<Cost> Build(Argument arg);

            /**
             * Return Name of `MinNopsCost`, used for debugging purposes currently
             */
            std::string name() override;

            /**
             * Call operator orders the instructions.
             */
            float cost(const InstructionStatus& inst) const override;
        };
    }
}
