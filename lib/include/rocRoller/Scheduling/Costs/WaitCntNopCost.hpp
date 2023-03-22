
#pragma once

#include <concepts>
#include <string>
#include <vector>

#include "Cost.hpp"
#include "WaitCntNopCost_fwd.hpp"

namespace rocRoller
{
    namespace Scheduling
    {

        /**
         * WaitCntNopCost: Orders the instructions based on the number of Nops and WaitCnts.
         */
        class WaitCntNopCost : public Cost
        {
        public:
            WaitCntNopCost(std::shared_ptr<Context>);

            using Base = Cost;

            static const std::string Basename;
            static const std::string Name;

            /**
             * Returns true if `CostFunction` is WaitCntNop
             */
            static bool Match(Argument arg);

            /**
             * Return shared pointer of `WaitCntNopCost` built from context
             */
            static std::shared_ptr<Cost> Build(Argument arg);

            /**
             * Return Name of `WaitCntNopCost`, used for debugging purposes currently
             */
            std::string name() const override;

            /**
             * Call operator orders the instructions.
             */
            float cost(const InstructionStatus& inst) const override;
        };
    }
}
