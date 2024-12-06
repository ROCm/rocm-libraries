
#pragma once

#include <concepts>
#include <string>
#include <vector>

#include <rocRoller/Scheduling/Costs/Cost.hpp>
#include <rocRoller/Scheduling/Costs/WaitCntNopCost_fwd.hpp>

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
            WaitCntNopCost(ContextPtr);

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
            float cost(Instruction const& inst, InstructionStatus const& status) const override;
        };
    }
}
