
#pragma once

#include <concepts>
#include <string>
#include <vector>

#include "Cost.hpp"
#include "NoneCost_fwd.hpp"

namespace rocRoller
{
    namespace Scheduling
    {

        /**
         * NoneCost: This cost can be used for cost-independent schedulers. If it's ever initialized an exception is thrown.
         */
        class NoneCost : public Cost
        {
        public:
            NoneCost(std::shared_ptr<Context>);

            using Base = Cost;

            static const std::string Basename;
            static const std::string Name;

            /**
             * Returns true if `CostFunction` is None
             */
            static bool Match(Argument arg);

            /**
             * Return shared pointer of `NoneCost` built from context
             */
            static std::shared_ptr<Cost> Build(Argument arg);

            /**
             * Return Name of `NoneCost`, used for debugging purposes currently
             */
            std::string name() const override;

            /**
             * Call operator orders the instructions.
             */
            float cost(const InstructionStatus& inst) const override;
        };
    }
}
