
#pragma once

#include <concepts>
#include <string>
#include <vector>

#include "Cost_fwd.hpp"

#include "../../CodeGen/Instruction.hpp"
#include "../../Context_fwd.hpp"
#include "../../Utilities/Component.hpp"
#include "../../Utilities/Generator.hpp"

namespace rocRoller
{
    namespace Scheduling
    {
        /**
         * A `Cost` is a base class for the different types of costs used to determine scheduling order.
         *
         * - This class should be able to be made into `ComponentBase` class
         */
        class Cost
        {
        public:
            using Argument = std::tuple<CostProcedure, std::shared_ptr<rocRoller::Context>>;
            using Result   = std::vector<std::tuple<int, float>>;

            Cost(ContextPtr);

            static const std::string Name;

            virtual std::string name()                               = 0;
            virtual float       cost(const InstructionStatus&) const = 0;

            Result operator()(std::vector<Generator<Instruction>::iterator>&) const;
            float  operator()(Generator<Instruction>::iterator&) const;

        protected:
            std::weak_ptr<rocRoller::Context> m_ctx;
        };

        std::ostream& operator<<(std::ostream&, CostProcedure);
    }
}

#include "Cost_impl.hpp"
