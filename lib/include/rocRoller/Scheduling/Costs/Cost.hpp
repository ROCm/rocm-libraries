
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
            using Argument = std::tuple<CostFunction, std::shared_ptr<rocRoller::Context>>;

            /**
             * @brief Collection of Tuples where the first member is the index of the generator
             * and the second member is the cost next instruction for the given iteration
             *
             */
            using Result = std::vector<std::tuple<int, float>>;

            Cost(ContextPtr);

            static const std::string Basename;

            virtual std::string name() const                         = 0;
            virtual float       cost(const InstructionStatus&) const = 0;

            /**
             * @brief Gets the sorted costs for the collection of generator iterators
             *
             * @return Sorted collection of Tuples  based on lowest cost where
             * the first member is the index of the generator
             * and the second member is the cost next instruction for the given iteration
             */
            Result operator()(std::vector<Generator<Instruction>::iterator>&) const;

            /**
             * @brief Gets the cost of one generator for the given iteration
             *
             * @return float
             */
            float operator()(Generator<Instruction>::iterator&) const;

        protected:
            std::weak_ptr<rocRoller::Context> m_ctx;
        };

        std::ostream& operator<<(std::ostream&, CostFunction);
    }
}

#include "Cost_impl.hpp"
