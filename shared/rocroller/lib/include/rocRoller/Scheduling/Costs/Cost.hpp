/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include <concepts>
#include <string>
#include <vector>

#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/Context_fwd.hpp>
#include <rocRoller/Scheduling/Costs/Cost_fwd.hpp>
#include <rocRoller/Utilities/Component.hpp>
#include <rocRoller/Utilities/Generator.hpp>

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
            using Argument = std::tuple<CostFunction, rocRoller::ContextPtr>;

            /**
             * @brief Collection of Tuples where the first member is the index of the generator
             * and the second member is the cost next instruction for the given iteration
             *
             */
            using Result = std::vector<std::tuple<int, float>>;

            Cost(ContextPtr);

            static const std::string Basename;

            virtual std::string name() const                                                   = 0;
            virtual float cost(Instruction const& inst, InstructionStatus const& status) const = 0;

            /**
             * @brief Gets the sorted costs for the collection of generator iterators
             *
             * @return Sorted collection of Tuples  based on lowest cost where
             * the first member is the index of the generator
             * and the second member is the cost next instruction for the given iteration
             */
            Result operator()(std::vector<Generator<Instruction>::iterator>&) const;

            float operator()(Instruction const& inst) const;

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

#include <rocRoller/Scheduling/Costs/Cost_impl.hpp>
