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
