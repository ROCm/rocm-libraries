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

#include <rocRoller/Scheduling/Costs/MinNopsCost.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        RegisterComponent(MinNopsCost);
        static_assert(Component::Component<MinNopsCost>);

        inline MinNopsCost::MinNopsCost(ContextPtr ctx)
            : Cost{ctx}
        {
        }

        inline bool MinNopsCost::Match(Argument arg)
        {
            return std::get<0>(arg) == CostFunction::MinNops;
        }

        inline std::shared_ptr<Cost> MinNopsCost::Build(Argument arg)
        {
            if(!Match(arg))
                return nullptr;

            return std::make_shared<MinNopsCost>(std::get<1>(arg));
        }

        inline std::string MinNopsCost::name() const
        {
            return Name;
        }

        inline float MinNopsCost::cost(Instruction const&       inst,
                                       InstructionStatus const& status) const
        {
            return static_cast<float>(status.nops);
        }
    }
}
