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

#include <rocRoller/Context.hpp>
#include <rocRoller/Scheduling/Costs/Cost.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        RegisterComponentBase(Cost);

        std::string toString(CostFunction proc)
        {
            switch(proc)
            {
            case CostFunction::None:
                return "None";
            case CostFunction::Uniform:
                return "Uniform";
            case CostFunction::MinNops:
                return "MinNops";
            case CostFunction::WaitCntNop:
                return "WaitCntNop";
            case CostFunction::LinearWeighted:
                return "LinearWeighted";
            case CostFunction::Count:
                return "Count";
            }

            Throw<FatalError>("Invalid Cost Function: ", ShowValue(static_cast<int>(proc)));
        }

        std::ostream& operator<<(std::ostream& stream, CostFunction proc)
        {
            return stream << toString(proc);
        }

        Cost::Result Cost::operator()(std::vector<Generator<Instruction>::iterator>& iters) const
        {
            Cost::Result retval;
            auto         context = m_ctx.lock();
            size_t       n       = iters.size();

            for(size_t i = 0; i < n; ++i)
            {
                if(iters[i] != std::default_sentinel_t{})
                {
                    auto const&            inst   = *(iters[i]);
                    auto                   status = context->peek(inst);
                    float                  peek   = cost(inst, status);
                    std::tuple<int, float> new_val{i, peek};

                    retval.insert(
                        std::upper_bound(
                            retval.begin(),
                            retval.end(),
                            new_val,
                            [](std::tuple<int, float> lhs, std::tuple<int, float> rhs) -> bool {
                                return (std::get<1>(lhs) < std::get<1>(rhs))
                                       || (std::get<1>(lhs) == std::get<1>(rhs)
                                           && std::get<0>(lhs) < std::get<0>(rhs));
                            }),
                        new_val);
                }
            }
            return retval;
        }

        float Cost::operator()(Instruction const& inst) const
        {
            auto status = m_ctx.lock()->peek(inst);
            return cost(inst, status);
        }

        float Cost::operator()(Generator<Instruction>::iterator& iter) const
        {
            auto const& inst = *iter;
            return (*this)(inst);
        }
    }
}
