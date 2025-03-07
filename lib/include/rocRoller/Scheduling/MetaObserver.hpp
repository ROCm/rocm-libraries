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
#include <tuple>
#include <vector>

#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/CodeGen/WaitCount.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/InstructionValues/Register.hpp>
#include <rocRoller/Scheduling/Scheduling.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        template <CObserver... Types>
        class MetaObserver : public IObserver
        {
        public:
            using Tup = std::tuple<Types...>;
            enum
            {
                Size = std::tuple_size<Tup>::value
            };

            MetaObserver();
            MetaObserver(ContextPtr ctx);
            MetaObserver(Tup const& tup);

            ~MetaObserver();

            //> Speculatively predict stalls if this instruction were scheduled now.
            virtual InstructionStatus peek(Instruction const& inst) const override;

            //> Add any waitcnt or nop instructions needed before `inst` if it were to be scheduled now.
            //> Throw an exception if it can't be scheduled now.
            virtual void modify(Instruction& inst) const override;

            //> This instruction _will_ be scheduled now, record any side effects.
            virtual void observe(Instruction const& inst) override;

        private:
            Tup m_tuple;
        };

    }
}

#include <rocRoller/Scheduling/MetaObserver_impl.hpp>
