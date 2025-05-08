/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2021-2025 AMD ROCm(TM) Software
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

#include <rocRoller/Scheduling/MetaObserver.hpp>
#include <rocRoller/Scheduling/Scheduling.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        namespace Detail
        {
            template <typename TupType, size_t... Indices>
            TupType ConstructMapping(ContextPtr ctx, std::index_sequence<Indices...>)
            {
                auto tup     = TupType();
                auto mapping = [&ctx]<CObserver T>(T head, auto... tail) { return T(ctx); };

                return {mapping(std::get<Indices>(tup))...};
            }
        }

        template <CObserver... Types>
        MetaObserver<Types...>::MetaObserver() = default;

        template <CObserver... Types>
        MetaObserver<Types...>::MetaObserver(ContextPtr ctx)
            : m_tuple(Detail::ConstructMapping<Tup>(ctx, std::make_index_sequence<Size>{}))
        {
        }

        template <CObserver... Types>
        MetaObserver<Types...>::MetaObserver(Tup const& tup)
            : m_tuple(tup)
        {
        }

        template <CObserver... Types>
        MetaObserver<Types...>::~MetaObserver() = default;

        namespace Detail
        {
            template <CObserver T, CObserver... Rest>
            InstructionStatus Peek(Instruction const& inst, T const& obs, Rest const&... rest)
            {
                auto rv = obs.peek(inst);
                if constexpr(sizeof...(rest) > 0)
                {
                    rv.combine(Peek(inst, rest...));
                }
                return rv;
            }
        }

        template <>
        inline InstructionStatus MetaObserver<>::peek(Instruction const& inst) const
        {
            return {};
        }

        template <CObserver... Types>
        InstructionStatus MetaObserver<Types...>::peek(Instruction const& inst) const
        {
            return std::apply([&inst](auto&&... args) { return Detail::Peek(inst, args...); },
                              m_tuple);
        }

        namespace Detail
        {
            template <CObserver T, CObserver... Rest>
            void Modify(Instruction& inst, T const& obs, Rest const&... rest)
            {
                obs.modify(inst);
                if constexpr(sizeof...(rest) > 0)
                {
                    Modify(inst, rest...);
                }
            }
        }

        template <>
        inline void MetaObserver<>::modify(Instruction& inst) const
        {
            return;
        }

        template <CObserver... Types>
        void MetaObserver<Types...>::modify(Instruction& inst) const
        {
            std::apply([&inst](auto&&... args) { return Detail::Modify(inst, args...); }, m_tuple);
        }

        namespace Detail
        {
            template <CObserver T, CObserver... Rest>
            void Observe(Instruction const& inst, T& obs, Rest&... rest)
            {
                obs.observe(inst);
                if constexpr(sizeof...(rest) > 0)
                {
                    Observe(inst, rest...);
                }
            }

        }

        template <>
        inline void MetaObserver<>::observe(Instruction const& inst)
        {
        }

        template <CObserver... Types>
        void MetaObserver<Types...>::observe(Instruction const& inst)
        {
            std::apply([&inst](auto&&... args) { Detail::Observe(inst, args...); }, m_tuple);
        }
    }
}
