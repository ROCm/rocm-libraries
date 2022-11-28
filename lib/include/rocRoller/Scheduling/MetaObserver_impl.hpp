/**
 * @brief
 * @copyright Copyright 2021 Advanced Micro Devices, Inc.
 */

#pragma once

#include "MetaObserver.hpp"
#include "Scheduling.hpp"

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
            InstructionStatus Observe(Instruction const& inst, T& obs, Rest&... rest)
            {
                auto rv = obs.observe(inst);
                if constexpr(sizeof...(rest) > 0)
                {
                    rv.combine(Observe(inst, rest...));
                }
                return rv;
            }

        }

        template <>
        inline InstructionStatus MetaObserver<>::observe(Instruction const& inst)
        {
            return {};
        }

        template <CObserver... Types>
        InstructionStatus MetaObserver<Types...>::observe(Instruction const& inst)
        {
            return std::apply([&inst](auto&&... args) { return Detail::Observe(inst, args...); },
                              m_tuple);
        }

        template <>
        inline bool MetaObserver<>::required(std::shared_ptr<Context>)
        {
            return true;
        }

        template <CObserver... Types>
        inline bool MetaObserver<Types...>::required(std::shared_ptr<Context> ctx)
        {
            bool result = true;
            auto tup    = Tup();
            std::apply(
                [&ctx, &result]<CObserver T>(T obs, auto... rest) { result &= obs.required(ctx); },
                tup);
            return result;
        }
    }
}
