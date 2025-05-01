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

#include <fstream>
#include <iostream>
#include <memory>
#include <ranges>
#include <string>

#include <rocRoller/Context_fwd.hpp>
#include <rocRoller/Scheduling/MetaObserver.hpp>
#include <rocRoller/Scheduling/Scheduling.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        /**
         * @brief Create a MetaObserver object
         *
         * This function builds a MetaObserver out of all observers that are required for the given context.
         *
         * @param ctx
         * @return std::shared_ptr<Scheduling::IObserver>
         */
        std::shared_ptr<Scheduling::IObserver> createObserver(ContextPtr const& ctx);

        template <CObserver... Types>
        struct PotentialObservers
        {
        };

        template <GPUArchitectureTarget Target, CObserver... Done>
        std::shared_ptr<Scheduling::IObserver> createMetaObserverFiltered(
            ContextPtr const& ctx, const PotentialObservers<>&, const Done&... observers)
        {
            using MyObserver                         = Scheduling::MetaObserver<Done...>;
            std::tuple<Done...> constructedObservers = {observers...};

            return std::make_shared<MyObserver>(constructedObservers);
        }

        template <GPUArchitectureTarget Target,
                  CObserver             Current,
                  CObserver... TypesRemaining,
                  CObserver... Done>
        std::shared_ptr<Scheduling::IObserver>
            createMetaObserverFiltered(ContextPtr const& ctx,
                                       const PotentialObservers<Current, TypesRemaining...>&,
                                       const Done&... observers)
        {
            static_assert(CObserverRuntime<Current> || CObserverConst<Current>);
            PotentialObservers<TypesRemaining...> remaining;

            if constexpr(CObserverConst<Current>)
            {
                if constexpr(Current::required(Target))
                {
                    Current obs(ctx);
                    return createMetaObserverFiltered<Target>(ctx, remaining, observers..., obs);
                }
                else
                {
                    return createMetaObserverFiltered<Target>(ctx, remaining, observers...);
                }
            }
            else
            {
                if(Current::runtimeRequired())
                {
                    Current obs(ctx);
                    return createMetaObserverFiltered<Target>(ctx, remaining, observers..., obs);
                }
                else
                {
                    return createMetaObserverFiltered<Target>(ctx, remaining, observers...);
                }
            }
        }

        template <size_t Idx = 0, CObserver... PotentialTypes>
        std::shared_ptr<Scheduling::IObserver>
            createMetaObserver(ContextPtr const&                            ctx,
                               const PotentialObservers<PotentialTypes...>& potentialObservers)
        {
            if constexpr(Idx < SupportedArchitectures.size())
            {
                constexpr auto Target = SupportedArchitectures.at(Idx);
                if(ctx->targetArchitecture().target() == Target)
                {
                    return createMetaObserverFiltered<Target>(ctx, potentialObservers);
                }
                return createMetaObserver<Idx + 1>(ctx, potentialObservers);
            }
            Throw<FatalError>("Unsupported Architecture",
                              ShowValue(ctx->targetArchitecture().target()));
        }
    }
}
