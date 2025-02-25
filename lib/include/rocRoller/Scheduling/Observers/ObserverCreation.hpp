
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
