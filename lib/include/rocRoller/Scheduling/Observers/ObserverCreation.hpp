
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

        template <class Remaining = PotentialObservers<>, CObserver... Done>
        std::shared_ptr<Scheduling::IObserver> createObserver_Conditional(ContextPtr const& ctx,
                                                                          const Remaining&,
                                                                          const Done&... observers)
        {
            using MyObserver                         = Scheduling::MetaObserver<Done...>;
            std::tuple<Done...> constructedObservers = {observers...};

            return std::make_shared<MyObserver>(constructedObservers);
        }

        template <CObserver Current,
                  CObserver... TypesRemaining,
                  template <CObserver...>
                  class Remaining,
                  CObserver... Done>
        std::shared_ptr<Scheduling::IObserver>
            createObserver_Conditional(ContextPtr const& ctx,
                                       const Remaining<Current, TypesRemaining...>&,
                                       const Done&... observers)
        {
            PotentialObservers<TypesRemaining...> remaining;
            if(Current::required(ctx->targetArchitecture().target()))
            {
                Current obs(ctx);
                return createObserver_Conditional(ctx, remaining, observers..., obs);
            }
            else
            {
                return createObserver_Conditional(ctx, remaining, observers...);
            }
        }
    };
}
