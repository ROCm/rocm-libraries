#include <rocRoller/Scheduling/Observers/ObserverCreation.hpp>

#include <rocRoller/Scheduling/MetaObserver.hpp>
#include <rocRoller/Scheduling/Observers/AllocatingObserver.hpp>
#include <rocRoller/Scheduling/Observers/FileWritingObserver.hpp>
#include <rocRoller/Scheduling/Observers/MFMA90aObserver.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/RegisterMapObserver.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/VALUWriteFollowedByMFMARead.hpp>
#include <rocRoller/Scheduling/Observers/WaitcntObserver.hpp>

namespace rocRoller
{
    namespace Scheduling
    {

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
            if(Current::required(ctx))
            {
                Current obs(ctx);
                return createObserver_Conditional(ctx, remaining, observers..., obs);
            }
            else
            {
                return createObserver_Conditional(ctx, remaining, observers...);
            }
        }

        std::shared_ptr<Scheduling::IObserver> createObserver(ContextPtr const& ctx)
        {
            PotentialObservers<
                Scheduling::RegisterMapObserver, // NOTE: RegisterMapObserver should be first
                Scheduling::AllocatingObserver,
                Scheduling::WaitcntObserver,
                Scheduling::FileWritingObserver,
                Scheduling::MFMA90aObserver,
                Scheduling::VALUWriteFollowedByMFMARead>
                potentialObservers;
            return createObserver_Conditional(ctx, potentialObservers);
        }
    };
}
