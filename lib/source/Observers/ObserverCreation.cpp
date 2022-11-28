#include <rocRoller/Scheduling/Observers/ObserverCreation.hpp>

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
        std::shared_ptr<Scheduling::IObserver> createObserver(ContextPtr const& ctx)
        {
            using AlwaysObservers
                = MetaObserver<RegisterMapObserver, // NOTE: RegisterMapObserver should be first
                               AllocatingObserver,
                               WaitcntObserver>;
            using Gfx908Observers = MetaObserver<MFMA90aObserver, VALUWriteFollowedByMFMARead>;
            using Gfx90aObservers = MetaObserver<VALUWriteFollowedByMFMARead>;
            using FileObservers   = MetaObserver<FileWritingObserver>;

            static_assert(CObserver<AlwaysObservers>);
            static_assert(CObserver<Gfx908Observers>);
            static_assert(CObserver<Gfx90aObservers>);
            static_assert(CObserver<FileObservers>);

            PotentialObservers<FileObservers, AlwaysObservers, Gfx908Observers, Gfx90aObservers>
                potentialObservers;
            return createObserver_Conditional(ctx, potentialObservers);
        }
    };
}
