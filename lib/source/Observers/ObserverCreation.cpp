#include <rocRoller/Scheduling/Observers/ObserverCreation.hpp>

#include <rocRoller/Scheduling/Observers/AllocatingObserver.hpp>
#include <rocRoller/Scheduling/Observers/FileWritingObserver.hpp>
#include <rocRoller/Scheduling/Observers/RegisterLivenessObserver.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/ACCVGPRReadWrite.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/ACCVGPRWriteWrite.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/CMPXWriteExec.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/DGEMM16x16x4Write.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/DGEMM4x4x4Write.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/DLWrite.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/RegisterMapObserver.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/VALUWrite.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/XDLReadSrcC908.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/XDLReadSrcC90a.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/XDLWrite908.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/XDLWrite90a.hpp>
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
            using Gfx908Observers   = MetaObserver<ACCVGPRReadWrite,
                                                 ACCVGPRWriteWrite,
                                                 CMPXWriteExec,
                                                 VALUWrite,
                                                 XDLReadSrcC908,
                                                 XDLWrite908>;
            using Gfx90aObservers   = MetaObserver<CMPXWriteExec,
                                                 DGEMM4x4x4Write,
                                                 DGEMM16x16x4Write,
                                                 DLWrite,
                                                 VALUWrite,
                                                 XDLReadSrcC90a,
                                                 XDLWrite90a>;
            using FileObservers     = MetaObserver<FileWritingObserver>;
            using AnalysisObservers = MetaObserver<RegisterLivenessObserver>;

            static_assert(CObserver<AlwaysObservers>);
            static_assert(CObserver<Gfx908Observers>);
            static_assert(CObserver<Gfx90aObservers>);
            static_assert(CObserver<FileObservers>);
            static_assert(CObserver<AnalysisObservers>);

            PotentialObservers<FileObservers,
                               AnalysisObservers,
                               AlwaysObservers,
                               Gfx908Observers,
                               Gfx90aObservers>
                potentialObservers;
            return createObserver_Conditional(ctx, potentialObservers);
        }
    };
}
