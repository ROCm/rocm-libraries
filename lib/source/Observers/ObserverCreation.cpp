#include <rocRoller/Scheduling/Observers/ObserverCreation.hpp>

#include <rocRoller/Scheduling/Observers/AllocatingObserver.hpp>
#include <rocRoller/Scheduling/Observers/FileWritingObserver.hpp>
#include <rocRoller/Scheduling/Observers/RegisterLivenessObserver.hpp>
#include <rocRoller/Scheduling/Observers/SupportedInstructionObserver.hpp>

#include <rocRoller/Scheduling/Observers/FunctionalUnit/MFMAObserver.hpp>

#include <rocRoller/Scheduling/Observers/WaitState/MFMA/ACCVGPRReadWrite.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/MFMA/ACCVGPRWriteWrite.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/MFMA/CMPXWriteExec.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/MFMA/DGEMM16x16x4Write.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/MFMA/DGEMM4x4x4Write.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/MFMA/DLWrite.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/MFMA/VALUWrite.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/MFMA/XDLReadSrcC908.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/MFMA/XDLReadSrcC90a.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/MFMA/XDLReadSrcC94x.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/MFMA/XDLWrite908.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/MFMA/XDLWrite90a.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/MFMA/XDLWrite94x.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/VALUTransWrite94x.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/VALUWriteReadlane94x.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/VALUWriteSGPRVCC94x.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/VALUWriteSGPRVMEM.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/VALUWriteVCCVDIVFMAS.hpp>
#include <rocRoller/Scheduling/Observers/WaitcntObserver.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        std::shared_ptr<Scheduling::IObserver> createObserver(ContextPtr const& ctx)
        {
            using AlwaysObservers = MetaObserver<AllocatingObserver, WaitcntObserver, MFMAObserver>;
            using Gfx908Observers = MetaObserver<ACCVGPRReadWrite,
                                                 ACCVGPRWriteWrite,
                                                 CMPXWriteExec,
                                                 VALUWrite,
                                                 VALUWriteSGPRVMEM,
                                                 VALUWriteVCCVDIVFMAS,
                                                 XDLReadSrcC908,
                                                 XDLWrite908>;
            using Gfx90aObservers = MetaObserver<CMPXWriteExec,
                                                 DGEMM4x4x4Write,
                                                 DGEMM16x16x4Write,
                                                 DLWrite,
                                                 VALUWrite,
                                                 VALUWriteSGPRVMEM,
                                                 VALUWriteVCCVDIVFMAS,
                                                 XDLReadSrcC90a,
                                                 XDLWrite90a>;
            using Gfx94xObservers = MetaObserver<CMPXWriteExec,
                                                 DGEMM4x4x4Write,
                                                 DGEMM16x16x4Write,
                                                 DLWrite,
                                                 VALUWrite,
                                                 VALUWriteReadlane94x,
                                                 VALUWriteSGPRVMEM,
                                                 VALUWriteVCCVDIVFMAS,
                                                 VALUTransWrite94x,
                                                 VALUWriteSGPRVCC94x,
                                                 XDLReadSrcC94x,
                                                 XDLWrite94x>;
            using FileObservers   = MetaObserver<FileWritingObserver>;
            using AnalysisObservers = MetaObserver<RegisterLivenessObserver>;
            using ErrorObservers    = MetaObserver<SupportedInstructionObserver>;

            static_assert(CObserver<AlwaysObservers>);
            static_assert(CObserver<Gfx908Observers>);
            static_assert(CObserver<Gfx90aObservers>);
            static_assert(CObserver<Gfx94xObservers>);
            static_assert(CObserver<FileObservers>);
            static_assert(CObserver<AnalysisObservers>);
            static_assert(CObserver<ErrorObservers>);

            PotentialObservers<FileObservers,
                               AnalysisObservers,
                               ErrorObservers,
                               AlwaysObservers,
                               Gfx908Observers,
                               Gfx90aObservers,
                               Gfx94xObservers>
                potentialObservers;
            return createObserver_Conditional(ctx, potentialObservers);
        }
    };
}
