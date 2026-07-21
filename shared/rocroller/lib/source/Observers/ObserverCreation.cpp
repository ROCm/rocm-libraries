// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/Scheduling/Observers/ObserverCreation.hpp>

#include <rocRoller/Scheduling/Observers/AllocatingObserver.hpp>
#include <rocRoller/Scheduling/Observers/FileWritingObserver.hpp>
#include <rocRoller/Scheduling/Observers/RegisterLivenessObserver.hpp>
#include <rocRoller/Scheduling/Observers/SupportedInstructionObserver.hpp>
#include <rocRoller/Scheduling/Observers/VGPRIndexingObserver.hpp>

#include <rocRoller/Scheduling/Observers/FunctionalUnit/MEMObserver.hpp>
#include <rocRoller/Scheduling/Observers/FunctionalUnit/MFMAObserver.hpp>
#include <rocRoller/Scheduling/Observers/FunctionalUnit/WMMAObserver.hpp>

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

#include <rocRoller/Scheduling/Observers/WaitState/BufferStoreDwordXXRead.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/OPSEL94x.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/VALUTransWrite94x.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/VALUWriteReadlane94x.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/VALUWriteSGPRVCC.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/VALUWriteSGPRVMEM.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/VALUWriteVCCVDIVFMAS.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/VCMPXWrite94x.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/WMMA/VALUReadDAfterWMMAOrSWMMAC.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/WMMA/VALUWriteAfterWMMAOrSWMMACRead.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/WMMA/VALUWriteAfterWMMAOrSWMMACWrite.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/WMMA/WMMAOrSWMMACReadDAfterWMMA.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/WMMA/WMMAReadSrcD.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/WMMA/WMMAWrite.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/WMMA/WMMAWriteSrcD.hpp>
#include <rocRoller/Scheduling/Observers/WaitcntObserver.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        std::shared_ptr<Scheduling::IObserver> createObserver(ContextPtr const& ctx)
        {
            PotentialObservers< // Always present
                AllocatingObserver,
                WaitcntObserver,
                MFMAObserver,
                MFMACoexecObserver,
                VMEMObserver,
                DSMEMObserver,
                WeightlessDSMemObserver,
                WMMAObserver,
                // Hazard Observers
                ACCVGPRReadWrite,
                ACCVGPRWriteWrite,
                BufferStoreDwordXXRead,
                CMPXWriteExec,
                DGEMM4x4x4Write,
                DGEMM16x16x4Write,
                DLWrite,
                OPSEL94x,
                VALUTransWrite94x,
                VALUWrite,
                VALUWriteReadlane94x,
                VALUWriteSGPRVCC,
                VALUWriteSGPRVMEM,
                VALUWriteVCCVDIVFMAS,
                VCMPXWrite94x,
                WMMAReadSrcD,
                WMMAWriteSrcD,
                WMMAWrite,
                WMMAOrSWMMACReadDAfterWMMA,
                VALUReadDAfterWMMAOrSWMMAC,
                VALUWriteAfterWMMAOrSWMMACWrite,
                VALUWriteAfterWMMAOrSWMMACRead,
                XDLReadSrcC908,
                XDLReadSrcC90a,
                XDLReadSrcC94x,
                XDLWrite908,
                XDLWrite90a,
                XDLWrite94x,
                // Other Observers
                FileWritingObserver,
                RegisterLivenessObserver,
                SupportedInstructionObserver,
                VGPRIndexingObserver>
                potentialObservers;

            return createMetaObserver(ctx, potentialObservers);
        }
    };
}
