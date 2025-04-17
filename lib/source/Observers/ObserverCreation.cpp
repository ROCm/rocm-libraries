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

#include <rocRoller/Scheduling/Observers/ObserverCreation.hpp>

#include <rocRoller/Scheduling/Observers/AllocatingObserver.hpp>
#include <rocRoller/Scheduling/Observers/FileWritingObserver.hpp>
#include <rocRoller/Scheduling/Observers/RegisterLivenessObserver.hpp>
#include <rocRoller/Scheduling/Observers/SupportedInstructionObserver.hpp>

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
                XDLReadSrcC908,
                XDLReadSrcC90a,
                XDLReadSrcC94x,
                XDLWrite908,
                XDLWrite90a,
                XDLWrite94x,
                // Other Observers
                FileWritingObserver,
                RegisterLivenessObserver,
                SupportedInstructionObserver>
                potentialObservers;

            return createMetaObserver(ctx, potentialObservers);
        }
    };
}
