/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
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
#include "driver.hpp"

#include <miopen/errors.hpp>

void Driver::AddGpuBufferCheckFlag(InputFlags& inflags)
{
    inflags.AddInputFlag("gpubuffer_check",
                         '~',
                         "0",
                         "Controls whether gpu buffers are sanitized during execution.  This is"
                         "\nonly supported for the HIP backend."
                         "\n0  No gpu buffer sanitation done (Default)."
                         "\n1  Check for invalid gpu memory accesses before the start of"
                         "\n   the gpu buffers."
                         "\n2  Check for invalid gpu memory accesses after the end of the"
                         "\n   gpu buffers.",
                         "int");
}

void Driver::AddHipGraphFlag(InputFlags& inflags)
{
#if MIOPEN_BACKEND_HIP
    inflags.AddInputFlag(
        "use_hip_graph",
        'J',
        "0",
        "Use HIP stream capture/graph replay for steady-state iterations (HIP only). "
        "0: disabled (default), 1: enabled.",
        "int");
#else
    // Suppress unused parameter warning for non-HIP backends
    (void)inflags;
#endif
}

GPUMem::Check Driver::GetGpuBufferCheck(const InputFlags& inflags) const
{
    auto check = inflags.GetValueInt("gpubuffer_check");
    switch(check)
    {
    case 0: return GPUMem::Check::None;
    case 1: return GPUMem::Check::Front;
    case 2: return GPUMem::Check::Back;
    default:
        MIOPEN_THROW(
            miopenStatusBadParm,
            "Invalid option " + std::to_string(check) +
                " used with --gpubuffer_check. Should be 0 (none), 1 (front), or 2 (back).");
    }
}

#if MIOPEN_BACKEND_HIP

int Driver::CaptureKernel(hipGraphFuncPtrType functPtr)
{
    bool use_hip_graph = GetInputFlags().GetValueInt("use_hip_graph") != 0;
    if(use_hip_graph)
    {
        miopenEnableProfiling(GetHandle(), false);
        int rc = CaptureKernelCapturing(functPtr);
        return rc;
    }
    else
    {
        hipGraphFuncPtr = functPtr; // just memorize to execute later
        return miopenStatusSuccess;
    }
}

int Driver::CaptureKernelCapturing(hipGraphFuncPtrType functPtr)
{

    hipError_t he = hipStreamBeginCapture(q, hipStreamCaptureModeGlobal);
    if(he != hipSuccess)
    {
        return miopenStatusInternalError;
    }

    int rc = functPtr();

    he = hipStreamEndCapture(q, &hipGraph);
    if(rc != miopenStatusSuccess || he != hipSuccess)
    {
        if(hipGraph != nullptr)
        {
            (void)hipGraphDestroy(hipGraph);
            hipGraph = nullptr;
        }
        if(rc != miopenStatusSuccess)
            return rc;
        return miopenStatusInternalError;
    }

    he = hipGraphInstantiate(&hipGraphExec, hipGraph, nullptr, nullptr, 0);
    if(he != hipSuccess)
    {
        (void)hipGraphDestroy(hipGraph);
        hipGraph = nullptr;
        return miopenStatusInternalError;
    }

    he       = hipGraphDestroy(hipGraph);
    hipGraph = nullptr; // Prevent double-free in FinalizeKernel()
    if(he != hipSuccess)
    {
        return miopenStatusInternalError;
    }

    return miopenStatusSuccess;
}

int Driver::ExecuteKernel()
{

    bool use_hip_graph = GetInputFlags().GetValueInt("use_hip_graph") == 1;

    if(use_hip_graph)
    {
        (void)hipEventCreate(&hipGraphStartEvent);
        (void)hipEventCreate(&hipGraphStopEvent);
        (void)hipEventRecord(hipGraphStartEvent, q);
        hipError_t he = hipGraphLaunch(hipGraphExec, q);
        (void)hipEventRecord(hipGraphStopEvent, q);
        (void)hipEventSynchronize(hipGraphStopEvent);
        (void)hipEventElapsedTime(
            &hipGraphLastExecutionTime, hipGraphStartEvent, hipGraphStopEvent);
        if(he == hipSuccess)
        {
            return miopenStatusSuccess;
        }
        else
        {
            (void)hipGraphExecDestroy(hipGraphExec);
            hipGraphExec = nullptr;
            if(hipGraph != nullptr)
            {
                (void)hipGraphDestroy(hipGraph);
                hipGraph = nullptr;
            }
            return miopenStatusInternalError;
        }
    }
    else
    {
        int rc = hipGraphFuncPtr(); // run without HIP graph
        return rc;
    }
}

void Driver::FinalizeKernel()
{

    bool use_hip_graph = GetInputFlags().GetValueInt("use_hip_graph") == 1;

    if(use_hip_graph)
    {
        (void)hipStreamSynchronize(q);
        if(hipGraphExec != nullptr)
        {
            (void)hipGraphExecDestroy(hipGraphExec);
            hipGraphExec = nullptr;
        }
        if(hipGraph != nullptr)
        {
            (void)hipGraphDestroy(hipGraph);
            hipGraph = nullptr;
        }
        if(hipGraphStartEvent != nullptr)
        {
            (void)hipEventDestroy(hipGraphStartEvent);
            hipGraphStartEvent = nullptr;
        }
        if(hipGraphStopEvent != nullptr)
        {
            (void)hipEventDestroy(hipGraphStopEvent);
            hipGraphStopEvent = nullptr;
        }
    }
}

#endif
