// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

// Test-only fault seam over the HIP stall-gate calls, shared by every test translation unit
// that benchmarks with the default timer. The link-time --wrap of the three symbols is set in
// tests/CMakeLists.txt; the wrappers are inline and [[gnu::used]] so every including file
// emits them and the linker keeps one definition.
#ifdef HIPDNN_TEST_HIP_STREAM_WAIT_FAILURES

#include <cstdint>

#include <hip/hip_runtime.h>

struct HipStreamWaitFaults
{
    bool enabled = false;
    bool unsupported = false;
    int capabilityQueries = 0;
    int armCalls = 0;
    int failFromArm = 0;
    int rejectedArms = 0;
    hipError_t lastError = hipSuccess;
    // Makes this many successful hipEventElapsedTime readings taken while a stall gate
    // wait was enqueued report a finite negative span, the shape seen on Windows.
    int negativeStalledReadings = 0;
    int stalledReadings = 0;
    bool waitEnqueued = false;
};

inline thread_local HipStreamWaitFaults gStreamWaitFaults;

// GNU ld requires these symbol names for test-only HIP call wrapping.
// NOLINTBEGIN(readability-identifier-naming, bugprone-reserved-identifier)
extern "C" hipError_t __real_hipStreamWaitValue32(
    hipStream_t stream, void* ptr, uint32_t value, unsigned int flags, uint32_t mask);
extern "C" hipError_t
    __real_hipDeviceGetAttribute(int* value, hipDeviceAttribute_t attribute, int device);
extern "C" hipError_t __real_hipEventElapsedTime(float* ms, hipEvent_t start, hipEvent_t stop);

extern "C" [[gnu::used]] inline hipError_t __wrap_hipStreamWaitValue32(
    hipStream_t stream, void* ptr, uint32_t value, unsigned int flags, uint32_t mask)
{
    if(gStreamWaitFaults.enabled)
    {
        ++gStreamWaitFaults.armCalls;
        if(gStreamWaitFaults.failFromArm > 0
           && gStreamWaitFaults.armCalls >= gStreamWaitFaults.failFromArm)
        {
            // HIP rejects nullptr before enqueueing anything. Exercise a real runtime
            // error while preserving the valid stream for ordinary event timing.
            gStreamWaitFaults.lastError
                = __real_hipStreamWaitValue32(stream, nullptr, value, flags, mask);
            ++gStreamWaitFaults.rejectedArms;
            return gStreamWaitFaults.lastError;
        }
    }
    const hipError_t status = __real_hipStreamWaitValue32(stream, ptr, value, flags, mask);
    gStreamWaitFaults.waitEnqueued = status == hipSuccess;
    return status;
}

extern "C" [[gnu::used]] inline hipError_t
    __wrap_hipDeviceGetAttribute(int* value, hipDeviceAttribute_t attribute, int device)
{
    if(gStreamWaitFaults.enabled && gStreamWaitFaults.unsupported
       && attribute == hipDeviceAttributeCanUseStreamWaitValue)
    {
        ++gStreamWaitFaults.capabilityQueries;
        *value = 0;
        return hipSuccess;
    }
    return __real_hipDeviceGetAttribute(value, attribute, device);
}

extern "C" [[gnu::used]] inline hipError_t
    __wrap_hipEventElapsedTime(float* ms, hipEvent_t start, hipEvent_t stop)
{
    const hipError_t status = __real_hipEventElapsedTime(ms, start, stop);
    if(gStreamWaitFaults.enabled && status == hipSuccess && gStreamWaitFaults.waitEnqueued)
    {
        ++gStreamWaitFaults.stalledReadings;
        if(gStreamWaitFaults.negativeStalledReadings > 0)
        {
            --gStreamWaitFaults.negativeStalledReadings;
            *ms = -0.01F;
        }
    }
    // The ingestor arms before every stalled sample, so the next wait decides again.
    gStreamWaitFaults.waitEnqueued = false;
    return status;
}
// NOLINTEND(readability-identifier-naming, bugprone-reserved-identifier)

#endif // HIPDNN_TEST_HIP_STREAM_WAIT_FAILURES
