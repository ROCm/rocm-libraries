// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

/// Skips the calling test unless the current device can wait on a stream value, the primitive
/// the stall gate arms with.
#define SKIP_UNLESS_STREAM_WAIT_VALUE()                                          \
    do                                                                           \
    {                                                                            \
        int streamWaitDevice = 0;                                                \
        int canUseStreamWaitValue = 0;                                           \
        ASSERT_EQ(hipGetDevice(&streamWaitDevice), hipSuccess);                  \
        ASSERT_EQ(hipDeviceGetAttribute(&canUseStreamWaitValue,                  \
                                        hipDeviceAttributeCanUseStreamWaitValue, \
                                        streamWaitDevice),                       \
                  hipSuccess);                                                   \
        if(canUseStreamWaitValue == 0)                                           \
        {                                                                        \
            GTEST_SKIP() << "Device does not support hipStreamWaitValue32";      \
        }                                                                        \
    } while(false)

// Test-only fault seam over the HIP stall-gate calls, shared by every test translation unit
// that benchmarks with the default timer. The link-time --wrap of the three symbols is set in
// tests/CMakeLists.txt; the wrappers are inline and [[gnu::used]] so every including file
// emits them and the linker keeps one definition.
#ifdef HIPDNN_TEST_HIP_STREAM_WAIT_FAILURES

#include <cstdint>

#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

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
    // When positive, every successful reading taken while a stall gate wait was enqueued reports
    // this span, so a stalled sample is distinguishable from an unstalled one by value.
    float stalledReadingMs = 0.0F;
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
        else if(gStreamWaitFaults.stalledReadingMs > 0.0F)
        {
            *ms = gStreamWaitFaults.stalledReadingMs;
        }
    }
    // The ingestor arms before every stalled sample, so the next wait decides again.
    gStreamWaitFaults.waitEnqueued = false;
    return status;
}
// NOLINTEND(readability-identifier-naming, bugprone-reserved-identifier)

/// Arms the fault seam for each test on a device, and disarms it after.
class StreamWaitFaultsFixture : public ::testing::Test
{
protected:
    void SetUp() override
    {
        gStreamWaitFaults = {};
        SKIP_IF_NO_DEVICES();
        gStreamWaitFaults.enabled = true;
    }

    void TearDown() override
    {
        gStreamWaitFaults = {};
        static_cast<void>(hipGetLastError());
    }
};

#endif // HIPDNN_TEST_HIP_STREAM_WAIT_FAILURES
