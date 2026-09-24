// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/ProfilingControlDescriptor.hpp"
#include "hipdnn_backend.h"
#include "mocks/MockHandle.hpp"

#include <gtest/gtest.h>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include <chrono>
#include <hipdnn_data_sdk/utilities/StallGate.hpp>
#include <string>
#include <thread>

using namespace hipdnn_backend;
using namespace hipdnn_backend::test_utilities;
using ::testing::NiceMock;
using ::testing::Return;

class TestProfilingControlDescriptor : public ::testing::Test
{
public:
    std::shared_ptr<ProfilingControlDescriptor> getDescriptor() const
    {
        return _wrapper->asDescriptor<ProfilingControlDescriptor>();
    }

protected:
    std::unique_ptr<HipdnnBackendDescriptor> _wrapper = nullptr;

    void SetUp() override
    {
        _wrapper = createDescriptor<ProfilingControlDescriptor>();
    }

    void TearDown() override
    {
        _wrapper.reset();
    }
};

TEST_F(TestProfilingControlDescriptor, CreateDescriptor)
{
    auto desc = getDescriptor();
    ASSERT_NE(desc, nullptr);
    ASSERT_FALSE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_PROFILING_CONTROL_EXT);
}

// ============================================================================
// Base-fixture guard coverage (no GPU)
//
// Each case targets a guard that throws before any hip* call, so a handle is
// never set and no device events are created. These run on every CI runner.
// For START/STOP the guard order is checkSetArgs(type) -> elementCount ->
// handle-set -> recorded-state, so each assertion targets the first guard that
// fires for the supplied inputs.
// ============================================================================

TEST_F(TestProfilingControlDescriptor, SetStartBeforeHandleThrows)
{
    auto desc = getDescriptor();
    bool value = true;
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_PROFILING_START_EXT, HIPDNN_TYPE_BOOLEAN, 1, &value),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestProfilingControlDescriptor, SetStopBeforeHandleThrows)
{
    auto desc = getDescriptor();
    bool value = true;
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_PROFILING_STOP_EXT, HIPDNN_TYPE_BOOLEAN, 1, &value),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestProfilingControlDescriptor, SetAttributeWrongElementCountThrows)
{
    auto desc = getDescriptor();
    bool value = true;
    // elementCount=2 fails the count guard (after the type check passes).
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_PROFILING_START_EXT, HIPDNN_TYPE_BOOLEAN, 2, &value),
        HIPDNN_STATUS_BAD_PARAM);
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_PROFILING_STOP_EXT, HIPDNN_TYPE_BOOLEAN, 2, &value),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestProfilingControlDescriptor, SetAttributeTypeMismatchThrows)
{
    auto desc = getDescriptor();
    bool value = true;
    // Wrong value type fails checkSetArgs (the first guard) for a boolean attr.
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_PROFILING_START_EXT, HIPDNN_TYPE_INT64, 1, &value),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestProfilingControlDescriptor, SetAttributeUnsupportedNameThrows)
{
    auto desc = getDescriptor();
    bool value = true;
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_BOOLEAN, 1, &value),
        HIPDNN_STATUS_NOT_SUPPORTED);
}

// STALL_USED_EXT is read-only: enforced by simply never appearing in setAttribute's switch,
// so a set attempt falls to the same unsupported-name guard as any other unknown name.
TEST_F(TestProfilingControlDescriptor, SetStallUsedThrowsNotSupported)
{
    auto desc = getDescriptor();
    bool value = true;
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_PROFILING_STALL_USED_EXT, HIPDNN_TYPE_BOOLEAN, 1, &value),
        HIPDNN_STATUS_NOT_SUPPORTED);
}

TEST_F(TestProfilingControlDescriptor, GetAttributeBeforeFinalizeThrows)
{
    auto desc = getDescriptor();
    float elapsed = 0.0f;
    int64_t elementCount = 0;
    ASSERT_THROW_HIPDNN_STATUS(
        desc->getAttribute(
            HIPDNN_ATTR_PROFILING_ELAPSED_MS_EXT, HIPDNN_TYPE_FLOAT, 1, &elementCount, &elapsed),
        HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestProfilingControlDescriptor, GetStallUsedBeforeFinalizeThrows)
{
    auto desc = getDescriptor();
    bool stallUsed = true;
    int64_t elementCount = 0;
    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_PROFILING_STALL_USED_EXT,
                                                  HIPDNN_TYPE_BOOLEAN,
                                                  1,
                                                  &elementCount,
                                                  &stallUsed),
                               HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestProfilingControlDescriptor, FinalizeBeforeHandleThrows)
{
    auto desc = getDescriptor();
    // Fresh descriptor: not finalized, but no handle/events created.
    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

// ============================================================================
// GPU-requiring lifecycle coverage
//
// Setting a handle creates real hipEvents on the device, so these tests need a
// device and are skipped on no-GPU runners via SKIP_IF_NO_DEVICES(). Mirrors
// TestGpuEngineHeuristicDescriptor.
// ============================================================================

class TestGpuProfilingControlDescriptor : public TestProfilingControlDescriptor
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        TestProfilingControlDescriptor::SetUp();
        ASSERT_EQ(hipStreamCreate(&_testStream), hipSuccess);
        _mockHandle = std::make_unique<NiceMock<MockHandle>>();
        ON_CALL(*_mockHandle, getStream()).WillByDefault(Return(_testStream));
    }

    void TearDown() override
    {
        if(_timingScratch != nullptr)
        {
            EXPECT_EQ(hipFree(_timingScratch), hipSuccess);
            _timingScratch = nullptr;
        }
        _mockHandle.reset();
        if(_testStream != nullptr)
        {
            EXPECT_EQ(hipStreamDestroy(_testStream), hipSuccess);
            _testStream = nullptr;
        }
        TestProfilingControlDescriptor::TearDown();
    }

    // A successful timed span needs enough same-stream device work to clear the
    // event timer's resolution. Allocate before arming: hipMalloc can synchronize.
    static constexpr size_t TIMED_WORK_BYTES = size_t{4} * 1024 * 1024;
    void prepareTimedWork()
    {
        if(_timingScratch == nullptr)
        {
            ASSERT_EQ(hipMalloc(&_timingScratch, TIMED_WORK_BYTES), hipSuccess);
        }
    }

    void enqueueTimedWork()
    {
        ASSERT_NE(_timingScratch, nullptr);
        ASSERT_EQ(hipMemsetAsync(_timingScratch, 0, TIMED_WORK_BYTES, _testStream), hipSuccess);
    }

    // Sets the handle on the descriptor, which creates the device events.
    void setHandle(const std::shared_ptr<ProfilingControlDescriptor>& desc) const
    {
        hipdnnHandle* handlePtr = _mockHandle.get();
        desc->setAttribute(HIPDNN_ATTR_PROFILING_HANDLE_EXT,
                           HIPDNN_TYPE_HANDLE,
                           1,
                           static_cast<const void*>(&handlePtr));
    }

    static void recordStart(const std::shared_ptr<ProfilingControlDescriptor>& desc)
    {
        bool value = true;
        desc->setAttribute(HIPDNN_ATTR_PROFILING_START_EXT, HIPDNN_TYPE_BOOLEAN, 1, &value);
    }

    static void recordStop(const std::shared_ptr<ProfilingControlDescriptor>& desc)
    {
        bool value = true;
        desc->setAttribute(HIPDNN_ATTR_PROFILING_STOP_EXT, HIPDNN_TYPE_BOOLEAN, 1, &value);
    }

    static void armStall(const std::shared_ptr<ProfilingControlDescriptor>& desc)
    {
        bool value = true;
        desc->setAttribute(HIPDNN_ATTR_PROFILING_STALL_ARM_EXT, HIPDNN_TYPE_BOOLEAN, 1, &value);
    }

    static void releaseStall(const std::shared_ptr<ProfilingControlDescriptor>& desc)
    {
        bool value = true;
        desc->setAttribute(HIPDNN_ATTR_PROFILING_STALL_RELEASE_EXT, HIPDNN_TYPE_BOOLEAN, 1, &value);
    }

    static void resetContext(const std::shared_ptr<ProfilingControlDescriptor>& desc)
    {
        bool value = true;
        desc->setAttribute(HIPDNN_ATTR_PROFILING_RESET_EXT, HIPDNN_TYPE_BOOLEAN, 1, &value);
    }

    // The stall needs hipStreamWaitValue32; a device without it degrades to the
    // unstalled path, which the timing assertions below would read as a failure. Skip
    // only that genuine no-support case. isUsable() is also false when a HIP call in the
    // constructor itself failed (hipGetDevice, the attribute query, or
    // hipExtMallocWithFlags) -- StallGate documents lastError() == hipSuccess as the
    // marker for "the query ran and reported no support"; anything else is a broken test
    // environment, not a capability gap, and must fail loudly rather than silently skip
    // every stall-gate test on this runner.
    static bool stallGateAvailable()
    {
        const hipdnn_data_sdk::utilities::StallGate gate;
        if(gate.isUsable())
        {
            return true;
        }
        EXPECT_EQ(gate.lastError(), hipSuccess)
            << "StallGate construction failed: " << gate.lastOperation() << " returned "
            << hipGetErrorString(gate.lastError());
        return false;
    }

    std::unique_ptr<NiceMock<MockHandle>> _mockHandle = nullptr;
    hipStream_t _testStream = nullptr;
    void* _timingScratch = nullptr;
};

TEST_F(TestGpuProfilingControlDescriptor, HappyPathCompletesLifecycle)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(setHandle(desc));
    ASSERT_NO_FATAL_FAILURE(prepareTimedWork());
    ASSERT_NO_THROW(recordStart(desc));
    ASSERT_NO_FATAL_FAILURE(enqueueTimedWork());
    ASSERT_NO_THROW(recordStop(desc));
    ASSERT_NO_THROW(desc->finalize());

    // getAttribute(ELAPSED_MS) round-trips: no throw, one element written.
    // With real device work in the measured span, the elapsed value is valid.
    float elapsed = -1.0f;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_PROFILING_ELAPSED_MS_EXT, HIPDNN_TYPE_FLOAT, 1, &elementCount, &elapsed));
    EXPECT_EQ(elementCount, 1);
    EXPECT_GE(elapsed, 0.0f);
}

TEST_F(TestGpuProfilingControlDescriptor, RebindingCannotChangeTheTimingContext)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(setHandle(desc));
    ASSERT_NO_FATAL_FAILURE(prepareTimedWork());
    ASSERT_NO_THROW(recordStart(desc));
    ASSERT_NO_FATAL_FAILURE(enqueueTimedWork());
    ASSERT_NO_THROW(recordStop(desc));
    ASSERT_NO_THROW(desc->finalize());
    ASSERT_NO_THROW(resetContext(desc));
    ASSERT_NO_THROW(setHandle(desc));

    NiceMock<MockHandle> other;
    ON_CALL(other, getStream()).WillByDefault(Return(_testStream));
    hipdnnHandle* handlePtr = &other;
    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_PROFILING_HANDLE_EXT,
                                                  HIPDNN_TYPE_HANDLE,
                                                  1,
                                                  static_cast<const void*>(&handlePtr)),
                               HIPDNN_STATUS_BAD_PARAM);

    ON_CALL(*_mockHandle, getStream()).WillByDefault(Return(nullptr));
    ASSERT_THROW_HIPDNN_STATUS(resetContext(desc), HIPDNN_STATUS_BAD_PARAM);
    ON_CALL(*_mockHandle, getStream()).WillByDefault(Return(_testStream));
    ASSERT_NO_THROW(resetContext(desc));
    ASSERT_NO_THROW(recordStart(desc));
    ASSERT_NO_FATAL_FAILURE(enqueueTimedWork());
    ASSERT_NO_THROW(recordStop(desc));
    ASSERT_NO_THROW(desc->finalize());
    EXPECT_TRUE(desc->isFinalized());
}

TEST_F(TestGpuProfilingControlDescriptor, StartRecordedTwiceThrows)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(setHandle(desc));
    ASSERT_NO_THROW(recordStart(desc));
    bool value = true;
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_PROFILING_START_EXT, HIPDNN_TYPE_BOOLEAN, 1, &value),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestGpuProfilingControlDescriptor, StopBeforeStartThrows)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(setHandle(desc));
    bool value = true;
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_PROFILING_STOP_EXT, HIPDNN_TYPE_BOOLEAN, 1, &value),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestGpuProfilingControlDescriptor, StopRecordedTwiceThrows)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(setHandle(desc));
    ASSERT_NO_THROW(recordStart(desc));
    ASSERT_NO_THROW(recordStop(desc));
    bool value = true;
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_PROFILING_STOP_EXT, HIPDNN_TYPE_BOOLEAN, 1, &value),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestGpuProfilingControlDescriptor, SetAttributeAfterFinalizeThrows)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(setHandle(desc));
    ASSERT_NO_FATAL_FAILURE(prepareTimedWork());
    ASSERT_NO_THROW(recordStart(desc));
    ASSERT_NO_FATAL_FAILURE(enqueueTimedWork());
    ASSERT_NO_THROW(recordStop(desc));
    ASSERT_NO_THROW(desc->finalize());
    bool value = true;
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_PROFILING_START_EXT, HIPDNN_TYPE_BOOLEAN, 1, &value),
        HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestGpuProfilingControlDescriptor, FinalizeAlreadyFinalizedThrows)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(setHandle(desc));
    ASSERT_NO_FATAL_FAILURE(prepareTimedWork());
    ASSERT_NO_THROW(recordStart(desc));
    ASSERT_NO_FATAL_FAILURE(enqueueTimedWork());
    ASSERT_NO_THROW(recordStop(desc));
    ASSERT_NO_THROW(desc->finalize());
    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestGpuProfilingControlDescriptor, FinalizeWithoutStopRecordedThrows)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(setHandle(desc));
    ASSERT_NO_THROW(recordStart(desc));
    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestGpuProfilingControlDescriptor, DeviceSyncSucceeds)
{
    auto desc = getDescriptor();
    bool value = true;
    ASSERT_NO_THROW(
        desc->setAttribute(HIPDNN_ATTR_PROFILING_DEVICE_SYNC_EXT, HIPDNN_TYPE_BOOLEAN, 1, &value));
}

TEST_F(TestGpuProfilingControlDescriptor, GetAttributeUnsupportedNameThrows)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(setHandle(desc));
    ASSERT_NO_FATAL_FAILURE(prepareTimedWork());
    ASSERT_NO_THROW(recordStart(desc));
    ASSERT_NO_FATAL_FAILURE(enqueueTimedWork());
    ASSERT_NO_THROW(recordStop(desc));
    ASSERT_NO_THROW(desc->finalize());

    // On a finalized descriptor, an unrelated attribute name hits the
    // unsupported-name guard past the finalized check.
    int64_t value = 0;
    int64_t elementCount = 0;
    ASSERT_THROW_HIPDNN_STATUS(
        desc->getAttribute(
            HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, &elementCount, &value),
        HIPDNN_STATUS_NOT_SUPPORTED);
}

// The defect the stall gate fixes: a start event recorded on an idle stream completes
// immediately, so every microsecond the host spends before the work is queued lands
// inside the measured span. The sleep stands in for descriptor validation, dispatch,
// and logging, which is host work of the same shape but not a fixed duration.
//
// Both runs enqueue the same above-resolution GPU work. Only the unstalled run
// can include the host delay in its elapsed time.
TEST_F(TestGpuProfilingControlDescriptor, StallGateExcludesHostSubmissionDelay)
{
    if(!stallGateAvailable())
    {
        GTEST_SKIP() << "Device does not support hipStreamWaitValue32";
    }

    constexpr auto HOST_DELAY = std::chrono::milliseconds(20);
    // 256 B of memset work is small enough that its device duration can round to zero
    // or even slip negative against ~1 us timer resolution, masking whatever the
    // stall gate does or does not exclude. 4 MiB clears that floor with margin while
    // still finishing well under the <5 ms stalled bound below (see
    // TestAutotunePlugin's identical TIMING_SCRATCH_SIZE precedent).
    constexpr size_t BUFFER_BYTES = size_t{4} * 1024 * 1024;

    void* buffer = nullptr;
    ASSERT_EQ(hipMalloc(&buffer, BUFFER_BYTES), hipSuccess);

    const auto measure = [&](bool useStall) {
        // A fresh descriptor per run: the fixture's descriptor is finalized by the first
        // measurement, and setAttribute rejects everything after finalize.
        const auto wrapper = createDescriptor<ProfilingControlDescriptor>();
        const auto desc = wrapper->asDescriptor<ProfilingControlDescriptor>();
        setHandle(desc);
        if(useStall)
        {
            armStall(desc);
        }
        recordStart(desc);
        std::this_thread::sleep_for(HOST_DELAY);
        EXPECT_EQ(hipMemsetAsync(buffer, 0, BUFFER_BYTES, _testStream), hipSuccess);
        recordStop(desc);
        if(useStall)
        {
            releaseStall(desc);
        }
        desc->finalize();

        float elapsed = -1.0f;
        int64_t elementCount = 0;
        desc->getAttribute(
            HIPDNN_ATTR_PROFILING_ELAPSED_MS_EXT, HIPDNN_TYPE_FLOAT, 1, &elementCount, &elapsed);
        return elapsed;
    };

    float unstalledMs = 0.0f;
    ASSERT_NO_THROW(unstalledMs = measure(/*useStall=*/false));
    float stalledMs = 0.0f;
    ASSERT_NO_THROW(stalledMs = measure(/*useStall=*/true));

    EXPECT_EQ(hipFree(buffer), hipSuccess);

    // Reported unconditionally: a timing bound that flakes in CI is not diagnosable
    // without the two numbers that produced it.
    GTEST_LOG_(INFO) << "unstalled=" << unstalledMs << " ms, stalled=" << stalledMs << " ms";
    EXPECT_GE(unstalledMs, 0.0f) << "unstalled timing returned an invalid elapsed value";
    EXPECT_GE(stalledMs, 0.0f) << "stalled timing returned an invalid elapsed value";

#if !defined(_WIN32)
    // The 20 ms host sleep lands inside the unstalled span on Linux. Windows/PAL
    // can defer event submission until a later flush, so the sleep can be absent
    // from both spans there.
    EXPECT_GE(unstalledMs, 15.0f) << "unstalled timing did not absorb the host delay";
    EXPECT_LT(stalledMs, unstalledMs) << "stalled timing did not exclude the host delay";
#endif
    // The gate's measured span excludes the sleep on every supported platform.
    EXPECT_LT(stalledMs, 5.0f) << "stalled timing still includes the host delay";
}

// The deadlock the watchdog exists for, reproduced exactly: work inside the timed
// region blocks the host on the stalled stream, and only the host can release. Without
// the watchdog this test hangs forever. With it, the write that ends the stall is also
// what the blocked host is waiting on, so hipStreamSynchronize returns.
//
// A timeout is scoped to its own attempt, not sticky: once the drained stream proves
// the watchdog's release actually retired, the very same gate can arm again, and an
// unrelated new gate is never affected in the first place.
TEST_F(TestGpuProfilingControlDescriptor, WatchdogBreaksSelfInflictedDeadlock)
{
    if(!stallGateAvailable())
    {
        GTEST_SKIP() << "Device does not support hipStreamWaitValue32";
    }

    constexpr auto TIMEOUT = std::chrono::milliseconds(300);
    hipdnn_data_sdk::utilities::StallGate gate(TIMEOUT);
    ASSERT_TRUE(gate.isUsable());
    // Sampled before arm(), because arm() sets the deadline to its own now() + TIMEOUT.
    // Starting the clock after arm() returns puts that gap outside the measured window, so
    // `waited` comes out just under TIMEOUT and the bound below only holds when watchdog
    // wakeup latency happens to cover the difference. Measured under TSAN: 299.995 ms.
    const auto begin = std::chrono::steady_clock::now();
    ASSERT_TRUE(gate.arm(_testStream));

    // Never returns unless something else releases the gate.
    EXPECT_EQ(hipStreamSynchronize(_testStream), hipSuccess);
    const auto waited = std::chrono::steady_clock::now() - begin;

    EXPECT_TRUE(gate.timedOut()) << "watchdog released but did not report it";
    EXPECT_GE(waited, TIMEOUT) << "watchdog fired before its deadline";

    // The synchronize above proves the stream drained past the released wait packet, so
    // the same gate can re-arm it immediately -- no sticky latch survives a timeout.
    ASSERT_TRUE(gate.arm(_testStream)) << "a drained stream must allow the same gate to re-arm";
    gate.release();
    ASSERT_EQ(hipStreamSynchronize(_testStream), hipSuccess);
    EXPECT_FALSE(gate.timedOut()) << "a fresh arm attempt clears the previous timeout";

    // An independent new gate was never in scope for the first gate's timeout at all.
    hipdnn_data_sdk::utilities::StallGate freshGate;
    ASSERT_TRUE(freshGate.isUsable());
    EXPECT_TRUE(freshGate.arm(_testStream))
        << "an independent new gate must not inherit another gate's timeout";
    freshGate.release();
    ASSERT_EQ(hipStreamSynchronize(_testStream), hipSuccess);
}

TEST_F(TestGpuProfilingControlDescriptor, NonPositiveTimeoutUsesDefaultBudget)
{
    if(!stallGateAvailable())
    {
        GTEST_SKIP() << "Device does not support hipStreamWaitValue32";
    }

    for(const auto timeout : {std::chrono::milliseconds(0), std::chrono::milliseconds(-1)})
    {
        hipdnn_data_sdk::utilities::StallGate gate(timeout);
        ASSERT_TRUE(gate.arm(_testStream));
        // Give an immediate watchdog deadline time to fire, well below the default 2 s.
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        gate.release();
        ASSERT_EQ(hipStreamSynchronize(_testStream), hipSuccess);
        EXPECT_FALSE(gate.timedOut());
    }
}

TEST_F(TestGpuProfilingControlDescriptor, DestructionReleasesAndRetiresWaitOnEitherStream)
{
    if(!stallGateAvailable())
    {
        GTEST_SKIP() << "Device does not support hipStreamWaitValue32";
    }

    for(const auto stream : {hipStream_t{nullptr}, _testStream})
    {
        hipEvent_t stop = nullptr;
        ASSERT_EQ(hipEventCreate(&stop), hipSuccess);
        const hipdnn_backend::HipEventGuard stopGuard(stop);
        {
            hipdnn_data_sdk::utilities::StallGate gate;
            ASSERT_TRUE(gate.arm(stream));
            ASSERT_EQ(hipEventRecord(stop, stream), hipSuccess);
        }
        // No explicit release or synchronization: destruction must retire the waiter.
        EXPECT_EQ(hipEventQuery(stop), hipSuccess);
    }
}

// A normal release must not depend on another GPU command making forward progress:
// some runtimes cannot execute a stream write while another stream waits on the signal.
// Repeating the cycle also proves that arm() resets the host-written signal for reuse.
TEST_F(TestGpuProfilingControlDescriptor, WatchdogDoesNotFireOnNormalRelease)
{
    if(!stallGateAvailable())
    {
        GTEST_SKIP() << "Device does not support hipStreamWaitValue32";
    }

    hipdnn_data_sdk::utilities::StallGate gate(std::chrono::milliseconds(5000));
    ASSERT_TRUE(gate.isUsable());
    for(int iteration = 0; iteration < 2; ++iteration)
    {
        ASSERT_TRUE(gate.arm(_testStream));
        gate.release();
        ASSERT_EQ(hipStreamSynchronize(_testStream), hipSuccess);
        EXPECT_FALSE(gate.timedOut());
    }
}

// A watchdog release must be visible through the public descriptor, so an external
// caller can discard the sample instead of averaging a timeout into its results.
TEST_F(TestGpuProfilingControlDescriptor, TimedOutAttributeIsFalseForAHealthyMeasurement)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(setHandle(desc));
    ASSERT_NO_FATAL_FAILURE(prepareTimedWork());
    ASSERT_NO_THROW(armStall(desc));
    ASSERT_NO_THROW(recordStart(desc));
    ASSERT_NO_FATAL_FAILURE(enqueueTimedWork());
    ASSERT_NO_THROW(recordStop(desc));
    ASSERT_NO_THROW(releaseStall(desc));
    ASSERT_NO_THROW(desc->finalize());

    bool timedOut = true;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_PROFILING_STALL_TIMED_OUT_EXT,
                                       HIPDNN_TYPE_BOOLEAN,
                                       1,
                                       &elementCount,
                                       &timedOut));
    EXPECT_FALSE(timedOut);
}

// An armed gate holds the stop event unsignalled, so a caller that arms and then hits an
// error path before releasing would hang in hipEventSynchronize forever. finalize()
// releases first. The watchdog also ends such a stall, so elapsed time alone cannot tell
// the two apart: STALL_TIMED_OUT_EXT must be false, which holds only if finalize()
// released.
TEST_F(TestGpuProfilingControlDescriptor, FinalizeReleasesUnreleasedStall)
{
    if(!stallGateAvailable())
    {
        GTEST_SKIP() << "Device does not support hipStreamWaitValue32";
    }

    auto desc = getDescriptor();
    ASSERT_NO_THROW(setHandle(desc));
    ASSERT_NO_FATAL_FAILURE(prepareTimedWork());
    ASSERT_NO_THROW(armStall(desc));
    ASSERT_NO_THROW(recordStart(desc));
    ASSERT_NO_FATAL_FAILURE(enqueueTimedWork());
    ASSERT_NO_THROW(recordStop(desc));
    ASSERT_NO_THROW(desc->finalize());

    bool timedOut = true;
    int64_t timedOutCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_PROFILING_STALL_TIMED_OUT_EXT,
                                       HIPDNN_TYPE_BOOLEAN,
                                       1,
                                       &timedOutCount,
                                       &timedOut));
    EXPECT_FALSE(timedOut) << "the watchdog released the stall, not finalize()";

    float elapsed = -1.0f;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_PROFILING_ELAPSED_MS_EXT, HIPDNN_TYPE_FLOAT, 1, &elementCount, &elapsed));
    EXPECT_GE(elapsed, 0.0f);
}

// finalize() must release an armed gate before any precondition check can throw, not
// only once every check has passed: a caller who arms and then hits a precondition error
// (forgot to record start) would otherwise leave the wait packet stalling the stream
// forever, with no later code path left to release it.
TEST_F(TestGpuProfilingControlDescriptor, FinalizeReleasesArmedStallBeforePreconditionThrows)
{
    if(!stallGateAvailable())
    {
        GTEST_SKIP() << "Device does not support hipStreamWaitValue32";
    }

    auto desc = getDescriptor();
    ASSERT_NO_THROW(setHandle(desc));
    ASSERT_NO_THROW(armStall(desc));
    // Start deliberately not recorded: finalize() must throw on that precondition, but
    // only after releasing the gate armed above.
    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);

    // A queued wait can remain not-ready briefly after the host release. Synchronize
    // instead of sampling it with hipStreamQuery; if finalize() did not release the
    // gate, this blocks until the watchdog and the timed-out flag below exposes it.
    ASSERT_EQ(hipStreamSynchronize(_testStream), hipSuccess);

    ASSERT_NO_FATAL_FAILURE(prepareTimedWork());
    ASSERT_NO_THROW(recordStart(desc));
    ASSERT_NO_FATAL_FAILURE(enqueueTimedWork());
    ASSERT_NO_THROW(recordStop(desc));
    ASSERT_NO_THROW(desc->finalize());
    bool timedOut = true;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_PROFILING_STALL_TIMED_OUT_EXT,
                                       HIPDNN_TYPE_BOOLEAN,
                                       1,
                                       &elementCount,
                                       &timedOut));
    EXPECT_FALSE(timedOut) << "the watchdog released the gate instead of the failed finalize()";
}

// Releasing a gate that was never armed is a no-op success, so a caller that arms
// conditionally need not track whether the arm took effect.
TEST_F(TestGpuProfilingControlDescriptor, StallReleaseWithoutArmSucceeds)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(releaseStall(desc));
}

// Arming creates no events of its own, so it must reject a descriptor with no handle
// rather than stalling a null stream.
TEST_F(TestGpuProfilingControlDescriptor, StallArmBeforeHandleThrows)
{
    auto desc = getDescriptor();
    bool value = true;
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_PROFILING_STALL_ARM_EXT, HIPDNN_TYPE_BOOLEAN, 1, &value),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestGpuProfilingControlDescriptor, StallArmTwiceThrows)
{
    if(!stallGateAvailable())
    {
        GTEST_SKIP() << "Device does not support hipStreamWaitValue32";
    }

    auto desc = getDescriptor();
    ASSERT_NO_THROW(setHandle(desc));
    ASSERT_NO_THROW(armStall(desc));
    bool value = true;
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_PROFILING_STALL_ARM_EXT, HIPDNN_TYPE_BOOLEAN, 1, &value),
        HIPDNN_STATUS_BAD_PARAM);
    ASSERT_NO_THROW(releaseStall(desc));
}

// Once start has recorded the begin timestamp, arming can no longer exclude host
// submission delay from this measurement -- that delay already happened -- so a late
// arm is a lifecycle error rather than a silently-partial stall.
TEST_F(TestGpuProfilingControlDescriptor, StallArmAfterStartThrows)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(setHandle(desc));
    ASSERT_NO_THROW(recordStart(desc));
    bool value = true;
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_PROFILING_STALL_ARM_EXT, HIPDNN_TYPE_BOOLEAN, 1, &value),
        HIPDNN_STATUS_BAD_PARAM);
}

// arm() succeeding for this measurement is what STALL_USED_EXT reports, independent of the
// armed state after release() (finalize() always releases first, so armed is always false
// by the time this is readable).
TEST_F(TestGpuProfilingControlDescriptor, StallUsedTrueWhenArmSucceeds)
{
    if(!stallGateAvailable())
    {
        GTEST_SKIP() << "Device does not support hipStreamWaitValue32";
    }

    auto desc = getDescriptor();
    ASSERT_NO_THROW(setHandle(desc));
    ASSERT_NO_FATAL_FAILURE(prepareTimedWork());
    ASSERT_NO_THROW(armStall(desc));
    ASSERT_NO_THROW(recordStart(desc));
    ASSERT_NO_FATAL_FAILURE(enqueueTimedWork());
    ASSERT_NO_THROW(recordStop(desc));
    ASSERT_NO_THROW(releaseStall(desc));
    ASSERT_NO_THROW(desc->finalize());

    bool stallUsed = false;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_PROFILING_STALL_USED_EXT, HIPDNN_TYPE_BOOLEAN, 1, &elementCount, &stallUsed));
    EXPECT_TRUE(stallUsed);
    EXPECT_EQ(elementCount, 1);
}

// No STALL_ARM_EXT call at all: the default is false, matching the plain unstalled
// lifecycle a caller gets by simply never setting the attribute.
TEST_F(TestGpuProfilingControlDescriptor, StallUsedFalseWithoutArm)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(setHandle(desc));
    ASSERT_NO_FATAL_FAILURE(prepareTimedWork());
    ASSERT_NO_THROW(recordStart(desc));
    ASSERT_NO_FATAL_FAILURE(enqueueTimedWork());
    ASSERT_NO_THROW(recordStop(desc));
    ASSERT_NO_THROW(desc->finalize());

    bool stallUsed = true;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_PROFILING_STALL_USED_EXT, HIPDNN_TYPE_BOOLEAN, 1, &elementCount, &stallUsed));
    EXPECT_FALSE(stallUsed);
}

// A watchdog release taken through the descriptor itself (not a raw gate): arm() had
// succeeded, so STALL_USED_EXT must stay true even though the watchdog -- not the caller --
// released the stall and STALL_TIMED_OUT_EXT is therefore also true. The two attributes are
// independent: neither implies the other, and a caller must check both to classify a
// measurement (used-and-healthy vs. used-but-invalid vs. never-stalled).
TEST_F(TestGpuProfilingControlDescriptor, StallUsedTrueAndTimedOutTrueOnWatchdogRelease)
{
    if(!stallGateAvailable())
    {
        GTEST_SKIP() << "Device does not support hipStreamWaitValue32";
    }

    // The descriptor's stall gate uses the default (2 s) watchdog timeout, so tripping it
    // for real costs a couple of seconds.

    auto desc = getDescriptor();
    ASSERT_NO_THROW(setHandle(desc));
    ASSERT_NO_THROW(armStall(desc));
    ASSERT_NO_THROW(recordStart(desc));

    // Blocks the host on the still-stalled stream; only the descriptor's own watchdog can
    // release it -- the exact deadlock the watchdog exists to break.
    EXPECT_EQ(hipStreamSynchronize(_testStream), hipSuccess);

    ASSERT_NO_THROW(recordStop(desc));
    ASSERT_NO_THROW(releaseStall(desc)); // no-op: the watchdog already released it
    ASSERT_NO_THROW(desc->finalize());

    bool stallUsed = false;
    bool timedOut = false;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_PROFILING_STALL_USED_EXT, HIPDNN_TYPE_BOOLEAN, 1, &elementCount, &stallUsed));
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_PROFILING_STALL_TIMED_OUT_EXT,
                                       HIPDNN_TYPE_BOOLEAN,
                                       1,
                                       &elementCount,
                                       &timedOut));

    EXPECT_TRUE(stallUsed) << "arm() succeeded, so STALL_USED_EXT must stay true";
    EXPECT_TRUE(timedOut) << "the watchdog released the stall, not the caller";
}

// RESET_EXT is the only attribute PROFILING_CONTROL accepts once finalized: it
// un-finalizes the context so a second full measurement can run on the same handle,
// stream, events, and gate.
TEST_F(TestGpuProfilingControlDescriptor, ResetAfterSuccessfulMeasurementAllowsReuse)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(resetContext(desc));
    ASSERT_NO_THROW(setHandle(desc));
    ASSERT_NO_FATAL_FAILURE(prepareTimedWork());
    ASSERT_NO_THROW(recordStart(desc));
    ASSERT_NO_FATAL_FAILURE(enqueueTimedWork());
    ASSERT_NO_THROW(recordStop(desc));
    ASSERT_NO_THROW(desc->finalize());
    ASSERT_TRUE(desc->isFinalized());

    ASSERT_NO_THROW(resetContext(desc));
    EXPECT_FALSE(desc->isFinalized()) << "reset() must un-finalize the context for reuse";

    // Start/stop are not stuck "already recorded" from the first pass.
    ASSERT_NO_THROW(recordStart(desc));
    ASSERT_NO_FATAL_FAILURE(enqueueTimedWork());
    ASSERT_NO_THROW(recordStop(desc));
    ASSERT_NO_THROW(desc->finalize());

    float elapsed = -1.0f;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_PROFILING_ELAPSED_MS_EXT, HIPDNN_TYPE_FLOAT, 1, &elementCount, &elapsed));
    EXPECT_GE(elapsed, 0.0f);
}

// The staleness this guards against: STALL_TIMED_OUT_EXT is latched at finalize(), not
// read live off the gate, so a later measurement that reuses the same gate but never
// arms it cannot inherit an earlier measurement's timeout. It also proves a timeout
// does not disable the gate for the descriptor's own later, stalled reuse.
TEST_F(TestGpuProfilingControlDescriptor, ResetAfterTimeoutThenUnstalledThenStalledReuse)
{
    if(!stallGateAvailable())
    {
        GTEST_SKIP() << "Device does not support hipStreamWaitValue32";
    }

    auto desc = getDescriptor();
    ASSERT_NO_THROW(setHandle(desc));
    ASSERT_NO_THROW(armStall(desc));
    ASSERT_NO_THROW(recordStart(desc));
    // Blocks the host on the still-stalled stream; only the watchdog can release it.
    EXPECT_EQ(hipStreamSynchronize(_testStream), hipSuccess);
    ASSERT_NO_THROW(recordStop(desc));
    ASSERT_NO_THROW(releaseStall(desc)); // no-op: the watchdog already released it
    ASSERT_NO_THROW(desc->finalize());

    bool timedOut = false;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_PROFILING_STALL_TIMED_OUT_EXT,
                                       HIPDNN_TYPE_BOOLEAN,
                                       1,
                                       &elementCount,
                                       &timedOut));
    ASSERT_TRUE(timedOut) << "setup did not actually trip the watchdog";

    // Reuse unstalled: reset, then a full lifecycle with no STALL_ARM_EXT at all.
    ASSERT_NO_THROW(resetContext(desc));
    ASSERT_NO_FATAL_FAILURE(prepareTimedWork());
    ASSERT_NO_THROW(recordStart(desc));
    ASSERT_NO_FATAL_FAILURE(enqueueTimedWork());
    ASSERT_NO_THROW(recordStop(desc));
    ASSERT_NO_THROW(desc->finalize());

    timedOut = true;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_PROFILING_STALL_TIMED_OUT_EXT,
                                       HIPDNN_TYPE_BOOLEAN,
                                       1,
                                       &elementCount,
                                       &timedOut));
    EXPECT_FALSE(timedOut) << "an unstalled reuse must not inherit the previous timeout";

    bool stallUsed = true;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_PROFILING_STALL_USED_EXT, HIPDNN_TYPE_BOOLEAN, 1, &elementCount, &stallUsed));
    EXPECT_FALSE(stallUsed);

    // Reuse stalled again: reset, then arm succeeds on the very same gate -- a timeout
    // does not disable later arming.
    ASSERT_NO_THROW(resetContext(desc));
    ASSERT_NO_THROW(armStall(desc));
    ASSERT_NO_THROW(recordStart(desc));
    ASSERT_NO_FATAL_FAILURE(enqueueTimedWork());
    ASSERT_NO_THROW(recordStop(desc));
    ASSERT_NO_THROW(releaseStall(desc));
    ASSERT_NO_THROW(desc->finalize());

    stallUsed = false;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_PROFILING_STALL_USED_EXT, HIPDNN_TYPE_BOOLEAN, 1, &elementCount, &stallUsed));
    EXPECT_TRUE(stallUsed) << "arm must succeed again on a drained, reset context";

    timedOut = true;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_PROFILING_STALL_TIMED_OUT_EXT,
                                       HIPDNN_TYPE_BOOLEAN,
                                       1,
                                       &elementCount,
                                       &timedOut));
    EXPECT_FALSE(timedOut);
}

// reset() must release an armed-but-unfinalized gate and drain the stream before
// returning, exactly like an error-path caller that abandons a measurement between
// start and stop. A caller that immediately re-arms afterward (below) would otherwise
// race the old wait packet's retirement and risk re-stalling already-released work.
TEST_F(TestGpuProfilingControlDescriptor, ResetAfterPartialMeasurementReleasesTheWait)
{
    if(!stallGateAvailable())
    {
        GTEST_SKIP() << "Device does not support hipStreamWaitValue32";
    }

    auto desc = getDescriptor();
    ASSERT_NO_THROW(setHandle(desc));
    hipEvent_t retired = nullptr;
    ASSERT_EQ(hipEventCreate(&retired), hipSuccess);
    const hipdnn_backend::HipEventGuard retiredGuard(retired);
    const auto beforeArm = std::chrono::steady_clock::now();
    ASSERT_NO_THROW(armStall(desc));
    ASSERT_NO_THROW(recordStart(desc));
    ASSERT_EQ(hipEventRecord(retired, _testStream), hipSuccess);
    // Abandon the measurement here: no stop, no release, no finalize(), as an error
    // path between start and stop would leave it.

    ASSERT_NO_THROW(resetContext(desc));
    EXPECT_LT(std::chrono::steady_clock::now() - beforeArm,
              hipdnn_data_sdk::utilities::StallGate::DEFAULT_TIMEOUT)
        << "reset waited for the watchdog instead of releasing the gate";
    EXPECT_EQ(hipEventQuery(retired), hipSuccess) << "reset did not retire the queued work";

    ASSERT_NO_FATAL_FAILURE(prepareTimedWork());
    ASSERT_NO_THROW(armStall(desc));
    ASSERT_NO_THROW(recordStart(desc));
    ASSERT_NO_FATAL_FAILURE(enqueueTimedWork());
    ASSERT_NO_THROW(recordStop(desc));
    ASSERT_NO_THROW(releaseStall(desc));
    ASSERT_NO_THROW(desc->finalize());

    bool stallUsed = false;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_PROFILING_STALL_USED_EXT, HIPDNN_TYPE_BOOLEAN, 1, &elementCount, &stallUsed));
    EXPECT_TRUE(stallUsed);

    bool timedOut = true;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_PROFILING_STALL_TIMED_OUT_EXT,
                                       HIPDNN_TYPE_BOOLEAN,
                                       1,
                                       &elementCount,
                                       &timedOut));
    EXPECT_FALSE(timedOut) << "reset() left an undrained waiter that stalled the reused "
                              "measurement";
}
