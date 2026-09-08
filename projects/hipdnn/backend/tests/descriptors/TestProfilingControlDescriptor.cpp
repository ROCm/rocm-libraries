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

// Guards against a silent 1/0 render of the boolean state fields in toString():
// a freshly created descriptor has all state flags false, so each must render as
// the literal token "false" (never "0" / "1").
TEST_F(TestProfilingControlDescriptor, ToStringRendersBooleanTokens)
{
    const std::string str = getDescriptor()->toString();

    EXPECT_NE(str.find("eventsCreated=false"), std::string::npos);
    EXPECT_NE(str.find("startRecorded=false"), std::string::npos);
    EXPECT_NE(str.find("stopRecorded=false"), std::string::npos);
    EXPECT_NE(str.find("finalized=false"), std::string::npos);

    EXPECT_EQ(str.find("eventsCreated=0"), std::string::npos);
    EXPECT_EQ(str.find("eventsCreated=1"), std::string::npos);
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
        // One case deliberately trips the watchdog, which disables stalling for the
        // whole process. Clear it per test so results cannot depend on test order.
        hipdnn_data_sdk::utilities::StallGate::resetDisabledProcessWideForTesting();
        ASSERT_EQ(hipStreamCreate(&_testStream), hipSuccess);
        _mockHandle = std::make_unique<NiceMock<MockHandle>>();
        ON_CALL(*_mockHandle, getStream()).WillByDefault(Return(_testStream));
    }

    void TearDown() override
    {
        _mockHandle.reset();
        if(_testStream != nullptr)
        {
            EXPECT_EQ(hipStreamDestroy(_testStream), hipSuccess);
            _testStream = nullptr;
        }
        TestProfilingControlDescriptor::TearDown();
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

    // The stall needs hipStreamWaitValue32; a device without it degrades to the
    // unstalled path, which the timing assertions below would read as a failure.
    static bool stallGateAvailable()
    {
        const hipdnn_data_sdk::utilities::StallGate gate;
        return gate.isUsable();
    }

    std::unique_ptr<NiceMock<MockHandle>> _mockHandle = nullptr;
    hipStream_t _testStream = nullptr;
};

TEST_F(TestGpuProfilingControlDescriptor, HappyPathCompletesLifecycle)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(setHandle(desc));
    ASSERT_NO_THROW(recordStart(desc));
    ASSERT_NO_THROW(recordStop(desc));
    ASSERT_NO_THROW(desc->finalize());

    // getAttribute(ELAPSED_MS) round-trips: no throw, one element written.
    // The elapsed value itself is driver-provided and not asserted.
    float elapsed = -1.0f;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_PROFILING_ELAPSED_MS_EXT, HIPDNN_TYPE_FLOAT, 1, &elementCount, &elapsed));
    EXPECT_EQ(elementCount, 1);
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
    ASSERT_NO_THROW(recordStart(desc));
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
    ASSERT_NO_THROW(recordStart(desc));
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
    ASSERT_NO_THROW(recordStart(desc));
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
// Both runs measure the same trivial device work, so the elapsed difference is the
// host delay and nothing else.
TEST_F(TestGpuProfilingControlDescriptor, StallGateExcludesHostSubmissionDelay)
{
    if(!stallGateAvailable())
    {
        GTEST_SKIP() << "Device does not support hipStreamWaitValue32";
    }

    constexpr auto HOST_DELAY = std::chrono::milliseconds(20);
    constexpr size_t BUFFER_BYTES = 256;

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

    // The 20 ms host sleep lands inside the unstalled span, and outside the stalled one.
    // The bounds are loose on both sides so scheduling jitter cannot flip the result.
    EXPECT_GE(unstalledMs, 15.0f) << "unstalled timing did not absorb the host delay";
    EXPECT_LT(stalledMs, 5.0f) << "stalled timing still includes the host delay";
}

// The deadlock the watchdog exists for, reproduced exactly: work inside the timed
// region blocks the host on the stalled stream, and only the host can release. Without
// the watchdog this test hangs forever. With it, the write that ends the stall is also
// what the blocked host is waiting on, so hipStreamSynchronize returns.
TEST_F(TestGpuProfilingControlDescriptor, WatchdogBreaksSelfInflictedDeadlock)
{
    if(!stallGateAvailable())
    {
        GTEST_SKIP() << "Device does not support hipStreamWaitValue32";
    }

    constexpr auto TIMEOUT = std::chrono::milliseconds(300);
    hipdnn_data_sdk::utilities::StallGate gate(TIMEOUT);
    ASSERT_TRUE(gate.isUsable());
    ASSERT_TRUE(gate.arm(_testStream));

    const auto begin = std::chrono::steady_clock::now();
    // Never returns unless something else releases the gate.
    EXPECT_EQ(hipStreamSynchronize(_testStream), hipSuccess);
    const auto waited = std::chrono::steady_clock::now() - begin;

    EXPECT_TRUE(gate.timedOut()) << "watchdog released but did not report it";
    EXPECT_GE(waited, TIMEOUT) << "watchdog fired before its deadline";

    // Sticky: the cause is a property of the measured code, so stalling stays off and a
    // later arm must decline rather than deadlock again.
    EXPECT_TRUE(hipdnn_data_sdk::utilities::StallGate::isDisabledProcessWide());
    EXPECT_FALSE(gate.arm(_testStream));
}

// The host released in time, so the watchdog must stay out of the way: no timeout
// reported, and stalling still enabled for everything after.
TEST_F(TestGpuProfilingControlDescriptor, WatchdogDoesNotFireOnNormalRelease)
{
    if(!stallGateAvailable())
    {
        GTEST_SKIP() << "Device does not support hipStreamWaitValue32";
    }

    hipdnn_data_sdk::utilities::StallGate gate(std::chrono::milliseconds(5000));
    ASSERT_TRUE(gate.isUsable());
    ASSERT_TRUE(gate.arm(_testStream));
    gate.release();
    ASSERT_EQ(hipStreamSynchronize(_testStream), hipSuccess);

    EXPECT_FALSE(gate.timedOut());
    EXPECT_FALSE(hipdnn_data_sdk::utilities::StallGate::isDisabledProcessWide());
}

// A watchdog release must be visible through the public descriptor, so an external
// caller can discard the sample instead of averaging a timeout into its results.
TEST_F(TestGpuProfilingControlDescriptor, TimedOutAttributeIsFalseForAHealthyMeasurement)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(setHandle(desc));
    ASSERT_NO_THROW(armStall(desc));
    ASSERT_NO_THROW(recordStart(desc));
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
// releases first; this test hangs on regression rather than reporting a wrong number.
TEST_F(TestGpuProfilingControlDescriptor, FinalizeReleasesUnreleasedStall)
{
    if(!stallGateAvailable())
    {
        GTEST_SKIP() << "Device does not support hipStreamWaitValue32";
    }

    auto desc = getDescriptor();
    ASSERT_NO_THROW(setHandle(desc));
    ASSERT_NO_THROW(armStall(desc));
    ASSERT_NO_THROW(recordStart(desc));
    ASSERT_NO_THROW(recordStop(desc));
    ASSERT_NO_THROW(desc->finalize());

    float elapsed = -1.0f;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_PROFILING_ELAPSED_MS_EXT, HIPDNN_TYPE_FLOAT, 1, &elementCount, &elapsed));
    EXPECT_GE(elapsed, 0.0f);
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
