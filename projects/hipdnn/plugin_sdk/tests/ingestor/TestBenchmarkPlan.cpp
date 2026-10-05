// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <chrono>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/StallGate.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_plugin_sdk/GlobalKnobDefines.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/ingestor/BenchmarkPlan.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/GenericPlanBuilder.hpp>
#include <hipdnn_plugin_sdk/ingestor/IDeviceResolver.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelIngestorStateManager.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>
#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "IngestorMocks.hpp"
#include "KernelIngestorTestFixtures.hpp"

#ifdef HIPDNN_TEST_HIP_STREAM_WAIT_FAILURES
namespace
{
struct HipStreamWaitFaults
{
    bool enabled = false;
    bool unsupported = false;
    int capabilityQueries = 0;
    int armCalls = 0;
    int failFromArm = 0;
    int rejectedArms = 0;
    hipError_t lastError = hipSuccess;
};

thread_local HipStreamWaitFaults gStreamWaitFaults;
} // namespace

// GNU ld requires these symbol names for test-only HIP call wrapping.
// NOLINTBEGIN(readability-identifier-naming, bugprone-reserved-identifier)
extern "C" hipError_t __real_hipStreamWaitValue32(
    hipStream_t stream, void* ptr, uint32_t value, unsigned int flags, uint32_t mask);
extern "C" hipError_t
    __real_hipDeviceGetAttribute(int* value, hipDeviceAttribute_t attribute, int device);

extern "C" hipError_t __wrap_hipStreamWaitValue32(
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
    return __real_hipStreamWaitValue32(stream, ptr, value, flags, mask);
}

extern "C" hipError_t
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
// NOLINTEND(readability-identifier-naming, bugprone-reserved-identifier)
#endif

/**
 * @file TestBenchmarkPlan.cpp
 * @brief Unit tests for BenchmarkPlan.hpp: construction, workspace sizing, execution
 *        delegation, and ranked capture/write-back, plus the oracle proving buildPlan()'s
 *        benchmarking-off path never reaches it.
 */
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hipdnn_plugin_sdk::ingestor::testing;
using ::testing::_;
using ::testing::ByMove;
using ::testing::Field;
using ::testing::Return;

/// A minimal TContext exposing the plan buildPlan() set, so a test can execute() it and
/// observe which candidate launched. Local to this file, mirroring
/// TestGenericPlanBuilder.cpp's own KnobFilterContext.
struct OracleContext
{
    void setExecutionSettings(const StubSettings& settings)
    {
        _settings = settings;
    }

    const StubSettings& executionSettings() const
    {
        return _settings;
    }

    void setPlan(std::unique_ptr<hipdnn_plugin_sdk::IPlan<StubHandle>> plan)
    {
        _plan = std::move(plan);
    }

    const hipdnn_plugin_sdk::IPlan<StubHandle>& plan() const
    {
        return *_plan;
    }

private:
    StubSettings _settings;
    std::unique_ptr<hipdnn_plugin_sdk::IPlan<StubHandle>> _plan;
};

using OraclePlanBuilder = GenericPlanBuilder<StubHandle, StubSettings, OracleContext>;

/// Three kernels with no matchers, so every kernel survives catalog construction and
/// only the heuristic decides rank.
std::unique_ptr<KernelIngestorStateManager<StubHandle>> makeThreeKernelStubStateManager()
{
    MetadataSchema schema;
    schema.id = SCHEMA_ID;
    schema.name = "test schema";
    schema.fields = {{BLOCK_SIZE, MetadataType::INT, MetadataValue{int64_t{64}}},
                     {DTYPE, MetadataType::STRING, std::nullopt}};

    KernelDescriptorPack pack;
    pack.id = PACK_ID;
    pack.name = "test pack";
    pack.engineId = ENGINE_ID;
    pack.dispatchId = DISPATCH_ID;
    pack.kernels = {makeTestKernel(testId(0x64), "kernel_64_float", 64, "FLOAT"),
                    makeTestKernel(testId(0x65), "kernel_256_float", 256, "FLOAT"),
                    makeTestKernel(testId(0x66), "kernel_64_half", 64, "HALF")};

    return std::make_unique<KernelIngestorStateManager<StubHandle>>(
        std::move(schema),
        std::vector<MatchDescriptor>{},
        makeStubDispatches(),
        std::vector<KernelDescriptorPack>{std::move(pack)},
        std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL),
        GRAPH_MATCH_SYMBOL);
}

/// With benchmarking off, buildPlan() builds one plain GenericPlan for the ranked front
/// and never constructs a BenchmarkPlan, launching exactly once. scoreByBlockSize ranks
/// by BLOCK_SIZE, so kernel_256_float (0x65) outranks the two 64-block kernels.
TEST(TestIngestorBenchmarkPlan, BenchmarkingOffBuildsAPlainPlanThatLaunchesTheRankedFrontOnce)
{
    // A leaked override must not make this look benchmarked: the environment must
    // genuinely be unset here.
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter forceBenchmarkingGuard(
        hipdnn_plugin_sdk::FORCE_BENCHMARKING_ENV_NAME);
    const ScopedTestSymbols symbols;

    const MockKernelDispatchHandler handler;
    const ScopedDispatchRegistration<StubHandle> dispatch("hipdnn.kernel_ingestor.test.dispatch",
                                                          handler);

    const auto manager = makeThreeKernelStubStateManager();
    const auto engine = makeEngineWithKnobs({BLOCK_SIZE});
    const StubDeviceResolver resolver;
    const OraclePlanBuilder builder(engine, *manager, resolver);

    const auto rankedFrontId = testId(0x65);
    EXPECT_CALL(handler, workspaceBytes(_, _, Field(&KernelDefinition::kernelId, rankedFrontId)))
        .WillOnce(Return(size_t{0}));
    EXPECT_CALL(handler, prepare(_, _, Field(&KernelDefinition::kernelId, rankedFrontId)))
        .WillOnce(Return(ByMove(std::make_unique<PreparedDispatch>())));
    EXPECT_CALL(handler, launch(_, _, _, _, _)).Times(1);

    const TestGraph graph(makeGraphId(0x50));
    // No knob set and an invalid config: readBenchmarkingEnabled() and the unset
    // HIPDNN_FORCE_BENCHMARKING override both read as off, matching a plain
    // hipdnnExecute with no autotune.
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper invalidConfig(nullptr,
                                                                                          0);

    StubSettings settings;
    builder.initializeExecutionSettings(StubHandle{}, graph, invalidConfig, settings);
    ASSERT_FALSE(settings.ingestorSettings.benchmarkingEnabled);

    OracleContext context;
    context.setExecutionSettings(settings);
    builder.buildPlan(StubHandle{}, graph, invalidConfig, context);

    const StubHandle handle;
    context.plan().execute(handle, nullptr, 0, nullptr);
}

// ---------------------------------------------------------------------------
// BenchmarkPlan's own unit: construction, resolution, delegation. These construct
// BenchmarkPlan directly rather than through buildPlan(), and inject a deterministic
// timer, so selection is provable without a device.
// ---------------------------------------------------------------------------

/// A handle satisfying HasGetStream, which BenchmarkPlan's constructor static_asserts.
/// StubHandle (used by the oracle above) has no getStream(). Defaults to the null stream,
/// which never matters for the injected timers; the watchdog case below passes a real one,
/// because the default timer arms the stall gate on whatever stream it is given.
struct BenchmarkTestHandle
{
    hipStream_t stream = nullptr;

    // Non-static: models a real handle's instance accessor, which is what HasGetStream
    // detects.
    // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
    hipStream_t getStream() const
    {
        return stream;
    }
};

/// Bytes of same-stream scratch a real-GPU-work FakePlan memsets per execute(). 4 MiB
/// clears hipEventElapsedTime's ~1 us resolution floor on every supported GPU with
/// margin, so bracketing this work with real HIP events cannot report a sub-resolution
/// or negative duration purely from having nothing to measure.
constexpr size_t GPU_WORK_SCRATCH_BYTES = size_t{4} * 1024 * 1024;

/// A minimal IPlan double recording every execute() call's arguments and count.
/// Throws on the first @p throwForCalls invocations (default 0, never throws), then
/// succeeds and counts a "launch". @p enqueueGpuWork additionally memsets a same-stream
/// scratch buffer on every execute(): the real-HIP-timer tests need actual device work
/// between the timer's start and stop events, or hipEventElapsedTime has nothing to
/// measure and can report a sub-resolution or negative duration.
class FakePlan : public hipdnn_plugin_sdk::IPlan<BenchmarkTestHandle>
{
public:
    explicit FakePlan(size_t workspaceSize = 0, int throwForCalls = 0, bool enqueueGpuWork = false)
        : _workspaceSize(workspaceSize)
        , _throwForCalls(throwForCalls)
    {
        if(enqueueGpuWork)
        {
            void* raw = nullptr;
            if(hipMalloc(&raw, GPU_WORK_SCRATCH_BYTES) != hipSuccess || raw == nullptr)
            {
                throw std::runtime_error("FakePlan: GPU scratch allocation failed");
            }
            _gpuScratch = {raw, [](void* ptr) { static_cast<void>(hipFree(ptr)); }};
        }
    }

    size_t getWorkspaceSize(const BenchmarkTestHandle& /*handle*/) const override
    {
        return _workspaceSize;
    }

    void execute(const BenchmarkTestHandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override
    {
        ++_callCount;
        _lastDeviceBuffers = deviceBuffers;
        _lastNumDeviceBuffers = numDeviceBuffers;
        _lastWorkspace = workspace;
        if(!_gpuScratch.isEmpty()
           && hipMemsetAsync(_gpuScratch.get(), 0, GPU_WORK_SCRATCH_BYTES, handle.getStream())
                  != hipSuccess)
        {
            throw std::runtime_error("FakePlan: GPU work enqueue failed");
        }
        if(_callCount <= _throwForCalls)
        {
            throw std::runtime_error("FakePlan: simulated failure");
        }
        ++_launchCount;
    }

    int launchCount() const
    {
        return _launchCount;
    }

    const hipdnnPluginDeviceBuffer_t* lastDeviceBuffers() const
    {
        return _lastDeviceBuffers;
    }

    uint32_t lastNumDeviceBuffers() const
    {
        return _lastNumDeviceBuffers;
    }

    void* lastWorkspace() const
    {
        return _lastWorkspace;
    }

private:
    size_t _workspaceSize;
    int _throwForCalls;
    hipdnn_data_sdk::utilities::ScopedResource<void*> _gpuScratch;
    mutable int _callCount = 0;
    mutable int _launchCount = 0;
    mutable const hipdnnPluginDeviceBuffer_t* _lastDeviceBuffers = nullptr;
    mutable uint32_t _lastNumDeviceBuffers = 0;
    mutable void* _lastWorkspace = nullptr;
};

/// Blocks the host on the handle's own stream from inside execute(). The GPU work
/// stays behind the stall gate until release; synchronizing it before the caller's
/// release still requires the watchdog. The same work gives the unstalled rerun a
/// meaningful interval to time.
class StreamSyncingPlan : public FakePlan
{
public:
    StreamSyncingPlan()
        : FakePlan(/*workspaceSize=*/0, /*throwForCalls=*/0, /*enqueueGpuWork=*/true)
    {
    }

    void execute(const BenchmarkTestHandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override
    {
        FakePlan::execute(handle, deviceBuffers, numDeviceBuffers, workspace);
        if(hipStreamSynchronize(handle.getStream()) != hipSuccess)
        {
            throw std::runtime_error("StreamSyncingPlan: stream synchronization failed");
        }
    }
};

using TestBenchmarkPlan = BenchmarkPlan<BenchmarkTestHandle>;

/// The launch count a candidate accrues from one sampling pass: the untimed warmups
/// plus the timed iterations. The winner adds one more for the delegated execute.
constexpr int SAMPLING_LAUNCHES = BENCHMARK_WARMUP_RUNS + BENCHMARK_ITERATIONS;

/// A timer returning a fixed duration per sub-plan, forwarding the real execute() so
/// launch counts still accrue. A plan absent from @p durations is untimeable, which
/// scores it unusable.
TestBenchmarkPlan::Timer
    fixedTimer(std::map<const hipdnn_plugin_sdk::IPlan<BenchmarkTestHandle>*, double> durations)
{
    return [durations
            = std::move(durations)](const hipdnn_plugin_sdk::IPlan<BenchmarkTestHandle>& plan,
                                    const BenchmarkTestHandle& handle,
                                    const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                    uint32_t numDeviceBuffers,
                                    void* workspace) -> std::optional<double> {
        const auto found = durations.find(&plan);
        if(found == durations.end())
        {
            return std::nullopt;
        }
        plan.execute(handle, deviceBuffers, numDeviceBuffers, workspace);
        return found->second;
    };
}

TEST(TestIngestorBenchmarkPlan, GetWorkspaceSizeIsTheMaxAcrossSubPlans)
{
    std::vector<TestBenchmarkPlan::Candidate> candidates;
    candidates.push_back({testId(0x01), std::make_unique<FakePlan>(64)});
    candidates.push_back({testId(0x02), std::make_unique<FakePlan>(256)});
    candidates.push_back({testId(0x03), std::make_unique<FakePlan>(128)});

    const BenchmarkTestHandle handle;
    const TestBenchmarkPlan plan(std::move(candidates), handle);

    EXPECT_EQ(plan.getWorkspaceSize(handle), 256U);
}

TEST(TestIngestorBenchmarkPlan, ConstructorThrowsInternalErrorOnAnEmptyCandidateVector)
{
    const BenchmarkTestHandle handle;

    try
    {
        const TestBenchmarkPlan plan(std::vector<TestBenchmarkPlan::Candidate>{}, handle);
        FAIL() << "expected HipdnnPluginException";
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& ex)
    {
        EXPECT_EQ(ex.getStatus(), HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR);
    }
}

/// The fastest candidate wins and takes the delegated execute; the loser is sampled and
/// then never touched again. The two exact counts also pin BENCHMARK_WARMUP_RUNS and
/// BENCHMARK_ITERATIONS: both candidates accrue exactly one sampling pass.
TEST(TestIngestorBenchmarkPlan, TheFastestCandidateWinsAndOnlyItReceivesTheDelegatedExecute)
{
    auto slow = std::make_unique<FakePlan>(64);
    auto fast = std::make_unique<FakePlan>(64);
    const auto* slowRaw = slow.get();
    const auto* fastRaw = fast.get();

    std::vector<TestBenchmarkPlan::Candidate> candidates;
    candidates.push_back({testId(0x01), std::move(slow)});
    candidates.push_back({testId(0x02), std::move(fast)});

    const BenchmarkTestHandle handle;
    const TestBenchmarkPlan plan(
        std::move(candidates), handle, fixedTimer({{slowRaw, 5.0}, {fastRaw, 2.0}}));

    plan.execute(handle, nullptr, 0, nullptr);

    EXPECT_EQ(fastRaw->launchCount(), SAMPLING_LAUNCHES + 1);
    EXPECT_EQ(slowRaw->launchCount(), SAMPLING_LAUNCHES);
}

/// A single lucky sample does not win the sweep. The reduction is a mean over the samples
/// left after the slow tail is trimmed, so a candidate that is usually slower cannot beat a
/// steady rival on one fast outlier -- it would then serve its typical, slower time on every
/// dispatch the cached ranking covers.
///
/// This is the case that separates the reduction from a minimum: LUCKY_SAMPLE is far below
/// anything the steady candidate produces, so a min-based reduction picks the wrong winner.
TEST(TestIngestorBenchmarkPlan, TheReductionIgnoresASingleLuckySample)
{
    auto lucky = std::make_unique<FakePlan>(64);
    auto steady = std::make_unique<FakePlan>(64);
    const auto* luckyRaw = lucky.get();
    const auto* steadyRaw = steady.get();

    std::vector<TestBenchmarkPlan::Candidate> candidates;
    candidates.push_back({testId(0x01), std::move(lucky)});
    candidates.push_back({testId(0x02), std::move(steady)});

    // The lucky candidate reports one very fast sample and is otherwise slower than the
    // steady one, whose constant sits between the two.
    constexpr double LUCKY_SAMPLE = 1.0;
    constexpr double LUCKY_TYPICAL = 12.0;
    constexpr double STEADY_SAMPLE = 10.0;

    static_assert(LUCKY_SAMPLE < STEADY_SAMPLE && STEADY_SAMPLE < LUCKY_TYPICAL,
                  "the lucky candidate must win on its best sample and lose on its typical "
                  "one, or the case stops separating the reduction from a minimum");

    int luckySampleIndex = 0;
    const TestBenchmarkPlan::Timer timer
        = [&](const hipdnn_plugin_sdk::IPlan<BenchmarkTestHandle>& plan,
              const BenchmarkTestHandle& planHandle,
              const hipdnnPluginDeviceBuffer_t* deviceBuffers,
              uint32_t numDeviceBuffers,
              void* workspace) -> std::optional<double> {
        plan.execute(planHandle, deviceBuffers, numDeviceBuffers, workspace);
        if(&plan == steadyRaw)
        {
            return STEADY_SAMPLE;
        }
        return luckySampleIndex++ == 0 ? LUCKY_SAMPLE : LUCKY_TYPICAL;
    };

    const BenchmarkTestHandle handle;
    const TestBenchmarkPlan plan(std::move(candidates), handle, timer);

    plan.execute(handle, nullptr, 0, nullptr);

    // Steady takes the delegated execute; the lucky candidate is sampled and dropped.
    EXPECT_EQ(steadyRaw->launchCount(), SAMPLING_LAUNCHES + 1);
    EXPECT_EQ(luckyRaw->launchCount(), SAMPLING_LAUNCHES);
}

/// A single slow sample does not lose the sweep either. Interference can make an iteration
/// slower but never faster, so the slow tail is contamination and is trimmed before the mean
/// is taken.
///
/// This is the case that separates the reduction from a plain mean: OUTLIER_SAMPLE is large
/// enough to drag the untrimmed average above the steady rival's constant.
TEST(TestIngestorBenchmarkPlan, TheReductionTrimsASingleSlowOutlier)
{
    auto spiky = std::make_unique<FakePlan>(64);
    auto steady = std::make_unique<FakePlan>(64);
    const auto* spikyRaw = spiky.get();
    const auto* steadyRaw = steady.get();

    std::vector<TestBenchmarkPlan::Candidate> candidates;
    candidates.push_back({testId(0x01), std::move(spiky)});
    candidates.push_back({testId(0x02), std::move(steady)});

    // Spiky is genuinely the faster kernel, but one contaminated sample pushes its raw mean
    // past steady's constant. Its remaining samples carry ordinary jitter: identical samples
    // would leave the median absolute deviation at zero, and the reduction keeps every
    // sample when it cannot measure a spread, which would defeat the case.
    constexpr std::array<double, BENCHMARK_ITERATIONS> SPIKY_SAMPLES
        = {40.0, 10.0, 10.5, 11.0, 9.5, 10.0, 10.5};
    constexpr double SPIKY_OUTLIER = SPIKY_SAMPLES[0];
    constexpr double STEADY_SAMPLE = 12.0;

    static_assert(SPIKY_SAMPLES[1] < STEADY_SAMPLE, "spiky must be the genuinely faster candidate");
    static_assert(SPIKY_OUTLIER > STEADY_SAMPLE * BENCHMARK_ITERATIONS
                                      - (SPIKY_SAMPLES[1] + SPIKY_SAMPLES[2] + SPIKY_SAMPLES[3]
                                         + SPIKY_SAMPLES[4] + SPIKY_SAMPLES[5] + SPIKY_SAMPLES[6]),
                  "the outlier must drag the untrimmed mean above steady, or the case stops "
                  "separating the reduction from a plain mean");

    size_t spikySampleIndex = 0;
    const TestBenchmarkPlan::Timer timer
        = [&](const hipdnn_plugin_sdk::IPlan<BenchmarkTestHandle>& plan,
              const BenchmarkTestHandle& planHandle,
              const hipdnnPluginDeviceBuffer_t* deviceBuffers,
              uint32_t numDeviceBuffers,
              void* workspace) -> std::optional<double> {
        plan.execute(planHandle, deviceBuffers, numDeviceBuffers, workspace);
        return &plan == steadyRaw ? STEADY_SAMPLE : SPIKY_SAMPLES.at(spikySampleIndex++);
    };

    const BenchmarkTestHandle handle;
    const TestBenchmarkPlan plan(std::move(candidates), handle, timer);

    plan.execute(handle, nullptr, 0, nullptr);

    // Spiky takes the delegated execute: the outlier is trimmed rather than averaged in.
    EXPECT_EQ(spikyRaw->launchCount(), SAMPLING_LAUNCHES + 1);
    EXPECT_EQ(steadyRaw->launchCount(), SAMPLING_LAUNCHES);
}

/// A candidate the timer cannot time is scored unusable and abandoned mid-sweep, after
/// its warmup and exactly one failed timed iteration.
TEST(TestIngestorBenchmarkPlan, AnUntimeableCandidateIsScoredUnusableAndLosesToATimedOne)
{
    auto untimeable = std::make_unique<FakePlan>(64);
    auto timed = std::make_unique<FakePlan>(64);
    const auto* untimeableRaw = untimeable.get();
    const auto* timedRaw = timed.get();

    std::vector<TestBenchmarkPlan::Candidate> candidates;
    candidates.push_back({testId(0x01), std::move(untimeable)});
    candidates.push_back({testId(0x02), std::move(timed)});

    const BenchmarkTestHandle handle;
    // Only the second candidate has a duration, so the first returns nullopt.
    const TestBenchmarkPlan plan(std::move(candidates), handle, fixedTimer({{timedRaw, 9.0}}));

    plan.execute(handle, nullptr, 0, nullptr);

    // The untimeable candidate ran its warmups, then bailed out of the timed loop
    // without launching: the timer returns nullopt before forwarding execute().
    EXPECT_EQ(untimeableRaw->launchCount(), BENCHMARK_WARMUP_RUNS);
    EXPECT_EQ(timedRaw->launchCount(), SAMPLING_LAUNCHES + 1);
}

/// The default timer runs against real HIP events and same-stream GPU work.
/// This case verifies event creation, reuse, recording, synchronization, and elapsed
/// readback without an injected timer; the watchdog and fault tests cover recovery.
///
/// Asserting the exact count is what makes it meaningful: the timer must have returned a
/// duration on all BENCHMARK_ITERATIONS samples. A single nullopt would score the
/// candidate unusable and drop the count to BENCHMARK_WARMUP_RUNS.
TEST(TestIngestorBenchmarkPlan, TheDefaultTimerTimesEverySampleAgainstRealHipEvents)
{
    SKIP_IF_NO_DEVICES();

    // Unsupported stream-wait devices must use ordinary event timing from the
    // first sample, without an extra discarded pass.
    auto sub = std::make_unique<FakePlan>(64, /*throwForCalls=*/0, /*enqueueGpuWork=*/true);
    const auto* subRaw = sub.get();

    std::vector<TestBenchmarkPlan::Candidate> candidates;
    candidates.push_back({testId(0x01), std::move(sub)});

    const BenchmarkTestHandle handle;
    std::vector<RankedEntry> recorded;
    const TestBenchmarkPlan plan(std::move(candidates),
                                 handle,
                                 TestBenchmarkPlan::Timer{},
                                 [&recorded](auto ranking) { recorded = std::move(ranking); });

    plan.execute(handle, nullptr, 0, nullptr);
    ASSERT_EQ(recorded.size(), 1U) << "the HIP-event timer did not produce a usable ranking";

    EXPECT_EQ(subRaw->launchCount(), SAMPLING_LAUNCHES + 1)
        << "the HIP-event timer failed a sample; the candidate was scored unusable";

    // The event pair is created once and re-recorded, so a second sweep-free execute()
    // still delegates exactly once.
    plan.execute(handle, nullptr, 0, nullptr);
    EXPECT_EQ(subRaw->launchCount(), SAMPLING_LAUNCHES + 2);
}

/// A watchdog timeout says the candidate cannot be measured with the stream stalled; it
/// says nothing about how fast the candidate is. The comparison aborts sampling the
/// instant the timeout is seen -- a candidate ordered after the deadlocking one is never
/// sampled stalled at all -- then re-measures every candidate unstalled from scratch, so
/// all three end up ranked.
///
/// This is the real default timer against real HIP: the middle candidate synchronizes
/// the very stream the gate stalls, which is the self-inflicted deadlock the watchdog
/// exists for. No module-wide latch is consulted or reset: this policy is local to each
/// comparison, proven by running a second, independent comparison afterward and seeing
/// its own gate stall and time out too.
TEST(TestIngestorBenchmarkPlan, AWatchdogTimeoutAbortsThePassImmediatelyAndRerunsUnstalled)
{
    SKIP_IF_NO_DEVICES();

    int canWaitValue = 0;
    int device = 0;
    ASSERT_EQ(hipGetDevice(&device), hipSuccess);
    ASSERT_EQ(hipDeviceGetAttribute(&canWaitValue, hipDeviceAttributeCanUseStreamWaitValue, device),
              hipSuccess);
    if(canWaitValue == 0)
    {
        GTEST_SKIP() << "Device does not support hipStreamWaitValue32";
    }

    // One run of a [normal, deadlocking, normal] comparison against a fresh stream and
    // plan: how long plan.execute() took, what the ranking recorded, and how many times
    // the candidates before/after the deadlocking one launched. Returned by value rather
    // than through out-params so two calls below cannot share any state.
    struct ComparisonResult
    {
        std::vector<RankedEntry> recorded;
        std::chrono::milliseconds wallTime;
        int beforeLaunches;
        int afterLaunches;
    };

    const auto runComparison = []() -> ComparisonResult {
        hipStream_t rawStream = nullptr;
        EXPECT_EQ(hipStreamCreate(&rawStream), hipSuccess);
        // ScopedResource, the same RAII pattern the default HIP-event timer uses for its
        // events: destroys the stream on every exit path instead of only fall-through.
        const hipdnn_data_sdk::utilities::ScopedResource<hipStream_t> stream(
            rawStream, [](hipStream_t s) { static_cast<void>(hipStreamDestroy(s)); });
        const BenchmarkTestHandle handle{stream.get()};

        auto before = std::make_unique<FakePlan>(64, /*throwForCalls=*/0, /*enqueueGpuWork=*/true);
        auto after = std::make_unique<FakePlan>(64, /*throwForCalls=*/0, /*enqueueGpuWork=*/true);
        const auto* beforeRaw = before.get();
        const auto* afterRaw = after.get();

        std::vector<TestBenchmarkPlan::Candidate> candidates;
        candidates.push_back({testId(0x01), std::move(before), testId(0xF0), testId(0xD0)});
        candidates.push_back(
            {testId(0x02), std::make_unique<StreamSyncingPlan>(), testId(0xF0), testId(0xD0)});
        candidates.push_back({testId(0x03), std::move(after), testId(0xF0), testId(0xD0)});

        std::vector<RankedEntry> recorded;
        const TestBenchmarkPlan plan(
            std::move(candidates),
            handle,
            TestBenchmarkPlan::Timer{},
            [&recorded](std::vector<RankedEntry> ranking) { recorded = std::move(ranking); });

        const auto start = std::chrono::steady_clock::now();
        plan.execute(handle, nullptr, 0U, nullptr);
        const auto wallTime = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start);

        return {std::move(recorded), wallTime, beforeRaw->launchCount(), afterRaw->launchCount()};
    };

    // plan.execute() samples every candidate and then delegates once more to whichever
    // one the ranking crowned the winner, so that candidate's launch count is one higher
    // than a pure sampling count. Compare against the actual winner rather than assuming
    // one: real HIP timing decides which candidate wins.
    const auto delegatedBonus
        = [](const std::vector<RankedEntry>& recorded, DescriptorId id) -> int {
        return !recorded.empty() && recorded.front().kernelId == id ? 1 : 0;
    };

    const auto first = runComparison();

    EXPECT_GE(first.wallTime, hipdnn_data_sdk::utilities::StallGate::DEFAULT_TIMEOUT)
        << "the watchdog never fired, so this case proved nothing";
    EXPECT_EQ(first.recorded.size(), 3U)
        << "the timed-out candidate was dropped instead of re-measured unstalled";

    // Immediate abort: the candidate ordered after the deadlocking one accrues exactly
    // one pass's launches (the unstalled rerun only), because the stalled pass never
    // reaches it. The candidate ordered before it finishes its stalled pass normally and
    // is then resampled unstalled, so it accrues two passes' worth. Either count gains
    // one more launch if that candidate turned out to be the delegated winner.
    EXPECT_EQ(first.beforeLaunches,
              2 * SAMPLING_LAUNCHES + delegatedBonus(first.recorded, testId(0x01)))
        << "the candidate before the timeout must be sampled once per pass";
    EXPECT_EQ(first.afterLaunches, SAMPLING_LAUNCHES + delegatedBonus(first.recorded, testId(0x03)))
        << "the candidate after the timeout must not be sampled during the aborted stalled "
           "pass, only during the unstalled rerun";

    // A second, wholly independent comparison must still be able to stall and time out:
    // nothing about the first comparison's fallback may have disabled stalling for it.
    const auto second = runComparison();

    EXPECT_GE(second.wallTime, hipdnn_data_sdk::utilities::StallGate::DEFAULT_TIMEOUT)
        << "a prior comparison's fallback suppressed stalling for this independent one";
    EXPECT_EQ(second.recorded.size(), 3U);
    EXPECT_EQ(second.afterLaunches,
              SAMPLING_LAUNCHES + delegatedBonus(second.recorded, testId(0x03)));
}

#ifdef HIPDNN_TEST_HIP_STREAM_WAIT_FAILURES
class TestIngestorHipTimerFaults : public ::testing::Test
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

TEST_F(TestIngestorHipTimerFaults, ArmErrorRestartsTheWholeComparisonUnstalled)
{
    int device = 0;
    int canWait = 0;
    ASSERT_EQ(hipGetDevice(&device), hipSuccess);
    ASSERT_EQ(hipDeviceGetAttribute(&canWait, hipDeviceAttributeCanUseStreamWaitValue, device),
              hipSuccess);
    if(canWait == 0)
    {
        GTEST_SKIP() << "Device does not support hipStreamWaitValue32";
    }

    std::array<const FakePlan*, 3> plans{};
    std::vector<TestBenchmarkPlan::Candidate> candidates;
    for(size_t index = 0; index < plans.size(); ++index)
    {
        auto candidate
            = std::make_unique<FakePlan>(64, /*throwForCalls=*/0, /*enqueueGpuWork=*/true);
        plans[index] = candidate.get();
        candidates.push_back({testId(static_cast<uint8_t>(index + 1)), std::move(candidate)});
    }

    // The first candidate finishes stalled. The next arm and any later attempts
    // fail in HIP, so recovery must stop arming and remeasure every candidate.
    gStreamWaitFaults.failFromArm = BENCHMARK_ITERATIONS + 1;
    std::vector<RankedEntry> recorded;
    const BenchmarkTestHandle handle;
    const TestBenchmarkPlan plan(std::move(candidates),
                                 handle,
                                 TestBenchmarkPlan::Timer{},
                                 [&recorded](auto ranking) { recorded = std::move(ranking); });
    plan.execute(handle, nullptr, 0U, nullptr);

    EXPECT_EQ(gStreamWaitFaults.lastError, hipErrorInvalidValue);
    EXPECT_EQ(gStreamWaitFaults.rejectedArms, 1);
    ASSERT_EQ(recorded.size(), plans.size());
    const std::array<int, 3> expectedLaunches{
        2 * SAMPLING_LAUNCHES, SAMPLING_LAUNCHES + BENCHMARK_WARMUP_RUNS + 1, SAMPLING_LAUNCHES};
    for(size_t index = 0; index < plans.size(); ++index)
    {
        const int delegated
            = recorded.front().kernelId == testId(static_cast<uint8_t>(index + 1)) ? 1 : 0;
        EXPECT_EQ(plans[index]->launchCount(), expectedLaunches[index] + delegated);
    }
}

TEST_F(TestIngestorHipTimerFaults, UnsupportedDeviceStartsUnstalledWithoutDiscardingSamples)
{
    gStreamWaitFaults.unsupported = true;
    auto candidate = std::make_unique<FakePlan>(64, /*throwForCalls=*/0, /*enqueueGpuWork=*/true);
    const auto* candidatePtr = candidate.get();
    std::vector<TestBenchmarkPlan::Candidate> candidates;
    candidates.push_back({testId(1), std::move(candidate)});

    std::vector<RankedEntry> recorded;
    const BenchmarkTestHandle handle;
    const TestBenchmarkPlan plan(std::move(candidates),
                                 handle,
                                 TestBenchmarkPlan::Timer{},
                                 [&recorded](auto ranking) { recorded = std::move(ranking); });
    plan.execute(handle, nullptr, 0U, nullptr);

    ASSERT_GT(gStreamWaitFaults.capabilityQueries, 0);
    EXPECT_EQ(gStreamWaitFaults.armCalls, 0);
    ASSERT_EQ(recorded.size(), 1U);
    EXPECT_EQ(candidatePtr->launchCount(), SAMPLING_LAUNCHES + 1);
}
#endif

/// A one-candidate composite still samples before delegating to the only candidate.
TEST(TestIngestorBenchmarkPlan, ASingleCandidateCompositeExecutesThatOne)
{
    auto sub = std::make_unique<FakePlan>(64);
    const auto* subRaw = sub.get();

    std::vector<TestBenchmarkPlan::Candidate> candidates;
    candidates.push_back({testId(0x01), std::move(sub)});

    const BenchmarkTestHandle handle;
    const TestBenchmarkPlan plan(std::move(candidates), handle, fixedTimer({{subRaw, 1.0}}));

    plan.execute(handle, nullptr, 0, nullptr);
    EXPECT_EQ(subRaw->launchCount(), SAMPLING_LAUNCHES + 1);

    plan.execute(handle, nullptr, 0, nullptr);
    EXPECT_EQ(subRaw->launchCount(), SAMPLING_LAUNCHES + 2);
}

/// A second execute() adds exactly one more launch to the winner and none to the loser:
/// the sampling sweep runs once for the plan's life.
TEST(TestIngestorBenchmarkPlan, TheWinnerIsResolvedOnceAcrossRepeatedExecuteCalls)
{
    auto slow = std::make_unique<FakePlan>(64);
    auto fast = std::make_unique<FakePlan>(64);
    const auto* slowRaw = slow.get();
    const auto* fastRaw = fast.get();

    std::vector<TestBenchmarkPlan::Candidate> candidates;
    candidates.push_back({testId(0x01), std::move(slow)});
    candidates.push_back({testId(0x02), std::move(fast)});

    const BenchmarkTestHandle handle;
    const TestBenchmarkPlan plan(
        std::move(candidates), handle, fixedTimer({{slowRaw, 5.0}, {fastRaw, 2.0}}));

    plan.execute(handle, nullptr, 0, nullptr);
    plan.execute(handle, nullptr, 0, nullptr);
    plan.execute(handle, nullptr, 0, nullptr);

    EXPECT_EQ(fastRaw->launchCount(), SAMPLING_LAUNCHES + 3);
    EXPECT_EQ(slowRaw->launchCount(), SAMPLING_LAUNCHES);
}

/// Every candidate throws on its first invocation, caught inside sampleCandidate()
/// before the timer is reached. resolveChosen() falls back to index 0 rather than
/// propagating, and the delegated call that follows succeeds, so execute() must not
/// throw.
TEST(TestIngestorBenchmarkPlan, AllCandidatesUnusableStillDelegatesToCandidateZero)
{
    auto first = std::make_unique<FakePlan>(64, /*throwForCalls=*/1);
    auto second = std::make_unique<FakePlan>(64, /*throwForCalls=*/1);
    const auto* firstRaw = first.get();
    const auto* secondRaw = second.get();

    std::vector<TestBenchmarkPlan::Candidate> candidates;
    candidates.push_back({testId(0x01), std::move(first)});
    candidates.push_back({testId(0x02), std::move(second)});

    const BenchmarkTestHandle handle;
    const TestBenchmarkPlan plan(
        std::move(candidates), handle, fixedTimer({{firstRaw, 5.0}, {secondRaw, 2.0}}));

    EXPECT_NO_THROW(plan.execute(handle, nullptr, 0, nullptr));

    // Candidate 0 is the documented fallback: its second call, the real delegate, must
    // have launched. Candidate 1 is faster by the timer, which never runs for a
    // candidate that throws during warmup.
    EXPECT_EQ(firstRaw->launchCount(), 1);
    EXPECT_EQ(secondRaw->launchCount(), 0);
}

TEST(TestIngestorBenchmarkPlan, BuffersAndWorkspaceArriveAtTheChosenSubPlanUnmodified)
{
    auto sub = std::make_unique<FakePlan>(64);
    const auto* subRaw = sub.get();

    std::vector<TestBenchmarkPlan::Candidate> candidates;
    candidates.push_back({testId(0x01), std::move(sub)});

    const BenchmarkTestHandle handle;
    const TestBenchmarkPlan plan(std::move(candidates), handle, fixedTimer({{subRaw, 1.0}}));

    const std::array<hipdnnPluginDeviceBuffer_t, 1> buffers{
        {{/*uid=*/9, /*ptr=*/reinterpret_cast<void*>(0x5678)}}};
    int workspaceStorage = 0;
    void* const workspace = &workspaceStorage;

    plan.execute(handle, buffers.data(), 1U, workspace);

    EXPECT_EQ(subRaw->lastDeviceBuffers(), buffers.data());
    EXPECT_EQ(subRaw->lastNumDeviceBuffers(), 1U);
    EXPECT_EQ(subRaw->lastWorkspace(), workspace);
}

// ---------------------------------------------------------------------------
// Ranked capture and write-back
// ---------------------------------------------------------------------------

/// Supplies deterministic times through the Timer seam so ordering, omission and the
/// all-unusable case are decided by the code under test, not by GPU availability. The
/// real hipEvent path is proven separately on gfx942.
///
/// Times are keyed by candidate identity, since the timer sees the plan rather than its
/// index.
inline TestBenchmarkPlan::Timer
    makeDeterministicTimer(const std::vector<TestBenchmarkPlan::Candidate>& candidates,
                           std::vector<std::optional<double>> times)
{
    std::vector<const hipdnn_plugin_sdk::IPlan<BenchmarkTestHandle>*> order;
    order.reserve(candidates.size());
    for(const auto& candidate : candidates)
    {
        order.push_back(candidate.plan.get());
    }
    return [order = std::move(order),
            times = std::move(times)](const hipdnn_plugin_sdk::IPlan<BenchmarkTestHandle>& plan,
                                      const BenchmarkTestHandle&,
                                      const hipdnnPluginDeviceBuffer_t*,
                                      uint32_t,
                                      void*) -> std::optional<double> {
        const auto found = std::find(order.begin(), order.end(), &plan);
        if(found == order.end())
        {
            return std::nullopt;
        }
        const auto index = static_cast<size_t>(std::distance(order.begin(), found));
        if(index < times.size())
        {
            return times[index];
        }
        return std::nullopt;
    };
}

/// Builds a plan whose sampling is driven by @p times, indexed by candidate order.
inline TestBenchmarkPlan makeDeterministicPlan(std::vector<TestBenchmarkPlan::Candidate> candidates,
                                               const BenchmarkTestHandle& handle,
                                               std::vector<std::optional<double>> times,
                                               TestBenchmarkPlan::RecordRankingFn recordRanking
                                               = {})
{
    auto timer = makeDeterministicTimer(candidates, std::move(times));
    return {std::move(candidates), handle, std::move(timer), std::move(recordRanking)};
}

std::vector<TestBenchmarkPlan::Candidate> threeCandidates()
{
    std::vector<TestBenchmarkPlan::Candidate> candidates;
    candidates.push_back(
        {testId(0x01), std::make_unique<FakePlan>(64), testId(0xF0), testId(0xD0)});
    candidates.push_back(
        {testId(0x02), std::make_unique<FakePlan>(64), testId(0xF0), testId(0xD0)});
    candidates.push_back(
        {testId(0x03), std::make_unique<FakePlan>(64), testId(0xF0), testId(0xD0)});
    return candidates;
}

TEST(TestIngestorBenchmarkPlan, SamplingRecordsEveryUsableCandidateInMeasuredOrder)
{
    std::vector<RankedEntry> recorded;
    const BenchmarkTestHandle handle;
    const auto plan = makeDeterministicPlan(
        threeCandidates(), handle, {5.0, 1.0, 3.0}, [&recorded](std::vector<RankedEntry> ranking) {
            recorded = std::move(ranking);
        });

    plan.execute(handle, nullptr, 0U, nullptr);

    ASSERT_EQ(recorded.size(), 3U);
    EXPECT_EQ(recorded[0].kernelId, testId(0x02)) << "the fastest candidate must rank first";
    EXPECT_EQ(recorded[1].kernelId, testId(0x03));
    EXPECT_EQ(recorded[2].kernelId, testId(0x01));
    EXPECT_EQ(recorded[0].packId, testId(0xF0)) << "the staleness ids must travel with the id";
    EXPECT_EQ(recorded[0].dispatchId, testId(0xD0));
}

TEST(TestIngestorBenchmarkPlan, ACandidateThatFailedSamplingNeverAppearsInTheRanking)
{
    std::vector<RankedEntry> recorded;
    const BenchmarkTestHandle handle;
    // Candidate 1 failed to time; it must be omitted, not ranked last.
    const auto plan = makeDeterministicPlan(
        threeCandidates(),
        handle,
        {5.0, std::nullopt, 3.0},
        [&recorded](std::vector<RankedEntry> ranking) { recorded = std::move(ranking); });

    plan.execute(handle, nullptr, 0U, nullptr);

    ASSERT_EQ(recorded.size(), 2U);
    for(const auto& entry : recorded)
    {
        EXPECT_NE(entry.kernelId, testId(0x02))
            << "a known-broken kernel recorded as a fallback would be served ahead of the "
               "normal ranked path on a later run";
    }
}

/// A malformed sample from the timer must never enter robustMean() or the ranking. NaN
/// and infinite samples score the candidate unusable immediately, with no retry. A
/// negative sample is instead retried in place, so a candidate that is persistently
/// negative -- this deterministic timer keeps returning the same malformed value on
/// every retry -- still exhausts MAX_NEGATIVE_SAMPLE_RETRIES and ends up scored unusable
/// exactly like a nullopt return, just after paying the retry budget first rather than
/// on the very first sample. Zero is a valid sample and stays in the ranking.
TEST(TestIngestorBenchmarkPlan, MalformedTimerSamplesScoreTheCandidateUnusable)
{
    constexpr double NEGATIVE = -1.0;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();

    for(const double malformed : {NEGATIVE, nan, inf})
    {
        std::vector<RankedEntry> recorded;
        const BenchmarkTestHandle handle;
        auto candidates = threeCandidates();
        const auto* malformedCandidate = candidates[1].plan.get();
        auto timer = makeDeterministicTimer(candidates, {5.0, malformed, 3.0});
        int malformedCalls = 0;
        const TestBenchmarkPlan plan(
            std::move(candidates),
            handle,
            [&](const hipdnn_plugin_sdk::IPlan<BenchmarkTestHandle>& subPlan,
                const BenchmarkTestHandle& planHandle,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* workspace) {
                if(&subPlan == malformedCandidate)
                {
                    ++malformedCalls;
                }
                return timer(subPlan, planHandle, deviceBuffers, numDeviceBuffers, workspace);
            },
            [&recorded](std::vector<RankedEntry> ranking) { recorded = std::move(ranking); });

        plan.execute(handle, nullptr, 0U, nullptr);
        EXPECT_EQ(malformedCalls, malformed < 0.0 ? MAX_NEGATIVE_SAMPLE_RETRIES + 1 : 1);

        ASSERT_EQ(recorded.size(), 2U);
        for(const auto& entry : recorded)
        {
            EXPECT_NE(entry.kernelId, testId(0x02))
                << "a malformed measurement recorded as a fallback would be served ahead of "
                   "the normal ranked path on a later run";
        }
    }
}

/// A single transient negative sample must not disqualify the candidate: sampleCandidate()
/// discards it and re-measures the same slot, so a candidate that only stumbles once
/// still accrues its full BENCHMARK_ITERATIONS valid samples and can win the sweep. This
/// is the case a naive "any negative sample means unusable" check would defeat.
TEST(TestIngestorBenchmarkPlan, ATransientNegativeSampleIsReplacedAndTheCandidateCanStillWin)
{
    auto flaky = std::make_unique<FakePlan>(64);
    auto steady = std::make_unique<FakePlan>(64);
    const auto* flakyRaw = flaky.get();
    const auto* steadyRaw = steady.get();

    std::vector<TestBenchmarkPlan::Candidate> candidates;
    candidates.push_back({testId(0x01), std::move(flaky)});
    candidates.push_back({testId(0x02), std::move(steady)});

    constexpr double FLAKY_SAMPLE = 1.0;
    constexpr double STEADY_SAMPLE = 10.0;

    // Only the very first timed sample is negative; every retry after it is a real,
    // valid measurement.
    int flakyCallIndex = 0;
    const TestBenchmarkPlan::Timer timer
        = [&](const hipdnn_plugin_sdk::IPlan<BenchmarkTestHandle>& plan,
              const BenchmarkTestHandle& planHandle,
              const hipdnnPluginDeviceBuffer_t* deviceBuffers,
              uint32_t numDeviceBuffers,
              void* workspace) -> std::optional<double> {
        plan.execute(planHandle, deviceBuffers, numDeviceBuffers, workspace);
        if(&plan == steadyRaw)
        {
            return STEADY_SAMPLE;
        }
        return flakyCallIndex++ == 0 ? -1.0 : FLAKY_SAMPLE;
    };

    std::vector<RankedEntry> recorded;
    const BenchmarkTestHandle handle;
    const TestBenchmarkPlan plan(std::move(candidates), handle, timer, [&recorded](auto ranking) {
        recorded = std::move(ranking);
    });

    plan.execute(handle, nullptr, 0, nullptr);

    ASSERT_EQ(recorded.size(), 2U) << "the transient negative must not drop the candidate";
    EXPECT_EQ(recorded.front().kernelId, testId(0x01))
        << "the candidate that recovered from a transient negative must still win on its "
           "real, faster time";
    EXPECT_EQ(recorded.front().timeMs, FLAKY_SAMPLE)
        << "a full set of BENCHMARK_ITERATIONS valid samples, all equal to FLAKY_SAMPLE, "
           "must reduce to exactly that value: the discarded negative sample never enters "
           "the reduction";

    // One retry beyond the normal sampling launches for the recovered candidate, plus
    // the delegated winner execute.
    EXPECT_EQ(flakyRaw->launchCount(), SAMPLING_LAUNCHES + 1 + 1);
    EXPECT_EQ(steadyRaw->launchCount(), SAMPLING_LAUNCHES);
}

/// Zero is a valid measurement (an unmeasurably fast launch): it must win and be cached
/// like any other real sample, not be treated as malformed.
TEST(TestIngestorBenchmarkPlan, ZeroTimerSampleIsValidAndCanWin)
{
    std::vector<RankedEntry> recorded;
    const BenchmarkTestHandle handle;
    const auto plan = makeDeterministicPlan(
        threeCandidates(), handle, {5.0, 0.0, 3.0}, [&recorded](std::vector<RankedEntry> ranking) {
            recorded = std::move(ranking);
        });

    plan.execute(handle, nullptr, 0U, nullptr);

    ASSERT_EQ(recorded.size(), 3U);
    EXPECT_EQ(recorded.front().kernelId, testId(0x02));
    EXPECT_EQ(recorded.front().timeMs, 0.0);
}

TEST(TestIngestorBenchmarkPlan, AnAllUnusableSweepRecordsNothing)
{
    bool invoked = false;
    const BenchmarkTestHandle handle;
    const auto plan
        = makeDeterministicPlan(threeCandidates(),
                                handle,
                                {std::nullopt, std::nullopt, std::nullopt},
                                [&invoked](const std::vector<RankedEntry>&) { invoked = true; });

    plan.execute(handle, nullptr, 0U, nullptr);

    EXPECT_FALSE(invoked) << "caching index 0 when nothing was usable would cache a guess";
}

/// The explicit no-caching path: every flag-off caller constructs a BenchmarkPlan
/// without a callback, and selection must be unaffected.
TEST(TestIngestorBenchmarkPlan, AnAbsentCallbackLeavesSelectionUnchanged)
{
    auto candidates = threeCandidates();
    // Hold the deterministic winner's sub-plan to count its launches.
    auto* const expectedWinner = static_cast<FakePlan*>(candidates[1].plan.get());

    const BenchmarkTestHandle handle;
    const auto plan = makeDeterministicPlan(std::move(candidates), handle, {5.0, 1.0, 3.0});

    plan.execute(handle, nullptr, 0U, nullptr);
    const int afterFirst = expectedWinner->launchCount();

    plan.execute(handle, nullptr, 0U, nullptr);

    EXPECT_GT(afterFirst, 0) << "the fastest candidate must be the one that ran";
    EXPECT_GT(expectedWinner->launchCount(), afterFirst)
        << "the second execute must delegate to the same already-chosen winner";
}

/// Ties resolve to the lowest candidate index. std::sort would reorder equal times
/// arbitrarily and silently change which kernel wins.
TEST(TestIngestorBenchmarkPlan, EqualTimesKeepTheLowestCandidateIndexFirst)
{
    std::vector<RankedEntry> recorded;
    const BenchmarkTestHandle handle;
    const auto plan = makeDeterministicPlan(
        threeCandidates(), handle, {2.0, 2.0, 2.0}, [&recorded](std::vector<RankedEntry> ranking) {
            recorded = std::move(ranking);
        });

    plan.execute(handle, nullptr, 0U, nullptr);

    ASSERT_EQ(recorded.size(), 3U);
    EXPECT_EQ(recorded[0].kernelId, testId(0x01));
    EXPECT_EQ(recorded[1].kernelId, testId(0x02));
    EXPECT_EQ(recorded[2].kernelId, testId(0x03));
}

/// The tie-break above cannot distinguish `sort` from `stable_sort`: libstdc++ drops to
/// insertion sort below its introsort threshold, and that fallback happens to be stable,
/// so three tied candidates order the same either way. This runs enough of them to clear
/// the threshold, where an unstable sort genuinely reorders equal elements.
TEST(TestIngestorBenchmarkPlan, EqualTimesKeepCandidateOrderPastTheInsertionSortThreshold)
{
    constexpr size_t TIED_CANDIDATES = 32;

    std::vector<TestBenchmarkPlan::Candidate> candidates;
    candidates.reserve(TIED_CANDIDATES);
    for(size_t index = 0; index < TIED_CANDIDATES; ++index)
    {
        candidates.push_back(
            {testId(static_cast<uint8_t>(index + 1)), std::make_unique<FakePlan>(64)});
    }

    std::vector<RankedEntry> recorded;
    const BenchmarkTestHandle handle;
    const auto plan = makeDeterministicPlan(
        std::move(candidates),
        handle,
        std::vector<std::optional<double>>(TIED_CANDIDATES, 2.0),
        [&recorded](std::vector<RankedEntry> ranking) { recorded = std::move(ranking); });

    plan.execute(handle, nullptr, 0U, nullptr);

    ASSERT_EQ(recorded.size(), TIED_CANDIDATES);
    for(size_t index = 0; index < TIED_CANDIDATES; ++index)
    {
        EXPECT_EQ(recorded[index].kernelId, testId(static_cast<uint8_t>(index + 1)))
            << "candidate at index " << index << " moved; equal times must keep input order";
    }
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
