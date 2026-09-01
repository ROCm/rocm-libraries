// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

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
/// StubHandle (used by the oracle above) has no getStream(). The injected timer never
/// records an event, so the null stream's behaviour never matters.
struct BenchmarkTestHandle
{
    // Non-static: models a real handle's instance accessor, which is what HasGetStream
    // detects.
    // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
    hipStream_t getStream() const
    {
        return nullptr;
    }
};

/// A minimal IPlan double recording every execute() call's arguments and count.
/// Throws on the first @p throwForCalls invocations (default 0, never throws), then
/// succeeds and counts a "launch".
class FakePlan : public hipdnn_plugin_sdk::IPlan<BenchmarkTestHandle>
{
public:
    explicit FakePlan(size_t workspaceSize = 0, int throwForCalls = 0)
        : _workspaceSize(workspaceSize)
        , _throwForCalls(throwForCalls)
    {
    }

    size_t getWorkspaceSize(const BenchmarkTestHandle& /*handle*/) const override
    {
        return _workspaceSize;
    }

    void execute(const BenchmarkTestHandle& /*handle*/,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override
    {
        ++_callCount;
        _lastDeviceBuffers = deviceBuffers;
        _lastNumDeviceBuffers = numDeviceBuffers;
        _lastWorkspace = workspace;
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
    mutable int _callCount = 0;
    mutable int _launchCount = 0;
    mutable const hipdnnPluginDeviceBuffer_t* _lastDeviceBuffers = nullptr;
    mutable uint32_t _lastNumDeviceBuffers = 0;
    mutable void* _lastWorkspace = nullptr;
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
        if(&plan == steadyRaw)
        {
            return STEADY_SAMPLE;
        }
        return SPIKY_SAMPLES.at(spikySampleIndex++);
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

/// The only case exercising the DEFAULT timer: no timer argument, so makeHipEventTimer()
/// runs for real against HIP. Every other case here injects a fake, which would let a
/// regression in event creation, event reuse across samples, the record/synchronize
/// pair, or the elapsed-time read pass unnoticed.
///
/// Asserting the exact count is what makes it meaningful: the timer must have returned a
/// duration on all BENCHMARK_ITERATIONS samples. A single nullopt would score the
/// candidate unusable and drop the count to BENCHMARK_WARMUP_RUNS.
TEST(TestIngestorBenchmarkPlan, TheDefaultTimerTimesEverySampleAgainstRealHipEvents)
{
    SKIP_IF_NO_DEVICES();

    auto sub = std::make_unique<FakePlan>(64);
    const auto* subRaw = sub.get();

    std::vector<TestBenchmarkPlan::Candidate> candidates;
    candidates.push_back({testId(0x01), std::move(sub)});

    const BenchmarkTestHandle handle;
    const TestBenchmarkPlan plan(std::move(candidates), handle);

    plan.execute(handle, nullptr, 0, nullptr);

    EXPECT_EQ(subRaw->launchCount(), SAMPLING_LAUNCHES + 1)
        << "the HIP-event timer failed a sample; the candidate was scored unusable";

    // The event pair is created once and re-recorded, so a second sweep-free execute()
    // still delegates exactly once.
    plan.execute(handle, nullptr, 0, nullptr);
    EXPECT_EQ(subRaw->launchCount(), SAMPLING_LAUNCHES + 2);
}

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
        return index < times.size() ? times[index] : std::nullopt;
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
