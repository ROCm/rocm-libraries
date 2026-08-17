// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
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

#include "IngestorMocks.hpp"
#include "KernelIngestorTestFixtures.hpp"

/**
 * @file TestBenchmarkPlan.cpp
 * @brief Unit tests for BenchmarkPlan.hpp: the composite plan GenericPlanBuilder
 *        constructs when `global.benchmarking` is on.
 *
 * Phase 1 ships exactly one case here: the no-knob oracle, proving the benchmarking-off
 * path through buildPlan() is provably untouched -- it never reaches BenchmarkPlan at
 * all. Phase 2 adds BenchmarkPlan's own cases (workspace-as-max, empty-candidate throw,
 * single-candidate execute, winner-resolved-once, all-unusable-falls-back-to-0, and
 * buffer/workspace pass-through) to this same file.
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
/// TestGenericPlanBuilder.cpp's own KnobFilterContext rather than widening a shared
/// fixture for one test's needs.
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

/// Three kernels, no matchers at all (pack.matcherIds is empty, exactly as
/// makeStubStateManager() sets up for its one-kernel case), so every kernel always
/// survives catalog construction and only the heuristic decides rank.
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

/// Phase 1 oracle (plan §8): with benchmarking off and a three-kernel catalog,
/// buildPlan() must take the exact branch it takes today -- one plain GenericPlan for
/// the ranked front, BenchmarkPlan never constructed -- and that plan must launch
/// exactly once, on the kernel the heuristic ranked first. scoreByBlockSize ranks by
/// BLOCK_SIZE, so kernel_256_float (0x65) outranks the two 64-block kernels.
TEST(TestIngestorBenchmarkPlan, BenchmarkingOffBuildsAPlainPlanThatLaunchesTheRankedFrontOnce)
{
    // A leaked override must not make this look benchmarked; the oracle asserts the
    // no-knob path is untouched, so the environment must genuinely be unset here.
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
    // hipdnnExecute with no autotune -- today's behaviour, unchanged.
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
// Task 2.3: BenchmarkPlan's own unit -- construction, resolution, delegation.
// These construct BenchmarkPlan directly, not through buildPlan(), so they never
// exercise GenericPlanBuilder's `if constexpr` branch.
// ---------------------------------------------------------------------------

/// A handle satisfying HasGetStream, local to this file: StubHandle (used by the
/// oracle above) has no getStream(), and BenchmarkPlan's constructor static_asserts
/// it. The null stream is a valid hipStream_t for these tests -- every case here is
/// hardware-independent by construction (see FakePlan below), so it never depends on
/// what the null stream actually does on a given machine.
struct BenchmarkTestHandle
{
    hipStream_t getStream() const
    {
        return nullptr;
    }
};

/// A minimal IPlan double recording every execute() call's arguments and count.
/// Throws on the first @p throwForCalls invocations (default 0, never throws), then
/// succeeds and counts a "launch" thereafter -- this is what makes the all-unusable
/// case hardware-independent: BENCHMARK_WARMUP_RUNS == 1, so a plan that throws on
/// call 1 is caught by sampleCandidate()'s try/catch before the timed loop ever
/// constructs a hipEvent, so the test's outcome cannot depend on whether this machine
/// has a working HIP device.
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

    int callCount() const
    {
        return _callCount;
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

TEST(TestIngestorBenchmarkPlan, ASingleCandidateCompositeExecutesThatOne)
{
    auto sub = std::make_unique<FakePlan>(64);
    const auto* subRaw = sub.get();

    std::vector<TestBenchmarkPlan::Candidate> candidates;
    candidates.push_back({testId(0x01), std::move(sub)});

    const BenchmarkTestHandle handle;
    const TestBenchmarkPlan plan(std::move(candidates), handle);

    plan.execute(handle, nullptr, 0, nullptr);

    EXPECT_EQ(subRaw->launchCount(), 1);
}

/// The winner is resolved once: a second execute() call must add exactly one more
/// launch to whichever candidate won -- never re-run the whole sampling pass, which
/// would add more than one launch (however many resolveChosen's own sampling calls
/// contributed the first time, a quantity that itself varies with HIP device
/// availability and is deliberately not asserted here).
TEST(TestIngestorBenchmarkPlan, TheWinnerIsResolvedOnceAcrossRepeatedExecuteCalls)
{
    auto first = std::make_unique<FakePlan>(64);
    auto second = std::make_unique<FakePlan>(64);
    const auto* firstRaw = first.get();
    const auto* secondRaw = second.get();

    std::vector<TestBenchmarkPlan::Candidate> candidates;
    candidates.push_back({testId(0x01), std::move(first)});
    candidates.push_back({testId(0x02), std::move(second)});

    const BenchmarkTestHandle handle;
    const TestBenchmarkPlan plan(std::move(candidates), handle);

    plan.execute(handle, nullptr, 0, nullptr);
    const int totalLaunchesAfterFirstCall = firstRaw->launchCount() + secondRaw->launchCount();
    ASSERT_GE(totalLaunchesAfterFirstCall, 1) << "the winner must have launched at least once";

    plan.execute(handle, nullptr, 0, nullptr);
    const int totalLaunchesAfterSecondCall = firstRaw->launchCount() + secondRaw->launchCount();

    EXPECT_EQ(totalLaunchesAfterSecondCall - totalLaunchesAfterFirstCall, 1);
}

/// Every candidate throws on its very first invocation -- caught inside
/// sampleCandidate() before any hipEvent is touched, so this is deterministic
/// regardless of whether this machine has a working HIP device. resolveChosen() must
/// fall back to index 0 (logging ERROR) rather than propagating, and the real delegate
/// call that follows -- FakePlan's second invocation -- succeeds, so execute() itself
/// must not throw.
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
    const TestBenchmarkPlan plan(std::move(candidates), handle);

    EXPECT_NO_THROW(plan.execute(handle, nullptr, 0, nullptr));

    // Candidate 0 is the documented fallback: its second call (the real delegate,
    // after its first sampling call threw) must have launched.
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
    const TestBenchmarkPlan plan(std::move(candidates), handle);

    const std::array<hipdnnPluginDeviceBuffer_t, 1> buffers{
        {{/*uid=*/9, /*ptr=*/reinterpret_cast<void*>(0x5678)}}};
    int workspaceStorage = 0;
    void* const workspace = &workspaceStorage;

    plan.execute(handle, buffers.data(), 1U, workspace);

    EXPECT_EQ(subRaw->lastDeviceBuffers(), buffers.data());
    EXPECT_EQ(subRaw->lastNumDeviceBuffers(), 1U);
    EXPECT_EQ(subRaw->lastWorkspace(), workspace);
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
