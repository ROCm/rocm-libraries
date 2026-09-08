// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Verification mode dispatch: AUTO/GOLDEN/GPU/CPU paths.

#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>

#include "BundleFixtureFiles.hpp"
#include "HarnessTestSupport.hpp"
#include "harness/ReferenceCapabilityError.hpp"
#include "harness/bundle/IntegrationBundleVerificationHarness.hpp"
#include "harness/bundle/IntegrationTestBundle.hpp"

// NOLINTBEGIN(readability-identifier-naming)

using namespace hipdnn_integration_tests;
using namespace hipdnn_integration_tests::bundle;

namespace
{

class TestVerificationModePathsFixture : public ::testing::Test
{
protected:
    std::optional<hipdnn_test_sdk::utilities::ScopedDirectory> _scopedDir;
    std::filesystem::path _tempDir;
    testing_support::HarnessMocks _mocks;

    void SetUp() override
    {
        testing_support::ensureTestConfigInitialized();
        _scopedDir.emplace(scratch::makeDir("vmode_test_"));
        _tempDir = _scopedDir->path();
    }

    std::shared_ptr<IntegrationTestBundle> loadBundle(const std::string& name,
                                                      bool includeGoldenOutput) const
    {
        return fixtures::loadBundle(_tempDir, name, includeGoldenOutput);
    }

    /// Builds a real harness wired to this fixture's mocks and drives it through
    /// SetUp()/TestBody(), capturing whatever gtest part results that run produces.
    void runCapturing(std::shared_ptr<IntegrationTestBundle> bundle,
                      VerificationMode mode,
                      ::testing::TestPartResultArray* results)
    {
        IntegrationBundleVerificationHarness harness(
            _mocks.dependencies(testing_support::hostPolicy(mode)));
        harness.setBundle(std::move(bundle), "vmode-test-bundle");
        testing_support::driveHarness(harness, results);
    }

    void useMatchingEngine()
    {
        testing_support::engineWrites(
            _mocks.engineRunner, &fixtures::writeOutput, fixtures::K_OUTPUT_VALUE);
    }

    void useMismatchingEngine()
    {
        testing_support::engineWrites(
            _mocks.engineRunner, &fixtures::writeOutput, fixtures::K_OUTPUT_VALUE + 100.0f);
    }

    static void writeReferenceOutput(::testing::NiceMock<MockReferenceGraphExecutor>& executor,
                                     float value)
    {
        using ::testing::_;
        ON_CALL(executor, execute(_, _, _))
            .WillByDefault([value](void*, size_t, const VariantPack& variantPack) {
                auto* ptr = static_cast<float*>(variantPack.at(fixtures::K_OUTPUT_UID));
                std::fill(ptr, ptr + fixtures::K_OUTPUT_ELEMS, value);
            });
    }

    /// Both reference executors answer with the golden value: whichever one the
    /// dispatch picks (GPU for explicit `gpu`/`auto`'s first try, CPU for explicit
    /// `cpu`/`auto`'s fallback) matches the engine.
    void useMatchingReference()
    {
        writeReferenceOutput(_mocks.cpuReference, fixtures::K_OUTPUT_VALUE);
        writeReferenceOutput(_mocks.gpuReference, fixtures::K_OUTPUT_VALUE);
    }

    /// Neither reference executor can run this op.
    void useCapabilityMissReference()
    {
        using ::testing::_;
        using ::testing::Throw;
        ON_CALL(_mocks.cpuReference, execute(_, _, _))
            .WillByDefault(Throw(ReferenceCapabilityError("stub: unsupported op")));
        ON_CALL(_mocks.gpuReference, execute(_, _, _))
            .WillByDefault(Throw(ReferenceCapabilityError("stub: unsupported op")));
    }

    /// GPU cannot run this op; CPU matches. Exercises AUTO's fallthrough.
    void useGpuMissCpuMatchReference()
    {
        using ::testing::_;
        using ::testing::Throw;
        ON_CALL(_mocks.gpuReference, execute(_, _, _))
            .WillByDefault(Throw(ReferenceCapabilityError("stub: no GPU ref plan")));
        writeReferenceOutput(_mocks.cpuReference, fixtures::K_OUTPUT_VALUE);
    }

    /// GPU throws a real error (not a capability miss); CPU matches. Exercises
    /// AUTO's RUNTIME_ERROR branch, which is distinct from a capability miss: it
    /// must publish a reference error before falling through to CPU.
    void useGpuErrorCpuMatchReference()
    {
        using ::testing::_;
        using ::testing::Throw;
        ON_CALL(_mocks.gpuReference, execute(_, _, _))
            .WillByDefault(Throw(std::runtime_error("stub: GPU ref crashed")));
        writeReferenceOutput(_mocks.cpuReference, fixtures::K_OUTPUT_VALUE);
    }
};

} // namespace

// ── AUTO mode ───────────────────────────────────────────────────────────────

TEST_F(TestVerificationModePathsFixture, AutoWithGoldenUsesGoldenAndPasses)
{
    using ::testing::_;
    useMatchingEngine();
    EXPECT_CALL(_mocks.referenceExecutors, get(_)).Times(0);

    ::testing::TestPartResultArray results;
    runCapturing(
        loadBundle("auto_golden", /*includeGoldenOutput=*/true), VerificationMode::AUTO, &results);

    EXPECT_FALSE(testing_support::anyFailed(results));
    EXPECT_FALSE(testing_support::anySkipped(results));
}

TEST_F(TestVerificationModePathsFixture, AutoWithGoldenMismatchFails)
{
    useMismatchingEngine();

    ::testing::TestPartResultArray results;
    runCapturing(loadBundle("auto_golden_mm", /*includeGoldenOutput=*/true),
                 VerificationMode::AUTO,
                 &results);

    EXPECT_TRUE(testing_support::anyFailed(results));
}

TEST_F(TestVerificationModePathsFixture, AutoNoGoldenRefSucceedsPasses)
{
    using ::testing::ReturnRef;
    useMatchingEngine();
    useMatchingReference();

    // AUTO tries GPU first when there is no golden data; both stubs return the
    // golden value, so only pinning the dispatch itself catches a swapped order.
    EXPECT_CALL(_mocks.referenceExecutors, get(ReferenceExecutorType::GPU))
        .Times(1)
        .WillRepeatedly(ReturnRef(_mocks.gpuReference));
    EXPECT_CALL(_mocks.referenceExecutors, get(ReferenceExecutorType::CPU)).Times(0);

    ::testing::TestPartResultArray results;
    runCapturing(
        loadBundle("auto_gpu", /*includeGoldenOutput=*/false), VerificationMode::AUTO, &results);

    EXPECT_FALSE(testing_support::anyFailed(results));
    EXPECT_FALSE(testing_support::anySkipped(results));
}

TEST_F(TestVerificationModePathsFixture, AutoNoGoldenRefMissFallsThroughToCpu)
{
    using ::testing::ReturnRef;
    useMatchingEngine();
    useGpuMissCpuMatchReference();

    // GPU must be consulted before CPU is tried. The "GPU miss" stub throws a
    // capability error either way, so only the call sequence — not the pass/fail
    // outcome — can catch a swapped dispatch order.
    {
        const ::testing::InSequence seq;
        EXPECT_CALL(_mocks.referenceExecutors, get(ReferenceExecutorType::GPU))
            .WillOnce(ReturnRef(_mocks.gpuReference));
        EXPECT_CALL(_mocks.referenceExecutors, get(ReferenceExecutorType::CPU))
            .WillOnce(ReturnRef(_mocks.cpuReference));
    }

    ::testing::TestPartResultArray results;
    runCapturing(loadBundle("auto_fallthrough", /*includeGoldenOutput=*/false),
                 VerificationMode::AUTO,
                 &results);

    EXPECT_FALSE(testing_support::anyFailed(results));
    EXPECT_FALSE(testing_support::anySkipped(results));
}

TEST_F(TestVerificationModePathsFixture, AutoNoGoldenBothRefsMissSkips)
{
    useMatchingEngine();
    useCapabilityMissReference();

    ::testing::TestPartResultArray results;
    runCapturing(loadBundle("auto_both_miss", /*includeGoldenOutput=*/false),
                 VerificationMode::AUTO,
                 &results);

    EXPECT_TRUE(testing_support::anySkipped(results));
    EXPECT_FALSE(testing_support::anyFailed(results));
}

// The other GPU miss form: a real runtime error, not a capability miss. AUTO
// still falls through to CPU and passes, but unlike a plain capability miss it
// must be loud about it — a reference error naming the GPU failure is published
// before CPU ever runs.
TEST_F(TestVerificationModePathsFixture, AutoNoGoldenRefRuntimeErrorFallsThroughToCpu)
{
    useMatchingEngine();
    useGpuErrorCpuMatchReference();

    std::vector<std::string> refErrors;
    testing_support::captureReferenceErrors(_mocks.reporter, refErrors);

    ::testing::TestPartResultArray results;
    runCapturing(loadBundle("auto_ref_runtime_error", /*includeGoldenOutput=*/false),
                 VerificationMode::AUTO,
                 &results);

    EXPECT_FALSE(testing_support::anyFailed(results));
    EXPECT_FALSE(testing_support::anySkipped(results));
    ASSERT_EQ(refErrors.size(), 1U);
    EXPECT_THAT(refErrors.front(), ::testing::HasSubstr("GPU reference errored"));
    EXPECT_THAT(refErrors.front(), ::testing::HasSubstr("stub: GPU ref crashed"));
}

// ── GOLDEN mode ─────────────────────────────────────────────────────────────

TEST_F(TestVerificationModePathsFixture, GoldenModeWithDataPasses)
{
    using ::testing::_;
    useMatchingEngine();
    EXPECT_CALL(_mocks.referenceExecutors, get(_)).Times(0);

    ::testing::TestPartResultArray results;
    runCapturing(
        loadBundle("golden_ok", /*includeGoldenOutput=*/true), VerificationMode::GOLDEN, &results);

    EXPECT_FALSE(testing_support::anyFailed(results));
    EXPECT_FALSE(testing_support::anySkipped(results));
}

// An explicit mode is a demand for a specific oracle, not a preference. Skipping
// when that oracle is absent means the run did not do what it was asked and still
// went green — `auto` is the mode with a fallback chain.
TEST_F(TestVerificationModePathsFixture, GoldenModeWithoutDataFails)
{
    useMatchingEngine();
    useMatchingReference();

    ::testing::TestPartResultArray results;
    runCapturing(loadBundle("golden_absent", /*includeGoldenOutput=*/false),
                 VerificationMode::GOLDEN,
                 &results);

    EXPECT_TRUE(testing_support::anyFailed(results));
    EXPECT_FALSE(testing_support::anySkipped(results));
}

// A live reference is available here and would have produced an answer, but the
// caller asked for golden. Falling back silently is the behaviour `auto` exists
// for; `golden` must not borrow it.
TEST_F(TestVerificationModePathsFixture, GoldenModeDoesNotFallBackToAReference)
{
    using ::testing::_;
    useMatchingEngine();
    useMatchingReference();
    EXPECT_CALL(_mocks.referenceExecutors, get(_)).Times(0);

    ::testing::TestPartResultArray results;
    runCapturing(loadBundle("golden_absent_ref_ok", /*includeGoldenOutput=*/false),
                 VerificationMode::GOLDEN,
                 &results);

    EXPECT_TRUE(testing_support::anyFailed(results));
}

// ── Explicit GPU mode ───────────────────────────────────────────────────────
// "Device" in these case names denotes VerificationMode::GPU (the device-side
// reference executor). The literal "Gpu" keyword is reserved by the test-name
// linter for the suite name and so cannot appear in the case name.

TEST_F(TestVerificationModePathsFixture, DeviceModeRefSucceedsPasses)
{
    using ::testing::ReturnRef;
    useMatchingEngine();
    useMatchingReference();

    // Explicit GPU mode must dispatch the GPU executor, never CPU — both stubs
    // return the golden value, so a wrong dispatch would otherwise pass unnoticed.
    EXPECT_CALL(_mocks.referenceExecutors, get(ReferenceExecutorType::GPU))
        .Times(1)
        .WillRepeatedly(ReturnRef(_mocks.gpuReference));
    EXPECT_CALL(_mocks.referenceExecutors, get(ReferenceExecutorType::CPU)).Times(0);

    ::testing::TestPartResultArray results;
    runCapturing(
        loadBundle("gpu_ok", /*includeGoldenOutput=*/true), VerificationMode::GPU, &results);

    EXPECT_FALSE(testing_support::anyFailed(results));
    EXPECT_FALSE(testing_support::anySkipped(results));
}

TEST_F(TestVerificationModePathsFixture, DeviceModeCapabilityMissSkips)
{
    useMatchingEngine();
    useCapabilityMissReference();

    ::testing::TestPartResultArray results;
    runCapturing(
        loadBundle("gpu_miss", /*includeGoldenOutput=*/true), VerificationMode::GPU, &results);

    EXPECT_TRUE(testing_support::anySkipped(results));
    EXPECT_FALSE(testing_support::anyFailed(results));
}

// ── Explicit CPU mode ───────────────────────────────────────────────────────

TEST_F(TestVerificationModePathsFixture, CpuModeRefSucceedsPasses)
{
    using ::testing::ReturnRef;
    useMatchingEngine();
    useMatchingReference();

    // Explicit CPU mode must dispatch the CPU executor, never GPU — both stubs
    // return the golden value, so a wrong dispatch would otherwise pass unnoticed.
    EXPECT_CALL(_mocks.referenceExecutors, get(ReferenceExecutorType::CPU))
        .Times(1)
        .WillRepeatedly(ReturnRef(_mocks.cpuReference));
    EXPECT_CALL(_mocks.referenceExecutors, get(ReferenceExecutorType::GPU)).Times(0);

    ::testing::TestPartResultArray results;
    runCapturing(
        loadBundle("cpu_ok", /*includeGoldenOutput=*/true), VerificationMode::CPU, &results);

    EXPECT_FALSE(testing_support::anyFailed(results));
    EXPECT_FALSE(testing_support::anySkipped(results));
}

TEST_F(TestVerificationModePathsFixture, CpuModeCapabilityMissSkips)
{
    useMatchingEngine();
    useCapabilityMissReference();

    ::testing::TestPartResultArray results;
    runCapturing(
        loadBundle("cpu_miss", /*includeGoldenOutput=*/true), VerificationMode::CPU, &results);

    EXPECT_TRUE(testing_support::anySkipped(results));
    EXPECT_FALSE(testing_support::anyFailed(results));
}

// ── Enforcement-level gate ──────────────────────────────────────────────────
// runComparison() routes on enforcement level alone: a non-FULL bundle reaches
// enforceAtLevel() regardless of --enforce-support-claims, which only controls
// what checkSupportClaims() does earlier in TestBody(). enforceAtLevel() is no
// longer stubbable, so these assert its real rung behaviour directly through the
// engine-runner mock instead of an intercepted EnforcementLevel value.
//
// Both rungs need a named engine to check applicability against, so these two
// construct the harness with one. Without it the real rung returns unverifiable
// ("enforcement_level requires --test-engine") — pinned separately in
// TestEnforcementRungs.EnforcementRungWithoutAnEngineIsUnverifiable.

TEST_F(TestVerificationModePathsFixture, NonFullBundleRoutesToEnforcementPath)
{
    using ::testing::_;
    auto bundle = loadBundle("enforce_gate", /*includeGoldenOutput=*/true);
    bundle->metadata.enforcementLevel = EnforcementLevel::APPLICABILITY;

    // APPLICABILITY with an accepted session (the default openGraph() stub) passes
    // without ever compiling plans, and never reaches the comparison path.
    EXPECT_CALL(_mocks.engineRunner, buildPlans(_, _)).Times(0);
    EXPECT_CALL(_mocks.engineRunner, execute(_, _, _)).Times(0);

    IntegrationBundleVerificationHarness harness(
        _mocks.dependencies(testing_support::hostPolicy(VerificationMode::AUTO)),
        LoadedEngine{11, "ENGINE_UNDER_TEST"});
    harness.setBundle(std::move(bundle), "vmode-test-bundle");

    ::testing::TestPartResultArray results;
    {
        const ::testing::ScopedFakeTestPartResultReporter reporter(
            ::testing::ScopedFakeTestPartResultReporter::INTERCEPT_ALL_THREADS, &results);
        harness.SetUp();
        harness.TestBody();
    }

    EXPECT_FALSE(testing_support::anyFailed(results))
        << "APPLICABILITY rung with an accepted session must pass";
    EXPECT_FALSE(testing_support::anySkipped(results))
        << "APPLICABILITY rung with an accepted session must pass, not skip";
}

TEST_F(TestVerificationModePathsFixture, BuildableBundleRoutesToBuildPlans)
{
    using ::testing::_;
    auto bundle = loadBundle("enforce_gate_buildable", /*includeGoldenOutput=*/true);
    bundle->metadata.enforcementLevel = EnforcementLevel::BUILDABLE;

    // BUILDABLE additionally compiles plans; the default buildPlans() stub
    // succeeds, so this passes without reaching the comparison path either.
    EXPECT_CALL(_mocks.engineRunner, buildPlans(_, _)).Times(1);
    EXPECT_CALL(_mocks.engineRunner, execute(_, _, _)).Times(0);

    IntegrationBundleVerificationHarness harness(
        _mocks.dependencies(testing_support::hostPolicy(VerificationMode::AUTO)),
        LoadedEngine{11, "ENGINE_UNDER_TEST"});
    harness.setBundle(std::move(bundle), "vmode-test-bundle");

    ::testing::TestPartResultArray results;
    {
        const ::testing::ScopedFakeTestPartResultReporter reporter(
            ::testing::ScopedFakeTestPartResultReporter::INTERCEPT_ALL_THREADS, &results);
        harness.SetUp();
        harness.TestBody();
    }

    EXPECT_FALSE(testing_support::anyFailed(results))
        << "BUILDABLE rung with a successful buildPlans() must pass";
    EXPECT_FALSE(testing_support::anySkipped(results))
        << "BUILDABLE rung with a successful buildPlans() must pass, not skip";
}

TEST_F(TestVerificationModePathsFixture, FullBundleRoutesToComparisonPath)
{
    using ::testing::_;
    auto bundle = loadBundle("enforce_gate_full", /*includeGoldenOutput=*/true);
    bundle->metadata.enforcementLevel = EnforcementLevel::FULL;
    useMatchingEngine();

    // FULL uses neither rung: no plans are compiled, and the comparison path's
    // engine execution runs instead.
    EXPECT_CALL(_mocks.engineRunner, buildPlans(_, _)).Times(0);
    EXPECT_CALL(_mocks.engineRunner, execute(_, _, _)).Times(1);

    IntegrationBundleVerificationHarness harness(
        _mocks.dependencies(testing_support::hostPolicy(VerificationMode::AUTO)));
    harness.setBundle(std::move(bundle), "vmode-test-bundle");

    ::testing::TestPartResultArray results;
    {
        const ::testing::ScopedFakeTestPartResultReporter reporter(
            ::testing::ScopedFakeTestPartResultReporter::INTERCEPT_ALL_THREADS, &results);
        harness.SetUp();
        harness.TestBody();
    }

    EXPECT_FALSE(testing_support::anyFailed(results)) << "FULL bundle must run the comparison path";
}

// NOLINTEND(readability-identifier-naming)
