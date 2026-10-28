// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <hipdnn_plugin_sdk/GlobalKnobDefines.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>
#include <hipdnn_test_sdk/utilities/ScratchDirectory.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "BundleFixtureFiles.hpp"
#include "HarnessTestSupport.hpp"
#include "harness/Benchmark.hpp"
#include "harness/BenchmarkReport.hpp"
#include "harness/ReferenceBenchmarkSync.hpp"
#include "harness/ReferenceCapabilityError.hpp"

// NOLINTBEGIN(readability-identifier-naming)

using namespace hipdnn_integration_tests;
using namespace hipdnn_integration_tests::bundle;
using hipdnn_frontend::AutotuneResult;
using hipdnn_frontend::TimingQuality;

namespace
{

BenchmarkSettings settings(int timed, int warmup)
{
    BenchmarkSettings result;
    result.timedIterations = timed;
    result.warmupIterations = warmup;
    return result;
}

AutotuneResult successfulResult()
{
    AutotuneResult result;
    result.engineName = "TEST_ENGINE";
    result.succeeded = true;
    result.iterationsRun = 3;
    result.minTimeMs = 0.25f;
    result.robustTimeMs = 0.375f;
    result.avgTimeMs = 0.5f;
    result.stddevMs = 0.125f;
    result.timingQuality = TimingQuality::DEVICE_ONLY;
    return result;
}

AutotuneResult failedResult(std::string message)
{
    AutotuneResult result;
    result.errorMessage = std::move(message);
    return result;
}

} // namespace

TEST(TestReferenceBenchmark, RejectsModesThatDoNotExecute)
{
    EXPECT_FALSE(validateReferenceBenchmark({}).has_value());
    EXPECT_FALSE(validateReferenceBenchmark({true}).has_value());
    for(const bool enabled : {true, false})
    {
        for(const auto& request : {ReferenceBenchmarkRequest{enabled, true, false, false},
                                   ReferenceBenchmarkRequest{enabled, false, true, false},
                                   ReferenceBenchmarkRequest{enabled, false, false, true}})
        {
            // Without the flag these modes are ordinary runs; refusing them would make
            // main() exit 1 on every run that never asked for reference timing.
            if(!enabled)
            {
                EXPECT_FALSE(validateReferenceBenchmark(request).has_value());
                continue;
            }
            ASSERT_TRUE(validateReferenceBenchmark(request).has_value());
            EXPECT_NE(validateReferenceBenchmark(request)->find("--benchmark-reference"),
                      std::string::npos);
        }
    }
}

TEST(TestReferenceBenchmark, StatusNamesAreThePublishedVocabulary)
{
    EXPECT_STREQ(toString(ReferenceBenchmarkStatus::PENDING), "pending");
    EXPECT_STREQ(toString(ReferenceBenchmarkStatus::VERIFIED), "verified");
    EXPECT_STREQ(toString(ReferenceBenchmarkStatus::MISMATCH), "mismatch");
    EXPECT_STREQ(toString(ReferenceBenchmarkStatus::DECLINED), "declined");
    EXPECT_STREQ(toString(ReferenceBenchmarkStatus::ERRORED), "error");
}

TEST(TestReferenceBenchmark, FailedSynchronizeIsAReferenceBenchmarkError)
{
    EXPECT_NO_THROW(throwIfReferenceSyncFailed(hipSuccess));
    try
    {
        throwIfReferenceSyncFailed(hipErrorInvalidValue);
        FAIL() << "a failed synchronize must throw";
    }
    catch(const ReferenceBenchmarkError& e)
    {
        EXPECT_NE(std::string(e.what()).find("synchronization failed"), std::string::npos);
    }
}

// The production timer must synchronize the device, or a GPU reference's timing
// covers only its kernel launches. Queue enough asynchronous work that it is still
// running when the work callable returns: the stream is idle afterwards only if the
// timer waited for it.
TEST(TestReferenceBenchmark, ProductionTimerWaitsForQueuedDeviceWork)
{
    SKIP_IF_NO_DEVICES();
    constexpr size_t BYTES = size_t{256} << 20;
    constexpr int FILLS = 16;
    void* buffer = nullptr;
    ASSERT_EQ(hipMalloc(&buffer, BYTES), hipSuccess);
    const auto timer = productionReferenceTimer();
    double elapsedMs = 0.0;
    EXPECT_NO_THROW(timer.measure(elapsedMs, [&] {
        for(int fill = 0; fill < FILLS; ++fill)
        {
            static_cast<void>(hipMemsetAsync(buffer, fill, BYTES, nullptr));
        }
    }));
    EXPECT_EQ(hipStreamQuery(nullptr), hipSuccess);
    EXPECT_GT(elapsedMs, 0.0);
    EXPECT_EQ(hipFree(buffer), hipSuccess);
}

TEST(TestReferenceBenchmark, ExcludesPendingEngineWorkAndIncludesReferenceCompletion)
{
    using Clock = ReferenceBenchmarkTimer::Clock;
    auto now = Clock::time_point{};
    int synchronizations = 0;
    const ReferenceBenchmarkTimer timer(
        [&] { now += std::chrono::milliseconds(++synchronizations == 1 ? 100 : 7); },
        [&] { return now; });
    ReferenceBenchmarkResult result;
    timer.measure(result.executeMs, [&] { now += std::chrono::milliseconds(3); });

    EXPECT_EQ(synchronizations, 2);
    EXPECT_DOUBLE_EQ(result.executeMs, 10.0);
    result.prepareMs = 2.0;
    result.materializeMs = 4.0;
    EXPECT_DOUBLE_EQ(result.totalMs(), 16.0);
}

TEST(TestReferenceBenchmark, GoldenOutputsAreNotReadWithSharedInputs)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory directory(
        hipdnn_test_sdk::utilities::claimScratchDirectory("reference_benchmark_golden"));
    const auto bundle = fixtures::loadBundle(directory.path(), "golden", true);
    auto tensors = bundle->loadInputs();
    EXPECT_EQ(tensors.count(fixtures::K_OUTPUT_UID), 0u);
    EXPECT_EQ(tensors.size(), 5u);
    bundle->loadGoldenOutputs(tensors);
    ASSERT_EQ(tensors.count(fixtures::K_OUTPUT_UID), 1u);
    const auto* data = static_cast<const float*>(tensors.at(fixtures::K_OUTPUT_UID)->rawHostData());
    EXPECT_FLOAT_EQ(data[0], fixtures::K_OUTPUT_VALUE);
    EXPECT_FLOAT_EQ(data[fixtures::K_OUTPUT_ELEMS - 1], fixtures::K_OUTPUT_VALUE);
}

TEST(TestReferenceBenchmark, ReportsFailedFallbackAttemptsWithoutSuccessfulSamples)
{
    using ::testing::_;
    using ::testing::Return;
    testing_support::ensureTestConfigInitialized();
    for(const bool throwDuringProbe : {false, true})
    {
        const hipdnn_test_sdk::utilities::ScopedDirectory directory(
            hipdnn_test_sdk::utilities::claimScratchDirectory("reference_benchmark_fallback"));
        testing_support::HarnessMocks mocks;
        testing_support::engineWrites(
            mocks.engineRunner, fixtures::writeOutput, fixtures::K_OUTPUT_VALUE);
        ON_CALL(mocks.gpuReference, isApplicable(_, _))
            .WillByDefault([throwDuringProbe](void*, size_t) {
                if(throwDuringProbe)
                {
                    throw std::runtime_error("probe failed");
                }
                return false;
            });
        ON_CALL(mocks.cpuReference, execute(_, _, _))
            .WillByDefault([](void*, size_t, const VariantPack& pack) {
                auto writable = pack;
                fixtures::writeOutput(writable, fixtures::K_OUTPUT_VALUE);
            });
        auto policy = testing_support::hostPolicy();
        policy.benchmarkReference = true;
        IntegrationBundleVerificationHarness harness(mocks.dependencies(policy));
        harness.setBundle(fixtures::loadBundle(directory.path(), "fallback", false), "fallback");
        ::testing::TestPartResultArray results;
        testing_support::driveHarness(harness, &results);

        EXPECT_FALSE(testing_support::anyFailed(results)) << testing_support::allMessages(results);
        const auto& timings = harness.referenceBenchmarks();
        ASSERT_EQ(timings.size(), 2u);
        EXPECT_EQ(timings[0].reference, "gpu");
        EXPECT_EQ(timings[0].status,
                  throwDuringProbe ? ReferenceBenchmarkStatus::ERRORED
                                   : ReferenceBenchmarkStatus::DECLINED);
        EXPECT_EQ(timings[1].reference, "cpu");
        EXPECT_EQ(timings[1].status, ReferenceBenchmarkStatus::VERIFIED);
    }
}

namespace
{

/// What one bundle run with a failing reference synchronize did.
struct SyncFailureRun
{
    int synchronizations = 0;
    int cpuReferenceRequests = 0;
    bool failed = false;
    std::string messages;
    std::vector<ReferenceBenchmarkResult> timings;
};

/// Drives one bundle under --benchmark-reference with a timer whose `failingCall`th
/// synchronize throws; 0 never throws, which counts the synchronizes a clean run makes.
SyncFailureRun runWithFailingSync(VerificationMode mode, bool golden, int failingCall)
{
    using ::testing::_;
    SyncFailureRun run;
    const hipdnn_test_sdk::utilities::ScopedDirectory directory(
        hipdnn_test_sdk::utilities::claimScratchDirectory("reference_benchmark_sync_error"));
    testing_support::HarnessMocks mocks;
    testing_support::engineWrites(
        mocks.engineRunner, fixtures::writeOutput, fixtures::K_OUTPUT_VALUE);
    ON_CALL(mocks.referenceExecutors, get(ReferenceExecutorType::CPU))
        .WillByDefault([&](ReferenceExecutorType) -> IReferenceGraphExecutor& {
            ++run.cpuReferenceRequests;
            return mocks.cpuReference;
        });
    for(auto* reference : {&mocks.cpuReference, &mocks.gpuReference})
    {
        ON_CALL(*reference, execute(_, _, _))
            .WillByDefault([](void*, size_t, const VariantPack& pack) {
                auto writable = pack;
                fixtures::writeOutput(writable, fixtures::K_OUTPUT_VALUE);
            });
    }
    auto policy = testing_support::hostPolicy(mode);
    policy.benchmarkReference = true;
    auto deps = mocks.dependencies(policy);
    deps.referenceTimer = ReferenceBenchmarkTimer([&run, failingCall] {
        if(++run.synchronizations == failingCall)
        {
            throw ReferenceBenchmarkError("reference benchmark synchronization failed");
        }
    });
    IntegrationBundleVerificationHarness harness(std::move(deps));
    harness.setBundle(fixtures::loadBundle(directory.path(), "sync_error", golden), "sync_error");
    ::testing::TestPartResultArray results;
    testing_support::driveHarness(harness, &results);
    run.failed = testing_support::anyFailed(results);
    run.messages = testing_support::allMessages(results);
    run.timings = harness.referenceBenchmarks();
    return run;
}

std::map<std::string, std::string> recordedProperties()
{
    const auto* result = ::testing::UnitTest::GetInstance()->current_test_info()->result();
    std::map<std::string, std::string> properties;
    for(int index = 0; index < result->test_property_count(); ++index)
    {
        const auto& property = result->GetTestProperty(index);
        properties[property.key()] = property.value();
    }
    return properties;
}

} // namespace

// Each reference phase (probe, prepare, execute, materialize) synchronizes before and
// after. A failure at any of them is the harness's, so it must fail the test rather
// than let AUTO fall through and report a verified CPU reference.
TEST(TestReferenceBenchmark, SynchronizationFailureInAnyPhaseStopsBeforeReferenceFallback)
{
    testing_support::ensureTestConfigInitialized();
    constexpr int REFERENCE_SYNCHRONIZATIONS = 8;
    const auto clean = runWithFailingSync(VerificationMode::AUTO, false, 0);
    ASSERT_FALSE(clean.failed) << clean.messages;
    ASSERT_EQ(clean.synchronizations, REFERENCE_SYNCHRONIZATIONS);

    for(int failingCall = 1; failingCall <= REFERENCE_SYNCHRONIZATIONS; ++failingCall)
    {
        SCOPED_TRACE("failing synchronize " + std::to_string(failingCall));
        const auto run = runWithFailingSync(VerificationMode::AUTO, false, failingCall);
        EXPECT_TRUE(run.failed);
        EXPECT_NE(run.messages.find("synchronization failed"), std::string::npos) << run.messages;
        EXPECT_EQ(run.synchronizations, failingCall);
        EXPECT_EQ(run.cpuReferenceRequests, 0);
        ASSERT_EQ(run.timings.size(), 1u);
        EXPECT_EQ(run.timings[0].reference, "gpu");
        EXPECT_EQ(run.timings[0].status, ReferenceBenchmarkStatus::ERRORED);
    }
}

TEST(TestReferenceBenchmark, SynchronizationFailureAroundGoldenOutputsIsAFailure)
{
    testing_support::ensureTestConfigInitialized();
    constexpr int GOLDEN_SYNCHRONIZATIONS = 4;
    const auto clean = runWithFailingSync(VerificationMode::GOLDEN, true, 0);
    ASSERT_FALSE(clean.failed) << clean.messages;
    ASSERT_EQ(clean.synchronizations, GOLDEN_SYNCHRONIZATIONS);

    for(int failingCall = 1; failingCall <= GOLDEN_SYNCHRONIZATIONS; ++failingCall)
    {
        SCOPED_TRACE("failing synchronize " + std::to_string(failingCall));
        const auto run = runWithFailingSync(VerificationMode::GOLDEN, true, failingCall);
        EXPECT_TRUE(run.failed);
        EXPECT_NE(run.messages.find("synchronization failed"), std::string::npos) << run.messages;
        EXPECT_EQ(run.synchronizations, failingCall);
        ASSERT_EQ(run.timings.size(), 1u);
        EXPECT_EQ(run.timings[0].reference, "golden");
        EXPECT_EQ(run.timings[0].status, ReferenceBenchmarkStatus::ERRORED);
    }
}

// isApplicable() said yes, but execute() found a capability gap: the attempt is
// declined, has no sample, and the next reference verifies.
TEST(TestReferenceBenchmark, AReferenceDecliningAtExecuteHasNoSample)
{
    using ::testing::_;
    testing_support::ensureTestConfigInitialized();
    const hipdnn_test_sdk::utilities::ScopedDirectory directory(
        hipdnn_test_sdk::utilities::claimScratchDirectory("reference_benchmark_declined"));
    testing_support::HarnessMocks mocks;
    testing_support::engineWrites(
        mocks.engineRunner, fixtures::writeOutput, fixtures::K_OUTPUT_VALUE);
    ON_CALL(mocks.gpuReference, execute(_, _, _))
        .WillByDefault([](void*, size_t, const VariantPack&) {
            throw ReferenceCapabilityError("no kernel for this op");
        });
    ON_CALL(mocks.cpuReference, execute(_, _, _))
        .WillByDefault([](void*, size_t, const VariantPack& pack) {
            auto writable = pack;
            fixtures::writeOutput(writable, fixtures::K_OUTPUT_VALUE);
        });
    auto policy = testing_support::hostPolicy();
    policy.benchmarkReference = true;
    IntegrationBundleVerificationHarness harness(mocks.dependencies(policy));
    harness.setBundle(fixtures::loadBundle(directory.path(), "declined", false), "declined");
    ::testing::TestPartResultArray results;
    testing_support::driveHarness(harness, &results);

    EXPECT_FALSE(testing_support::anyFailed(results)) << testing_support::allMessages(results);
    const auto& timings = harness.referenceBenchmarks();
    ASSERT_EQ(timings.size(), 2u);
    EXPECT_EQ(timings[0].reference, "gpu");
    EXPECT_EQ(timings[0].status, ReferenceBenchmarkStatus::DECLINED);
    EXPECT_FALSE(timings[0].hasSample());
    EXPECT_EQ(timings[1].reference, "cpu");
    EXPECT_EQ(timings[1].status, ReferenceBenchmarkStatus::VERIFIED);
    const auto properties = recordedProperties();
    EXPECT_EQ(properties.at("reference_benchmark_gpu_status"), "declined");
    EXPECT_EQ(properties.count("reference_benchmark_gpu_total_ms"), 0u);
    EXPECT_EQ(properties.at("reference_benchmark_cpu_status"), "verified");
}

// Under the flag the golden outputs are read inside the reference timing instead of
// with the inputs. An unreadable blob must still fail the test the same way.
TEST(TestReferenceBenchmark, UnreadableGoldenBlobFailsAsItDoesWithoutTheFlag)
{
    testing_support::ensureTestConfigInitialized();
    for(const bool benchmarkReference : {false, true})
    {
        SCOPED_TRACE(benchmarkReference ? "with --benchmark-reference"
                                        : "without --benchmark-reference");
        const hipdnn_test_sdk::utilities::ScopedDirectory directory(
            hipdnn_test_sdk::utilities::claimScratchDirectory("reference_benchmark_corrupt"));
        const auto bundleDir = directory.path() / "corrupt";
        fixtures::writeBundleFiles(bundleDir, "corrupt", true);
        std::filesystem::resize_file(
            bundleDir / ("corrupt.tensor" + std::to_string(fixtures::K_OUTPUT_UID) + ".bin"),
            sizeof(float));
        testing_support::HarnessMocks mocks;
        testing_support::engineWrites(
            mocks.engineRunner, fixtures::writeOutput, fixtures::K_OUTPUT_VALUE);
        auto policy = testing_support::hostPolicy(VerificationMode::GOLDEN);
        policy.benchmarkReference = benchmarkReference;
        IntegrationBundleVerificationHarness harness(mocks.dependencies(policy));
        harness.setBundle(fixtures::loadWrittenBundle(bundleDir, "corrupt"), "corrupt");
        ::testing::TestPartResultArray results;
        testing_support::driveHarness(harness, &results);

        EXPECT_TRUE(testing_support::anyFailed(results));
        EXPECT_NE(testing_support::allMessages(results).find("tensor data failed to load"),
                  std::string::npos)
            << testing_support::allMessages(results);
        if(benchmarkReference)
        {
            ASSERT_EQ(harness.referenceBenchmarks().size(), 1u);
            EXPECT_EQ(harness.referenceBenchmarks()[0].reference, "golden");
            EXPECT_EQ(harness.referenceBenchmarks()[0].status, ReferenceBenchmarkStatus::ERRORED);
        }
        else
        {
            EXPECT_TRUE(harness.referenceBenchmarks().empty());
        }
    }
}

TEST(TestReferenceBenchmark, PropertiesCarryEveryPhaseOfEachSampleAndTotalsOnlyVerified)
{
    auto* unitTest = ::testing::UnitTest::GetInstance();
    ReferenceBenchmarkSummary summary;
    summary.OnTestIterationStart(*unitTest, 0);

    ReferenceBenchmarkResult declined;
    declined.reference = "gpu";
    declined.status = ReferenceBenchmarkStatus::DECLINED;
    declined.prepareMs = 64.0;

    ReferenceBenchmarkResult verified;
    verified.reference = "cpu";
    verified.validator = "cpu";
    verified.status = ReferenceBenchmarkStatus::VERIFIED;
    verified.prepareMs = 1.0;
    verified.executeMs = 2.0;
    verified.materializeMs = 4.0;

    ReferenceBenchmarkResult mismatch;
    mismatch.reference = "golden";
    mismatch.validator = "gpu";
    mismatch.status = ReferenceBenchmarkStatus::MISMATCH;
    mismatch.prepareMs = 8.0;
    mismatch.executeMs = 16.0;
    mismatch.materializeMs = 32.0;

    ::testing::internal::CaptureStdout();
    reportReferenceBenchmark(declined);
    reportReferenceBenchmark(verified);
    reportReferenceBenchmark(mismatch);
    const std::string lines = ::testing::internal::GetCapturedStdout();

    const auto properties = recordedProperties();
    EXPECT_EQ(properties.at("reference_benchmark_gpu_status"), "declined");
    for(const char* field : {"samples", "prepare_ms", "execute_ms", "materialize_ms", "total_ms"})
    {
        EXPECT_EQ(properties.count(std::string("reference_benchmark_gpu_") + field), 0u) << field;
    }
    const std::map<std::string, std::string> expected{
        {"reference_benchmark_cpu_status", "verified"},
        {"reference_benchmark_cpu_validator", "cpu"},
        {"reference_benchmark_cpu_samples", "1"},
        {"reference_benchmark_cpu_prepare_ms", "1.000000"},
        {"reference_benchmark_cpu_execute_ms", "2.000000"},
        {"reference_benchmark_cpu_materialize_ms", "4.000000"},
        {"reference_benchmark_cpu_total_ms", "7.000000"},
        {"reference_benchmark_golden_status", "mismatch"},
        {"reference_benchmark_golden_validator", "gpu"},
        {"reference_benchmark_golden_samples", "1"},
        {"reference_benchmark_golden_prepare_ms", "8.000000"},
        {"reference_benchmark_golden_execute_ms", "16.000000"},
        {"reference_benchmark_golden_materialize_ms", "32.000000"},
        {"reference_benchmark_golden_total_ms", "56.000000"},
    };
    for(const auto& [key, value] : expected)
    {
        ASSERT_EQ(properties.count(key), 1u) << key;
        EXPECT_EQ(properties.at(key), value) << key;
    }

    EXPECT_NE(lines.find(" reference=gpu status=declined validator=\n"), std::string::npos)
        << lines;
    EXPECT_NE(lines.find(" reference=golden status=mismatch validator=gpu samples=1 "
                         "prepare_ms=8.000000 execute_ms=16.000000 materialize_ms=32.000000 "
                         "total_ms=56.000000\n"),
              std::string::npos)
        << lines;

    // A mismatch is a sample but not a verified reference.
    ASSERT_EQ(ReferenceBenchmarkSummary::totals().size(), 1u);
    EXPECT_EQ(ReferenceBenchmarkSummary::totals().at("cpu").count, 1u);
    EXPECT_DOUBLE_EQ(ReferenceBenchmarkSummary::totals().at("cpu").milliseconds, 7.0);

    ::testing::internal::CaptureStdout();
    summary.OnTestIterationEnd(*unitTest, 0);
    const std::string summaryLines = ::testing::internal::GetCapturedStdout();
    EXPECT_EQ(summaryLines,
              "[ REFERENCE BENCHMARK SUMMARY ] reference=cpu measured=1 total_ms=7\n");

    summary.OnTestIterationStart(*unitTest, 1);
    EXPECT_TRUE(ReferenceBenchmarkSummary::totals().empty());
    ::testing::internal::CaptureStdout();
    summary.OnTestIterationEnd(*unitTest, 1);
    EXPECT_EQ(::testing::internal::GetCapturedStdout(),
              "[ REFERENCE BENCHMARK SUMMARY ] measured=0\n");
}

// ---- resolveBenchmark -----------------------------------------------------

TEST(TestResolveBenchmark, OffWithoutFlagsHasNoSettingsAndNoError)
{
    const auto resolution = resolveBenchmark({});

    EXPECT_FALSE(resolution.settings.has_value());
    EXPECT_FALSE(resolution.error.has_value());
}

TEST(TestResolveBenchmark, EnabledAloneUsesTheDefaults)
{
    BenchmarkRequest request;
    request.enabled = true;

    const auto resolution = resolveBenchmark(request);

    ASSERT_TRUE(resolution.settings.has_value()) << resolution.error.value_or("");
    EXPECT_EQ(resolution.settings->timedIterations, 20);
    EXPECT_EQ(resolution.settings->warmupIterations, 30);
}

TEST(TestResolveBenchmark, ExplicitCountsOverrideTheDefaults)
{
    BenchmarkRequest request;
    request.enabled = true;
    request.timedIterations = 7;
    request.warmupIterations = 0;

    const auto resolution = resolveBenchmark(request);

    ASSERT_TRUE(resolution.settings.has_value()) << resolution.error.value_or("");
    EXPECT_EQ(resolution.settings->timedIterations, 7);
    EXPECT_EQ(resolution.settings->warmupIterations, 0);
}

TEST(TestResolveBenchmark, CountsWithoutBenchmarkAreRefused)
{
    BenchmarkRequest iterations;
    iterations.timedIterations = 5;
    BenchmarkRequest warmup;
    warmup.warmupIterations = 1;

    for(const auto& request : {iterations, warmup})
    {
        const auto resolution = resolveBenchmark(request);
        EXPECT_FALSE(resolution.settings.has_value());
        ASSERT_TRUE(resolution.error.has_value());
        EXPECT_NE(resolution.error->find("require --benchmark"), std::string::npos);
    }
}

TEST(TestResolveBenchmark, OutOfRangeCountsAreRefused)
{
    BenchmarkRequest noIterations;
    noIterations.enabled = true;
    noIterations.timedIterations = 0;
    BenchmarkRequest negativeWarmup;
    negativeWarmup.enabled = true;
    negativeWarmup.warmupIterations = -1;

    EXPECT_EQ(resolveBenchmark(noIterations).error,
              std::optional<std::string>("--benchmark-iterations must be at least 1"));
    EXPECT_EQ(resolveBenchmark(negativeWarmup).error,
              std::optional<std::string>("--benchmark-warmup must not be negative"));
}

TEST(TestResolveBenchmark, ModesThatDoNotExecuteAreRefused)
{
    BenchmarkRequest skip;
    skip.skipGraphValidation = true;
    BenchmarkRequest capture;
    capture.captureBundles = true;
    BenchmarkRequest write;
    write.writeSupportClaims = true;

    for(auto request : {skip, capture, write})
    {
        // Without --benchmark these modes are untouched.
        EXPECT_FALSE(resolveBenchmark(request).error.has_value());

        request.enabled = true;
        const auto resolution = resolveBenchmark(request);
        EXPECT_FALSE(resolution.settings.has_value());
        ASSERT_TRUE(resolution.error.has_value());
        EXPECT_NE(resolution.error->find("do not execute the graph"), std::string::npos);
    }
}

// ---- selectEngineResult ---------------------------------------------------

namespace
{

AutotuneResult resultFor(int64_t engineId, bool excludedByCaller = false)
{
    auto result = successfulResult();
    result.engineId = engineId;
    result.excludedByCaller = excludedByCaller;
    return result;
}

const hipdnn_frontend::Error K_GOOD{hipdnn_frontend::ErrorCode::OK, ""};
const hipdnn_frontend::Error K_NO_ENGINE_SUCCEEDED{hipdnn_frontend::ErrorCode::HIPDNN_BACKEND_ERROR,
                                                   "no engine succeeded"};

} // namespace

TEST(TestSelectEngineResult, ReturnsTheResultOfTheRequestedEngine)
{
    auto other = resultFor(7);
    other.minTimeMs = 99.0f;
    auto wanted = resultFor(42);
    wanted.minTimeMs = 1.0f;

    const auto selected = selectEngineResult(42, {other, wanted}, K_GOOD);

    EXPECT_EQ(selected.engineId, 42);
    EXPECT_TRUE(selected.succeeded);
    EXPECT_FLOAT_EQ(selected.minTimeMs, 1.0f);
}

TEST(TestSelectEngineResult, TheEngineOwnFailureWinsOverTheAutotuneStatus)
{
    auto failed = resultFor(42);
    failed.succeeded = false;
    failed.errorMessage = "Benchmark failed on iteration 2";

    const auto selected = selectEngineResult(42, {failed}, K_NO_ENGINE_SUCCEEDED);

    EXPECT_FALSE(selected.succeeded);
    EXPECT_EQ(selected.errorMessage, "Benchmark failed on iteration 2");
}

TEST(TestSelectEngineResult, NeverReportsAnExcludedOrDifferentEngine)
{
    for(const auto& results : {std::vector<AutotuneResult>{resultFor(42, true)},
                               std::vector<AutotuneResult>{resultFor(7)}})
    {
        const auto selected = selectEngineResult(42, results, K_GOOD);

        EXPECT_FALSE(selected.succeeded);
        EXPECT_EQ(selected.engineId, 42);
        EXPECT_EQ(selected.errorMessage, "autotune returned no result for the engine under test");
    }
}

TEST(TestSelectEngineResult, NoResultsReportsTheAutotuneStatus)
{
    const auto good = selectEngineResult(42, {}, K_GOOD);
    const auto bad = selectEngineResult(42, {}, K_NO_ENGINE_SUCCEEDED);

    EXPECT_FALSE(good.succeeded);
    EXPECT_EQ(good.errorMessage, "autotune returned no result for the engine under test");
    EXPECT_FALSE(bad.succeeded);
    EXPECT_EQ(bad.errorMessage, "no engine succeeded");
}

// ---- toString(TimingQuality) ------------------------------------------------

TEST(TestTimingQualityName, NamesEveryValueAndRefusesUnknownValues)
{
    EXPECT_STREQ(toString(TimingQuality::DEVICE_ONLY), "DEVICE_ONLY");
    EXPECT_STREQ(toString(TimingQuality::UNSTALLED), "UNSTALLED");
    EXPECT_STREQ(toString(TimingQuality::INVALID), "INVALID");
    EXPECT_THROW(toString(static_cast<TimingQuality>(99)), std::invalid_argument);
}

// ---- reportBenchmark ------------------------------------------------------

TEST(TestReportBenchmark, RecordsTheResultAsProperties)
{
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter forceBenchmarking(
        hipdnn_plugin_sdk::FORCE_BENCHMARKING_ENV_NAME, "1");
    auto benchmarked = successfulResult();
    benchmarked.supportsExhaustive = true;

    reportBenchmark(benchmarked);

    const auto* result = ::testing::UnitTest::GetInstance()->current_test_info()->result();
    std::map<std::string, std::string> properties;
    for(int i = 0; i < result->test_property_count(); ++i)
    {
        const auto& property = result->GetTestProperty(i);
        properties[property.key()] = property.value();
    }

    EXPECT_EQ(properties["benchmark_engine"], "TEST_ENGINE");
    EXPECT_EQ(properties["benchmark_timing"], "DEVICE_ONLY");
    EXPECT_EQ(properties["benchmark_samples"], "3");
    EXPECT_EQ(properties["benchmark_min_ms"], "0.2500");
    EXPECT_EQ(properties["benchmark_robust_ms"], "0.3750");
    EXPECT_EQ(properties["benchmark_mean_ms"], "0.5000");
    EXPECT_EQ(properties["benchmark_stddev_ms"], "0.1250");
    EXPECT_EQ(properties["benchmark_tuned"], "1");
}

// The harness sets no knobs, so only HIPDNN_FORCE_BENCHMARKING can turn benchmarking
// on, and only an engine with the global.benchmarking knob reads it.
TEST(TestReportBenchmark, TunedOnlyWhenForcedOnAnEngineWithTheKnob)
{
    struct Case
    {
        bool engineHasKnob;
        const char* forceBenchmarking;
        bool tuned;
    };
    const std::array<Case, 5> cases{{
        {true, "", false},
        {true, "1", true},
        {true, "0", false},
        {false, "1", false},
        {false, "", false},
    }};

    for(const auto& c : cases)
    {
        const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter forceBenchmarking(
            hipdnn_plugin_sdk::FORCE_BENCHMARKING_ENV_NAME, c.forceBenchmarking);
        auto result = successfulResult();
        result.supportsExhaustive = c.engineHasKnob;

        EXPECT_EQ(selectedByBenchmarking(result), c.tuned)
            << "knob=" << c.engineHasKnob << " HIPDNN_FORCE_BENCHMARKING='" << c.forceBenchmarking
            << "'";
    }
}

TEST(TestReportBenchmark, AFailedBenchmarkAddsAFailure)
{
    EXPECT_NONFATAL_FAILURE(reportBenchmark(failedResult("Benchmark failed on iteration 2")),
                            "benchmark failed: Benchmark failed on iteration 2");
    EXPECT_NONFATAL_FAILURE(reportBenchmark(AutotuneResult{}),
                            "benchmark failed: the engine was not benchmarked");
}

// ---- bundle harness -------------------------------------------------------

namespace
{

/// One way the harness verifies a bundle. Each way has its own benchmark hook.
struct OracleCase
{
    const char* name;
    VerificationMode mode;
    bool hasGoldenOutput;
    Verifier verifier;

    friend void PrintTo(const OracleCase& oracleCase, std::ostream* os)
    {
        *os << oracleCase.name;
    }
};

/// AUTO with golden data and explicit GOLDEN compare against the golden file.
/// AUTO without golden data falls through to the reference oracle chain.
const std::array<OracleCase, 5> ORACLE_CASES{{
    {"AutoGolden", VerificationMode::AUTO, true, Verifier::GOLDEN},
    {"AutoReference", VerificationMode::AUTO, false, Verifier::GPU_REFERENCE},
    {"GoldenMode", VerificationMode::GOLDEN, true, Verifier::GOLDEN},
    {"CpuReference", VerificationMode::CPU, false, Verifier::CPU_REFERENCE},
    {"GpuReference", VerificationMode::GPU, false, Verifier::GPU_REFERENCE},
}};

/// Expectations are checked after the harness returns: a gmock or EXPECT failure
/// raised inside driveHarness() lands in its intercepted results, not in this test.
class TestBundleHarnessBenchmark : public ::testing::TestWithParam<OracleCase>
{
protected:
    std::optional<hipdnn_test_sdk::utilities::ScopedDirectory> _scopedDir;
    testing_support::HarnessMocks _mocks;

    /// What the mocked runner returns from benchmark().
    AutotuneResult _benchmarkResult = successfulResult();
    /// The settings of every benchmark() call, and whether its variant pack held
    /// the verified output tensor.
    std::vector<BenchmarkSettings> _benchmarkCalls;
    bool _sawVerifiedOutput = false;
    /// When set, the mocked runner throws it from benchmark().
    std::optional<std::string> _benchmarkThrows;
    /// The verifier the harness reported for each bundle it ran.
    std::vector<Verifier> _verifiers;
    ReferenceBenchmarkTimer::Clock::time_point _now;
    int _referenceSynchronizations = 0;
    std::vector<ReferenceBenchmarkResult> _referenceResults;
    /// The harness being driven, so a clock read can look at what it has loaded.
    IntegrationBundleVerificationHarness* _harness = nullptr;
    /// At each reference clock read, whether the golden output had been loaded.
    std::vector<bool> _goldenLoadedAtClockRead;

    void SetUp() override
    {
        using ::testing::_;
        testing_support::ensureTestConfigInitialized();
        _scopedDir.emplace(hipdnn_test_sdk::utilities::claimScratchDirectory("bundle_benchmark"));
        testing_support::captureVerifiers(_mocks.reporter, _verifiers);

        ON_CALL(_mocks.engineRunner, benchmark(_, _, _))
            .WillByDefault(
                [this](GraphSession&, VariantPack& variantPack, const BenchmarkSettings& s) {
                    _benchmarkCalls.push_back(s);
                    // Long enough that any of it landing in a reference phase shows.
                    _now += std::chrono::milliseconds(1000);
                    _sawVerifiedOutput = variantPack.count(fixtures::K_OUTPUT_UID) == 1;
                    if(_benchmarkThrows.has_value())
                    {
                        throw std::runtime_error(*_benchmarkThrows);
                    }
                    return _benchmarkResult;
                });

        // Both references produce the golden value, so the reference path and the
        // golden path agree on what a match is.
        for(auto* reference : {&_mocks.cpuReference, &_mocks.gpuReference})
        {
            ON_CALL(*reference, execute(_, _, _))
                .WillByDefault([this](void*, size_t, const VariantPack& variantPack) {
                    _now += std::chrono::milliseconds(3);
                    auto* ptr = static_cast<float*>(variantPack.at(fixtures::K_OUTPUT_UID));
                    std::fill(ptr, ptr + fixtures::K_OUTPUT_ELEMS, fixtures::K_OUTPUT_VALUE);
                });
        }
    }

    /// Drives a bundle through the real harness with the parameter's oracle. The
    /// mocked engine writes `engineOutput`; the expected output is
    /// fixtures::K_OUTPUT_VALUE.
    void run(float engineOutput,
             std::optional<BenchmarkSettings> benchmark,
             ::testing::TestPartResultArray* results,
             bool benchmarkReference = false)
    {
        using ::testing::_;
        ON_CALL(_mocks.engineRunner, execute(_, _, _))
            .WillByDefault([this, engineOutput](GraphSession&,
                                                const std::optional<LoadedEngine>&,
                                                VariantPack& variantPack) {
                _now += std::chrono::milliseconds(100);
                fixtures::writeOutput(variantPack, engineOutput);
                return EngineOpResult::succeeded();
            });

        auto policy = testing_support::hostPolicy(GetParam().mode);
        policy.benchmark = benchmark;
        policy.benchmarkReference = benchmarkReference;
        auto deps = _mocks.dependencies(policy);
        deps.referenceTimer = ReferenceBenchmarkTimer(
            [this] {
                ++_referenceSynchronizations;
                _now += std::chrono::milliseconds(2);
            },
            [this] {
                _goldenLoadedAtClockRead.push_back(_harness->inputs().count(fixtures::K_OUTPUT_UID)
                                                   == 1);
                return _now;
            });
        IntegrationBundleVerificationHarness harness(std::move(deps));
        _harness = &harness;
        harness.setBundle(
            fixtures::loadBundle(_scopedDir->path(), "bench", GetParam().hasGoldenOutput), "bench");

        testing_support::driveHarness(harness, results);
        _referenceResults = harness.referenceBenchmarks();
        _harness = nullptr;
    }
};

} // namespace

TEST_P(TestBundleHarnessBenchmark, ReferenceTimingExcludesEngineAndAccountsForEveryPhase)
{
    using ::testing::_;
    EXPECT_CALL(_mocks.engineRunner, execute(_, _, _)).Times(1);
    ::testing::TestPartResultArray results;
    run(fixtures::K_OUTPUT_VALUE, std::nullopt, &results, true);

    EXPECT_FALSE(testing_support::anyFailed(results)) << testing_support::allMessages(results);
    ASSERT_EQ(_referenceResults.size(), 1u);
    const auto& timing = _referenceResults.front();
    EXPECT_EQ(timing.status, ReferenceBenchmarkStatus::VERIFIED);
    EXPECT_EQ(timing.validator, "cpu");
    EXPECT_DOUBLE_EQ(timing.materializeMs, 2.0);
    if(GetParam().hasGoldenOutput)
    {
        EXPECT_EQ(timing.reference, "golden");
        EXPECT_DOUBLE_EQ(timing.prepareMs, 0.0);
        EXPECT_DOUBLE_EQ(timing.executeMs, 2.0);
        EXPECT_DOUBLE_EQ(timing.totalMs(), 4.0);
        EXPECT_EQ(_referenceSynchronizations, 4);
    }
    else
    {
        EXPECT_EQ(timing.reference, GetParam().mode == VerificationMode::CPU ? "cpu" : "gpu");
        EXPECT_DOUBLE_EQ(timing.prepareMs, 4.0);
        EXPECT_DOUBLE_EQ(timing.executeMs, 5.0);
        EXPECT_DOUBLE_EQ(timing.totalMs(), 11.0);
        EXPECT_EQ(_referenceSynchronizations, 8);
    }
    EXPECT_TRUE(_benchmarkCalls.empty());
}

// The golden reference's execute phase is reading the golden outputs, so they must
// not already be loaded with the inputs when it starts.
TEST_P(TestBundleHarnessBenchmark, GoldenOutputsAreReadInsideTheExecutePhase)
{
    ::testing::TestPartResultArray results;
    run(fixtures::K_OUTPUT_VALUE, std::nullopt, &results, true);

    EXPECT_FALSE(testing_support::anyFailed(results)) << testing_support::allMessages(results);
    if(!GetParam().hasGoldenOutput)
    {
        EXPECT_EQ(
            std::count(_goldenLoadedAtClockRead.begin(), _goldenLoadedAtClockRead.end(), true), 0);
        return;
    }
    // Execute's start and stop, then materialize's.
    const std::vector<bool> expected{false, true, true, true};
    EXPECT_EQ(_goldenLoadedAtClockRead, expected);
}

TEST_P(TestBundleHarnessBenchmark, EngineBenchmarkStaysOutOfTheReferenceTiming)
{
    ::testing::TestPartResultArray results;
    run(fixtures::K_OUTPUT_VALUE, settings(3, 1), &results, true);

    EXPECT_FALSE(testing_support::anyFailed(results)) << testing_support::allMessages(results);
    ASSERT_EQ(_benchmarkCalls.size(), 1u);
    EXPECT_EQ(_benchmarkCalls.front().timedIterations, 3);
    ASSERT_EQ(_referenceResults.size(), 1u);
    EXPECT_EQ(_referenceResults.front().status, ReferenceBenchmarkStatus::VERIFIED);
    // The same totals as ReferenceTimingExcludesEngineAndAccountsForEveryPhase, which
    // runs without --benchmark.
    EXPECT_DOUBLE_EQ(_referenceResults.front().totalMs(), GetParam().hasGoldenOutput ? 4.0 : 11.0);
}

TEST_P(TestBundleHarnessBenchmark, DisabledReferenceTimingDoesNotSynchronizeOrReport)
{
    ::testing::TestPartResultArray results;
    run(fixtures::K_OUTPUT_VALUE, std::nullopt, &results);

    EXPECT_FALSE(testing_support::anyFailed(results));
    EXPECT_EQ(_referenceSynchronizations, 0);
    EXPECT_TRUE(_referenceResults.empty());
}

TEST_P(TestBundleHarnessBenchmark, ReferenceTimingLabelsMismatches)
{
    ::testing::TestPartResultArray results;
    run(fixtures::K_OUTPUT_VALUE + 100.0f, std::nullopt, &results, true);

    EXPECT_TRUE(testing_support::anyFailed(results));
    ASSERT_EQ(_referenceResults.size(), 1u);
    EXPECT_EQ(_referenceResults.front().status, ReferenceBenchmarkStatus::MISMATCH);
}

TEST_P(TestBundleHarnessBenchmark, AVerifiedPassIsTimedWithThePolicySettings)
{
    ::testing::TestPartResultArray results;
    run(fixtures::K_OUTPUT_VALUE, settings(4, 1), &results);

    EXPECT_FALSE(testing_support::anyFailed(results)) << testing_support::allMessages(results);
    EXPECT_FALSE(testing_support::anySkipped(results));
    ASSERT_EQ(_benchmarkCalls.size(), 1u);
    EXPECT_EQ(_benchmarkCalls[0].timedIterations, 4);
    EXPECT_EQ(_benchmarkCalls[0].warmupIterations, 1);
    EXPECT_TRUE(_sawVerifiedOutput) << "the benchmark must run on the verified tensors";
    EXPECT_EQ(_verifiers, std::vector<Verifier>{GetParam().verifier});
}

TEST_P(TestBundleHarnessBenchmark, AMismatchIsNotTimed)
{
    ::testing::TestPartResultArray results;
    run(fixtures::K_OUTPUT_VALUE + 100.0f, settings(4, 1), &results);

    EXPECT_TRUE(testing_support::anyFailed(results));
    EXPECT_TRUE(_benchmarkCalls.empty());
}

TEST_P(TestBundleHarnessBenchmark, WithoutBenchmarkAMatchPassesUntimed)
{
    ::testing::TestPartResultArray results;
    run(fixtures::K_OUTPUT_VALUE, std::nullopt, &results);

    EXPECT_FALSE(testing_support::anyFailed(results)) << testing_support::allMessages(results);
    EXPECT_FALSE(testing_support::anySkipped(results));
    EXPECT_TRUE(_benchmarkCalls.empty());
}

TEST_P(TestBundleHarnessBenchmark, WithoutBenchmarkAMismatchStillFails)
{
    ::testing::TestPartResultArray results;
    run(fixtures::K_OUTPUT_VALUE + 100.0f, std::nullopt, &results);

    EXPECT_TRUE(testing_support::anyFailed(results));
    EXPECT_TRUE(_benchmarkCalls.empty());
}

TEST_P(TestBundleHarnessBenchmark, ABenchmarkFailureFailsAVerifiedBundle)
{
    _benchmarkResult = failedResult("timed execution failed: device lost");

    ::testing::TestPartResultArray results;
    run(fixtures::K_OUTPUT_VALUE, settings(4, 1), &results);

    EXPECT_EQ(_benchmarkCalls.size(), 1u);
    EXPECT_TRUE(testing_support::anyFailed(results));
    EXPECT_NE(testing_support::allMessages(results).find("benchmark failed: timed execution "
                                                         "failed: device lost"),
              std::string::npos)
        << testing_support::allMessages(results);
}

// A throw from the benchmark must not turn the verified PASS into a harness failure,
// which would also change the support-claim verdict.
TEST_P(TestBundleHarnessBenchmark, AThrowingBenchmarkFailsButKeepsTheVerifiedOutcome)
{
    _benchmarkThrows = "hipMalloc failed";

    ::testing::TestPartResultArray results;
    run(fixtures::K_OUTPUT_VALUE, settings(4, 1), &results);

    EXPECT_EQ(_benchmarkCalls.size(), 1u);
    EXPECT_TRUE(testing_support::anyFailed(results));
    EXPECT_NE(testing_support::allMessages(results).find("benchmark failed: hipMalloc failed"),
              std::string::npos)
        << testing_support::allMessages(results);
    EXPECT_EQ(_verifiers, std::vector<Verifier>{GetParam().verifier})
        << "the outcome the harness recorded must still be the verified pass";
}

INSTANTIATE_TEST_SUITE_P(Oracles,
                         TestBundleHarnessBenchmark,
                         ::testing::ValuesIn(ORACLE_CASES),
                         [](const ::testing::TestParamInfo<OracleCase>& info) {
                             return std::string(info.param.name);
                         });

// ---- materialize on a device ---------------------------------------------

namespace
{

/// One way the expected outputs have to move to reach the validator.
struct DeviceSiteCase
{
    const char* name;
    VerificationMode mode;
    bool golden;
    /// Whether the reference writes its outputs to device memory.
    bool referenceOnDevice;
    ValidatorDevice validator;
    /// Reference clock reads in a clean run; the last one stops materialize.
    int clockReads;

    friend void PrintTo(const DeviceSiteCase& deviceCase, std::ostream* os)
    {
        *os << deviceCase.name;
    }
};

const std::array<DeviceSiteCase, 3> DEVICE_SITE_CASES{{
    {"GoldenUploadedToGpuValidator",
     VerificationMode::GOLDEN,
     true,
     false,
     ValidatorDevice::GPU,
     4},
    {"HostReferenceUploadedToGpuValidator",
     VerificationMode::CPU,
     false,
     false,
     ValidatorDevice::GPU,
     8},
    {"DeviceReferenceDownloadedToCpuValidator",
     VerificationMode::GPU,
     false,
     true,
     ValidatorDevice::CPU,
     8},
}};

class TestReferenceBenchmarkDeviceSite : public ::testing::TestWithParam<DeviceSiteCase>
{
};

} // namespace

// Materialize moves the expected outputs to the validator, so the copy has to be
// complete when its clock stops. Overwriting the copy's source at that clock read
// tells the two apart: a copy made inside the phase holds the reference's values,
// and one left for the comparison picks up the overwrite and mismatches. Runs the
// real device synchronize, so it needs a device.
TEST_P(TestReferenceBenchmarkDeviceSite, ExpectedOutputsReachTheValidatorInsideMaterialize)
{
    SKIP_IF_NO_DEVICES();
    using ::testing::_;
    using ::testing::Return;
    testing_support::ensureTestConfigInitialized();
    const auto& param = GetParam();
    constexpr size_t BYTES = fixtures::K_OUTPUT_ELEMS * sizeof(float);
    const std::vector<float> expected(fixtures::K_OUTPUT_ELEMS, fixtures::K_OUTPUT_VALUE);
    const std::vector<float> overwrite(fixtures::K_OUTPUT_ELEMS, -1.0f);

    const hipdnn_test_sdk::utilities::ScopedDirectory directory(
        hipdnn_test_sdk::utilities::claimScratchDirectory("reference_benchmark_device_site"));
    testing_support::HarnessMocks mocks;
    // DEVICE placement, so the engine's outputs are device buffers.
    ON_CALL(mocks.engineRunner, execute(_, _, _))
        .WillByDefault(
            [&](GraphSession&, const std::optional<LoadedEngine>&, VariantPack& variantPack) {
                EXPECT_EQ(hipMemcpy(variantPack.at(fixtures::K_OUTPUT_UID),
                                    expected.data(),
                                    BYTES,
                                    hipMemcpyHostToDevice),
                          hipSuccess);
                return EngineOpResult::succeeded();
            });
    void* referenceOutput = nullptr;
    for(auto* reference : {&mocks.cpuReference, &mocks.gpuReference})
    {
        ON_CALL(*reference, requiresDeviceMemory()).WillByDefault(Return(param.referenceOnDevice));
        ON_CALL(*reference, execute(_, _, _))
            .WillByDefault([&](void*, size_t, const VariantPack& variantPack) {
                referenceOutput = variantPack.at(fixtures::K_OUTPUT_UID);
                if(param.referenceOnDevice)
                {
                    EXPECT_EQ(
                        hipMemcpy(referenceOutput, expected.data(), BYTES, hipMemcpyHostToDevice),
                        hipSuccess);
                }
                else
                {
                    std::copy(
                        expected.begin(), expected.end(), static_cast<float*>(referenceOutput));
                }
            });
    }

    auto policy = testing_support::hostPolicy(param.mode);
    policy.placement = TensorPlacement::DEVICE;
    policy.validator = param.validator;
    policy.benchmarkReference = true;
    auto deps = mocks.dependencies(policy);
    IntegrationBundleVerificationHarness* harness = nullptr;
    int clockReads = 0;
    deps.referenceTimer = ReferenceBenchmarkTimer(synchronizeReferenceBenchmark, [&] {
        if(++clockReads == param.clockReads)
        {
            if(param.golden)
            {
                auto* golden = static_cast<float*>(
                    harness->inputs().at(fixtures::K_OUTPUT_UID)->rawHostData());
                std::copy(overwrite.begin(), overwrite.end(), golden);
            }
            else if(param.referenceOnDevice)
            {
                EXPECT_EQ(
                    hipMemcpy(referenceOutput, overwrite.data(), BYTES, hipMemcpyHostToDevice),
                    hipSuccess);
            }
            else
            {
                std::copy(overwrite.begin(), overwrite.end(), static_cast<float*>(referenceOutput));
            }
        }
        return ReferenceBenchmarkTimer::Clock::now();
    });
    IntegrationBundleVerificationHarness driven(std::move(deps));
    harness = &driven;
    driven.setBundle(fixtures::loadBundle(directory.path(), "device_site", param.golden),
                     "device_site");
    ::testing::TestPartResultArray results;
    testing_support::driveHarness(driven, &results);

    EXPECT_FALSE(testing_support::anyFailed(results)) << testing_support::allMessages(results);
    EXPECT_EQ(clockReads, param.clockReads);
    ASSERT_EQ(driven.referenceBenchmarks().size(), 1u);
    const auto& timing = driven.referenceBenchmarks().front();
    EXPECT_EQ(timing.status, ReferenceBenchmarkStatus::VERIFIED);
    EXPECT_EQ(timing.validator, param.validator == ValidatorDevice::GPU ? "gpu" : "cpu");
}

INSTANTIATE_TEST_SUITE_P(Sites,
                         TestReferenceBenchmarkDeviceSite,
                         ::testing::ValuesIn(DEVICE_SITE_CASES),
                         [](const ::testing::TestParamInfo<DeviceSiteCase>& info) {
                             return std::string(info.param.name);
                         });

// NOLINTEND(readability-identifier-naming)
