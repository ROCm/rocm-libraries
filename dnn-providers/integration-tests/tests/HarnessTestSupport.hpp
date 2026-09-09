// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include "ScratchDirectory.hpp"
#include "harness/TestConfig.hpp"
#include "harness/bundle/HarnessDependencies.hpp"
#include "harness/bundle/IntegrationBundleVerificationHarness.hpp"
#include "mocks/MockGraphEngineRunner.hpp"
#include "mocks/MockReferenceExecutors.hpp"
#include "mocks/MockSupportClaimObserver.hpp"
#include "mocks/MockVerificationReporter.hpp"

namespace hipdnn_integration_tests::bundle::testing_support
{

/// The harness still reads the TOML skip list and tolerance overrides through the
/// TestConfig singleton, which throws until somebody initializes it. Initializing
/// twice is itself an error, so ask first — another suite in this binary may have
/// got there already.
inline void ensureTestConfigInitialized()
{
    if(!TestConfig::isInitialized())
    {
        TestConfig::initialize(TestConfigOptions{});
    }
}

/// A policy for a deviceless run: host pointers, so the mocked engine can write
/// straight into the variant pack and no ITensor ever hipMallocs.
inline HarnessPolicy hostPolicy(VerificationMode mode = VerificationMode::AUTO,
                                bool enforceSupportClaims = false)
{
    HarnessPolicy policy;
    policy.mode = mode;
    policy.enforceSupportClaims = enforceSupportClaims;
    policy.placement = TensorPlacement::HOST;
    policy.arch = "gfx942";
    policy.platform = "linux";
    policy.deviceVramMb = 0;
    return policy;
}

/// Hands a mock to the harness without handing over ownership: the test keeps the
/// mock as a member so it can set expectations on it after construction.
template <typename T>
std::shared_ptr<T> nonOwning(T& instance)
{
    return std::shared_ptr<T>(&instance, [](T*) {});
}

/// A session that says "the engine takes this graph" and carries no real graph.
inline GraphSession acceptedSession()
{
    GraphSession session;
    session.engines.accepted = true;
    return session;
}

/// A session that says "no engine takes this graph".
inline GraphSession declinedSession()
{
    return GraphSession{};
}

/// A session whose from_binary failed.
inline GraphSession buildErrorSession(std::string error)
{
    GraphSession session;
    session.buildError = std::move(error);
    return session;
}

/// Every mock the harness needs, owned by the test, plus the wiring.
///
/// NiceMock throughout: these suites assert on what the harness *decided*, so an
/// incidental call the test did not name is not interesting and must not print a
/// warning for every bundle.
struct HarnessMocks
{
    ::testing::NiceMock<MockGraphEngineRunner> engineRunner;
    ::testing::NiceMock<MockReferenceExecutors> referenceExecutors;
    ::testing::NiceMock<MockSupportClaimObserver> claimObserver;
    ::testing::NiceMock<MockVerificationReporter> reporter;
    ::testing::NiceMock<MockReferenceGraphExecutor> cpuReference;
    ::testing::NiceMock<MockReferenceGraphExecutor> gpuReference;

    HarnessMocks()
    {
        using ::testing::_;
        using ::testing::Return;
        using ::testing::ReturnRef;

        ON_CALL(referenceExecutors, get(ReferenceExecutorType::CPU))
            .WillByDefault(ReturnRef(cpuReference));
        ON_CALL(referenceExecutors, get(ReferenceExecutorType::GPU))
            .WillByDefault(ReturnRef(gpuReference));

        // Applicable unless a test says otherwise; an inapplicable reference is the
        // interesting case and should have to be asked for.
        ON_CALL(cpuReference, isApplicable(_, _)).WillByDefault(Return(true));
        ON_CALL(gpuReference, isApplicable(_, _)).WillByDefault(Return(true));

        ON_CALL(engineRunner, openGraph(_, _))
            .WillByDefault([](const IntegrationTestBundle&, const std::optional<LoadedEngine>&) {
                return acceptedSession();
            });
        ON_CALL(engineRunner, buildPlans(_, _)).WillByDefault(Return(EngineOpResult::succeeded()));
        ON_CALL(engineRunner, execute(_, _, _)).WillByDefault(Return(EngineOpResult::succeeded()));
    }

    HarnessDependencies dependencies(HarnessPolicy policy)
    {
        HarnessDependencies deps;
        deps.engineRunner = nonOwning<IGraphEngineRunner>(engineRunner);
        deps.referenceExecutors = nonOwning<IReferenceExecutors>(referenceExecutors);
        deps.claimObserver = nonOwning<ISupportClaimObserver>(claimObserver);
        deps.reporter = nonOwning<IVerificationReporter>(reporter);
        deps.policy = std::move(policy);
        return deps;
    }
};

/// Makes the mocked engine write `value` into the bundle's output tensor and
/// report success — the "the engine ran and produced this" case.
inline void engineWrites(::testing::NiceMock<MockGraphEngineRunner>& runner,
                         void (*writer)(VariantPack&, float),
                         float value)
{
    using ::testing::_;
    ON_CALL(runner, execute(_, _, _))
        .WillByDefault([writer, value](GraphSession&,
                                       const std::optional<LoadedEngine>&,
                                       VariantPack& variantPack) {
            writer(variantPack, value);
            return EngineOpResult::succeeded();
        });
}

/// Collects every verdict the harness publishes, through the mock rather than a
/// second hand-written double.
inline void captureVerdicts(::testing::NiceMock<MockVerificationReporter>& reporter,
                            std::vector<SupportResult>& out)
{
    using ::testing::_;
    ON_CALL(reporter, recordVerdict(_)).WillByDefault([&out](const SupportResult& record) {
        out.push_back(record);
    });
}

/// Collects every coverage update the harness publishes.
inline void captureCoverage(::testing::NiceMock<MockVerificationReporter>& reporter,
                            std::vector<CoverageUpdate>& out)
{
    using ::testing::_;
    ON_CALL(reporter, recordCoverage(_)).WillByDefault([&out](const CoverageUpdate& update) {
        out.push_back(update);
    });
}

/// Collects the reasons the harness recorded as unverifiable.
inline void captureUnverifiable(::testing::NiceMock<MockVerificationReporter>& reporter,
                                std::vector<std::string>& out)
{
    using ::testing::_;
    ON_CALL(reporter, recordUnverifiable(_, _))
        .WillByDefault(
            [&out](const std::string&, const std::string& reason) { out.push_back(reason); });
}

/// Collects the reasons the harness recorded as reference errors.
inline void captureReferenceErrors(::testing::NiceMock<MockVerificationReporter>& reporter,
                                   std::vector<std::string>& out)
{
    using ::testing::_;
    ON_CALL(reporter, recordReferenceError(_, _))
        .WillByDefault(
            [&out](const std::string&, const std::string& reason) { out.push_back(reason); });
}

inline bool anyFailed(const ::testing::TestPartResultArray& results)
{
    for(int i = 0; i < results.size(); ++i)
    {
        if(results.GetTestPartResult(i).failed())
        {
            return true;
        }
    }
    return false;
}

inline bool anySkipped(const ::testing::TestPartResultArray& results)
{
    for(int i = 0; i < results.size(); ++i)
    {
        if(results.GetTestPartResult(i).skipped())
        {
            return true;
        }
    }
    return false;
}

/// Every message GTest recorded, joined — for asserting on what a failure said.
inline std::string allMessages(const ::testing::TestPartResultArray& results)
{
    std::string joined;
    for(int i = 0; i < results.size(); ++i)
    {
        joined += results.GetTestPartResult(i).message();
        joined += "\n";
    }
    return joined;
}

/// Runs one harness the way GTest's own runner would, capturing every disposition
/// it issues into `results`.
///
/// The skip guard is the whole point. `Test::Run()` checks IsSkipped() after SetUp()
/// and does not call TestBody() when it is set, so a driver that always calls
/// TestBody() diverges from production the moment a metadata guard or a TOML skip
/// fires in SetUp() — easy to omit, and silent when omitted.
///
/// Bundle and locator wiring stays with the caller: the suites disagree on what to
/// set up (some pass a locator, some tag metadata, some build the harness from a
/// mode), and folding those differences into one signature buys nothing. This owns
/// only the part they must agree on.
inline void driveHarness(IntegrationBundleVerificationHarness& harness,
                         ::testing::TestPartResultArray* results)
{
    const ::testing::ScopedFakeTestPartResultReporter reporter(
        ::testing::ScopedFakeTestPartResultReporter::INTERCEPT_ALL_THREADS, results);
    harness.SetUp();
    if(!anySkipped(*results))
    {
        harness.TestBody();
    }
}

} // namespace hipdnn_integration_tests::bundle::testing_support
