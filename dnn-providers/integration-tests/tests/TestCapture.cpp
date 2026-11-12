// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <hipdnn_test_sdk/utilities/ScratchDirectory.hpp>

#include "BundleFixtureFiles.hpp"
#include "HarnessTestSupport.hpp"
#include "harness/Capture.hpp"

// NOLINTBEGIN(readability-identifier-naming)

using namespace hipdnn_integration_tests;
using namespace hipdnn_integration_tests::bundle;

// ---- resolveCapture -------------------------------------------------------

TEST(TestResolveCapture, OffHasNoSettingsAndNoError)
{
    const auto resolution = resolveCapture(CaptureRequest{});

    EXPECT_FALSE(resolution.settings.has_value());
    EXPECT_FALSE(resolution.error.has_value());
}

TEST(TestResolveCapture, EnabledAloneUsesTheDefaults)
{
    CaptureRequest request;
    request.enabled = true;
    const auto resolution = resolveCapture(request);

    EXPECT_FALSE(resolution.error.has_value());
    ASSERT_TRUE(resolution.settings.has_value());
    EXPECT_EQ(resolution.settings->replays, CaptureSettings::DEFAULT_REPLAYS);
}

TEST(TestResolveCapture, BenchmarkIsRefused)
{
    CaptureRequest request;
    request.benchmark = true;
    EXPECT_FALSE(resolveCapture(request).error.has_value());

    request.enabled = true;
    const auto resolution = resolveCapture(request);
    EXPECT_FALSE(resolution.settings.has_value());
    ASSERT_TRUE(resolution.error.has_value());
    EXPECT_NE(resolution.error->find("--benchmark"), std::string::npos);
}

TEST(TestResolveCapture, ModesThatDoNotExecuteAreRefused)
{
    CaptureRequest skip;
    skip.skipGraphValidation = true;
    CaptureRequest bundles;
    bundles.captureBundles = true;
    CaptureRequest write;
    write.writeSupportClaims = true;

    for(auto request : {skip, bundles, write})
    {
        EXPECT_FALSE(resolveCapture(request).error.has_value());

        request.enabled = true;
        const auto resolution = resolveCapture(request);
        EXPECT_FALSE(resolution.settings.has_value());
        ASSERT_TRUE(resolution.error.has_value());
        EXPECT_NE(resolution.error->find("do not execute the graph"), std::string::npos);
    }
}

TEST(TestResolveCapture, DisabledBundlesAreRefused)
{
    CaptureRequest request;
    request.bundlesDisabled = true;
    EXPECT_FALSE(resolveCapture(request).error.has_value());

    request.enabled = true;
    const auto resolution = resolveCapture(request);
    EXPECT_FALSE(resolution.settings.has_value());
    ASSERT_TRUE(resolution.error.has_value());
    EXPECT_NE(resolution.error->find("HIPDNN_TEST_ALLOW_BUNDLES"), std::string::npos);
}

TEST(TestDeclinedForMissingNote, NamesTheEngineAndTheIdTheRunnerReported)
{
    constexpr int64_t ENGINE_ID = 0x1234567890abcdefLL;
    const auto result = declinedForMissingNote(ENGINE_ID);

    EXPECT_EQ(result.status, CaptureResult::Status::DECLINED);
    EXPECT_NE(result.message.find(hipdnn_data_sdk::utilities::engineNameOrHex(ENGINE_ID)),
              std::string::npos)
        << result.message;
    EXPECT_NE(result.message.find("(id " + std::to_string(ENGINE_ID) + ")"), std::string::npos)
        << result.message;
    EXPECT_NE(result.message.find("SUPPORTS_GRAPH_CAPTURE"), std::string::npos);
}

// ---- bundle harness -------------------------------------------------------

namespace
{

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

/// The replay is judged by the oracle that judged the eager run, so each way of
/// verifying a bundle needs its own coverage.
const std::array<OracleCase, 3> ORACLE_CASES{{
    {"AutoGolden", VerificationMode::AUTO, true, Verifier::GOLDEN},
    {"AutoReference", VerificationMode::AUTO, false, Verifier::GPU_REFERENCE},
    {"GoldenMode", VerificationMode::GOLDEN, true, Verifier::GOLDEN},
}};

/// What the mocked runner does inside captureReplay().
enum class Replay
{
    /// Reset, write the expected value, let the harness check it.
    MATCHES,
    /// Reset, write a value the oracle rejects.
    WRITES_WRONG_VALUE,
    /// Reset, write nothing: only the sentinel is left to check.
    WRITES_NOTHING,
    DECLINES,
    FAILS,
    THROWS,
};

class TestBundleHarnessCapture : public ::testing::TestWithParam<OracleCase>
{
protected:
    std::optional<hipdnn_test_sdk::utilities::ScopedDirectory> _scopedDir;
    testing_support::HarnessMocks _mocks;

    Replay _replay = Replay::MATCHES;
    int _captureCalls = 0;
    std::vector<int> _settingsSeen;
    bool _sawVerifiedOutput = false;
    /// Whether the output was back at the sentinel after the reset hook ran.
    bool _outputWasResetByHook = false;
    std::vector<Verifier> _verifiers;

    static constexpr const char* FAILED_STEP = "hipGraphLaunch failed: hipErrorLaunchFailure";
    static constexpr const char* DECLINE_REASON
        = "engine 7 does not report the SUPPORTS_GRAPH_CAPTURE note";

    void SetUp() override
    {
        using ::testing::_;
        testing_support::ensureTestConfigInitialized();
        _scopedDir.emplace(hipdnn_test_sdk::utilities::claimScratchDirectory("bundle_capture"));
        testing_support::captureVerifiers(_mocks.reporter, _verifiers);

        ON_CALL(_mocks.engineRunner, captureReplay(_, _, _, _))
            .WillByDefault([this](GraphSession&,
                                  VariantPack& variantPack,
                                  const CaptureSettings& settings,
                                  const ReplayHooks& hooks) {
                ++_captureCalls;
                _settingsSeen.push_back(settings.replays);
                _sawVerifiedOutput = variantPack.count(fixtures::K_OUTPUT_UID) == 1;
                switch(_replay)
                {
                case Replay::DECLINES:
                    return CaptureResult::declined(DECLINE_REASON);
                case Replay::FAILS:
                    return CaptureResult::failed(FAILED_STEP);
                case Replay::THROWS:
                    throw std::runtime_error("hipStreamBeginCapture exploded");
                default:
                    break;
                }

                auto* output = static_cast<float*>(variantPack.at(fixtures::K_OUTPUT_UID));
                hooks.resetOutputs();
                _outputWasResetByHook = std::isnan(output[0]);
                if(_replay == Replay::MATCHES)
                {
                    std::fill(output, output + fixtures::K_OUTPUT_ELEMS, fixtures::K_OUTPUT_VALUE);
                }
                else if(_replay == Replay::WRITES_WRONG_VALUE)
                {
                    std::fill(output,
                              output + fixtures::K_OUTPUT_ELEMS,
                              fixtures::K_OUTPUT_VALUE + 100.f);
                }

                auto result = CaptureResult::replayed();
                result.outputsMismatched = !hooks.checkOutputs();
                return result;
            });

        for(auto* reference : {&_mocks.cpuReference, &_mocks.gpuReference})
        {
            ON_CALL(*reference, execute(_, _, _))
                .WillByDefault([](void*, size_t, const VariantPack& variantPack) {
                    auto* ptr = static_cast<float*>(variantPack.at(fixtures::K_OUTPUT_UID));
                    std::fill(ptr, ptr + fixtures::K_OUTPUT_ELEMS, fixtures::K_OUTPUT_VALUE);
                });
        }
    }

    void run(float engineOutput,
             std::optional<CaptureSettings> capture,
             ::testing::TestPartResultArray* results)
    {
        testing_support::engineWrites(_mocks.engineRunner, &fixtures::writeOutput, engineOutput);

        auto policy = testing_support::hostPolicy(GetParam().mode);
        policy.graphCapture = capture;
        IntegrationBundleVerificationHarness harness(_mocks.dependencies(policy));
        harness.setBundle(
            fixtures::loadBundle(_scopedDir->path(), "capture", GetParam().hasGoldenOutput),
            "capture");

        testing_support::driveHarness(harness, results);
    }

    static CaptureSettings replays(int count)
    {
        CaptureSettings settings;
        settings.replays = count;
        return settings;
    }
};

} // namespace

TEST_P(TestBundleHarnessCapture, WithoutTheLaneNothingIsCaptured)
{
    ::testing::TestPartResultArray results;
    run(fixtures::K_OUTPUT_VALUE, std::nullopt, &results);

    EXPECT_FALSE(testing_support::anyFailed(results)) << testing_support::allMessages(results);
    EXPECT_FALSE(testing_support::anySkipped(results));
    EXPECT_EQ(_captureCalls, 0);
}

TEST_P(TestBundleHarnessCapture, AVerifiedPassIsCapturedOnTheVerifiedTensors)
{
    ::testing::TestPartResultArray results;
    run(fixtures::K_OUTPUT_VALUE, replays(5), &results);

    EXPECT_FALSE(testing_support::anyFailed(results)) << testing_support::allMessages(results);
    EXPECT_FALSE(testing_support::anySkipped(results));
    ASSERT_EQ(_captureCalls, 1);
    EXPECT_EQ(_settingsSeen, std::vector<int>{5});
    EXPECT_TRUE(_sawVerifiedOutput);
    EXPECT_TRUE(_outputWasResetByHook) << "the reset hook must put the sentinel back in place";
    EXPECT_EQ(_verifiers, std::vector<Verifier>{GetParam().verifier});
}

TEST_P(TestBundleHarnessCapture, AMismatchedEagerRunIsNotCaptured)
{
    ::testing::TestPartResultArray results;
    run(fixtures::K_OUTPUT_VALUE + 100.0f, replays(3), &results);

    EXPECT_TRUE(testing_support::anyFailed(results));
    EXPECT_EQ(_captureCalls, 0);
}

TEST_P(TestBundleHarnessCapture, AReplayThatWritesAWrongValueFails)
{
    _replay = Replay::WRITES_WRONG_VALUE;

    ::testing::TestPartResultArray results;
    run(fixtures::K_OUTPUT_VALUE, replays(3), &results);

    EXPECT_EQ(_captureCalls, 1);
    EXPECT_TRUE(testing_support::anyFailed(results));
    EXPECT_NE(testing_support::allMessages(results).find("graph-capture replay"), std::string::npos)
        << "the failure must say it came from the replay: "
        << testing_support::allMessages(results);
}

// The eager run already left the right values in the buffer, so a replay that writes
// nothing only fails if the outputs were reset first.
TEST_P(TestBundleHarnessCapture, AReplayThatWritesNothingFails)
{
    _replay = Replay::WRITES_NOTHING;

    ::testing::TestPartResultArray results;
    run(fixtures::K_OUTPUT_VALUE, replays(3), &results);

    EXPECT_TRUE(_outputWasResetByHook);
    EXPECT_TRUE(testing_support::anyFailed(results));
}

TEST_P(TestBundleHarnessCapture, AnEngineWithoutTheNoteIsSkippedNotFailed)
{
    _replay = Replay::DECLINES;

    ::testing::TestPartResultArray results;
    run(fixtures::K_OUTPUT_VALUE, replays(3), &results);

    EXPECT_FALSE(testing_support::anyFailed(results)) << testing_support::allMessages(results);
    EXPECT_TRUE(testing_support::anySkipped(results));
    EXPECT_NE(testing_support::allMessages(results).find("SUPPORTS_GRAPH_CAPTURE"),
              std::string::npos)
        << testing_support::allMessages(results);
    EXPECT_EQ(_verifiers, std::vector<Verifier>{GetParam().verifier})
        << "the verified outcome must still be recorded";
}

TEST_P(TestBundleHarnessCapture, ACaptureFailureFailsAndNamesTheStep)
{
    _replay = Replay::FAILS;

    ::testing::TestPartResultArray results;
    run(fixtures::K_OUTPUT_VALUE, replays(3), &results);

    EXPECT_TRUE(testing_support::anyFailed(results));
    EXPECT_FALSE(testing_support::anySkipped(results));
    EXPECT_NE(testing_support::allMessages(results).find(FAILED_STEP), std::string::npos)
        << testing_support::allMessages(results);
}

// A throw from the runner must not turn the verified PASS into a harness failure,
// which would also change the support-claim verdict.
TEST_P(TestBundleHarnessCapture, AThrowingCaptureFailsButKeepsTheVerifiedOutcome)
{
    _replay = Replay::THROWS;

    ::testing::TestPartResultArray results;
    run(fixtures::K_OUTPUT_VALUE, replays(3), &results);

    EXPECT_TRUE(testing_support::anyFailed(results));
    EXPECT_NE(testing_support::allMessages(results).find("hipStreamBeginCapture exploded"),
              std::string::npos)
        << testing_support::allMessages(results);
    EXPECT_EQ(_verifiers, std::vector<Verifier>{GetParam().verifier});
}

INSTANTIATE_TEST_SUITE_P(Oracles,
                         TestBundleHarnessCapture,
                         ::testing::ValuesIn(ORACLE_CASES),
                         [](const ::testing::TestParamInfo<OracleCase>& info) {
                             return std::string(info.param.name);
                         });

// NOLINTEND(readability-identifier-naming)
