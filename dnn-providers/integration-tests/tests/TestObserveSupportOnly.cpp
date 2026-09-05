// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <hipdnn_frontend/Error.hpp>

#include "HarnessTestSupport.hpp"
#include "harness/bundle/GraphSession.hpp"
#include "harness/bundle/IntegrationBundleVerificationHarness.hpp"
#include "harness/bundle/LoadedEngine.hpp"
#include "harness/bundle/SupportClaims.hpp"

using hipdnn_frontend::ErrorCode;
using namespace hipdnn_integration_tests::bundle;
using namespace hipdnn_integration_tests::bundle::testing_support;

// NOLINTBEGIN(readability-identifier-naming)

namespace
{

constexpr int64_t ENGINE_A_ID = 1;
constexpr int64_t ENGINE_B_ID = 2;

const LoadedEngine ENGINE_A{ENGINE_A_ID, "ENGINE_A"};
const LoadedEngine ENGINE_B{ENGINE_B_ID, "ENGINE_B"};

GraphSession resolvedSession(const std::vector<int64_t>& rankedIds)
{
    GraphSession session;
    session.engines.status = hipdnn_frontend::Error{ErrorCode::OK, ""};
    session.engines.rankedIds = rankedIds;
    return session;
}

GraphSession unresolvedSession()
{
    GraphSession session;
    session.engines.status
        = hipdnn_frontend::Error{ErrorCode::HIPDNN_BACKEND_ERROR, "backend timed out"};
    return session;
}

class TestObserveSupportOnly : public ::testing::Test
{
protected:
    testing_support::HarnessMocks _mocks;
};

TEST_F(TestObserveSupportOnly, BuildErrorReturnsEmpty)
{
    IntegrationBundleVerificationHarness harness(
        _mocks.dependencies(testing_support::hostPolicy()));
    harness.setBundle(nullptr, "fake/bundle.json", singleGraphClaimLocator("fake/bundle.json"));

    auto session = testing_support::buildErrorSession("from_binary failed");
    auto observations = harness.observeSupportOnly(session, {ENGINE_A, ENGINE_B});

    EXPECT_TRUE(observations.empty());
}

TEST_F(TestObserveSupportOnly, UnresolvedQueryReturnsEmpty)
{
    IntegrationBundleVerificationHarness harness(
        _mocks.dependencies(testing_support::hostPolicy()));
    harness.setBundle(nullptr, "fake/bundle.json", singleGraphClaimLocator("fake/bundle.json"));

    auto session = unresolvedSession();
    auto observations = harness.observeSupportOnly(session, {ENGINE_A, ENGINE_B});

    EXPECT_TRUE(observations.empty());
}

TEST_F(TestObserveSupportOnly, ResolvedQueryRecordsAllEngines)
{
    IntegrationBundleVerificationHarness harness(
        _mocks.dependencies(testing_support::hostPolicy()));
    harness.setBundle(nullptr, "fake/bundle.json", singleGraphClaimLocator("fake/bundle.json"));

    auto session = resolvedSession({ENGINE_A_ID});
    auto observations = harness.observeSupportOnly(session, {ENGINE_A, ENGINE_B});

    ASSERT_EQ(observations.size(), 2u);
    EXPECT_EQ(observations[0].engineName, "ENGINE_A");
    EXPECT_TRUE(observations[0].engineIsSupported);
    EXPECT_EQ(observations[1].engineName, "ENGINE_B");
    EXPECT_FALSE(observations[1].engineIsSupported);
}

TEST_F(TestObserveSupportOnly, ModeCNarrowsToEngineUnderTest)
{
    IntegrationBundleVerificationHarness harness(_mocks.dependencies(testing_support::hostPolicy()),
                                                 ENGINE_A);
    harness.setBundle(nullptr, "fake/bundle.json", singleGraphClaimLocator("fake/bundle.json"));

    auto session = resolvedSession({ENGINE_A_ID});
    auto observations = harness.observeSupportOnly(session, {ENGINE_A, ENGINE_B});

    ASSERT_EQ(observations.size(), 1u);
    EXPECT_EQ(observations[0].engineName, "ENGINE_A");
    EXPECT_TRUE(observations[0].engineIsSupported);
}

TEST_F(TestObserveSupportOnly, CarriesArchAndPlatformFromPolicy)
{
    IntegrationBundleVerificationHarness harness(
        _mocks.dependencies(testing_support::hostPolicy()));
    harness.setBundle(nullptr, "fake/bundle.json", singleGraphClaimLocator("fake/bundle.json"));

    auto session = resolvedSession({});
    auto observations = harness.observeSupportOnly(session, {ENGINE_A});

    ASSERT_EQ(observations.size(), 1u);
    EXPECT_EQ(observations[0].arch, "gfx942");
    EXPECT_EQ(observations[0].platform, "linux");
    EXPECT_FALSE(observations[0].engineIsSupported);
}

} // namespace

// NOLINTEND(readability-identifier-naming)
