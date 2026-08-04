// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <sstream>

#include "harness/bundle/SupportEnforcementReport.hpp"

using hipdnn_integration_tests::bundle::SupportQueryGuard;
using hipdnn_integration_tests::bundle::UnclaimedSupportReport;

// NOLINTBEGIN(readability-identifier-naming) -- gtest macro-generated names

class TestSupportQueryGuard : public ::testing::Test
{
protected:
    void SetUp() override
    {
        SupportQueryGuard::get().reset();
    }

    void TearDown() override
    {
        SupportQueryGuard::get().reset();
    }
};

TEST_F(TestSupportQueryGuard, SingletonIdentity)
{
    EXPECT_EQ(&SupportQueryGuard::get(), &SupportQueryGuard::get());
}

TEST_F(TestSupportQueryGuard, FreshGuardIsNotTripped)
{
    EXPECT_FALSE(SupportQueryGuard::get().enforcementExpected());
    EXPECT_EQ(SupportQueryGuard::get().queriesObserved(), 0u);
    EXPECT_FALSE(SupportQueryGuard::get().tripped());
}

TEST_F(TestSupportQueryGuard, NoEnforcementExpectedNeverTripsRegardlessOfQueries)
{
    // No claim-bearing bundle registered -- zero queries is fine, nothing to enforce.
    EXPECT_FALSE(SupportQueryGuard::get().tripped());
}

TEST_F(TestSupportQueryGuard, EnforcementExpectedWithZeroQueriesTrips)
{
    SupportQueryGuard::get().noteClaimBearingBundleRegistered();
    EXPECT_TRUE(SupportQueryGuard::get().enforcementExpected());
    EXPECT_EQ(SupportQueryGuard::get().queriesObserved(), 0u);
    EXPECT_TRUE(SupportQueryGuard::get().tripped());
}

TEST_F(TestSupportQueryGuard, EnforcementExpectedWithAtLeastOneQueryDoesNotTrip)
{
    SupportQueryGuard::get().noteClaimBearingBundleRegistered();
    SupportQueryGuard::get().noteQueryObserved();
    EXPECT_FALSE(SupportQueryGuard::get().tripped());
    EXPECT_EQ(SupportQueryGuard::get().queriesObserved(), 1u);
}

TEST_F(TestSupportQueryGuard, CountsMultipleRegistrationsAndQueries)
{
    SupportQueryGuard::get().noteClaimBearingBundleRegistered();
    SupportQueryGuard::get().noteClaimBearingBundleRegistered();
    SupportQueryGuard::get().noteQueryObserved();
    SupportQueryGuard::get().noteQueryObserved();
    SupportQueryGuard::get().noteQueryObserved();

    EXPECT_EQ(SupportQueryGuard::get().claimBearingBundleCount(), 2u);
    EXPECT_EQ(SupportQueryGuard::get().queriesObserved(), 3u);
}

class TestUnclaimedSupportReport : public ::testing::Test
{
protected:
    void SetUp() override
    {
        UnclaimedSupportReport::get().reset();
    }

    void TearDown() override
    {
        UnclaimedSupportReport::get().reset();
    }
};

TEST_F(TestUnclaimedSupportReport, SingletonIdentity)
{
    EXPECT_EQ(&UnclaimedSupportReport::get(), &UnclaimedSupportReport::get());
}

TEST_F(TestUnclaimedSupportReport, EmptyReportPrintsNothing)
{
    std::ostringstream os;
    UnclaimedSupportReport::get().print(os);
    EXPECT_TRUE(os.str().empty());
}

TEST_F(TestUnclaimedSupportReport, RecordIsRetrievable)
{
    UnclaimedSupportReport::get().record("bundleA", "MIOPEN_ENGINE", "gfx942", "linux");

    const auto records = UnclaimedSupportReport::get().getRecords();
    ASSERT_EQ(records.size(), 1u);
    EXPECT_EQ(records[0].bundle, "bundleA");
    EXPECT_EQ(records[0].engine, "MIOPEN_ENGINE");
    EXPECT_EQ(records[0].arch, "gfx942");
    EXPECT_EQ(records[0].platform, "linux");
}

TEST_F(TestUnclaimedSupportReport, PrintIncludesEveryRecordedField)
{
    UnclaimedSupportReport::get().record("bundleA", "MIOPEN_ENGINE", "gfx942", "linux");

    std::ostringstream os;
    UnclaimedSupportReport::get().print(os);
    const auto text = os.str();
    EXPECT_NE(text.find("bundleA"), std::string::npos);
    EXPECT_NE(text.find("MIOPEN_ENGINE"), std::string::npos);
    EXPECT_NE(text.find("gfx942"), std::string::npos);
    EXPECT_NE(text.find("linux"), std::string::npos);
}

TEST_F(TestUnclaimedSupportReport, ResetClearsRecords)
{
    UnclaimedSupportReport::get().record("bundleA", "MIOPEN_ENGINE", "gfx942", "linux");
    UnclaimedSupportReport::get().reset();
    EXPECT_TRUE(UnclaimedSupportReport::get().getRecords().empty());
}

// NOLINTEND(readability-identifier-naming)
