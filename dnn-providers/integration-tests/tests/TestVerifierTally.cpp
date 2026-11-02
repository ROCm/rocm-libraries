// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <sstream>

#include "harness/bundle/VerificationOutcome.hpp"
#include "harness/bundle/VerifierTally.hpp"

using hipdnn_integration_tests::bundle::Verifier;
using hipdnn_integration_tests::bundle::VerifierTally;
using hipdnn_integration_tests::bundle::VerifierTallyIterationReset;

// NOLINTBEGIN(readability-identifier-naming)

namespace
{

class TestVerifierTally : public ::testing::Test
{
protected:
    std::ostringstream _out;

    void SetUp() override
    {
        VerifierTally::get().reset();
    }

    void TearDown() override
    {
        VerifierTally::get().reset();
    }
};

} // namespace

TEST_F(TestVerifierTally, CountsEachVerifierUnderItsOwnName)
{
    auto& tally = VerifierTally::get();
    tally.record("a.json", Verifier::GOLDEN, _out);
    tally.record("b.json", Verifier::GPU_REFERENCE, _out);
    tally.record("c.json", Verifier::GPU_REFERENCE, _out);
    tally.record("d.json", Verifier::CPU_REFERENCE, _out);
    tally.record("e.json", Verifier::NONE, _out);

    const auto counts = tally.counts();
    EXPECT_EQ(counts.golden, 1u);
    EXPECT_EQ(counts.gpuReference, 2u);
    EXPECT_EQ(counts.cpuReference, 1u);
    EXPECT_EQ(counts.none, 1u);
    EXPECT_EQ(counts.total(), 5u);
}

// With --gtest_repeat=N, main() prints gtest's counts for the last iteration. The
// "Verified by" line beside them must count that iteration too, not all N of them.
TEST_F(TestVerifierTally, EachIterationStartsFromZero)
{
    auto& tally = VerifierTally::get();
    tally.record("a.json", Verifier::GOLDEN, _out);
    tally.record("b.json", Verifier::GPU_REFERENCE, _out);

    VerifierTallyIterationReset listener;
    listener.OnTestIterationStart(*::testing::UnitTest::GetInstance(), 1);
    tally.record("b.json", Verifier::GPU_REFERENCE, _out);

    const auto counts = tally.counts();
    EXPECT_EQ(counts.golden, 0u);
    EXPECT_EQ(counts.gpuReference, 1u);
    EXPECT_EQ(counts.total(), 1u);
}

// NOLINTEND(readability-identifier-naming)
