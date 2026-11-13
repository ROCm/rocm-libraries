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
    tally.record("a.json", Verifier::GOLDEN, {}, _out);
    tally.record("b.json", Verifier::GPU_REFERENCE, {}, _out);
    tally.record("c.json", Verifier::GPU_REFERENCE, {}, _out);
    tally.record("d.json", Verifier::CPU_REFERENCE, {}, _out);
    tally.record("e.json", Verifier::NONE, {}, _out);
    tally.record("f.json", Verifier::EXPECTED_REJECTION, "ranked_list", _out);

    const auto counts = tally.counts();
    EXPECT_EQ(counts.golden, 1u);
    EXPECT_EQ(counts.gpuReference, 2u);
    EXPECT_EQ(counts.cpuReference, 1u);
    EXPECT_EQ(counts.rejected, 1u);
    EXPECT_EQ(counts.none, 1u);
    EXPECT_EQ(counts.total(), 6u);
}

// An expected rejection says where the engine refused the bundle on the per-test line,
// which is the only place a reader of a passing run can see it. Every other verifier
// keeps its bare form.
TEST_F(TestVerifierTally, DetailIsPrintedInParenthesesAfterTheVerifier)
{
    auto& tally = VerifierTally::get();
    tally.record("a.json", Verifier::EXPECTED_REJECTION, "plan_build_failed", _out);
    tally.record("b.json", Verifier::GOLDEN, {}, _out);

    EXPECT_EQ(_out.str(),
              "[ VERIFIER ] rejected(plan_build_failed): a.json\n"
              "[ VERIFIER ] golden: b.json\n");
}

// With --gtest_repeat=N, main() prints gtest's counts for the last iteration. The
// "Verified by" line beside them must count that iteration too, not all N of them.
TEST_F(TestVerifierTally, EachIterationStartsFromZero)
{
    auto& tally = VerifierTally::get();
    tally.record("a.json", Verifier::GOLDEN, {}, _out);
    tally.record("b.json", Verifier::GPU_REFERENCE, {}, _out);

    VerifierTallyIterationReset listener;
    listener.OnTestIterationStart(*::testing::UnitTest::GetInstance(), 1);
    tally.record("b.json", Verifier::GPU_REFERENCE, {}, _out);

    const auto counts = tally.counts();
    EXPECT_EQ(counts.golden, 0u);
    EXPECT_EQ(counts.gpuReference, 1u);
    EXPECT_EQ(counts.total(), 1u);
}

// NOLINTEND(readability-identifier-naming)
