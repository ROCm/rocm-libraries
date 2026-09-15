// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only unit tests for benchmark_collective.hpp: no GPU/HIP calls, pure logic.

#include "benchmark_collective.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

using hipblaslt_bench::agree_flag;
using hipblaslt_bench::agree_value;
using hipblaslt_bench::AgreeOp;
using hipblaslt_bench::CollectiveAgreement;

namespace
{
    // Stands in for a rank group: every call reduces the caller's contribution
    // against the fixed contributions of the other ranks.
    CollectiveAgreement group(std::vector<double> others_value, std::vector<bool> others_flag)
    {
        CollectiveAgreement a;
        a.value = [others_value](double mine, AgreeOp op) {
            double acc = mine;
            for(double other : others_value)
                acc = (op == AgreeOp::Max) ? std::max(acc, other) : std::min(acc, other);
            return acc;
        };
        a.flag = [others_flag](bool mine, AgreeOp op) {
            bool acc = mine;
            for(bool other : others_flag)
                acc = (op == AgreeOp::All) ? (acc && other) : (acc || other);
            return acc;
        };
        return a;
    }
} // namespace

TEST(benchmark_collective, empty_agreement_is_identity)
{
    const CollectiveAgreement none;

    EXPECT_EQ(agree_value(none, 12.5, AgreeOp::Max), 12.5);
    EXPECT_EQ(agree_value(none, 12.5, AgreeOp::Min), 12.5);
    EXPECT_TRUE(agree_flag(none, true, AgreeOp::All));
    EXPECT_FALSE(agree_flag(none, false, AgreeOp::All));
    EXPECT_TRUE(agree_flag(none, true, AgreeOp::Any));
    EXPECT_FALSE(agree_flag(none, false, AgreeOp::Any));
}

TEST(benchmark_collective, value_reduces_both_directions)
{
    const CollectiveAgreement a = group({3.0, 7.0}, {});

    EXPECT_EQ(agree_value(a, 5.0, AgreeOp::Max), 7.0);
    EXPECT_EQ(agree_value(a, 5.0, AgreeOp::Min), 3.0);
}

TEST(benchmark_collective, flag_all_requires_every_rank)
{
    EXPECT_TRUE(agree_flag(group({}, {true, true}), true, AgreeOp::All));
    EXPECT_FALSE(agree_flag(group({}, {true, false}), true, AgreeOp::All));
}

TEST(benchmark_collective, flag_any_takes_one_rank)
{
    EXPECT_TRUE(agree_flag(group({}, {false, true}), false, AgreeOp::Any));
    EXPECT_FALSE(agree_flag(group({}, {false, false}), false, AgreeOp::Any));
}
