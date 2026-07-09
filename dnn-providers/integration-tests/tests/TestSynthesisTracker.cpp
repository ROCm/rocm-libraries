// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>

#include "harness/golden/input_init/SynthesisTracker.hpp"

// NOLINTBEGIN(readability-identifier-naming)

using namespace hipdnn_integration_tests::golden;

namespace
{

InputTensorMap makeTensors(const std::vector<int64_t>& uids)
{
    InputTensorMap map;
    for(const int64_t uid : uids)
    {
        map[uid] = std::make_unique<hipdnn_data_sdk::utilities::Tensor<float>>(
            std::vector<int64_t>{2, 3}, std::vector<int64_t>{3, 1});
        map[uid]->fillTensorWithValue(0.f);
    }
    return map;
}

} // namespace

// All owned inputs declared FREE -> ok().
TEST(TestSynthesisTracker, AllFreeSucceeds)
{
    auto inputs = makeTensors({1, 2, 3});
    const std::vector<int64_t> owned = {1, 2, 3};
    std::mt19937 rng(42);

    SynthesisTracker tracker(owned, inputs);
    tracker.fillFree(1, -1.f, 1.f, rng);
    tracker.fillFree(2, -1.f, 1.f, rng);
    tracker.fillFree(3, -1.f, 1.f, rng);

    const auto result = tracker.finish("TestOp");
    EXPECT_TRUE(result.filled);
}

// An owned input left undeclared -> unsupported().
TEST(TestSynthesisTracker, UndeclaredInputFails)
{
    auto inputs = makeTensors({1, 2, 3});
    const std::vector<int64_t> owned = {1, 2, 3};
    std::mt19937 rng(42);

    SynthesisTracker tracker(owned, inputs);
    tracker.fillFree(1, -1.f, 1.f, rng);
    // uid 2 and 3 never declared

    const auto result = tracker.finish("TestOp");
    EXPECT_FALSE(result.filled);
    EXPECT_NE(result.reason.find("uid=2"), std::string::npos);
    EXPECT_NE(result.reason.find("uid=3"), std::string::npos);
}

// A STRUCTURED input -> unsupported() with diagnostic.
TEST(TestSynthesisTracker, StructuredInputFails)
{
    auto inputs = makeTensors({1, 2});
    const std::vector<int64_t> owned = {1, 2};
    std::mt19937 rng(42);

    SynthesisTracker tracker(owned, inputs);
    tracker.fillFree(1, -1.f, 1.f, rng);
    tracker.markStructured(2, "page_table");

    const auto result = tracker.finish("TestOp");
    EXPECT_FALSE(result.filled);
    EXPECT_NE(result.reason.find("page_table"), std::string::npos);
    EXPECT_NE(result.reason.find("structured"), std::string::npos);
}

// A DERIVED input -> unsupported() with diagnostic.
TEST(TestSynthesisTracker, DerivedInputFails)
{
    auto inputs = makeTensors({1, 2});
    const std::vector<int64_t> owned = {1, 2};
    std::mt19937 rng(42);

    SynthesisTracker tracker(owned, inputs);
    tracker.fillFree(1, -1.f, 1.f, rng);
    tracker.markDerived(2, "forward_output");

    const auto result = tracker.finish("TestOp");
    EXPECT_FALSE(result.filled);
    EXPECT_NE(result.reason.find("forward_output"), std::string::npos);
    EXPECT_NE(result.reason.find("derived"), std::string::npos);
}

// uid 0 (absent optional tensor) is silently ignored, not treated as owned.
TEST(TestSynthesisTracker, ZeroUidIgnored)
{
    auto inputs = makeTensors({1});
    const std::vector<int64_t> owned = {1};
    std::mt19937 rng(42);

    SynthesisTracker tracker(owned, inputs);
    tracker.fillFree(1, -1.f, 1.f, rng);
    tracker.markStructured(0, "absent_optional");

    const auto result = tracker.finish("TestOp");
    EXPECT_TRUE(result.filled);
}

// A uid not in the owned set is silently ignored.
TEST(TestSynthesisTracker, NonOwnedUidIgnored)
{
    auto inputs = makeTensors({1, 99});
    const std::vector<int64_t> owned = {1};
    std::mt19937 rng(42);

    SynthesisTracker tracker(owned, inputs);
    tracker.fillFree(1, -1.f, 1.f, rng);
    tracker.fillFree(99, -1.f, 1.f, rng); // not owned, ignored

    const auto result = tracker.finish("TestOp");
    EXPECT_TRUE(result.filled);
}

// Empty owned set -> ok() trivially (no inputs to account for).
TEST(TestSynthesisTracker, EmptyOwnedSucceeds)
{
    InputTensorMap inputs;
    const std::vector<int64_t> owned;

    const SynthesisTracker tracker(owned, inputs);

    const auto result = tracker.finish("TestOp");
    EXPECT_TRUE(result.filled);
}

// Mixed: some FREE, one STRUCTURED, one undeclared -> both problems reported.
TEST(TestSynthesisTracker, MixedFailuresReportAll)
{
    auto inputs = makeTensors({1, 2, 3});
    const std::vector<int64_t> owned = {1, 2, 3};
    std::mt19937 rng(42);

    SynthesisTracker tracker(owned, inputs);
    tracker.fillFree(1, -1.f, 1.f, rng);
    tracker.markStructured(2, "seed");
    // uid 3 undeclared

    const auto result = tracker.finish("TestOp");
    EXPECT_FALSE(result.filled);
    EXPECT_NE(result.reason.find("seed"), std::string::npos);
    EXPECT_NE(result.reason.find("uid=3"), std::string::npos);
}

// SynthesisResult::ok() and ::unsupported() factory methods.
TEST(TestSynthesisResult, FactoryMethods)
{
    const auto ok = SynthesisResult::ok();
    EXPECT_TRUE(ok.filled);
    EXPECT_TRUE(ok.reason.empty());

    const auto bad = SynthesisResult::unsupported("cannot synthesize X");
    EXPECT_FALSE(bad.filled);
    EXPECT_EQ(bad.reason, "cannot synthesize X");
}

// NOLINTEND(readability-identifier-naming)
