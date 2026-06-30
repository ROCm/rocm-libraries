// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>

#include "harness/input_init/InputSynthesizer.hpp"

// NOLINTBEGIN(readability-identifier-naming)

using namespace hipdnn_integration_tests;

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

// All owned inputs implicitly FREE -> ok().
TEST(TestInputSynthesizer, AllFreeSucceeds)
{
    auto inputs = makeTensors({1, 2, 3});
    const std::vector<int64_t> owned = {1, 2, 3};

    InputSynthesizer synth(owned, inputs);

    const auto result = synth.synthesize("TestOp");
    EXPECT_TRUE(result.filled);
}

// Op-specific range override works.
TEST(TestInputSynthesizer, RangeOverrideSucceeds)
{
    auto inputs = makeTensors({1, 2});
    const std::vector<int64_t> owned = {1, 2};

    InputSynthesizer synth(owned, inputs);
    synth.range(2, 0.5f, 1.5f);

    const auto result = synth.synthesize("TestOp");
    EXPECT_TRUE(result.filled);
}

// A STRUCTURED input -> unsupported() with diagnostic.
TEST(TestInputSynthesizer, StructuredInputFails)
{
    auto inputs = makeTensors({1, 2});
    const std::vector<int64_t> owned = {1, 2};

    InputSynthesizer synth(owned, inputs);
    synth.markStructured(2, "page_table");

    const auto result = synth.synthesize("TestOp");
    EXPECT_FALSE(result.filled);
    EXPECT_NE(result.reason.find("page_table"), std::string::npos);
    EXPECT_NE(result.reason.find("structured"), std::string::npos);
}

// A DERIVED input -> unsupported() with diagnostic.
TEST(TestInputSynthesizer, DerivedInputFails)
{
    auto inputs = makeTensors({1, 2});
    const std::vector<int64_t> owned = {1, 2};

    InputSynthesizer synth(owned, inputs);
    synth.markDerived(2, "forward_output");

    const auto result = synth.synthesize("TestOp");
    EXPECT_FALSE(result.filled);
    EXPECT_NE(result.reason.find("forward_output"), std::string::npos);
    EXPECT_NE(result.reason.find("derived"), std::string::npos);
}

// uid 0 (absent optional tensor) is silently ignored, not treated as owned.
TEST(TestInputSynthesizer, ZeroUidIgnored)
{
    auto inputs = makeTensors({1});
    const std::vector<int64_t> owned = {1};

    InputSynthesizer synth(owned, inputs);
    synth.markStructured(0, "absent_optional");

    const auto result = synth.synthesize("TestOp");
    EXPECT_TRUE(result.filled);
}

// A uid not in the owned set is silently ignored by range().
TEST(TestInputSynthesizer, NonOwnedUidIgnored)
{
    auto inputs = makeTensors({1, 99});
    const std::vector<int64_t> owned = {1};

    InputSynthesizer synth(owned, inputs);
    synth.range(99, 0.0f, 1.0f); // not owned, ignored

    const auto result = synth.synthesize("TestOp");
    EXPECT_TRUE(result.filled);
}

// Empty owned set -> ok() trivially (no inputs to account for).
TEST(TestInputSynthesizer, EmptyOwnedSucceeds)
{
    InputTensorMap inputs;
    const std::vector<int64_t> owned;

    InputSynthesizer synth(owned, inputs);

    const auto result = synth.synthesize("TestOp");
    EXPECT_TRUE(result.filled);
}

// Mixed: one STRUCTURED -> reported as refusal.
TEST(TestInputSynthesizer, StructuredRefusalReported)
{
    auto inputs = makeTensors({1, 2, 3});
    const std::vector<int64_t> owned = {1, 2, 3};

    InputSynthesizer synth(owned, inputs);
    synth.markStructured(2, "seed");

    const auto result = synth.synthesize("TestOp");
    EXPECT_FALSE(result.filled);
    EXPECT_NE(result.reason.find("seed"), std::string::npos);
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
