// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file TestEngineOverrideConfig.cpp
 * @brief Unit tests for the rule-matching internals of EngineOverrideConfig
 *        (op + dim + stride wildcards, exact/wildcard partition ordering,
 *        JSON parsing). The end-to-end policy behavior driven by
 *        HIPDNN_HEUR_CONFIG_PATH is covered in TestConfigBuiltIn.cpp.
 */

#include "heuristics/config/EngineOverrideConfig.hpp"

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>

#include <cstdint>
#include <vector>

using namespace hipdnn_backend::heuristics::config;
using namespace hipdnn_data_sdk::utilities;

namespace
{

struct TensorData
{
    std::vector<int64_t> dim;
    std::vector<int64_t> stride;
};

TensorView viewOf(const TensorData& t)
{
    return TensorView{&t.dim, &t.stride};
}

std::vector<TensorView> viewsOf(const std::vector<TensorData>& ts)
{
    std::vector<TensorView> views;
    views.reserve(ts.size());
    for(const auto& t : ts)
    {
        views.push_back(viewOf(t));
    }
    return views;
}

TensorPattern makePattern(std::vector<int64_t> dim)
{
    TensorPattern p;
    p.dim = std::move(dim);
    return p;
}

TensorPattern makePatternWithStride(std::vector<int64_t> dim, std::vector<int64_t> stride)
{
    TensorPattern p;
    p.dim = std::move(dim);
    p.stride = std::move(stride);
    return p;
}

EngineOverrideConfig makeConfig(std::vector<OperationRule> rules)
{
    return EngineOverrideConfig(std::move(rules));
}

} // namespace

// ── Test 1: exact dim match, single rule ────────────────────────────────────

TEST(TestEngineOverrideConfig, ExactDimMatchSingleRule)
{
    OperationRule rule;
    rule.op = "conv_fprop";
    rule.engineName = MIOPEN_ENGINE_NAME;
    rule.tensors = {makePattern({1, 3, 224, 224}), makePattern({64, 3, 7, 7})};

    const auto config = makeConfig({std::move(rule)});

    const std::vector<TensorData> tensors = {{{1, 3, 224, 224}, {}}, {{64, 3, 7, 7}, {}}};

    auto result = config.matchOperation("conv_fprop", viewsOf(tensors));
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, MIOPEN_ENGINE_ID);
}

// ── Test 2: first matching rule wins ────────────────────────────────────────

TEST(TestEngineOverrideConfig, FirstMatchingRuleWins)
{
    OperationRule rule1;
    rule1.op = "conv_fprop";
    rule1.engineName = MIOPEN_ENGINE_NAME;
    rule1.tensors = {makePattern({1, 3, 224, 224})};

    OperationRule rule2;
    rule2.op = "conv_fprop";
    rule2.engineName = HIPBLASLT_ENGINE_NAME;
    rule2.tensors = {makePattern({1, 3, 224, 224})};

    const auto config = makeConfig({std::move(rule1), std::move(rule2)});

    const std::vector<TensorData> tensors = {{{1, 3, 224, 224}, {}}};

    auto result = config.matchOperation("conv_fprop", viewsOf(tensors));
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, MIOPEN_ENGINE_ID);
}

// ── Test 3: no rule matches (wrong dims) ────────────────────────────────────

TEST(TestEngineOverrideConfig, NoRuleMatchesWrongDims)
{
    OperationRule rule;
    rule.op = "conv_fprop";
    rule.engineName = MIOPEN_ENGINE_NAME;
    rule.tensors = {makePattern({1, 3, 224, 224})};

    const auto config = makeConfig({std::move(rule)});

    const std::vector<TensorData> tensors = {{{1, 3, 112, 112}, {}}};

    auto result = config.matchOperation("conv_fprop", viewsOf(tensors));
    EXPECT_FALSE(result.has_value());
}

// ── Test 4: wildcard (-1) in one dimension ──────────────────────────────────

TEST(TestEngineOverrideConfig, WildcardInOneDimension)
{
    OperationRule rule;
    rule.op = "conv_fprop";
    rule.engineName = HIPBLASLT_ENGINE_NAME;
    rule.tensors = {makePattern({-1, 64, 56, 56})};

    const auto config = makeConfig({std::move(rule)});

    for(const int64_t batch : {1, 4, 8, 32})
    {
        const std::vector<TensorData> tensors = {{{batch, 64, 56, 56}, {}}};
        auto result = config.matchOperation("conv_fprop", viewsOf(tensors));
        ASSERT_TRUE(result.has_value()) << "batch=" << batch << " should match";
        EXPECT_EQ(*result, HIPBLASLT_ENGINE_ID);
    }

    const std::vector<TensorData> tensors = {{{4, 128, 56, 56}, {}}};
    EXPECT_FALSE(config.matchOperation("conv_fprop", viewsOf(tensors)).has_value());
}

// ── Test 5: all-wildcard rule matches any shape ─────────────────────────────

TEST(TestEngineOverrideConfig, AllWildcardRuleMatchesAnyShape)
{
    OperationRule rule;
    rule.op = "conv_fprop";
    rule.engineName = FUSILLI_ENGINE_NAME;
    rule.tensors = {makePattern({-1, -1, -1, -1})};

    const auto config = makeConfig({std::move(rule)});

    for(const auto& shape :
        std::vector<std::vector<int64_t>>{{1, 3, 224, 224}, {8, 64, 56, 56}, {32, 256, 14, 14}})
    {
        const std::vector<TensorData> tensors = {{shape, {}}};
        auto result = config.matchOperation("conv_fprop", viewsOf(tensors));
        ASSERT_TRUE(result.has_value());
        EXPECT_EQ(*result, FUSILLI_ENGINE_ID);
    }
}

// ── Test 6: wrong op name → nullopt ─────────────────────────────────────────

TEST(TestEngineOverrideConfig, WrongOpNameReturnsNullopt)
{
    OperationRule rule;
    rule.op = "conv_fprop";
    rule.engineName = MIOPEN_ENGINE_NAME;
    rule.tensors = {makePattern({1, 3, 224, 224})};

    const auto config = makeConfig({std::move(rule)});

    const std::vector<TensorData> tensors = {{{1, 3, 224, 224}, {}}};

    EXPECT_FALSE(config.matchOperation("conv_dgrad", viewsOf(tensors)).has_value());
    EXPECT_FALSE(config.matchOperation("conv_wgrad", viewsOf(tensors)).has_value());
    EXPECT_FALSE(config.matchOperation("matmul", viewsOf(tensors)).has_value());
}

// ── Test 7: wrong tensor count in rule → nullopt ────────────────────────────

TEST(TestEngineOverrideConfig, WrongTensorCountReturnsNullopt)
{
    OperationRule rule;
    rule.op = "conv_fprop";
    rule.engineName = MIOPEN_ENGINE_NAME;
    rule.tensors = {makePattern({1, 3, 224, 224}), makePattern({64, 3, 7, 7})};

    const auto config = makeConfig({std::move(rule)});

    const std::vector<TensorData> tensors1 = {{{1, 3, 224, 224}, {}}};
    EXPECT_FALSE(config.matchOperation("conv_fprop", viewsOf(tensors1)).has_value());

    const std::vector<TensorData> tensors3
        = {{{1, 3, 224, 224}, {}}, {{64, 3, 7, 7}, {}}, {{64, 1, 1, 1}, {}}};
    EXPECT_FALSE(config.matchOperation("conv_fprop", viewsOf(tensors3)).has_value());
}

// ── Tests 11–12: cross-partition ordering (exact vs wildcard) ───────────────

TEST(TestEngineOverrideConfig, WildcardBeforeExactBothMatch)
{
    OperationRule wildcard;
    wildcard.op = "conv_fprop";
    wildcard.engineName = FUSILLI_ENGINE_NAME;
    wildcard.tensors = {makePattern({-1, 3, 224, 224})};

    OperationRule exact;
    exact.op = "conv_fprop";
    exact.engineName = HIPBLASLT_ENGINE_NAME;
    exact.tensors = {makePattern({1, 3, 224, 224})};

    const auto config = makeConfig({std::move(wildcard), std::move(exact)});

    const std::vector<TensorData> tensors = {{{1, 3, 224, 224}, {}}};
    auto result = config.matchOperation("conv_fprop", viewsOf(tensors));
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, FUSILLI_ENGINE_ID);
}

TEST(TestEngineOverrideConfig, ExactBeforeWildcardBothMatch)
{
    OperationRule exact;
    exact.op = "conv_fprop";
    exact.engineName = HIPBLASLT_ENGINE_NAME;
    exact.tensors = {makePattern({1, 3, 224, 224})};

    OperationRule wildcard;
    wildcard.op = "conv_fprop";
    wildcard.engineName = FUSILLI_ENGINE_NAME;
    wildcard.tensors = {makePattern({-1, 3, 224, 224})};

    const auto config = makeConfig({std::move(exact), std::move(wildcard)});

    const std::vector<TensorData> tensors = {{{1, 3, 224, 224}, {}}};
    auto result = config.matchOperation("conv_fprop", viewsOf(tensors));
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, HIPBLASLT_ENGINE_ID);
}

// ── Stride matching tests ────────────────────────────────────────────────────

TEST(TestEngineOverrideConfig, ExactStrideMatchSelectsEngine)
{
    OperationRule rule;
    rule.op = "conv_fprop";
    rule.engineName = MIOPEN_ENGINE_NAME;
    rule.tensors = {makePatternWithStride({1, 3, 224, 224}, {150528, 50176, 224, 1})};

    const auto config = makeConfig({std::move(rule)});

    const std::vector<TensorData> matching = {{{1, 3, 224, 224}, {150528, 50176, 224, 1}}};
    auto result = config.matchOperation("conv_fprop", viewsOf(matching));
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, MIOPEN_ENGINE_ID);

    const std::vector<TensorData> wrongStride
        = {{{1, 3, 224, 224}, {1, 224, int64_t{224} * 3, int64_t{224} * 3 * 224}}};
    EXPECT_FALSE(config.matchOperation("conv_fprop", viewsOf(wrongStride)).has_value());
}

TEST(TestEngineOverrideConfig, WildcardStrideElement)
{
    OperationRule rule;
    rule.op = "conv_fprop";
    rule.engineName = HIPBLASLT_ENGINE_NAME;
    rule.tensors = {makePatternWithStride({1, 3, 224, 224}, {150528, 50176, -1, -1})};

    const auto config = makeConfig({std::move(rule)});

    for(const int64_t s2 : {224, 112, 56})
    {
        const std::vector<TensorData> tensors = {{{1, 3, 224, 224}, {150528, 50176, s2, 1}}};
        auto result = config.matchOperation("conv_fprop", viewsOf(tensors));
        ASSERT_TRUE(result.has_value()) << "stride[2]=" << s2;
        EXPECT_EQ(*result, HIPBLASLT_ENGINE_ID);
    }

    const std::vector<TensorData> wrongStride = {{{1, 3, 224, 224}, {999, 50176, 224, 1}}};
    EXPECT_FALSE(config.matchOperation("conv_fprop", viewsOf(wrongStride)).has_value());
}

TEST(TestEngineOverrideConfig, EmptyStridePatternMatchesAnyStride)
{
    OperationRule rule;
    rule.op = "conv_fprop";
    rule.engineName = FUSILLI_ENGINE_NAME;
    rule.tensors = {makePattern({1, 3, 224, 224})};

    const auto config = makeConfig({std::move(rule)});

    for(const auto& strides : std::vector<std::vector<int64_t>>{
            {150528, 50176, 224, 1}, {1, 3, 672, 150528}, {999, 888, 777, 666}})
    {
        const std::vector<TensorData> tensors = {{{1, 3, 224, 224}, strides}};
        auto result = config.matchOperation("conv_fprop", viewsOf(tensors));
        ASSERT_TRUE(result.has_value());
        EXPECT_EQ(*result, FUSILLI_ENGINE_ID);
    }
}

// ── JSON-dependent ──────────────────────────────────────────────────────────

TEST(TestEngineOverrideConfig, LoadFromValidJsonFile)
{
    constexpr const char* CONTENTS = R"({
  "engine_overrides": [
    {
      "comment": "test rule for ResNet first conv",
      "op": "conv_fprop",
      "engine_name": "MIOPEN_ENGINE",
      "tensors": [
        { "dim": [1, 3, 224, 224] },
        { "dim": [64, 3, 7, 7] }
      ]
    },
    {
      "comment": "wildcard catch-all",
      "op": "conv_fprop",
      "engine_name": "FUSILLI_ENGINE",
      "tensors": [
        { "dim": [-1, -1, -1, -1] },
        { "dim": [-1, -1, -1, -1] }
      ]
    }
  ]
})";

    auto config = EngineOverrideConfig::loadFromContent(CONTENTS);
    ASSERT_TRUE(config.has_value());

    const std::vector<TensorData> exact = {{{1, 3, 224, 224}, {}}, {{64, 3, 7, 7}, {}}};
    auto r1 = config->matchOperation("conv_fprop", viewsOf(exact));
    ASSERT_TRUE(r1.has_value());
    EXPECT_EQ(*r1, MIOPEN_ENGINE_ID);

    const std::vector<TensorData> other = {{{8, 64, 56, 56}, {}}, {{64, 64, 3, 3}, {}}};
    auto r2 = config->matchOperation("conv_fprop", viewsOf(other));
    ASSERT_TRUE(r2.has_value());
    EXPECT_EQ(*r2, FUSILLI_ENGINE_ID);
}

TEST(TestEngineOverrideConfig, LoadFromMissingFileReturnsNullopt)
{
    auto config = EngineOverrideConfig::load("/nonexistent/path/hipdnn_no_such_file.json");
    EXPECT_FALSE(config.has_value());
}

TEST(TestEngineOverrideConfig, EnvVarUnsetReturnsNullopt)
{
    EXPECT_FALSE(EngineOverrideConfig::loadFromEnv().has_value());
}

// ── Malformed JSON: parser must return nullopt, not throw ──────────────────

TEST(TestEngineOverrideConfig, LoadFromContentRejectsInvalidJsonSyntax)
{
    // Trailing comma + unterminated array — nlohmann::json::parse_error.
    constexpr const char* CONTENTS = R"({ "engine_overrides": [)";
    EXPECT_FALSE(EngineOverrideConfig::loadFromContent(CONTENTS).has_value());
}

TEST(TestEngineOverrideConfig, LoadFromContentRejectsMissingTopLevelKey)
{
    // Valid JSON, but no "engine_overrides" key — at() throws out_of_range.
    constexpr const char* CONTENTS = R"({ "other_key": [] })";
    EXPECT_FALSE(EngineOverrideConfig::loadFromContent(CONTENTS).has_value());
}

TEST(TestEngineOverrideConfig, LoadFromContentRejectsMissingEntryFields)
{
    // engine_overrides entry missing "engine_name" — at() throws out_of_range.
    constexpr const char* CONTENTS = R"({
  "engine_overrides": [
    { "op": "conv_fprop", "tensors": [ { "dim": [1, 3, 224, 224] } ] }
  ]
})";
    EXPECT_FALSE(EngineOverrideConfig::loadFromContent(CONTENTS).has_value());
}

TEST(TestEngineOverrideConfig, LoadFromContentRejectsMissingTensorDim)
{
    // tensor entry missing "dim" — at() throws out_of_range.
    constexpr const char* CONTENTS = R"({
  "engine_overrides": [
    {
      "op": "conv_fprop",
      "engine_name": "MIOPEN_ENGINE",
      "tensors": [ { "stride": [1, 2, 3, 4] } ]
    }
  ]
})";
    EXPECT_FALSE(EngineOverrideConfig::loadFromContent(CONTENTS).has_value());
}

TEST(TestEngineOverrideConfig, LoadFromContentRejectsWrongFieldType)
{
    // "op" is a number, not a string — get<string> throws type_error.
    constexpr const char* CONTENTS = R"({
  "engine_overrides": [
    {
      "op": 42,
      "engine_name": "MIOPEN_ENGINE",
      "tensors": [ { "dim": [1, 3, 224, 224] } ]
    }
  ]
})";
    EXPECT_FALSE(EngineOverrideConfig::loadFromContent(CONTENTS).has_value());
}

TEST(TestEngineOverrideConfig, JsonWithStrideConstraint)
{
    constexpr const char* CONTENTS = R"({
  "engine_overrides": [
    {
      "op": "conv_fprop",
      "engine_name": "MIOPEN_ENGINE",
      "tensors": [
        { "dim": [1, 3, 224, 224], "stride": [150528, 50176, 224, 1] },
        { "dim": [64, 3, 7, 7] }
      ]
    }
  ]
})";

    auto config = EngineOverrideConfig::loadFromContent(CONTENTS);
    ASSERT_TRUE(config.has_value());

    const std::vector<TensorData> matching
        = {{{1, 3, 224, 224}, {150528, 50176, 224, 1}}, {{64, 3, 7, 7}, {}}};
    auto r1 = config->matchOperation("conv_fprop", viewsOf(matching));
    ASSERT_TRUE(r1.has_value());
    EXPECT_EQ(*r1, MIOPEN_ENGINE_ID);

    const std::vector<TensorData> wrong
        = {{{1, 3, 224, 224}, {1, 224, int64_t{224} * 3, int64_t{224} * 3 * 224}},
           {{64, 3, 7, 7}, {}}};
    EXPECT_FALSE(config->matchOperation("conv_fprop", viewsOf(wrong)).has_value());
}
