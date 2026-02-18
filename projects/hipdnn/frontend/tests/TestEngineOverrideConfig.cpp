// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend/detail/EngineOverrideConfig.hpp>

#ifndef HIPDNN_FRONTEND_SKIP_JSON_LIB
#include <cstdio>
#include <fstream>
#endif

using namespace hipdnn_frontend::engine_override;
using namespace hipdnn_frontend::graph;

// ── helpers ─────────────────────────────────────────────────────────────────

static std::shared_ptr<TensorAttributes> makeTensor(std::vector<int64_t> dims)
{
    auto t = std::make_shared<TensorAttributes>();
    t->set_dim(std::move(dims));
    return t;
}

static TensorPattern makePattern(std::vector<int64_t> dims)
{
    TensorPattern p;
    p.dim = std::move(dims);
    return p;
}

// Construct a single-rule config inline (no JSON required).
static EngineOverrideConfig makeConfig(std::vector<OperationRule> rules)
{
    return EngineOverrideConfig(std::move(rules));
}

// ── Test 1: exact dim match, single rule ────────────────────────────────────

TEST(TestMatch, ExactDimMatchSingleRule)
{
    OperationRule rule;
    rule.op = "conv_fprop";
    rule.engine_id = 3;
    rule.tensors = {makePattern({1, 3, 224, 224}), makePattern({64, 3, 7, 7})};

    auto config = makeConfig({std::move(rule)});

    std::vector<std::shared_ptr<TensorAttributes>> tensors
        = {makeTensor({1, 3, 224, 224}), makeTensor({64, 3, 7, 7})};

    auto result = config.matchOperation("conv_fprop", tensors);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 3);
}

// ── Test 2: first matching rule wins ────────────────────────────────────────

TEST(TestMatch, FirstMatchingRuleWins)
{
    OperationRule rule1;
    rule1.op = "conv_fprop";
    rule1.engine_id = 3;
    rule1.tensors = {makePattern({1, 3, 224, 224})};

    OperationRule rule2;
    rule2.op = "conv_fprop";
    rule2.engine_id = 7;
    rule2.tensors = {makePattern({1, 3, 224, 224})};

    auto config = makeConfig({std::move(rule1), std::move(rule2)});

    std::vector<std::shared_ptr<TensorAttributes>> tensors = {makeTensor({1, 3, 224, 224})};

    auto result = config.matchOperation("conv_fprop", tensors);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 3); // first rule wins
}

// ── Test 3: no rule matches (wrong dims) ────────────────────────────────────

TEST(TestMatch, NoRuleMatchesWrongDims)
{
    OperationRule rule;
    rule.op = "conv_fprop";
    rule.engine_id = 3;
    rule.tensors = {makePattern({1, 3, 224, 224})};

    auto config = makeConfig({std::move(rule)});

    std::vector<std::shared_ptr<TensorAttributes>> tensors = {
        makeTensor({1, 3, 112, 112}) // different spatial dims
    };

    auto result = config.matchOperation("conv_fprop", tensors);
    EXPECT_FALSE(result.has_value());
}

// ── Test 4: wildcard (-1) in one dimension ──────────────────────────────────

TEST(TestMatch, WildcardInOneDimension)
{
    OperationRule rule;
    rule.op = "conv_fprop";
    rule.engine_id = 7;
    rule.tensors = {makePattern({-1, 64, 56, 56})}; // batch dim is wildcard

    auto config = makeConfig({std::move(rule)});

    for(int64_t batch : {1, 4, 8, 32})
    {
        std::vector<std::shared_ptr<TensorAttributes>> tensors = {makeTensor({batch, 64, 56, 56})};
        auto result = config.matchOperation("conv_fprop", tensors);
        ASSERT_TRUE(result.has_value()) << "batch=" << batch << " should match";
        EXPECT_EQ(*result, 7);
    }

    // Non-matching channel dim should still fail
    std::vector<std::shared_ptr<TensorAttributes>> tensors = {makeTensor({4, 128, 56, 56})};
    EXPECT_FALSE(config.matchOperation("conv_fprop", tensors).has_value());
}

// ── Test 5: all-wildcard rule matches any shape ─────────────────────────────

TEST(TestMatch, AllWildcardRuleMatchesAnyShape)
{
    OperationRule rule;
    rule.op = "conv_fprop";
    rule.engine_id = 1;
    rule.tensors = {makePattern({-1, -1, -1, -1})};

    auto config = makeConfig({std::move(rule)});

    for(const auto& shape :
        std::vector<std::vector<int64_t>>{{1, 3, 224, 224}, {8, 64, 56, 56}, {32, 256, 14, 14}})
    {
        std::vector<std::shared_ptr<TensorAttributes>> tensors = {makeTensor(shape)};
        auto result = config.matchOperation("conv_fprop", tensors);
        ASSERT_TRUE(result.has_value());
        EXPECT_EQ(*result, 1);
    }
}

// ── Test 6: wrong op name → nullopt ─────────────────────────────────────────

TEST(TestMatch, WrongOpNameReturnsNullopt)
{
    OperationRule rule;
    rule.op = "conv_fprop";
    rule.engine_id = 3;
    rule.tensors = {makePattern({1, 3, 224, 224})};

    auto config = makeConfig({std::move(rule)});

    std::vector<std::shared_ptr<TensorAttributes>> tensors = {makeTensor({1, 3, 224, 224})};

    EXPECT_FALSE(config.matchOperation("conv_dgrad", tensors).has_value());
    EXPECT_FALSE(config.matchOperation("conv_wgrad", tensors).has_value());
    EXPECT_FALSE(config.matchOperation("matmul", tensors).has_value());
}

// ── Test 7: wrong tensor count in rule → nullopt ────────────────────────────

TEST(TestMatch, WrongTensorCountReturnsNullopt)
{
    OperationRule rule;
    rule.op = "conv_fprop";
    rule.engine_id = 3;
    rule.tensors = {makePattern({1, 3, 224, 224}), makePattern({64, 3, 7, 7})}; // 2 patterns

    auto config = makeConfig({std::move(rule)});

    // Provide only 1 tensor where 2 are expected
    std::vector<std::shared_ptr<TensorAttributes>> tensors = {makeTensor({1, 3, 224, 224})};
    EXPECT_FALSE(config.matchOperation("conv_fprop", tensors).has_value());

    // Provide 3 tensors where 2 are expected
    std::vector<std::shared_ptr<TensorAttributes>> tensors3
        = {makeTensor({1, 3, 224, 224}), makeTensor({64, 3, 7, 7}), makeTensor({64, 1, 1, 1})};
    EXPECT_FALSE(config.matchOperation("conv_fprop", tensors3).has_value());
}

// ── Tests 11–12: cross-partition ordering (exact vs wildcard) ───────────────
//
// These tests verify that first-match-wins semantics are preserved when an
// exact rule and a wildcard rule sit in different partitions.

// Test 11: wildcard declared before exact — wildcard must win
TEST(TestMatch, WildcardBeforeExactBothMatch)
{
    OperationRule wildcard;
    wildcard.op = "conv_fprop";
    wildcard.engine_id = 10;
    wildcard.tensors = {makePattern({-1, 3, 224, 224})}; // order 0, wildcard

    OperationRule exact;
    exact.op = "conv_fprop";
    exact.engine_id = 20;
    exact.tensors = {makePattern({1, 3, 224, 224})}; // order 1, exact

    auto config = makeConfig({std::move(wildcard), std::move(exact)});

    std::vector<std::shared_ptr<TensorAttributes>> tensors = {makeTensor({1, 3, 224, 224})};
    auto result = config.matchOperation("conv_fprop", tensors);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 10); // wildcard (order 0) beats exact (order 1)
}

// Test 12: exact declared before wildcard — exact must win
TEST(TestMatch, ExactBeforeWildcardBothMatch)
{
    OperationRule exact;
    exact.op = "conv_fprop";
    exact.engine_id = 20;
    exact.tensors = {makePattern({1, 3, 224, 224})}; // order 0, exact

    OperationRule wildcard;
    wildcard.op = "conv_fprop";
    wildcard.engine_id = 10;
    wildcard.tensors = {makePattern({-1, 3, 224, 224})}; // order 1, wildcard

    auto config = makeConfig({std::move(exact), std::move(wildcard)});

    std::vector<std::shared_ptr<TensorAttributes>> tensors = {makeTensor({1, 3, 224, 224})};
    auto result = config.matchOperation("conv_fprop", tensors);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 20); // exact (order 0) beats wildcard (order 1)
}

// ── Tests 8–10: JSON-dependent ──────────────────────────────────────────────

#ifndef HIPDNN_FRONTEND_SKIP_JSON_LIB

// Test 8: load from valid JSON file → parses rules, matches correctly

TEST(TestMatch, LoadFromValidJsonFile)
{
    constexpr const char* kPath = "/tmp/hipdnn_match_test_case8.json";

    {
        std::ofstream f(kPath);
        f << R"({
  "engine_overrides": [
    {
      "comment": "test rule for ResNet first conv",
      "op": "conv_fprop",
      "engine_id": 5,
      "tensors": [
        { "dim": [1, 3, 224, 224] },
        { "dim": [64, 3, 7, 7] }
      ]
    },
    {
      "comment": "wildcard catch-all",
      "op": "conv_fprop",
      "engine_id": 1,
      "tensors": [
        { "dim": [-1, -1, -1, -1] },
        { "dim": [-1, -1, -1, -1] }
      ]
    }
  ]
})";
    }

    auto config = EngineOverrideConfig::load(kPath);
    ASSERT_TRUE(config.has_value());

    // Exact match hits the first rule
    std::vector<std::shared_ptr<TensorAttributes>> exact
        = {makeTensor({1, 3, 224, 224}), makeTensor({64, 3, 7, 7})};
    auto r1 = config->matchOperation("conv_fprop", exact);
    ASSERT_TRUE(r1.has_value());
    EXPECT_EQ(*r1, 5);

    // Different shape falls through to the wildcard rule
    std::vector<std::shared_ptr<TensorAttributes>> other
        = {makeTensor({8, 64, 56, 56}), makeTensor({64, 64, 3, 3})};
    auto r2 = config->matchOperation("conv_fprop", other);
    ASSERT_TRUE(r2.has_value());
    EXPECT_EQ(*r2, 1);

    std::remove(kPath);
}

// Test 9: load from missing file → nullopt, no crash

TEST(TestMatch, LoadFromMissingFileReturnsNullopt)
{
    auto config = EngineOverrideConfig::load("/nonexistent/path/hipdnn_no_such_file.json");
    EXPECT_FALSE(config.has_value());
}

// Test 10: env var unset → loadFromEnv() returns nullopt

TEST(TestMatch, EnvVarUnsetReturnsNullopt)
{
    // Use a name guaranteed not to be set in the test environment.
    auto config = EngineOverrideConfig::loadFromEnv("HIPDNN_MATCH_TEST_UNSET_VAR_XYZ_99");
    EXPECT_FALSE(config.has_value());
}

#endif // HIPDNN_FRONTEND_SKIP_JSON_LIB
