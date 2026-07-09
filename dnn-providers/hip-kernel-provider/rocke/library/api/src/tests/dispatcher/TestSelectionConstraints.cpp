// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <functional>
#include <map>
#include <string>
#include <vector>

#include "dispatcher/SelectionConstraints.hpp"
#include "tests/dispatcher/DispatcherFixtures.hpp"

namespace rocke_client::dispatcher
{
namespace
{

using test::InstanceParams;
using test::makeInstance;
using test::makeMatchingProblem;

// Single-field selection matrix: start from a matching problem, mutate exactly
// one field, and assert whether the instance still serves it. Add a row to
// extend coverage of a new selection key.
struct SatisfyCase
{
    std::string name;
    std::function<void(SdpaProblem&)> mutate;
    bool expected;
};

class SatisfiesShapeMatrix : public testing::TestWithParam<SatisfyCase>
{
};

TEST_P(SatisfiesShapeMatrix, MatchesExpected)
{
    const AotInstance instance = makeInstance(InstanceParams{});
    SdpaProblem problem = makeMatchingProblem(InstanceParams{});
    GetParam().mutate(problem);
    EXPECT_EQ(satisfies(instance, problem), GetParam().expected);
}

INSTANTIATE_TEST_SUITE_P(
    SelectionConstraints,
    SatisfiesShapeMatrix,
    testing::Values(
        SatisfyCase{"Matching", [](SdpaProblem&) {}, true},
        SatisfyCase{"WrongHeadSize", [](SdpaProblem& p) { p.headSize = 128; }, false},
        SatisfyCase{"WrongSeqlenQ", [](SdpaProblem& p) { p.seqlenQ = 128; }, false},
        SatisfyCase{"WrongSeqlenK", [](SdpaProblem& p) { p.seqlenK = 128; }, false},
        SatisfyCase{"WrongQueryHeads", [](SdpaProblem& p) { p.numQueryHeads = 8; }, false},
        SatisfyCase{"WrongKvHeads", [](SdpaProblem& p) { p.numKvHeads = 2; }, false},
        SatisfyCase{"WrongDtype", [](SdpaProblem& p) { p.dtype = "bf16"; }, false},
        SatisfyCase{"LayoutBhsd", [](SdpaProblem& p) { p.layout = TensorLayout::BHSD; }, false},
        SatisfyCase{"LayoutOther", [](SdpaProblem& p) { p.layout = TensorLayout::OTHER; }, false},
        SatisfyCase{"Dropout", [](SdpaProblem& p) { p.dropoutProbability = 0.1; }, false},
        SatisfyCase{"PaddingMask", [](SdpaProblem& p) { p.paddingMask = true; }, false},
        SatisfyCase{"AlibiMask", [](SdpaProblem& p) { p.alibiMask = true; }, false},
        SatisfyCase{"MaskMode", [](SdpaProblem& p) { p.maskMode = "causal_top_left"; }, false},
        SatisfyCase{"ScalePolicy", [](SdpaProblem& p) { p.scalePolicy = "explicit"; }, false}),
    [](const testing::TestParamInfo<SatisfyCase>& info) { return info.param.name; });

TEST(TestSelectionConstraints, EnforcesInclusiveBatchRange)
{
    InstanceParams params;
    params.batchMin = 2;
    params.batchMax = 8;
    const AotInstance instance = makeInstance(params);

    EXPECT_FALSE(satisfies(instance, makeMatchingProblem(params, /*batch=*/1)));
    EXPECT_TRUE(satisfies(instance, makeMatchingProblem(params, /*batch=*/2)));
    EXPECT_TRUE(satisfies(instance, makeMatchingProblem(params, /*batch=*/8)));
    EXPECT_FALSE(satisfies(instance, makeMatchingProblem(params, /*batch=*/9)));
}

TEST(TestSelectionConstraints, AttributesMatchConstraintsOperators)
{
    const std::map<std::string, AttrValue> attributes = {
        {"mask_mode", AttrValue{std::string("none")}},
        {"dropout_probability", AttrValue{0.0}},
        {"heads", AttrValue{std::int64_t{4}}},
    };

    // equals
    {
        AttributeConstraints c;
        AttributeRule rule;
        rule.equals = AttrValue{std::string("none")};
        c.emplace("mask_mode", rule);
        EXPECT_TRUE(attributesMatchConstraints(attributes, c));
        c["mask_mode"].equals = AttrValue{std::string("causal_top_left")};
        EXPECT_FALSE(attributesMatchConstraints(attributes, c));
    }

    // not_equals
    {
        AttributeConstraints c;
        AttributeRule rule;
        rule.notEquals = AttrValue{std::string("causal_top_left")};
        c.emplace("mask_mode", rule);
        EXPECT_TRUE(attributesMatchConstraints(attributes, c));
        c["mask_mode"].notEquals = AttrValue{std::string("none")};
        EXPECT_FALSE(attributesMatchConstraints(attributes, c));
    }

    // one_of
    {
        AttributeConstraints c;
        AttributeRule rule;
        rule.oneOf = std::vector<AttrValue>{AttrValue{std::int64_t{2}}, AttrValue{std::int64_t{4}}};
        c.emplace("heads", rule);
        EXPECT_TRUE(attributesMatchConstraints(attributes, c));
        c["heads"].oneOf
            = std::vector<AttrValue>{AttrValue{std::int64_t{2}}, AttrValue{std::int64_t{8}}};
        EXPECT_FALSE(attributesMatchConstraints(attributes, c));
    }

    // a constrained attribute the problem does not expose never matches
    {
        AttributeConstraints c;
        AttributeRule rule;
        rule.equals = AttrValue{false};
        c.emplace("absent_attribute", rule);
        EXPECT_FALSE(attributesMatchConstraints(attributes, c));
    }
}

TEST(TestSelectionConstraints, EmptyRuleNeverMatches)
{
    const std::map<std::string, AttrValue> attributes = {
        {"mask_mode", AttrValue{std::string("none")}},
    };

    // A rule with no operator set is malformed (the Python producer rejects it at
    // parse time). Selection treats it as unsatisfiable so a malformed instance is
    // never dispatched, even when the attribute is present.
    AttributeConstraints c;
    c.emplace("mask_mode", AttributeRule{});
    EXPECT_FALSE(attributesMatchConstraints(attributes, c));
}

TEST(TestSelectionConstraints, AttributeRuleCombinesOperatorsWithAnd)
{
    const std::map<std::string, AttrValue> attributes = {
        {"mask_mode", AttrValue{std::string("none")}},
    };

    // one_of AND not_equals on the same attribute: both must hold.
    AttributeConstraints constraints;
    AttributeRule rule;
    rule.oneOf = std::vector<AttrValue>{AttrValue{std::string("none")},
                                        AttrValue{std::string("causal_top_left")}};
    rule.notEquals = AttrValue{std::string("causal_top_left")};
    constraints.emplace("mask_mode", rule);
    EXPECT_TRUE(attributesMatchConstraints(attributes, constraints));

    // Tighten not_equals to exclude the actual value: the AND now fails even
    // though one_of still holds.
    constraints["mask_mode"].notEquals = AttrValue{std::string("none")};
    EXPECT_FALSE(attributesMatchConstraints(attributes, constraints));
}

} // namespace
} // namespace rocke_client::dispatcher
