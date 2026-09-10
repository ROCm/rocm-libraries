// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include <hipdnn_flatbuffers_sdk/utilities/MoeGroupedMatmulValidation.hpp>

using namespace hipdnn_flatbuffers_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::data_objects;

namespace
{

// Routing configurations that satisfy the contract, used as the baselines that
// individual cases perturb.
MoeGroupedMatmulRouting validNone()
{
    MoeGroupedMatmulRouting routing;
    routing.mode = MoeGroupedMatmulMode::NONE;
    routing.hasTokenIndex = false;
    routing.hasTokenKs = false;
    routing.firstTokenOffsetDataType = DataType::INT32;
    routing.tokenIndexDataType = DataType::UNSET;
    routing.tokenKsDataType = DataType::UNSET;
    routing.topK = 0;
    routing.expertCount = 4;
    return routing;
}

MoeGroupedMatmulRouting validGather()
{
    MoeGroupedMatmulRouting routing = validNone();
    routing.mode = MoeGroupedMatmulMode::GATHER;
    routing.hasTokenIndex = true;
    routing.tokenIndexDataType = DataType::INT32;
    return routing;
}

MoeGroupedMatmulRouting validScatter()
{
    MoeGroupedMatmulRouting routing = validNone();
    routing.mode = MoeGroupedMatmulMode::SCATTER;
    routing.hasTokenIndex = true;
    routing.hasTokenKs = true;
    routing.tokenIndexDataType = DataType::INT32;
    routing.tokenKsDataType = DataType::INT32;
    routing.topK = 2;
    return routing;
}

struct RoutingCase
{
    const char* name;
    MoeGroupedMatmulRouting routing;
    const char* expectedReason; ///< nullptr when the configuration must be accepted
};

template <typename Perturbation>
MoeGroupedMatmulRouting with(MoeGroupedMatmulRouting routing, Perturbation perturbation)
{
    perturbation(routing);
    return routing;
}

std::vector<RoutingCase> getRoutingCases()
{
    return {
        RoutingCase{"AcceptsValidNoneMode", validNone(), nullptr},
        RoutingCase{"AcceptsValidGatherMode", validGather(), nullptr},
        RoutingCase{"AcceptsValidScatterMode", validScatter(), nullptr},
        RoutingCase{
            "Rule1FirstTokenOffsetMustBeInt32",
            with(validNone(), [](auto& r) { r.firstTokenOffsetDataType = DataType::FLOAT; }),
            "FIRST_TOKEN_OFFSET tensor must have INT32 data type"},
        RoutingCase{"Rule2NoneForbidsTokenIndex",
                    with(validNone(),
                         [](auto& r) {
                             r.hasTokenIndex = true;
                             r.tokenIndexDataType = DataType::INT32;
                         }),
                    "NONE mode forbids the TOKEN_INDEX tensor"},
        RoutingCase{"Rule3NoneForbidsTokenKs",
                    with(validNone(),
                         [](auto& r) {
                             r.hasTokenKs = true;
                             r.tokenKsDataType = DataType::INT32;
                         }),
                    "NONE mode forbids the TOKEN_KS tensor"},
        RoutingCase{"Rule4NoneRequiresZeroTopK",
                    with(validNone(), [](auto& r) { r.topK = 1; }),
                    "NONE mode requires top_k to equal 0"},
        RoutingCase{"Rule5GatherRequiresTokenIndex",
                    with(validGather(),
                         [](auto& r) {
                             r.hasTokenIndex = false;
                             r.tokenIndexDataType = DataType::UNSET;
                         }),
                    "GATHER mode requires the TOKEN_INDEX tensor"},
        RoutingCase{"Rule6GatherForbidsTokenKs",
                    with(validGather(),
                         [](auto& r) {
                             r.hasTokenKs = true;
                             r.tokenKsDataType = DataType::INT32;
                         }),
                    "GATHER mode forbids the TOKEN_KS tensor"},
        RoutingCase{"Rule7GatherRequiresZeroTopK",
                    with(validGather(), [](auto& r) { r.topK = 1; }),
                    "GATHER mode requires top_k to equal 0"},
        RoutingCase{"Rule8ScatterRequiresTokenIndex",
                    with(validScatter(),
                         [](auto& r) {
                             r.hasTokenIndex = false;
                             r.tokenIndexDataType = DataType::UNSET;
                         }),
                    "SCATTER mode requires the TOKEN_INDEX tensor"},
        RoutingCase{"Rule9ScatterRequiresTokenKs",
                    with(validScatter(),
                         [](auto& r) {
                             r.hasTokenKs = false;
                             r.tokenKsDataType = DataType::UNSET;
                         }),
                    "SCATTER mode requires the TOKEN_KS tensor"},
        RoutingCase{"Rule10ScatterRequiresPositiveTopK",
                    with(validScatter(), [](auto& r) { r.topK = 0; }),
                    "SCATTER mode requires top_k to be at least 1"},
        RoutingCase{"Rule11ScatterRequiresPositiveExpertCount",
                    with(validScatter(), [](auto& r) { r.expertCount = 0; }),
                    "expert count must be positive to bound top_k"},
        RoutingCase{"Rule12ScatterTopKMustNotExceedExpertCount",
                    with(validScatter(),
                         [](auto& r) {
                             r.expertCount = 1;
                             r.topK = 2;
                         }),
                    "top_k must not exceed the number of experts"},
        RoutingCase{"Rule13TokenIndexMustBeInt32",
                    with(validGather(), [](auto& r) { r.tokenIndexDataType = DataType::FLOAT; }),
                    "TOKEN_INDEX tensor must have INT32 data type"},
        RoutingCase{"Rule14TokenKsMustBeInt32",
                    with(validScatter(), [](auto& r) { r.tokenKsDataType = DataType::FLOAT; }),
                    "TOKEN_KS tensor must have INT32 data type"},
        RoutingCase{
            "Rule15UnknownModeIsRejected",
            with(validNone(), [](auto& r) { r.mode = static_cast<MoeGroupedMatmulMode>(-1); }),
            "unknown routing mode"},
    };
}

} // namespace

class TestMoeGroupedMatmulValidation : public ::testing::TestWithParam<RoutingCase>
{
};

TEST_P(TestMoeGroupedMatmulValidation, MatchesRoutingContract)
{
    const auto& testCase = GetParam();
    const char* reason = checkMoeGroupedMatmulRouting(testCase.routing);

    if(testCase.expectedReason == nullptr)
    {
        EXPECT_EQ(reason, nullptr) << "unexpected rejection: " << (reason != nullptr ? reason : "");
    }
    else
    {
        ASSERT_NE(reason, nullptr) << "expected rejection: " << testCase.expectedReason;
        EXPECT_STREQ(reason, testCase.expectedReason);
    }
}

INSTANTIATE_TEST_SUITE_P(Contract,
                         TestMoeGroupedMatmulValidation,
                         testing::ValuesIn(getRoutingCases()),
                         [](const testing::TestParamInfo<RoutingCase>& info) {
                             return std::string(info.param.name);
                         });
