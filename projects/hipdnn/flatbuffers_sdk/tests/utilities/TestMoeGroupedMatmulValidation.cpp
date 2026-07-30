// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/utilities/MoeGroupedMatmulValidation.hpp>

using namespace hipdnn_flatbuffers_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::data_objects;

namespace
{

// A routing configuration that satisfies the contract for NONE mode, used as the
// baseline that individual test cases perturb.
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

} // namespace

TEST(TestMoeGroupedMatmulValidation, AcceptsValidNoneMode)
{
    EXPECT_EQ(checkMoeGroupedMatmulRouting(validNone()), nullptr);
}

TEST(TestMoeGroupedMatmulValidation, AcceptsValidGatherMode)
{
    EXPECT_EQ(checkMoeGroupedMatmulRouting(validGather()), nullptr);
}

TEST(TestMoeGroupedMatmulValidation, AcceptsValidScatterMode)
{
    EXPECT_EQ(checkMoeGroupedMatmulRouting(validScatter()), nullptr);
}

TEST(TestMoeGroupedMatmulValidation, Rule1FirstTokenOffsetMustBeInt32)
{
    auto routing = validNone();
    routing.firstTokenOffsetDataType = DataType::FLOAT;
    EXPECT_STREQ(checkMoeGroupedMatmulRouting(routing),
                 "FIRST_TOKEN_OFFSET tensor must have INT32 data type");
}

TEST(TestMoeGroupedMatmulValidation, Rule2NoneForbidsTokenIndex)
{
    auto routing = validNone();
    routing.hasTokenIndex = true;
    routing.tokenIndexDataType = DataType::INT32;
    EXPECT_STREQ(checkMoeGroupedMatmulRouting(routing), "NONE mode forbids the TOKEN_INDEX tensor");
}

TEST(TestMoeGroupedMatmulValidation, Rule3NoneForbidsTokenKs)
{
    auto routing = validNone();
    routing.hasTokenKs = true;
    routing.tokenKsDataType = DataType::INT32;
    EXPECT_STREQ(checkMoeGroupedMatmulRouting(routing), "NONE mode forbids the TOKEN_KS tensor");
}

TEST(TestMoeGroupedMatmulValidation, Rule4NoneRequiresZeroTopK)
{
    auto routing = validNone();
    routing.topK = 1;
    EXPECT_STREQ(checkMoeGroupedMatmulRouting(routing), "NONE mode requires top_k to equal 0");
}

TEST(TestMoeGroupedMatmulValidation, Rule5GatherRequiresTokenIndex)
{
    auto routing = validGather();
    routing.hasTokenIndex = false;
    routing.tokenIndexDataType = DataType::UNSET;
    EXPECT_STREQ(checkMoeGroupedMatmulRouting(routing),
                 "GATHER mode requires the TOKEN_INDEX tensor");
}

TEST(TestMoeGroupedMatmulValidation, Rule6GatherForbidsTokenKs)
{
    auto routing = validGather();
    routing.hasTokenKs = true;
    routing.tokenKsDataType = DataType::INT32;
    EXPECT_STREQ(checkMoeGroupedMatmulRouting(routing), "GATHER mode forbids the TOKEN_KS tensor");
}

TEST(TestMoeGroupedMatmulValidation, Rule7GatherRequiresZeroTopK)
{
    auto routing = validGather();
    routing.topK = 1;
    EXPECT_STREQ(checkMoeGroupedMatmulRouting(routing), "GATHER mode requires top_k to equal 0");
}

TEST(TestMoeGroupedMatmulValidation, Rule8ScatterRequiresTokenIndex)
{
    auto routing = validScatter();
    routing.hasTokenIndex = false;
    routing.tokenIndexDataType = DataType::UNSET;
    EXPECT_STREQ(checkMoeGroupedMatmulRouting(routing),
                 "SCATTER mode requires the TOKEN_INDEX tensor");
}

TEST(TestMoeGroupedMatmulValidation, Rule9ScatterRequiresTokenKs)
{
    auto routing = validScatter();
    routing.hasTokenKs = false;
    routing.tokenKsDataType = DataType::UNSET;
    EXPECT_STREQ(checkMoeGroupedMatmulRouting(routing),
                 "SCATTER mode requires the TOKEN_KS tensor");
}

TEST(TestMoeGroupedMatmulValidation, Rule10ScatterRequiresPositiveTopK)
{
    auto routing = validScatter();
    routing.topK = 0;
    EXPECT_STREQ(checkMoeGroupedMatmulRouting(routing),
                 "SCATTER mode requires top_k to be at least 1");
}

TEST(TestMoeGroupedMatmulValidation, Rule11ScatterRequiresPositiveExpertCount)
{
    auto routing = validScatter();
    routing.expertCount = 0;
    EXPECT_STREQ(checkMoeGroupedMatmulRouting(routing),
                 "expert count must be positive to bound top_k");
}

TEST(TestMoeGroupedMatmulValidation, Rule12ScatterTopKMustNotExceedExpertCount)
{
    auto routing = validScatter();
    routing.expertCount = 1;
    routing.topK = 2;
    EXPECT_STREQ(checkMoeGroupedMatmulRouting(routing),
                 "top_k must not exceed the number of experts");
}

TEST(TestMoeGroupedMatmulValidation, Rule13TokenIndexMustBeInt32)
{
    auto routing = validGather();
    routing.tokenIndexDataType = DataType::FLOAT;
    EXPECT_STREQ(checkMoeGroupedMatmulRouting(routing),
                 "TOKEN_INDEX tensor must have INT32 data type");
}

TEST(TestMoeGroupedMatmulValidation, Rule14TokenKsMustBeInt32)
{
    auto routing = validScatter();
    routing.tokenKsDataType = DataType::FLOAT;
    EXPECT_STREQ(checkMoeGroupedMatmulRouting(routing),
                 "TOKEN_KS tensor must have INT32 data type");
}

TEST(TestMoeGroupedMatmulValidation, Rule15UnknownModeIsRejected)
{
    auto routing = validNone();
    routing.mode = static_cast<MoeGroupedMatmulMode>(-1);
    EXPECT_STREQ(checkMoeGroupedMatmulRouting(routing), "unknown routing mode");
}
