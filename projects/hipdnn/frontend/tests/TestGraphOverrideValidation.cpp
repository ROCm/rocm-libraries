// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_frontend/detail/GraphOverrideValidation.hpp>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

namespace
{

std::shared_ptr<TensorAttributes> makeTensor(const std::vector<int64_t>& dims,
                                             const std::vector<int64_t>& strides)
{
    auto tensor = std::make_shared<TensorAttributes>();
    tensor->set_uid(1).set_dim(dims).set_stride(strides).set_data_type(DataType::FLOAT);
    return tensor;
}

std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>>
    makeTensorMap(const std::vector<int64_t>& dims = {1, 3, 4, 4},
                  const std::vector<int64_t>& strides = {48, 16, 4, 1})
{
    return {{1, makeTensor(dims, strides)}};
}

struct PlanOnlyRejectCase
{
    const char* name;
    std::vector<int64_t> uids;
    std::vector<std::vector<int64_t>> shapes;
    std::vector<std::vector<int64_t>> strides;
    const char* expectedMessage;
};

class TestPlanOnlyOverrideValidation : public ::testing::TestWithParam<PlanOnlyRejectCase>
{
};

struct GraphBackedRejectCase
{
    const char* name;
    std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>> tensorsByUid;
    std::vector<int64_t> uids;
    std::vector<std::vector<int64_t>> shapes;
    std::vector<std::vector<int64_t>> strides;
    const char* expectedMessage;
};

class TestGraphBackedOverrideValidation : public ::testing::TestWithParam<GraphBackedRejectCase>
{
};

} // namespace

TEST(TestPlanOnlyOverrideValidation, AcceptsStructurallyValidOverrides)
{
    auto result = detail::validatePlanOnlyOverrideArguments(
        {1, 2}, {{1, 3, 4, 4}, {1, 3, 2, 2}}, {{48, 16, 4, 1}, {12, 4, 2, 1}});

    EXPECT_TRUE(result.is_good()) << result.err_msg;
}

TEST_P(TestPlanOnlyOverrideValidation, RejectsInvalidOverrides)
{
    const auto& testCase = GetParam();
    auto result = detail::validatePlanOnlyOverrideArguments(
        testCase.uids, testCase.shapes, testCase.strides);

    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE);
    EXPECT_NE(result.err_msg.find(testCase.expectedMessage), std::string::npos) << result.err_msg;
}

INSTANTIATE_TEST_SUITE_P(
    InvalidPlanOnlyOverrides,
    TestPlanOnlyOverrideValidation,
    ::testing::Values(
        PlanOnlyRejectCase{"ArraySizeMismatch",
                           {1, 2},
                           {{1, 3, 4, 4}},
                           {{48, 16, 4, 1}, {48, 16, 4, 1}},
                           "inconsistent sizes"},
        PlanOnlyRejectCase{
            "ShapeStrideRankMismatch", {1}, {{1, 3, 4}}, {{48, 16, 4, 1}}, "rank mismatch"},
        PlanOnlyRejectCase{"ZeroRankOverride", {1}, {{}}, {{}}, "must be non-zero"},
        PlanOnlyRejectCase{
            "NonPositiveShape", {1}, {{1, 3, 0, 4}}, {{48, 16, 4, 1}}, "must be positive"},
        PlanOnlyRejectCase{
            "NonPositiveStride", {1}, {{1, 3, 4, 4}}, {{48, 16, 0, 1}}, "must be positive"},
        PlanOnlyRejectCase{"DuplicateUid",
                           {1, 1},
                           {{1, 3, 4, 4}, {1, 3, 4, 4}},
                           {{48, 16, 4, 1}, {48, 16, 4, 1}},
                           "Duplicate UID"}),
    [](const auto& info) { return std::string(info.param.name); });

TEST(TestGraphBackedOverrideValidation, AcceptsValidOverrides)
{
    auto result = detail::validateGraphBackedOverrideArguments(
        makeTensorMap(), {1}, {{1, 3, 4, 4}}, {{48, 16, 4, 1}});

    EXPECT_TRUE(result.is_good()) << result.err_msg;
}

TEST_P(TestGraphBackedOverrideValidation, RejectsInvalidOverrides)
{
    const auto& testCase = GetParam();
    auto result = detail::validateGraphBackedOverrideArguments(
        testCase.tensorsByUid, testCase.uids, testCase.shapes, testCase.strides);

    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE);
    EXPECT_NE(result.err_msg.find(testCase.expectedMessage), std::string::npos) << result.err_msg;
}

INSTANTIATE_TEST_SUITE_P(
    InvalidGraphBackedOverrides,
    TestGraphBackedOverrideValidation,
    ::testing::Values(
        GraphBackedRejectCase{"ArraySizeMismatch",
                              makeTensorMap(),
                              {1, 2},
                              {{1, 3, 4, 4}},
                              {{48, 16, 4, 1}, {48, 16, 4, 1}},
                              "rule 1"},
        GraphBackedRejectCase{"DuplicateUid",
                              makeTensorMap(),
                              {1, 1},
                              {{1, 3, 4, 4}, {1, 3, 4, 4}},
                              {{48, 16, 4, 1}, {48, 16, 4, 1}},
                              "rule 5"},
        GraphBackedRejectCase{
            "UnknownUid", makeTensorMap(), {99}, {{1, 3, 4, 4}}, {{48, 16, 4, 1}}, "rule 2"},
        GraphBackedRejectCase{
            "RankMismatch", makeTensorMap(), {1}, {{1, 3, 4}}, {{16, 4, 1}}, "rule 3"},
        GraphBackedRejectCase{
            "NonPositiveShape", makeTensorMap(), {1}, {{1, 3, 0, 4}}, {{48, 16, 4, 1}}, "rule 6"},
        GraphBackedRejectCase{
            "NonPositiveStride", makeTensorMap(), {1}, {{1, 3, 4, 4}}, {{48, 16, 0, 1}}, "rule 7"},
        GraphBackedRejectCase{"ShapeExceedsDeclared",
                              makeTensorMap(),
                              {1},
                              {{1, 3, 8, 4}},
                              {{96, 32, 4, 1}},
                              "rule 4"},
        GraphBackedRejectCase{"DeclaredStrideRankMismatch",
                              makeTensorMap({1, 3, 4, 4}, {16, 4, 1}),
                              {1},
                              {{1, 3, 4, 4}},
                              {{48, 16, 4, 1}},
                              "rule 8"},
        GraphBackedRejectCase{"StrideOrderingMismatch",
                              makeTensorMap(),
                              {1},
                              {{1, 3, 4, 4}},
                              {{1, 4, 16, 48}},
                              "rule 8"}),
    [](const auto& info) { return std::string(info.param.name); });
