// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// GPU-free unit tests for the autotune config-file match-key selector. These
// tests build real frontend graphs through SelectorUnitGraph, then assert the
// operation string, criteria, and canonical input tensor order the writer emits.

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/detail/AutotuneConfigNames.hpp>
#include <hipdnn_frontend/detail/GraphMatchKey.hpp>
#include <hipdnn_test_sdk/utilities/SelectorUnitGraph.hpp>

#include <string>
#include <string_view>
#include <vector>

using namespace hipdnn_frontend;
using hipdnn_test_sdk::utilities::OperationType;
using hipdnn_test_sdk::utilities::SelectorUnitGraph;

namespace
{
namespace config_criterion = hipdnn_data_sdk::detail::autotune_config::criterion;
namespace config_op = hipdnn_data_sdk::detail::autotune_config::op;
namespace config_tensor = hipdnn_data_sdk::detail::autotune_config::tensor;

void expectTensorUidOrder(const detail::AutotuneConfigMatchKey& key,
                          const std::vector<std::shared_ptr<graph::TensorAttributes>>& tensors)
{
    ASSERT_EQ(key.tensors.size(), tensors.size());
    for(size_t i = 0; i < tensors.size(); ++i)
    {
        EXPECT_EQ(key.tensors[i]->get_uid(), tensors[i]->get_uid());
        EXPECT_EQ(key.tensors[i]->get_dim(), tensors[i]->get_dim());
        EXPECT_EQ(key.tensors[i]->get_stride(), tensors[i]->get_stride());
    }
}

void expectTensorIdOrder(const detail::AutotuneConfigMatchKey& key,
                         const std::vector<std::string_view>& tensorIds)
{
    ASSERT_EQ(key.tensors.size(), tensorIds.size());
    for(size_t i = 0; i < tensorIds.size(); ++i)
    {
        EXPECT_EQ(key.tensors[i].tensorId, tensorIds[i]);
    }
}

} // namespace

class TestGraphMatchKey : public ::testing::Test
{
};

struct GraphMatchKeyCase
{
    OperationType op;
    std::string_view expectedOpName;
    detail::AutotuneConfigCriteria expectedCriteria;
    std::vector<std::string> expectedTensorNames;
    std::vector<std::string_view> expectedTensorIds;
};

class TestGraphMatchKeyParameterized : public ::testing::TestWithParam<GraphMatchKeyCase>
{
};

TEST_P(TestGraphMatchKeyParameterized, SupportedOperationEmitsExpectedMatchKey)
{
    const auto& testCase = GetParam();
    SelectorUnitGraph unitGraph(testCase.op);

    const auto key = detail::getAutotuneConfigMatchKey(unitGraph.graph());
    ASSERT_TRUE(key.has_value());
    EXPECT_EQ(key->opName, testCase.expectedOpName);
    EXPECT_EQ(key->criteria, testCase.expectedCriteria);
    expectTensorIdOrder(*key, testCase.expectedTensorIds);
    ASSERT_EQ(key->tensors.size(), testCase.expectedTensorNames.size());
    for(size_t i = 0; i < testCase.expectedTensorNames.size(); ++i)
    {
        const auto& expectedTensor = unitGraph.byName(testCase.expectedTensorNames[i]);
        EXPECT_EQ(key->tensors[i]->get_uid(), expectedTensor->get_uid());
        EXPECT_EQ(key->tensors[i]->get_dim(), expectedTensor->get_dim());
        EXPECT_EQ(key->tensors[i]->get_stride(), expectedTensor->get_stride());
        EXPECT_TRUE(key->tensors[i]->has_uid());
    }
}

static std::vector<GraphMatchKeyCase> graphMatchKeyCases()
{
    std::vector<GraphMatchKeyCase> cases{
        {OperationType::CONV_FORWARD,
         config_op::CONV_FPROP,
         {},
         {"x", "w"},
         {config_tensor::X, config_tensor::W}},
        {OperationType::CONV_BACKWARD_DATA,
         config_op::CONV_DGRAD,
         {},
         {"dy", "w"},
         {config_tensor::DY, config_tensor::W}},
        {OperationType::CONV_BACKWARD_WEIGHTS,
         config_op::CONV_WGRAD,
         {},
         {"x", "dy"},
         {config_tensor::X, config_tensor::DY}},
        {OperationType::CONV_FWD_BIAS_ACTIV,
         config_op::CONV_FPROP,
         {},
         {"x", "w"},
         {config_tensor::X, config_tensor::W}},
        {OperationType::MATMUL,
         config_op::MATMUL,
         {},
         {"A", "B"},
         {config_tensor::A, config_tensor::B}},
        {OperationType::BATCHNORM_TRAINING,
         config_op::BATCHNORM_TRAINING,
         {},
         {"x", "scale", "bias", "epsilon"},
         {config_tensor::X, config_tensor::SCALE, config_tensor::BIAS, config_tensor::EPSILON}},
        {OperationType::BATCHNORM_INFERENCE,
         config_op::BATCHNORM_INFERENCE,
         {},
         {"x", "mean", "invVariance", "scale", "bias"},
         {config_tensor::X,
          config_tensor::MEAN,
          config_tensor::INV_VARIANCE,
          config_tensor::SCALE,
          config_tensor::BIAS}},
        {OperationType::BATCHNORM_INFERENCE_VARIANCE_EXT,
         config_op::BATCHNORM_INFERENCE_VARIANCE_EXT,
         {},
         {"x", "mean", "variance", "scale", "bias", "epsilon"},
         {config_tensor::X,
          config_tensor::MEAN,
          config_tensor::VARIANCE,
          config_tensor::SCALE,
          config_tensor::BIAS,
          config_tensor::EPSILON}},
        {OperationType::BATCHNORM_BACKWARD,
         config_op::BATCHNORM_BACKWARD,
         {},
         {"dy", "x", "scale"},
         {config_tensor::DY, config_tensor::X, config_tensor::SCALE}},
        {OperationType::LAYERNORM,
         config_op::LAYERNORM,
         {{config_criterion::NORM_FWD_PHASE, HIPDNN_NORM_FWD_INFERENCE}},
         {"x", "scale", "bias", "epsilon"},
         {config_tensor::X, config_tensor::SCALE, config_tensor::BIAS, config_tensor::EPSILON}},
        {OperationType::RMSNORM,
         config_op::RMSNORM,
         {{config_criterion::NORM_FWD_PHASE, HIPDNN_NORM_FWD_INFERENCE}},
         {"x", "scale", "epsilon"},
         {config_tensor::X, config_tensor::SCALE, config_tensor::EPSILON}},
        {OperationType::RMSNORM_BACKWARD,
         config_op::RMSNORM_BACKWARD,
         {},
         {"dy", "x", "scale", "invRms"},
         {config_tensor::DY, config_tensor::X, config_tensor::SCALE, config_tensor::INV_RMS}},
        {OperationType::REDUCTION,
         config_op::REDUCTION,
         {{config_criterion::REDUCTION_MODE, HIPDNN_REDUCE_TENSOR_ADD}},
         {"x"},
         {config_tensor::INPUT}},
        {OperationType::RESAMPLE_FWD,
         config_op::RESAMPLE_FWD,
         {{config_criterion::RESAMPLE_MODE, HIPDNN_RESAMPLE_AVGPOOL_EXCLUDE_PADDING},
          {config_criterion::PADDING_MODE, HIPDNN_PADDING_ZERO_PAD}},
         {"x"},
         {config_tensor::X}},
        {OperationType::POINTWISE_UNARY,
         config_op::POINTWISE,
         {{config_criterion::POINTWISE_MODE, HIPDNN_POINTWISE_RELU_FWD}},
         {"x"},
         {config_tensor::IN_0}},
        {OperationType::POINTWISE_BINARY,
         config_op::POINTWISE,
         {{config_criterion::POINTWISE_MODE, HIPDNN_POINTWISE_ADD}},
         {"x", "y"},
         {config_tensor::IN_0, config_tensor::IN_1}},
    };

#ifdef HIPDNN_ENABLE_SDPA
    cases.push_back({OperationType::SDPA_FORWARD,
                     config_op::SDPA_FWD,
                     {},
                     {"Q", "K", "V"},
                     {config_tensor::Q, config_tensor::K, config_tensor::V}});
    cases.push_back({OperationType::SDPA_BACKWARD,
                     config_op::SDPA_BWD,
                     {},
                     {"Q", "K", "V", "O", "dO", "stats"},
                     {config_tensor::Q,
                      config_tensor::K,
                      config_tensor::V,
                      config_tensor::O,
                      config_tensor::DO,
                      config_tensor::STATS}});
#endif

    return cases;
}

INSTANTIATE_TEST_SUITE_P(SupportedOps,
                         TestGraphMatchKeyParameterized,
                         ::testing::ValuesIn(graphMatchKeyCases()),
                         [](const ::testing::TestParamInfo<GraphMatchKeyCase>& info) {
                             return hipdnn_test_sdk::utilities::operationTypeToString(
                                 info.param.op);
                         });

TEST_F(TestGraphMatchKey, ConvFpropOpStringAndTensorOrder)
{
    SelectorUnitGraph unitGraph(OperationType::CONV_FORWARD);

    const auto key = detail::getAutotuneConfigMatchKey(unitGraph.graph());
    ASSERT_TRUE(key.has_value());
    EXPECT_EQ(key->opName, config_op::CONV_FPROP);
    EXPECT_TRUE(key->criteria.empty());
    expectTensorUidOrder(*key, {unitGraph.byName("x"), unitGraph.byName("w")});
    expectTensorIdOrder(*key, {config_tensor::X, config_tensor::W});
}

TEST_F(TestGraphMatchKey, ConvDgradOpStringAndTensorOrder)
{
    SelectorUnitGraph unitGraph(OperationType::CONV_BACKWARD_DATA);

    const auto key = detail::getAutotuneConfigMatchKey(unitGraph.graph());
    ASSERT_TRUE(key.has_value());
    EXPECT_EQ(key->opName, config_op::CONV_DGRAD);
    EXPECT_TRUE(key->criteria.empty());
    expectTensorUidOrder(*key, {unitGraph.byName("dy"), unitGraph.byName("w")});
}

TEST_F(TestGraphMatchKey, ConvWgradOpStringAndTensorOrder)
{
    SelectorUnitGraph unitGraph(OperationType::CONV_BACKWARD_WEIGHTS);

    const auto key = detail::getAutotuneConfigMatchKey(unitGraph.graph());
    ASSERT_TRUE(key.has_value());
    EXPECT_EQ(key->opName, config_op::CONV_WGRAD);
    EXPECT_TRUE(key->criteria.empty());
    expectTensorUidOrder(*key, {unitGraph.byName("x"), unitGraph.byName("dy")});
}

TEST_F(TestGraphMatchKey, FusedConvBiasActivUsesConvPrimaryKey)
{
    SelectorUnitGraph unitGraph(OperationType::CONV_FWD_BIAS_ACTIV);

    const auto key = detail::getAutotuneConfigMatchKey(unitGraph.graph());
    ASSERT_TRUE(key.has_value());
    EXPECT_EQ(key->opName, config_op::CONV_FPROP);
    EXPECT_TRUE(key->criteria.empty());
    expectTensorUidOrder(*key, {unitGraph.byName("x"), unitGraph.byName("w")});
}

TEST_F(TestGraphMatchKey, ReductionIncludesReductionModeCriterion)
{
    SelectorUnitGraph unitGraph(OperationType::REDUCTION);

    const auto key = detail::getAutotuneConfigMatchKey(unitGraph.graph());
    ASSERT_TRUE(key.has_value());
    EXPECT_EQ(key->opName, config_op::REDUCTION);
    EXPECT_EQ(key->criteria,
              (detail::AutotuneConfigCriteria{
                  {config_criterion::REDUCTION_MODE, HIPDNN_REDUCE_TENSOR_ADD}}));
    expectTensorUidOrder(*key, {unitGraph.byName("x")});
}

TEST_F(TestGraphMatchKey, PointwiseUnaryIncludesPointwiseModeCriterion)
{
    SelectorUnitGraph unitGraph(OperationType::POINTWISE_UNARY);

    const auto key = detail::getAutotuneConfigMatchKey(unitGraph.graph());
    ASSERT_TRUE(key.has_value());
    EXPECT_EQ(key->opName, config_op::POINTWISE);
    EXPECT_EQ(key->criteria,
              (detail::AutotuneConfigCriteria{
                  {config_criterion::POINTWISE_MODE, HIPDNN_POINTWISE_RELU_FWD}}));
    expectTensorUidOrder(*key, {unitGraph.byName("x")});
}

TEST_F(TestGraphMatchKey, PointwiseBinaryIncludesPointwiseModeCriterion)
{
    SelectorUnitGraph unitGraph(OperationType::POINTWISE_BINARY);

    const auto key = detail::getAutotuneConfigMatchKey(unitGraph.graph());
    ASSERT_TRUE(key.has_value());
    EXPECT_EQ(key->opName, config_op::POINTWISE);
    EXPECT_EQ(
        key->criteria,
        (detail::AutotuneConfigCriteria{{config_criterion::POINTWISE_MODE, HIPDNN_POINTWISE_ADD}}));
    expectTensorUidOrder(*key, {unitGraph.byName("x"), unitGraph.byName("y")});
    expectTensorIdOrder(*key, {config_tensor::IN_0, config_tensor::IN_1});
}
