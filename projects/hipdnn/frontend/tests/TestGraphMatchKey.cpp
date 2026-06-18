// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// GPU-free unit tests for the autotune config-file match-key selector. These
// tests build real frontend graphs through SelectorUnitGraph, then assert the
// operation string, criteria, and canonical input tensor order the writer emits.

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/detail/AutotuneConfigNames.hpp>
#include <hipdnn_frontend/detail/GraphMatchKey.hpp>
#include <hipdnn_test_sdk/utilities/SelectorUnitGraph.hpp>

#include <string_view>

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
