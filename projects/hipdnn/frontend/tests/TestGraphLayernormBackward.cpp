// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/attributes/LayernormAttributes.hpp>
#include <hipdnn_frontend/attributes/LayernormBackwardAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

#include <memory>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

namespace
{

std::shared_ptr<TensorAttributes> makeFloatTensor(const std::vector<int64_t>& dims,
                                                  const std::vector<int64_t>& strides)
{
    auto tensor = std::make_shared<TensorAttributes>();
    tensor->set_dim(dims).set_stride(strides).set_data_type(DataType::FLOAT);
    return tensor;
}

// Adds a layernorm_backward node whose dy and x have xDims, scale has scaleDims and saved
// mean and inv_variance have statsDims; dx, dscale and dbias are graph outputs.
void addLayernormBackward(Graph& graph,
                          const std::vector<int64_t>& xDims,
                          const std::vector<int64_t>& xStrides,
                          const std::vector<int64_t>& scaleDims,
                          const std::vector<int64_t>& scaleStrides,
                          const std::vector<int64_t>& statsDims)
{
    graph.set_compute_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    const std::vector<int64_t> statsStrides(statsDims.size(), 1);
    LayernormBackwardAttributes attributes;
    attributes.set_name("LayernormBackwardNode");
    attributes.set_saved_mean_and_inv_variance(makeFloatTensor(statsDims, statsStrides),
                                               makeFloatTensor(statsDims, statsStrides));
    attributes.set_epsilon(makeFloatTensor({1}, {1}));

    auto [dx, dscale, dbias] = graph.layernorm_backward(makeFloatTensor(xDims, xStrides),
                                                        makeFloatTensor(xDims, xStrides),
                                                        makeFloatTensor(scaleDims, scaleStrides),
                                                        attributes);
    dx->set_output(true).set_dim(xDims).set_stride(xStrides);
    dscale->set_output(true).set_dim(scaleDims).set_stride(scaleStrides);
    dbias->set_output(true).set_dim(scaleDims).set_stride(scaleStrides);
}

} // namespace

TEST(TestGraphLayernormBackward, BuildGraph)
{
    Graph graph;
    graph.set_compute_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    // Create attributes
    LayernormBackwardAttributes attributes;
    attributes.set_name("LayernormBackwardNode");

    // Create input tensors
    const auto dy = std::make_shared<TensorAttributes>();
    dy->set_dim({16, 64, 32, 32}).set_stride({65536, 1024, 32, 1}).set_data_type(DataType::FLOAT);

    const auto x = std::make_shared<TensorAttributes>();
    x->set_dim({16, 64, 32, 32}).set_stride({65536, 1024, 32, 1}).set_data_type(DataType::FLOAT);

    const auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 64, 32, 32}).set_stride({65536, 1024, 32, 1}).set_data_type(DataType::FLOAT);

    auto mean = std::make_shared<TensorAttributes>();
    mean->set_dim({16, 1, 1, 1}).set_stride({1, 1, 1, 1}).set_data_type(DataType::FLOAT);
    attributes.set_mean(std::move(mean));

    auto invVariance = std::make_shared<TensorAttributes>();
    invVariance->set_dim({16, 1, 1, 1}).set_stride({1, 1, 1, 1}).set_data_type(DataType::FLOAT);
    attributes.set_inv_variance(std::move(invVariance));

    auto epsilon = std::make_shared<TensorAttributes>();
    epsilon->set_dim({1}).set_stride({1}).set_data_type(DataType::FLOAT);
    attributes.set_epsilon(std::move(epsilon));

    // Call graph method
    auto results = graph.layernorm_backward(dy, x, scale, attributes);

    // Verify returned dx tensor is non-null
    ASSERT_NE(results[0], nullptr);
    EXPECT_EQ(results[0]->get_name(), "LayernormBackwardNode::DX");
    EXPECT_TRUE(results[0]->get_is_virtual());

    // Verify returned dscale tensor is non-null
    ASSERT_NE(results[1], nullptr);
    EXPECT_EQ(results[1]->get_name(), "LayernormBackwardNode::DSCALE");
    EXPECT_TRUE(results[1]->get_is_virtual());

    // Verify returned dbias tensor is non-null
    ASSERT_NE(results[2], nullptr);
    EXPECT_EQ(results[2]->get_name(), "LayernormBackwardNode::DBIAS");
    EXPECT_TRUE(results[2]->get_is_virtual());

    // Verify graph validates successfully
    const auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();
}

TEST(TestGraphLayernormBackward, ValidateRejectsStatsBatchMismatchOnFirstValidate)
{
    Graph graph;
    addLayernormBackward(
        graph, {4, 6, 8, 8}, {384, 64, 8, 1}, {1, 6, 8, 8}, {384, 64, 8, 1}, {3, 1, 1, 1});

    auto validationResult = graph.validate();
    EXPECT_EQ(validationResult.code, error_code_t::INVALID_VALUE) << validationResult.get_message();
}

TEST(TestGraphLayernormBackward, RevalidateAcceptsUnitDimAfterFirstNormalizedDim)
{
    Graph graph;
    addLayernormBackward(graph,
                         {2, 3, 1, 129},
                         {387, 129, 129, 1},
                         {1, 3, 1, 129},
                         {387, 129, 129, 1},
                         {2, 1, 1, 1});

    auto validationResult = graph.validate();
    ASSERT_TRUE(validationResult.is_good()) << validationResult.get_message();

    // build_operation_graph validates again with the normalized dim count inferred above
    validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();
}

TEST(TestGraphLayernormBackward, ValidateAcceptsForwardStatsWithRankReducedScale)
{
    // x [B, S, D] with scale [D]: the forward node infers mean and inv_variance [B, S, 1]
    Graph graph;
    graph.set_compute_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    auto x = makeFloatTensor({2, 4, 8}, {32, 8, 1});
    auto scale = makeFloatTensor({8}, {1});
    auto epsilon = std::make_shared<TensorAttributes>();
    epsilon->set_value(1e-5f);

    LayernormAttributes forwardAttributes;
    forwardAttributes.set_name("LayernormForward");
    forwardAttributes.set_forward_phase(NormFwdPhase::TRAINING);
    forwardAttributes.set_epsilon(epsilon);
    auto [y, mean, invVariance]
        = graph.layernorm(x, scale, makeFloatTensor({8}, {1}), forwardAttributes);
    y->set_output(true);

    LayernormBackwardAttributes backwardAttributes;
    backwardAttributes.set_name("LayernormBackward");
    backwardAttributes.set_saved_mean_and_inv_variance(mean, invVariance);
    backwardAttributes.set_epsilon(makeFloatTensor({1}, {1}));
    auto [dx, dscale, dbias] = graph.layernorm_backward(
        makeFloatTensor({2, 4, 8}, {32, 8, 1}), x, scale, backwardAttributes);
    dx->set_output(true);
    dscale->set_output(true);
    dbias->set_output(true);

    auto validationResult = graph.validate();
    ASSERT_TRUE(validationResult.is_good()) << validationResult.get_message();
    EXPECT_EQ(mean->get_dim(), (std::vector<int64_t>{2, 4, 1}));
}
