// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <memory>
#include <unordered_map>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/utilities/IntegrationTestFixture.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_data_sdk::utilities;

namespace
{

class IntegrationMoeGroupedMatmul : public hipdnn_tests::IntegrationTestFixture
{
};

TEST_F(IntegrationMoeGroupedMatmul, ScatterGraphDispatchesToProvider)
{
    Tensor<float> tokenTensor({1, 8, 16});
    Tensor<float> weightTensor({2, 16, 32});
    Tensor<int32_t> firstTokenOffsetTensor({2, 1, 1});
    Tensor<int32_t> tokenIndexTensor({1, 8, 1});
    Tensor<int32_t> tokenKsTensor({1, 8, 1});
    Tensor<float> outputTensor({1, 8, 32});

    tokenTensor.fillWithValue(1.0F);
    weightTensor.fillWithValue(1.0F);
    firstTokenOffsetTensor.fillWithValue(0);
    tokenIndexTensor.fillWithValue(0);
    tokenKsTensor.fillWithValue(0);
    outputTensor.fillWithValue(0.0F);

    auto graph = std::make_shared<Graph>();
    graph->set_name("MoeGroupedMatmulDispatch")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto token = std::make_shared<TensorAttributes>(
        makeTensorAttributes("token", getDataTypeEnumFromType<float>(), tokenTensor));
    token->set_uid(1);
    auto weight = std::make_shared<TensorAttributes>(
        makeTensorAttributes("weight", getDataTypeEnumFromType<float>(), weightTensor));
    weight->set_uid(2);
    auto firstTokenOffset = std::make_shared<TensorAttributes>(makeTensorAttributes(
        "first_token_offset", getDataTypeEnumFromType<int32_t>(), firstTokenOffsetTensor));
    firstTokenOffset->set_uid(3);
    auto tokenIndex = std::make_shared<TensorAttributes>(
        makeTensorAttributes("token_index", getDataTypeEnumFromType<int32_t>(), tokenIndexTensor));
    tokenIndex->set_uid(4);
    auto tokenKs = std::make_shared<TensorAttributes>(
        makeTensorAttributes("token_ks", getDataTypeEnumFromType<int32_t>(), tokenKsTensor));
    tokenKs->set_uid(5);

    MoeGroupedMatmulAttributes attributes;
    attributes.set_name("moe_grouped_matmul").set_mode(MoeGroupedMatmulMode::SCATTER).set_top_k(2);
    auto output = graph->moe_grouped_matmul(
        token, weight, firstTokenOffset, tokenIndex, tokenKs, attributes);
    output->set_uid(6).set_output(true).set_name("output");

    auto result = graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph->create_execution_plans();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph->check_support();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph->build_plans();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    int64_t workspaceSize = 0;
    result = graph->get_workspace_size(workspaceSize);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    const Workspace workspace(static_cast<size_t>(workspaceSize));

    std::unordered_map<int64_t, void*> variantPack = {
        {token->get_uid(), tokenTensor.memory().deviceData()},
        {weight->get_uid(), weightTensor.memory().deviceData()},
        {firstTokenOffset->get_uid(), firstTokenOffsetTensor.memory().deviceData()},
        {tokenIndex->get_uid(), tokenIndexTensor.memory().deviceData()},
        {tokenKs->get_uid(), tokenKsTensor.memory().deviceData()},
        {output->get_uid(), outputTensor.memory().deviceData()},
    };
    result = graph->execute(_handle, variantPack, workspace.get());
    EXPECT_EQ(result.code, ErrorCode::OK) << result.err_msg;
}

} // namespace
