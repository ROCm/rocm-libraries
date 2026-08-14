// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <cstdint>
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

class IntegrationMoeGroupedMatmulBwd : public hipdnn_tests::IntegrationTestFixture
{
};

TEST_F(IntegrationMoeGroupedMatmulBwd, GraphDispatchesToProvider)
{
    constexpr int64_t K_BATCH = 1;
    constexpr int64_t K_TOKENS = 8;
    constexpr int64_t K_DIM_K = 16;
    constexpr int64_t K_DIM_N = 32;
    constexpr int64_t K_EXPERTS = 2;

    Tensor<float> doutputTensor({K_BATCH, K_TOKENS, K_DIM_N});
    Tensor<float> tokenTensor({K_BATCH, K_TOKENS, K_DIM_K});
    Tensor<int32_t> firstTokenOffsetTensor({K_EXPERTS, 1, 1});
    Tensor<float> dweightTensor({K_EXPERTS, K_DIM_K, K_DIM_N});

    doutputTensor.fillWithValue(1.0F);
    tokenTensor.fillWithValue(1.0F);
    firstTokenOffsetTensor.fillWithValue(0);
    dweightTensor.fillWithValue(0.0F);

    auto graph = std::make_shared<Graph>();
    graph->set_name("MoeGroupedMatmulBwdDispatch")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto doutput = std::make_shared<TensorAttributes>(
        makeTensorAttributes("doutput", getDataTypeEnumFromType<float>(), doutputTensor));
    doutput->set_uid(1);
    auto token = std::make_shared<TensorAttributes>(
        makeTensorAttributes("token", getDataTypeEnumFromType<float>(), tokenTensor));
    token->set_uid(2);
    auto firstTokenOffset = std::make_shared<TensorAttributes>(makeTensorAttributes(
        "first_token_offset", getDataTypeEnumFromType<int32_t>(), firstTokenOffsetTensor));
    firstTokenOffset->set_uid(3);

    MoeGroupedMatmulBwdAttributes attributes;
    attributes.set_name("moe_grouped_matmul_bwd");
    auto dweight = graph->moe_grouped_matmul_bwd(doutput, token, firstTokenOffset, attributes);
    dweight->set_uid(4).set_output(true).set_name("dweight");
    dweight->set_dim({K_EXPERTS, K_DIM_K, K_DIM_N});

    auto result = graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    EXPECT_EQ(dweight->get_dim(), (std::vector<int64_t>{K_EXPERTS, K_DIM_K, K_DIM_N}));

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
        {doutput->get_uid(), doutputTensor.memory().deviceData()},
        {token->get_uid(), tokenTensor.memory().deviceData()},
        {firstTokenOffset->get_uid(), firstTokenOffsetTensor.memory().deviceData()},
        {dweight->get_uid(), dweightTensor.memory().deviceData()},
    };
    result = graph->execute(_handle, variantPack, workspace.get());
    EXPECT_EQ(result.code, ErrorCode::OK) << result.err_msg;
}

} // namespace
