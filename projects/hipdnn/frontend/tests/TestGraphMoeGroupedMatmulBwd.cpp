// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/attributes/MoeGroupedMatmulBwdAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

#include <memory>
#include <string>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

TEST(TestGraphMoeGroupedMatmulBwd, BuildGraph)
{
    Graph graph;
    graph.set_compute_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    // Create input tensors
    auto dOutput = std::make_shared<TensorAttributes>();
    dOutput->set_dim({1, 8, 32}).set_stride({256, 32, 1}).set_data_type(DataType::FLOAT);

    auto token = std::make_shared<TensorAttributes>();
    token->set_dim({1, 8, 16}).set_stride({128, 16, 1}).set_data_type(DataType::FLOAT);

    auto firstTokenOffset = std::make_shared<TensorAttributes>();
    firstTokenOffset->set_dim({2, 1, 1}).set_stride({1, 1, 1}).set_data_type(DataType::INT32);

    // Create attributes
    MoeGroupedMatmulBwdAttributes attributes;
    attributes.set_name("MoeGroupedMatmulBwdNode");

    // Call graph method
    auto dweight = graph.moe_grouped_matmul_bwd(dOutput, token, firstTokenOffset, attributes);

    // Verify returned tensor is non-null
    ASSERT_NE(dweight, nullptr);
    EXPECT_EQ(dweight->get_name(), "MoeGroupedMatmulBwdNode::DWEIGHT");
    EXPECT_TRUE(dweight->get_is_virtual());

    // Verify graph validates successfully
    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();

    // dweight's dims are left unset above, so validate() infers them as
    // [first_token_offset.dim[0], token.dim[2], doutput.dim[2]].
    const std::vector<int64_t> expectedDweightDims{2, 16, 32};
    EXPECT_EQ(dweight->get_dim(), expectedDweightDims);
}

namespace
{

// Identifies which required input to null out, and the tensor name expected in the resulting
// pre_validate_node() error message.
struct NullInputCase
{
    MoeGroupedMatmulBwdAttributes::InputNames nullInput;
    const char* name;
    const char* expectedMessageFragment;
};

} // namespace

// Each required input is dereferenced by moe_grouped_matmul_bwd() while auto-naming tensors, so
// every one of them needs its own null guard; a null must reach pre_validate_node() and be
// reported rather than crashing the graph builder.
class TestGraphMoeGroupedMatmulBwdNullInput : public ::testing::TestWithParam<NullInputCase>
{
};

TEST_P(TestGraphMoeGroupedMatmulBwdNullInput, ReportsMissingInput)
{
    using InputNames = MoeGroupedMatmulBwdAttributes::InputNames;
    const auto param = GetParam();

    auto dOutput = std::make_shared<TensorAttributes>();
    dOutput->set_dim({1, 8, 32}).set_stride({256, 32, 1}).set_data_type(DataType::FLOAT);

    auto token = std::make_shared<TensorAttributes>();
    token->set_dim({1, 8, 16}).set_stride({128, 16, 1}).set_data_type(DataType::FLOAT);

    auto firstTokenOffset = std::make_shared<TensorAttributes>();
    firstTokenOffset->set_dim({2, 1, 1}).set_stride({1, 1, 1}).set_data_type(DataType::INT32);

    switch(param.nullInput)
    {
    case InputNames::DOUTPUT:
        dOutput.reset();
        break;
    case InputNames::TOKEN:
        token.reset();
        break;
    case InputNames::FIRST_TOKEN_OFFSET:
        firstTokenOffset.reset();
        break;
    default:
        FAIL() << "Unhandled InputNames enum value";
        break;
    }

    Graph graph;
    graph.set_compute_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    MoeGroupedMatmulBwdAttributes attributes;
    attributes.set_name("MoeGroupedMatmulBwdNullInput");

    // Must not crash even though one input is null.
    auto dweight = graph.moe_grouped_matmul_bwd(dOutput, token, firstTokenOffset, attributes);
    ASSERT_NE(dweight, nullptr);

    const auto result = graph.validate();
    EXPECT_EQ(result.code, ErrorCode::ATTRIBUTE_NOT_SET);
    EXPECT_NE(result.get_message().find(param.expectedMessageFragment), std::string::npos)
        << result.get_message();
}

INSTANTIATE_TEST_SUITE_P(
    RequiredInputs,
    TestGraphMoeGroupedMatmulBwdNullInput,
    ::testing::Values(
        NullInputCase{MoeGroupedMatmulBwdAttributes::InputNames::DOUTPUT, "Doutput", "doutput"},
        NullInputCase{MoeGroupedMatmulBwdAttributes::InputNames::TOKEN, "Token", "token"},
        NullInputCase{MoeGroupedMatmulBwdAttributes::InputNames::FIRST_TOKEN_OFFSET,
                      "FirstTokenOffset",
                      "first_token_offset"}),
    [](const ::testing::TestParamInfo<NullInputCase>& info) { return info.param.name; });
