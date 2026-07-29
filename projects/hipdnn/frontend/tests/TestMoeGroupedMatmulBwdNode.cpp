// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/MoeGroupedMatmulBwdAttributes.hpp>
#include <hipdnn_frontend/node/MoeGroupedMatmulBwdNode.hpp>

#include <memory>
#include <unordered_set>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

// NOLINTBEGIN(misc-const-correctness)

// --- Helper: create fully configured attributes for a valid node ---
namespace
{

MoeGroupedMatmulBwdAttributes createValidAttributes()
{
    MoeGroupedMatmulBwdAttributes attrs;

    auto doutputTensor = std::make_shared<TensorAttributes>();
    doutputTensor->set_uid(1910);
    doutputTensor->set_dim({1, 8, 32});
    doutputTensor->set_stride({256, 32, 1});
    doutputTensor->set_data_type(DataType::FLOAT);
    attrs.set_doutput(doutputTensor);
    auto tokenTensor = std::make_shared<TensorAttributes>();
    tokenTensor->set_uid(1911);
    tokenTensor->set_dim({1, 8, 16});
    tokenTensor->set_stride({128, 16, 1});
    tokenTensor->set_data_type(DataType::FLOAT);
    attrs.set_token(tokenTensor);
    auto firstTokenOffsetTensor = std::make_shared<TensorAttributes>();
    firstTokenOffsetTensor->set_uid(1912);
    firstTokenOffsetTensor->set_dim({2, 1, 1});
    firstTokenOffsetTensor->set_stride({1, 1, 1});
    firstTokenOffsetTensor->set_data_type(DataType::INT32);
    attrs.set_first_token_offset(firstTokenOffsetTensor);
    auto dweightTensor = std::make_shared<TensorAttributes>();
    dweightTensor->set_uid(1913);
    dweightTensor->set_dim({2, 16, 32});
    dweightTensor->set_stride({512, 32, 1});
    dweightTensor->set_data_type(DataType::FLOAT);
    attrs.set_dweight(dweightTensor);

    return attrs;
}

} // namespace

// --- GetNodeType ---

TEST(TestMoeGroupedMatmulBwdNode, GetNodeTypeReturnsMoeGroupedMatmulBwd)
{
    const GraphAttributes graphAttrs;
    const MoeGroupedMatmulBwdNode node(MoeGroupedMatmulBwdAttributes{}, graphAttrs);
    EXPECT_EQ(node.getNodeType(), NodeType::MOE_GROUPED_MATMUL_BWD);
}

// --- PreValidateNode (success case) ---

TEST(TestMoeGroupedMatmulBwdNode, PreValidateNode)
{
    auto attrs = createValidAttributes();

    const GraphAttributes graphAttributes;
    const MoeGroupedMatmulBwdNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

// --- PreValidateNode: missing required tensors ---

TEST(TestMoeGroupedMatmulBwdNode, PreValidateNodeMissingDoutputTensor)
{
    MoeGroupedMatmulBwdAttributes attrs;

    // Set all required tensors except doutput
    auto tokenTensor = std::make_shared<TensorAttributes>();
    tokenTensor->set_dim({1, 8, 16});
    tokenTensor->set_stride({128, 16, 1});
    tokenTensor->set_data_type(DataType::FLOAT);
    attrs.set_token(tokenTensor);
    auto firstTokenOffsetTensor = std::make_shared<TensorAttributes>();
    firstTokenOffsetTensor->set_dim({2, 1, 1});
    firstTokenOffsetTensor->set_stride({1, 1, 1});
    firstTokenOffsetTensor->set_data_type(DataType::INT32);
    attrs.set_first_token_offset(firstTokenOffsetTensor);
    auto dweightTensor = std::make_shared<TensorAttributes>();
    dweightTensor->set_dim({2, 16, 32});
    dweightTensor->set_stride({512, 32, 1});
    dweightTensor->set_data_type(DataType::FLOAT);
    attrs.set_dweight(dweightTensor);

    // doutput tensor is missing
    const GraphAttributes graphAttributes;
    const MoeGroupedMatmulBwdNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestMoeGroupedMatmulBwdNode, PreValidateNodeMissingTokenTensor)
{
    MoeGroupedMatmulBwdAttributes attrs;

    // Set all required tensors except token
    auto doutputTensor = std::make_shared<TensorAttributes>();
    doutputTensor->set_dim({1, 8, 32});
    doutputTensor->set_stride({256, 32, 1});
    doutputTensor->set_data_type(DataType::FLOAT);
    attrs.set_doutput(doutputTensor);
    auto firstTokenOffsetTensor = std::make_shared<TensorAttributes>();
    firstTokenOffsetTensor->set_dim({2, 1, 1});
    firstTokenOffsetTensor->set_stride({1, 1, 1});
    firstTokenOffsetTensor->set_data_type(DataType::INT32);
    attrs.set_first_token_offset(firstTokenOffsetTensor);
    auto dweightTensor = std::make_shared<TensorAttributes>();
    dweightTensor->set_dim({2, 16, 32});
    dweightTensor->set_stride({512, 32, 1});
    dweightTensor->set_data_type(DataType::FLOAT);
    attrs.set_dweight(dweightTensor);

    // token tensor is missing
    const GraphAttributes graphAttributes;
    const MoeGroupedMatmulBwdNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestMoeGroupedMatmulBwdNode, PreValidateNodeMissingFirstTokenOffsetTensor)
{
    MoeGroupedMatmulBwdAttributes attrs;

    // Set all required tensors except first_token_offset
    auto doutputTensor = std::make_shared<TensorAttributes>();
    doutputTensor->set_dim({1, 8, 32});
    doutputTensor->set_stride({256, 32, 1});
    doutputTensor->set_data_type(DataType::FLOAT);
    attrs.set_doutput(doutputTensor);
    auto tokenTensor = std::make_shared<TensorAttributes>();
    tokenTensor->set_dim({1, 8, 16});
    tokenTensor->set_stride({128, 16, 1});
    tokenTensor->set_data_type(DataType::FLOAT);
    attrs.set_token(tokenTensor);
    auto dweightTensor = std::make_shared<TensorAttributes>();
    dweightTensor->set_dim({2, 16, 32});
    dweightTensor->set_stride({512, 32, 1});
    dweightTensor->set_data_type(DataType::FLOAT);
    attrs.set_dweight(dweightTensor);

    // first_token_offset tensor is missing
    const GraphAttributes graphAttributes;
    const MoeGroupedMatmulBwdNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestMoeGroupedMatmulBwdNode, PreValidateNodeMissingDweightTensor)
{
    MoeGroupedMatmulBwdAttributes attrs;

    // Set all required tensors except dweight
    auto doutputTensor = std::make_shared<TensorAttributes>();
    doutputTensor->set_dim({1, 8, 32});
    doutputTensor->set_stride({256, 32, 1});
    doutputTensor->set_data_type(DataType::FLOAT);
    attrs.set_doutput(doutputTensor);
    auto tokenTensor = std::make_shared<TensorAttributes>();
    tokenTensor->set_dim({1, 8, 16});
    tokenTensor->set_stride({128, 16, 1});
    tokenTensor->set_data_type(DataType::FLOAT);
    attrs.set_token(tokenTensor);
    auto firstTokenOffsetTensor = std::make_shared<TensorAttributes>();
    firstTokenOffsetTensor->set_dim({2, 1, 1});
    firstTokenOffsetTensor->set_stride({1, 1, 1});
    firstTokenOffsetTensor->set_data_type(DataType::INT32);
    attrs.set_first_token_offset(firstTokenOffsetTensor);

    // dweight tensor is missing
    const GraphAttributes graphAttributes;
    const MoeGroupedMatmulBwdNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestMoeGroupedMatmulBwdNode, PreValidateNodeAllValuesSet)
{
    auto attrs = createValidAttributes();

    const GraphAttributes graphAttributes;
    const MoeGroupedMatmulBwdNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

// --- PreValidateNode: dweight/token/doutput consistency checks ---

TEST(TestMoeGroupedMatmulBwdNode, PreValidateNodeRejectsDweightKDimensionMismatch)
{
    auto attrs = createValidAttributes();
    // dweight K dimension (dim[1]) no longer matches token K dimension (dim[2] == 16)
    attrs.get_dweight()->set_dim({2, 8, 32});

    const GraphAttributes graphAttributes;
    const MoeGroupedMatmulBwdNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestMoeGroupedMatmulBwdNode, PreValidateNodeRejectsDweightNDimensionMismatch)
{
    auto attrs = createValidAttributes();
    // dweight N dimension (dim[2]) no longer matches doutput N dimension (dim[2] == 32)
    attrs.get_dweight()->set_dim({2, 16, 64});

    const GraphAttributes graphAttributes;
    const MoeGroupedMatmulBwdNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestMoeGroupedMatmulBwdNode, PreValidateNodeRejectsNonInt32FirstTokenOffset)
{
    auto attrs = createValidAttributes();
    // first_token_offset must be INT32; this is a frontend-only check (the YAML's
    // expected_data_type only generates a backend finalize() check).
    attrs.get_first_token_offset()->set_data_type(DataType::FLOAT);

    const GraphAttributes graphAttributes;
    const MoeGroupedMatmulBwdNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

// --- InferPropertiesNode ---

TEST(TestMoeGroupedMatmulBwdNode, InferPropertiesNode)
{
    auto attrs = createValidAttributes();

    const GraphAttributes graphAttributes;
    MoeGroupedMatmulBwdNode node(std::move(attrs), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

TEST(TestMoeGroupedMatmulBwdNode, InferPropertiesNodeRejectsUnsetDweightDims)
{
    auto attrs = createValidAttributes();
    // dweight dimensions are caller-supplied and never inferred (expert count is not
    // derivable from any input tensor).
    attrs.get_dweight()->set_dim({});

    const GraphAttributes graphAttributes;
    MoeGroupedMatmulBwdNode node(std::move(attrs), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

// --- GatherHipdnnTensors ---

TEST(TestMoeGroupedMatmulBwdNode, GatherHipdnnTensor)
{
    MoeGroupedMatmulBwdAttributes attrs;

    auto doutputTensor = std::make_shared<TensorAttributes>();
    doutputTensor->set_uid(1910).set_name("DoutputTensor");
    attrs.set_doutput(doutputTensor);
    auto tokenTensor = std::make_shared<TensorAttributes>();
    tokenTensor->set_uid(1911).set_name("TokenTensor");
    attrs.set_token(tokenTensor);
    auto firstTokenOffsetTensor = std::make_shared<TensorAttributes>();
    firstTokenOffsetTensor->set_uid(1912).set_name("FirstTokenOffsetTensor");
    attrs.set_first_token_offset(firstTokenOffsetTensor);
    auto dweightTensor = std::make_shared<TensorAttributes>();
    dweightTensor->set_uid(1913).set_name("DweightTensor");
    attrs.set_dweight(dweightTensor);

    const GraphAttributes graphAttributes;
    const MoeGroupedMatmulBwdNode node(std::move(attrs), graphAttributes);

    std::unordered_set<std::shared_ptr<TensorAttributes>> allTensors;

    node.gather_hipdnn_tensors(allTensors);

    EXPECT_TRUE(allTensors.find(doutputTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(tokenTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(firstTokenOffsetTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(dweightTensor) != allTensors.end());
    EXPECT_EQ(allTensors.size(), 4u);
}

// NOLINTEND(misc-const-correctness)
