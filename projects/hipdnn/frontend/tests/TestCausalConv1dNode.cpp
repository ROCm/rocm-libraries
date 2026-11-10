// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/CausalConv1dAttributes.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/node/CausalConv1dNode.hpp>

#include <memory>
#include <unordered_set>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

// NOLINTBEGIN(misc-const-correctness)

// --- Helper: create fully configured attributes for a valid node ---
namespace
{

CausalConv1dAttributes createValidAttributes()
{
    CausalConv1dAttributes attrs;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({8, 4});
    xTensor->set_stride({4, 1});
    xTensor->set_data_type(DataType::FLOAT);
    attrs.set_x(xTensor);
    auto weightTensor = std::make_shared<TensorAttributes>();
    weightTensor->set_dim({4, 4});
    weightTensor->set_stride({4, 1});
    weightTensor->set_data_type(DataType::FLOAT);
    attrs.set_weight(weightTensor);
    auto cuSeqlensTensor = std::make_shared<TensorAttributes>();
    cuSeqlensTensor->set_dim({3});
    cuSeqlensTensor->set_stride({1});
    cuSeqlensTensor->set_data_type(DataType::INT32);
    attrs.set_cu_seqlens(cuSeqlensTensor);
    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_dim({8, 4});
    yTensor->set_stride({4, 1});
    yTensor->set_data_type(DataType::FLOAT);
    attrs.set_y(yTensor);

    return attrs;
}

} // namespace

// --- GetNodeType ---

TEST(TestCausalConv1dNode, GetNodeTypeReturnsCausalConv1d)
{
    const GraphAttributes graphAttrs;
    const CausalConv1dNode node(CausalConv1dAttributes{}, graphAttrs);
    EXPECT_EQ(node.getNodeType(), NodeType::CAUSAL_CONV1D);
}

// --- PreValidateNode (success case) ---

TEST(TestCausalConv1dNode, PreValidateNode)
{
    auto attrs = createValidAttributes();

    const GraphAttributes graphAttributes;
    const CausalConv1dNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

// --- PreValidateNode: missing required tensors ---

TEST(TestCausalConv1dNode, PreValidateNodeMissingXTensor)
{
    CausalConv1dAttributes attrs;

    // Set all required tensors except x
    auto weightTensor = std::make_shared<TensorAttributes>();
    weightTensor->set_dim({4, 4});
    weightTensor->set_stride({4, 1});
    weightTensor->set_data_type(DataType::FLOAT);
    attrs.set_weight(weightTensor);
    auto cuSeqlensTensor = std::make_shared<TensorAttributes>();
    cuSeqlensTensor->set_dim({3});
    cuSeqlensTensor->set_stride({1});
    cuSeqlensTensor->set_data_type(DataType::INT32);
    attrs.set_cu_seqlens(cuSeqlensTensor);
    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_dim({8, 4});
    yTensor->set_stride({4, 1});
    yTensor->set_data_type(DataType::FLOAT);
    attrs.set_y(yTensor);

    // x tensor is missing
    const GraphAttributes graphAttributes;
    const CausalConv1dNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestCausalConv1dNode, PreValidateNodeMissingWeightTensor)
{
    CausalConv1dAttributes attrs;

    // Set all required tensors except weight
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({8, 4});
    xTensor->set_stride({4, 1});
    xTensor->set_data_type(DataType::FLOAT);
    attrs.set_x(xTensor);
    auto cuSeqlensTensor = std::make_shared<TensorAttributes>();
    cuSeqlensTensor->set_dim({3});
    cuSeqlensTensor->set_stride({1});
    cuSeqlensTensor->set_data_type(DataType::INT32);
    attrs.set_cu_seqlens(cuSeqlensTensor);
    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_dim({8, 4});
    yTensor->set_stride({4, 1});
    yTensor->set_data_type(DataType::FLOAT);
    attrs.set_y(yTensor);

    // weight tensor is missing
    const GraphAttributes graphAttributes;
    const CausalConv1dNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestCausalConv1dNode, PreValidateNodeMissingCuSeqlensTensor)
{
    CausalConv1dAttributes attrs;

    // Set all required tensors except cu_seqlens
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({8, 4});
    xTensor->set_stride({4, 1});
    xTensor->set_data_type(DataType::FLOAT);
    attrs.set_x(xTensor);
    auto weightTensor = std::make_shared<TensorAttributes>();
    weightTensor->set_dim({4, 4});
    weightTensor->set_stride({4, 1});
    weightTensor->set_data_type(DataType::FLOAT);
    attrs.set_weight(weightTensor);
    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_dim({8, 4});
    yTensor->set_stride({4, 1});
    yTensor->set_data_type(DataType::FLOAT);
    attrs.set_y(yTensor);

    // cu_seqlens tensor is missing
    const GraphAttributes graphAttributes;
    const CausalConv1dNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestCausalConv1dNode, PreValidateNodeMissingYTensor)
{
    CausalConv1dAttributes attrs;

    // Set all required tensors except y
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({8, 4});
    xTensor->set_stride({4, 1});
    xTensor->set_data_type(DataType::FLOAT);
    attrs.set_x(xTensor);
    auto weightTensor = std::make_shared<TensorAttributes>();
    weightTensor->set_dim({4, 4});
    weightTensor->set_stride({4, 1});
    weightTensor->set_data_type(DataType::FLOAT);
    attrs.set_weight(weightTensor);
    auto cuSeqlensTensor = std::make_shared<TensorAttributes>();
    cuSeqlensTensor->set_dim({3});
    cuSeqlensTensor->set_stride({1});
    cuSeqlensTensor->set_data_type(DataType::INT32);
    attrs.set_cu_seqlens(cuSeqlensTensor);

    // y tensor is missing
    const GraphAttributes graphAttributes;
    const CausalConv1dNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestCausalConv1dNode, PreValidateNodeAllValuesSet)
{
    auto attrs = createValidAttributes();

    const GraphAttributes graphAttributes;
    const CausalConv1dNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

// --- InferPropertiesNode ---

TEST(TestCausalConv1dNode, InferPropertiesNode)
{
    auto attrs = createValidAttributes();

    const GraphAttributes graphAttributes;
    CausalConv1dNode node(std::move(attrs), graphAttributes);

    auto error = node.infer_properties_node();
    // Stub implementation: verify the method can be called without error
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

// --- Shape rules (T = 8, C = 4, W = 4, N = 2) ---

namespace
{

std::shared_ptr<TensorAttributes> makeTensor(const std::vector<int64_t>& dims,
                                             DataType dataType = DataType::FLOAT)
{
    auto tensor = std::make_shared<TensorAttributes>();
    tensor->set_dim(dims);
    tensor->set_data_type(dataType);
    return tensor;
}

Error preValidate(CausalConv1dAttributes attrs)
{
    const GraphAttributes graphAttributes;
    const CausalConv1dNode node(std::move(attrs), graphAttributes);
    return node.pre_validate_node();
}

} // namespace

TEST(TestCausalConv1dNode, PreValidateAcceptsAllOptionalInputs)
{
    auto attrs = createValidAttributes();
    attrs.set_bias(makeTensor({4}));
    attrs.set_initial_state(makeTensor({5, 4, 3}));
    attrs.set_state_indices(makeTensor({2}, DataType::INT64));
    attrs.set_final_state(makeTensor({5, 4, 3}));
    attrs.set_output_final_state(true);
    attrs.get_cu_seqlens()->set_data_type(DataType::INT64);
    auto error = preValidate(std::move(attrs));
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

TEST(TestCausalConv1dNode, PreValidateFailsXNotRank2)
{
    auto attrs = createValidAttributes();
    attrs.get_x()->set_dim({1, 8, 4});
    EXPECT_EQ(preValidate(std::move(attrs)).code, error_code_t::INVALID_VALUE);
}

TEST(TestCausalConv1dNode, PreValidateFailsWeightNotRank2)
{
    auto attrs = createValidAttributes();
    attrs.get_weight()->set_dim({4, 1, 4});
    EXPECT_EQ(preValidate(std::move(attrs)).code, error_code_t::INVALID_VALUE);
}

TEST(TestCausalConv1dNode, PreValidateFailsWeightChannelMismatch)
{
    auto attrs = createValidAttributes();
    attrs.get_weight()->set_dim({3, 4});
    EXPECT_EQ(preValidate(std::move(attrs)).code, error_code_t::INVALID_VALUE);
}

TEST(TestCausalConv1dNode, PreValidateFailsKernelWidthBelowTwo)
{
    auto attrs = createValidAttributes();
    attrs.get_weight()->set_dim({4, 1});
    EXPECT_EQ(preValidate(std::move(attrs)).code, error_code_t::INVALID_VALUE);
}

TEST(TestCausalConv1dNode, PreValidateFailsCuSeqlensNotRank1)
{
    auto attrs = createValidAttributes();
    attrs.get_cu_seqlens()->set_dim({3, 1});
    EXPECT_EQ(preValidate(std::move(attrs)).code, error_code_t::INVALID_VALUE);
}

TEST(TestCausalConv1dNode, PreValidateFailsCuSeqlensWithoutSequences)
{
    auto attrs = createValidAttributes();
    attrs.get_cu_seqlens()->set_dim({1});
    EXPECT_EQ(preValidate(std::move(attrs)).code, error_code_t::INVALID_VALUE);
}

TEST(TestCausalConv1dNode, PreValidateFailsCuSeqlensNotIndexType)
{
    auto attrs = createValidAttributes();
    attrs.get_cu_seqlens()->set_data_type(DataType::FLOAT);
    EXPECT_EQ(preValidate(std::move(attrs)).code, error_code_t::INVALID_VALUE);
}

TEST(TestCausalConv1dNode, PreValidateFailsBiasShape)
{
    auto attrs = createValidAttributes();
    attrs.set_bias(makeTensor({3}));
    EXPECT_EQ(preValidate(std::move(attrs)).code, error_code_t::INVALID_VALUE);
}

TEST(TestCausalConv1dNode, PreValidateFailsStateIndicesWithoutInitialState)
{
    auto attrs = createValidAttributes();
    attrs.set_state_indices(makeTensor({2}, DataType::INT32));
    EXPECT_EQ(preValidate(std::move(attrs)).code, error_code_t::INVALID_VALUE);
}

TEST(TestCausalConv1dNode, PreValidateFailsStateIndicesLength)
{
    auto attrs = createValidAttributes();
    attrs.set_initial_state(makeTensor({5, 4, 3}));
    attrs.set_state_indices(makeTensor({3}, DataType::INT32));
    EXPECT_EQ(preValidate(std::move(attrs)).code, error_code_t::INVALID_VALUE);
}

TEST(TestCausalConv1dNode, PreValidateFailsStateIndicesNotIndexType)
{
    auto attrs = createValidAttributes();
    attrs.set_initial_state(makeTensor({5, 4, 3}));
    attrs.set_state_indices(makeTensor({2}, DataType::FLOAT));
    EXPECT_EQ(preValidate(std::move(attrs)).code, error_code_t::INVALID_VALUE);
}

TEST(TestCausalConv1dNode, PreValidateFailsInitialStateRank)
{
    auto attrs = createValidAttributes();
    attrs.set_initial_state(makeTensor({2, 12}));
    EXPECT_EQ(preValidate(std::move(attrs)).code, error_code_t::INVALID_VALUE);
}

TEST(TestCausalConv1dNode, PreValidateFailsInitialStateSequenceCountWithoutIndices)
{
    auto attrs = createValidAttributes();
    attrs.set_initial_state(makeTensor({5, 4, 3}));
    EXPECT_EQ(preValidate(std::move(attrs)).code, error_code_t::INVALID_VALUE);
}

TEST(TestCausalConv1dNode, PreValidateFailsInitialStateChannels)
{
    auto attrs = createValidAttributes();
    attrs.set_initial_state(makeTensor({2, 3, 3}));
    EXPECT_EQ(preValidate(std::move(attrs)).code, error_code_t::INVALID_VALUE);
}

TEST(TestCausalConv1dNode, PreValidateFailsInitialStateWidth)
{
    auto attrs = createValidAttributes();
    attrs.set_initial_state(makeTensor({2, 4, 4}));
    EXPECT_EQ(preValidate(std::move(attrs)).code, error_code_t::INVALID_VALUE);
}

TEST(TestCausalConv1dNode, PreValidateFailsOutputFinalStateWithoutTensor)
{
    auto attrs = createValidAttributes();
    attrs.set_output_final_state(true);
    EXPECT_EQ(preValidate(std::move(attrs)).code, error_code_t::INVALID_VALUE);
}

TEST(TestCausalConv1dNode, PreValidateFailsFinalStateWithoutOutputFinalState)
{
    auto attrs = createValidAttributes();
    attrs.set_final_state(makeTensor({2, 4, 3}));
    EXPECT_EQ(preValidate(std::move(attrs)).code, error_code_t::INVALID_VALUE);
}

TEST(TestCausalConv1dNode, PreValidateFailsFinalStateShape)
{
    auto attrs = createValidAttributes();
    attrs.set_final_state(makeTensor({2, 4, 2}));
    attrs.set_output_final_state(true);
    EXPECT_EQ(preValidate(std::move(attrs)).code, error_code_t::INVALID_VALUE);
}

TEST(TestCausalConv1dNode, PreValidateFailsFinalStatePoolMismatch)
{
    auto attrs = createValidAttributes();
    attrs.set_initial_state(makeTensor({5, 4, 3}));
    attrs.set_state_indices(makeTensor({2}, DataType::INT32));
    attrs.set_final_state(makeTensor({6, 4, 3}));
    attrs.set_output_final_state(true);
    EXPECT_EQ(preValidate(std::move(attrs)).code, error_code_t::INVALID_VALUE);
}

TEST(TestCausalConv1dNode, InferPropertiesInfersYAndFinalStateShapes)
{
    auto attrs = createValidAttributes();
    attrs.set_y(std::make_shared<TensorAttributes>());
    attrs.set_final_state(std::make_shared<TensorAttributes>());
    attrs.set_output_final_state(true);

    const GraphAttributes graphAttributes;
    CausalConv1dNode node(std::move(attrs), graphAttributes);
    auto error = node.infer_properties_node();
    ASSERT_EQ(error.code, error_code_t::OK) << error.err_msg;

    EXPECT_EQ(node.attributes.get_y()->get_dim(), (std::vector<int64_t>{8, 4}));
    EXPECT_EQ(node.attributes.get_y()->get_stride(), (std::vector<int64_t>{4, 1}));
    EXPECT_EQ(node.attributes.get_final_state()->get_dim(), (std::vector<int64_t>{2, 4, 3}));
    EXPECT_EQ(node.attributes.get_final_state()->get_stride(), (std::vector<int64_t>{12, 3, 1}));
}

TEST(TestCausalConv1dNode, InferPropertiesInfersFinalStatePoolRows)
{
    auto attrs = createValidAttributes();
    attrs.set_initial_state(makeTensor({5, 4, 3}));
    attrs.set_state_indices(makeTensor({2}, DataType::INT32));
    attrs.set_final_state(std::make_shared<TensorAttributes>());
    attrs.set_output_final_state(true);

    const GraphAttributes graphAttributes;
    CausalConv1dNode node(std::move(attrs), graphAttributes);
    auto error = node.infer_properties_node();
    ASSERT_EQ(error.code, error_code_t::OK) << error.err_msg;
    EXPECT_EQ(node.attributes.get_final_state()->get_dim(), (std::vector<int64_t>{5, 4, 3}));
}

TEST(TestCausalConv1dNode, InferPropertiesFailsYShapeMismatch)
{
    auto attrs = createValidAttributes();
    attrs.get_y()->set_dim({8, 3});

    const GraphAttributes graphAttributes;
    CausalConv1dNode node(std::move(attrs), graphAttributes);
    EXPECT_EQ(node.infer_properties_node().code, error_code_t::INVALID_VALUE);
}

// --- GatherHipdnnTensors ---

TEST(TestCausalConv1dNode, GatherHipdnnTensor)
{
    CausalConv1dAttributes attrs;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(9600).set_name("XTensor");
    attrs.set_x(xTensor);
    auto weightTensor = std::make_shared<TensorAttributes>();
    weightTensor->set_uid(9601).set_name("WeightTensor");
    attrs.set_weight(weightTensor);
    auto cuSeqlensTensor = std::make_shared<TensorAttributes>();
    cuSeqlensTensor->set_uid(9602).set_name("CuSeqlensTensor");
    attrs.set_cu_seqlens(cuSeqlensTensor);
    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_uid(9603).set_name("YTensor");
    attrs.set_y(yTensor);

    const GraphAttributes graphAttributes;
    const CausalConv1dNode node(std::move(attrs), graphAttributes);

    std::unordered_set<std::shared_ptr<TensorAttributes>> allTensors;

    node.gather_hipdnn_tensors(allTensors);

    EXPECT_TRUE(allTensors.find(xTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(weightTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(cuSeqlensTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(yTensor) != allTensors.end());
    EXPECT_EQ(allTensors.size(), 4u);
}

// NOLINTEND(misc-const-correctness)
