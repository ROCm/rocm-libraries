// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/RopeAttributes.hpp>
#include <hipdnn_frontend/node/RopeNode.hpp>

#include <memory>
#include <unordered_set>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

// NOLINTBEGIN(misc-const-correctness)

// --- Helper: create fully configured attributes for a valid node ---
namespace
{

RopeAttributes createValidAttributes()
{
    RopeAttributes attrs;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 4, 16, 8});
    xTensor->set_stride({512, 128, 8, 1});
    xTensor->set_data_type(DataType::FLOAT);
    attrs.set_x(xTensor);
    auto freqsTensor = std::make_shared<TensorAttributes>();
    freqsTensor->set_dim({1, 1, 16, 4});
    freqsTensor->set_stride({64, 64, 4, 1});
    freqsTensor->set_data_type(DataType::FLOAT);
    attrs.set_freqs(freqsTensor);
    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_dim({2, 4, 16, 8});
    yTensor->set_stride({512, 128, 8, 1});
    yTensor->set_data_type(DataType::FLOAT);
    attrs.set_y(yTensor);

    attrs.set_output_scale(0.5);
    attrs.set_rope_dim(4);

    return attrs;
}

} // namespace

// --- GetNodeType ---

TEST(TestRopeNode, GetNodeTypeReturnsRope)
{
    const GraphAttributes graphAttrs;
    const RopeNode node(RopeAttributes{}, graphAttrs);
    EXPECT_EQ(node.getNodeType(), NodeType::ROPE);
}

// --- PreValidateNode (success case) ---

TEST(TestRopeNode, PreValidateNode)
{
    auto attrs = createValidAttributes();

    const GraphAttributes graphAttributes;
    const RopeNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

// --- PreValidateNode: missing required tensors ---

TEST(TestRopeNode, PreValidateNodeMissingXTensor)
{
    RopeAttributes attrs;

    // Set all required tensors except x
    auto freqsTensor = std::make_shared<TensorAttributes>();
    freqsTensor->set_dim({1, 1, 16, 4});
    freqsTensor->set_stride({64, 64, 4, 1});
    freqsTensor->set_data_type(DataType::FLOAT);
    attrs.set_freqs(freqsTensor);
    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_dim({2, 4, 16, 8});
    yTensor->set_stride({512, 128, 8, 1});
    yTensor->set_data_type(DataType::FLOAT);
    attrs.set_y(yTensor);

    attrs.set_output_scale(0.5);
    attrs.set_rope_dim(4);

    // x tensor is missing
    const GraphAttributes graphAttributes;
    const RopeNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestRopeNode, PreValidateNodeMissingFreqsTensor)
{
    RopeAttributes attrs;

    // Set all required tensors except freqs
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 4, 16, 8});
    xTensor->set_stride({512, 128, 8, 1});
    xTensor->set_data_type(DataType::FLOAT);
    attrs.set_x(xTensor);
    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_dim({2, 4, 16, 8});
    yTensor->set_stride({512, 128, 8, 1});
    yTensor->set_data_type(DataType::FLOAT);
    attrs.set_y(yTensor);

    attrs.set_output_scale(0.5);
    attrs.set_rope_dim(4);

    // freqs tensor is missing
    const GraphAttributes graphAttributes;
    const RopeNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestRopeNode, PreValidateNodeMissingYTensor)
{
    RopeAttributes attrs;

    // Set all required tensors except y
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 4, 16, 8});
    xTensor->set_stride({512, 128, 8, 1});
    xTensor->set_data_type(DataType::FLOAT);
    attrs.set_x(xTensor);
    auto freqsTensor = std::make_shared<TensorAttributes>();
    freqsTensor->set_dim({1, 1, 16, 4});
    freqsTensor->set_stride({64, 64, 4, 1});
    freqsTensor->set_data_type(DataType::FLOAT);
    attrs.set_freqs(freqsTensor);

    attrs.set_output_scale(0.5);
    attrs.set_rope_dim(4);

    // y tensor is missing
    const GraphAttributes graphAttributes;
    const RopeNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestRopeNode, PreValidateNodeAllValuesSet)
{
    auto attrs = createValidAttributes();

    const GraphAttributes graphAttributes;
    const RopeNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

// --- InferPropertiesNode ---

TEST(TestRopeNode, InferPropertiesNode)
{
    auto attrs = createValidAttributes();

    const GraphAttributes graphAttributes;
    RopeNode node(std::move(attrs), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

TEST(TestRopeNode, InferPropertiesNodeSetsOutputDimsAndStridesFromX)
{
    auto attrs = createValidAttributes();
    auto y = std::make_shared<TensorAttributes>();
    attrs.set_y(y);

    const GraphAttributes graphAttributes;
    RopeNode node(std::move(attrs), graphAttributes);

    auto error = node.infer_properties_node();
    ASSERT_EQ(error.code, error_code_t::OK) << error.err_msg;
    EXPECT_EQ(y->get_dim(), (std::vector<int64_t>{2, 4, 16, 8}));
    EXPECT_EQ(y->get_stride(), (std::vector<int64_t>{512, 128, 8, 1}));
}

TEST(TestRopeNode, InferPropertiesNodeRejectsMismatchedOutputDims)
{
    auto attrs = createValidAttributes();
    attrs.get_y()->set_dim({2, 4, 16, 4});

    const GraphAttributes graphAttributes;
    RopeNode node(std::move(attrs), graphAttributes);

    EXPECT_EQ(node.infer_properties_node().code, error_code_t::INVALID_VALUE);
}

// --- PreValidateNode: schema rules ---

namespace
{

void expectPreValidateFails(RopeAttributes attrs,
                            error_code_t expected = error_code_t::INVALID_VALUE)
{
    const GraphAttributes graphAttributes;
    const RopeNode node(std::move(attrs), graphAttributes);
    EXPECT_EQ(node.pre_validate_node().code, expected);
}

void expectPreValidateSucceeds(RopeAttributes attrs)
{
    const GraphAttributes graphAttributes;
    const RopeNode node(std::move(attrs), graphAttributes);
    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

} // namespace

TEST(TestRopeNode, PreValidateNodeRejectsRankOneX)
{
    auto attrs = createValidAttributes();
    attrs.get_x()->set_dim({8}).set_stride({1});
    attrs.get_freqs()->set_dim({4}).set_stride({1});
    expectPreValidateFails(std::move(attrs));
}

TEST(TestRopeNode, PreValidateNodeRejectsFreqsRankMismatch)
{
    auto attrs = createValidAttributes();
    attrs.get_freqs()->set_dim({16, 4}).set_stride({4, 1});
    expectPreValidateFails(std::move(attrs));
}

TEST(TestRopeNode, PreValidateNodeRejectsNonFloatFreqs)
{
    auto attrs = createValidAttributes();
    attrs.get_freqs()->set_data_type(DataType::HALF);
    expectPreValidateFails(std::move(attrs));
}

TEST(TestRopeNode, PreValidateNodeRejectsOddRopeDim)
{
    auto attrs = createValidAttributes();
    attrs.set_rope_dim(3);
    attrs.get_freqs()->set_dim({1, 1, 16, 3}).set_stride({48, 48, 3, 1});
    expectPreValidateFails(std::move(attrs));
}

TEST(TestRopeNode, PreValidateNodeRejectsNegativeRopeDim)
{
    auto attrs = createValidAttributes();
    attrs.set_rope_dim(-2);
    expectPreValidateFails(std::move(attrs));
}

TEST(TestRopeNode, PreValidateNodeRejectsRopeDimAboveHeadDim)
{
    auto attrs = createValidAttributes();
    attrs.set_rope_dim(10);
    attrs.get_freqs()->set_dim({1, 1, 16, 10}).set_stride({160, 160, 10, 1});
    expectPreValidateFails(std::move(attrs));
}

TEST(TestRopeNode, PreValidateNodeZeroRopeDimMeansFullRotation)
{
    auto attrs = createValidAttributes();
    attrs.set_rope_dim(0);
    attrs.get_freqs()->set_dim({1, 1, 16, 8}).set_stride({128, 128, 8, 1});
    expectPreValidateSucceeds(std::move(attrs));
}

TEST(TestRopeNode, PreValidateNodeRejectsFreqsLastDimNotRopeDim)
{
    auto attrs = createValidAttributes();
    attrs.set_rope_dim(0);
    expectPreValidateFails(std::move(attrs));
}

TEST(TestRopeNode, PreValidateNodeAcceptsLeadingOffset)
{
    auto attrs = createValidAttributes();
    attrs.set_rope_offset(0);
    expectPreValidateSucceeds(std::move(attrs));
}

TEST(TestRopeNode, PreValidateNodeRejectsOffsetPastHeadDim)
{
    auto attrs = createValidAttributes();
    attrs.set_rope_offset(6);
    expectPreValidateFails(std::move(attrs));
}

TEST(TestRopeNode, PreValidateNodeRejectsNegativeOffset)
{
    auto attrs = createValidAttributes();
    attrs.set_rope_offset(-1);
    expectPreValidateFails(std::move(attrs));
}

TEST(TestRopeNode, PreValidateNodeAcceptsPerBatchFreqs)
{
    auto attrs = createValidAttributes();
    attrs.get_freqs()->set_dim({2, 1, 16, 4}).set_stride({64, 64, 4, 1});
    expectPreValidateSucceeds(std::move(attrs));
}

TEST(TestRopeNode, PreValidateNodeRejectsNonBroadcastableFreqs)
{
    auto attrs = createValidAttributes();
    attrs.get_freqs()->set_dim({1, 1, 8, 4}).set_stride({32, 32, 4, 1});
    expectPreValidateFails(std::move(attrs));
}

TEST(TestRopeNode, PreValidateNodeRejectsOutputDimsMismatch)
{
    auto attrs = createValidAttributes();
    attrs.get_y()->set_dim({2, 4, 16, 4});
    expectPreValidateFails(std::move(attrs));
}

// --- GatherHipdnnTensors ---

TEST(TestRopeNode, GatherHipdnnTensor)
{
    RopeAttributes attrs;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(9100).set_name("XTensor");
    attrs.set_x(xTensor);
    auto freqsTensor = std::make_shared<TensorAttributes>();
    freqsTensor->set_uid(9101).set_name("FreqsTensor");
    attrs.set_freqs(freqsTensor);
    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_uid(9102).set_name("YTensor");
    attrs.set_y(yTensor);

    attrs.set_output_scale(0.5);
    attrs.set_rope_dim(4);

    const GraphAttributes graphAttributes;
    const RopeNode node(std::move(attrs), graphAttributes);

    std::unordered_set<std::shared_ptr<TensorAttributes>> allTensors;

    node.gather_hipdnn_tensors(allTensors);

    EXPECT_TRUE(allTensors.find(xTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(freqsTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(yTensor) != allTensors.end());
    EXPECT_EQ(allTensors.size(), 3u);
}

// NOLINTEND(misc-const-correctness)
