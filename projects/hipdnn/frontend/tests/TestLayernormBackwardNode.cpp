// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/LayernormBackwardAttributes.hpp>
#include <hipdnn_frontend/node/LayernormBackwardNode.hpp>

#include <memory>
#include <unordered_set>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

// --- Helper: create fully configured attributes for a valid node ---
namespace
{

LayernormBackwardAttributes createValidAttributes()
{
    LayernormBackwardAttributes attrs;

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({16, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dy(dyTensor);
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({16, 64, 32, 32});
    xTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_x(xTensor);
    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 32, 32});
    scaleTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_scale(scaleTensor);
    auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_dim({16, 64, 32, 32});
    dxTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dx(dxTensor);
    auto dscaleTensor = std::make_shared<TensorAttributes>();
    dscaleTensor->set_dim({1, 64, 32, 32});
    dscaleTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dscale(dscaleTensor);
    auto dbiasTensor = std::make_shared<TensorAttributes>();
    dbiasTensor->set_dim({1, 64, 32, 32});
    dbiasTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dbias(dbiasTensor);

    return attrs;
}

} // namespace

// --- GetNodeType ---

TEST(TestLayernormBackwardNode, GetNodeTypeReturnsLayernormBackward)
{
    const GraphAttributes graphAttrs;
    const LayernormBackwardNode node(LayernormBackwardAttributes{}, graphAttrs);
    EXPECT_EQ(node.getNodeType(), NodeType::LAYERNORM_BACKWARD);
}

// --- PreValidateNode (success case) ---

TEST(TestLayernormBackwardNode, PreValidateNode)
{
    auto attrs = createValidAttributes();

    const GraphAttributes graphAttributes;
    const LayernormBackwardNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

// --- PreValidateNode: missing required tensors ---

TEST(TestLayernormBackwardNode, PreValidateNodeMissingDyTensor)
{
    LayernormBackwardAttributes attrs;

    // Set all required tensors except dy
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({16, 64, 32, 32});
    xTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_x(xTensor);
    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 32, 32});
    scaleTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_scale(scaleTensor);
    auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_dim({16, 64, 32, 32});
    dxTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dx(dxTensor);
    auto dscaleTensor = std::make_shared<TensorAttributes>();
    dscaleTensor->set_dim({1, 64, 32, 32});
    dscaleTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dscale(dscaleTensor);
    auto dbiasTensor = std::make_shared<TensorAttributes>();
    dbiasTensor->set_dim({1, 64, 32, 32});
    dbiasTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dbias(dbiasTensor);

    // dy tensor is missing
    const GraphAttributes graphAttributes;
    const LayernormBackwardNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestLayernormBackwardNode, PreValidateNodeMissingXTensor)
{
    LayernormBackwardAttributes attrs;

    // Set all required tensors except x
    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({16, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dy(dyTensor);
    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 32, 32});
    scaleTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_scale(scaleTensor);
    auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_dim({16, 64, 32, 32});
    dxTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dx(dxTensor);
    auto dscaleTensor = std::make_shared<TensorAttributes>();
    dscaleTensor->set_dim({1, 64, 32, 32});
    dscaleTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dscale(dscaleTensor);
    auto dbiasTensor = std::make_shared<TensorAttributes>();
    dbiasTensor->set_dim({1, 64, 32, 32});
    dbiasTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dbias(dbiasTensor);

    // x tensor is missing
    const GraphAttributes graphAttributes;
    const LayernormBackwardNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestLayernormBackwardNode, PreValidateNodeMissingScaleTensor)
{
    LayernormBackwardAttributes attrs;

    // Set all required tensors except scale
    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({16, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dy(dyTensor);
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({16, 64, 32, 32});
    xTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_x(xTensor);
    auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_dim({16, 64, 32, 32});
    dxTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dx(dxTensor);
    auto dscaleTensor = std::make_shared<TensorAttributes>();
    dscaleTensor->set_dim({1, 64, 32, 32});
    dscaleTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dscale(dscaleTensor);
    auto dbiasTensor = std::make_shared<TensorAttributes>();
    dbiasTensor->set_dim({1, 64, 32, 32});
    dbiasTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dbias(dbiasTensor);

    // scale tensor is missing
    const GraphAttributes graphAttributes;
    const LayernormBackwardNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestLayernormBackwardNode, PreValidateNodeMissingDxTensor)
{
    LayernormBackwardAttributes attrs;

    // Set all required tensors except dx
    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({16, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dy(dyTensor);
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({16, 64, 32, 32});
    xTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_x(xTensor);
    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 32, 32});
    scaleTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_scale(scaleTensor);
    auto dscaleTensor = std::make_shared<TensorAttributes>();
    dscaleTensor->set_dim({1, 64, 32, 32});
    dscaleTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dscale(dscaleTensor);
    auto dbiasTensor = std::make_shared<TensorAttributes>();
    dbiasTensor->set_dim({1, 64, 32, 32});
    dbiasTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dbias(dbiasTensor);

    // dx tensor is missing
    const GraphAttributes graphAttributes;
    const LayernormBackwardNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestLayernormBackwardNode, PreValidateNodeMissingDscaleTensor)
{
    LayernormBackwardAttributes attrs;

    // Set all required tensors except dscale
    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({16, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dy(dyTensor);
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({16, 64, 32, 32});
    xTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_x(xTensor);
    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 32, 32});
    scaleTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_scale(scaleTensor);
    auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_dim({16, 64, 32, 32});
    dxTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dx(dxTensor);
    auto dbiasTensor = std::make_shared<TensorAttributes>();
    dbiasTensor->set_dim({1, 64, 32, 32});
    dbiasTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dbias(dbiasTensor);

    // dscale tensor is missing
    const GraphAttributes graphAttributes;
    const LayernormBackwardNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestLayernormBackwardNode, PreValidateNodeMissingDbiasTensor)
{
    LayernormBackwardAttributes attrs;

    // Set all required tensors except dbias
    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({16, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dy(dyTensor);
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({16, 64, 32, 32});
    xTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_x(xTensor);
    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 32, 32});
    scaleTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_scale(scaleTensor);
    auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_dim({16, 64, 32, 32});
    dxTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dx(dxTensor);
    auto dscaleTensor = std::make_shared<TensorAttributes>();
    dscaleTensor->set_dim({1, 64, 32, 32});
    dscaleTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dscale(dscaleTensor);

    // dbias tensor is missing
    const GraphAttributes graphAttributes;
    const LayernormBackwardNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestLayernormBackwardNode, PreValidateNodeAllValuesSet)
{
    auto attrs = createValidAttributes();

    const GraphAttributes graphAttributes;
    const LayernormBackwardNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

// --- InferPropertiesNode ---

TEST(TestLayernormBackwardNode, InferPropertiesNode)
{
    auto attrs = createValidAttributes();

    const GraphAttributes graphAttributes;
    LayernormBackwardNode node(std::move(attrs), graphAttributes);

    auto error = node.infer_properties_node();
    // Stub implementation: verify the method can be called without error
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

// --- GatherHipdnnTensors ---

TEST(TestLayernormBackwardNode, GatherHipdnnTensor)
{
    LayernormBackwardAttributes attrs;

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_uid(10).set_name("DyTensor");
    attrs.set_dy(dyTensor);
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(11).set_name("XTensor");
    attrs.set_x(xTensor);
    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_uid(12).set_name("ScaleTensor");
    attrs.set_scale(scaleTensor);
    auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_uid(16).set_name("DxTensor");
    attrs.set_dx(dxTensor);
    auto dscaleTensor = std::make_shared<TensorAttributes>();
    dscaleTensor->set_uid(17).set_name("DscaleTensor");
    attrs.set_dscale(dscaleTensor);
    auto dbiasTensor = std::make_shared<TensorAttributes>();
    dbiasTensor->set_uid(18).set_name("DbiasTensor");
    attrs.set_dbias(dbiasTensor);

    const GraphAttributes graphAttributes;
    const LayernormBackwardNode node(std::move(attrs), graphAttributes);

    std::unordered_set<std::shared_ptr<TensorAttributes>> allTensors;

    node.gather_hipdnn_tensors(allTensors);

    EXPECT_TRUE(allTensors.find(dyTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(xTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(scaleTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(dxTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(dscaleTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(dbiasTensor) != allTensors.end());
    EXPECT_EQ(allTensors.size(), 6u);
}
