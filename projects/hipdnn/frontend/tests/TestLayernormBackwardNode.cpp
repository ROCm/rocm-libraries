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

    const auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({16, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dy(dyTensor);
    const auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({16, 64, 32, 32});
    xTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_x(xTensor);
    const auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 32, 32});
    scaleTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_scale(scaleTensor);
    const auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_dim({16, 64, 32, 32});
    dxTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dx(dxTensor);
    const auto dscaleTensor = std::make_shared<TensorAttributes>();
    dscaleTensor->set_dim({1, 64, 32, 32});
    dscaleTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dscale(dscaleTensor);
    const auto dbiasTensor = std::make_shared<TensorAttributes>();
    dbiasTensor->set_dim({1, 64, 32, 32});
    dbiasTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dbias(dbiasTensor);

    return attrs;
}

LayernormBackwardAttributes createMinimalAttributes()
{
    LayernormBackwardAttributes attrs;

    const auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({16, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dy(dyTensor);
    const auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({16, 64, 32, 32});
    xTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_x(xTensor);
    const auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 32, 32});
    scaleTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_scale(scaleTensor);
    const auto dxTensor = std::make_shared<TensorAttributes>();
    attrs.set_dx(dxTensor);
    const auto dscaleTensor = std::make_shared<TensorAttributes>();
    attrs.set_dscale(dscaleTensor);
    const auto dbiasTensor = std::make_shared<TensorAttributes>();
    attrs.set_dbias(dbiasTensor);

    return attrs;
}

// Attributes whose dy, dx follow xDims, whose dscale, dbias follow scaleDims and whose saved
// mean and inv_variance have statsDims. normalized_dim_count is left unset.
LayernormBackwardAttributes createShapedAttributes(const std::vector<int64_t>& xDims,
                                                   const std::vector<int64_t>& scaleDims,
                                                   const std::vector<int64_t>& statsDims)
{
    const auto makeTensor = [](const std::vector<int64_t>& dims) {
        auto tensor = std::make_shared<TensorAttributes>();
        tensor->set_dim(dims);
        return tensor;
    };

    LayernormBackwardAttributes attrs;
    attrs.set_dy(makeTensor(xDims));
    attrs.set_x(makeTensor(xDims));
    attrs.set_scale(makeTensor(scaleDims));
    attrs.set_mean(makeTensor(statsDims));
    attrs.set_inv_variance(makeTensor(statsDims));
    attrs.set_dx(makeTensor(xDims));
    attrs.set_dscale(makeTensor(scaleDims));
    attrs.set_dbias(makeTensor(scaleDims));

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

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

// --- PreValidateNode: missing required tensors ---

TEST(TestLayernormBackwardNode, PreValidateNodeMissingDyTensor)
{
    LayernormBackwardAttributes attrs;

    // Set all required tensors except dy
    const auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({16, 64, 32, 32});
    xTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_x(xTensor);
    const auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 32, 32});
    scaleTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_scale(scaleTensor);
    const auto meanTensor = std::make_shared<TensorAttributes>();
    meanTensor->set_dim({16, 1, 1, 1});
    meanTensor->set_stride({1, 1, 1, 1});
    attrs.set_mean(meanTensor);
    const auto invVarianceTensor = std::make_shared<TensorAttributes>();
    invVarianceTensor->set_dim({16, 1, 1, 1});
    invVarianceTensor->set_stride({1, 1, 1, 1});
    attrs.set_inv_variance(invVarianceTensor);
    const auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_dim({16, 64, 32, 32});
    dxTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dx(dxTensor);
    const auto dscaleTensor = std::make_shared<TensorAttributes>();
    dscaleTensor->set_dim({1, 64, 32, 32});
    dscaleTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dscale(dscaleTensor);
    const auto dbiasTensor = std::make_shared<TensorAttributes>();
    dbiasTensor->set_dim({1, 64, 32, 32});
    dbiasTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dbias(dbiasTensor);
    attrs.set_normalized_dim_count(3);

    // dy tensor is missing
    const GraphAttributes graphAttributes;
    const LayernormBackwardNode node(std::move(attrs), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestLayernormBackwardNode, PreValidateNodeMissingXTensor)
{
    LayernormBackwardAttributes attrs;

    // Set all required tensors except x
    const auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({16, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dy(dyTensor);
    const auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 32, 32});
    scaleTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_scale(scaleTensor);
    const auto meanTensor = std::make_shared<TensorAttributes>();
    meanTensor->set_dim({16, 1, 1, 1});
    meanTensor->set_stride({1, 1, 1, 1});
    attrs.set_mean(meanTensor);
    const auto invVarianceTensor = std::make_shared<TensorAttributes>();
    invVarianceTensor->set_dim({16, 1, 1, 1});
    invVarianceTensor->set_stride({1, 1, 1, 1});
    attrs.set_inv_variance(invVarianceTensor);
    const auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_dim({16, 64, 32, 32});
    dxTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dx(dxTensor);
    const auto dscaleTensor = std::make_shared<TensorAttributes>();
    dscaleTensor->set_dim({1, 64, 32, 32});
    dscaleTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dscale(dscaleTensor);
    const auto dbiasTensor = std::make_shared<TensorAttributes>();
    dbiasTensor->set_dim({1, 64, 32, 32});
    dbiasTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dbias(dbiasTensor);
    attrs.set_normalized_dim_count(3);

    // x tensor is missing
    const GraphAttributes graphAttributes;
    const LayernormBackwardNode node(std::move(attrs), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestLayernormBackwardNode, PreValidateNodeMissingScaleTensor)
{
    LayernormBackwardAttributes attrs;

    // Set all required tensors except scale
    const auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({16, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dy(dyTensor);
    const auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({16, 64, 32, 32});
    xTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_x(xTensor);
    const auto meanTensor = std::make_shared<TensorAttributes>();
    meanTensor->set_dim({16, 1, 1, 1});
    meanTensor->set_stride({1, 1, 1, 1});
    attrs.set_mean(meanTensor);
    const auto invVarianceTensor = std::make_shared<TensorAttributes>();
    invVarianceTensor->set_dim({16, 1, 1, 1});
    invVarianceTensor->set_stride({1, 1, 1, 1});
    attrs.set_inv_variance(invVarianceTensor);
    const auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_dim({16, 64, 32, 32});
    dxTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dx(dxTensor);
    const auto dscaleTensor = std::make_shared<TensorAttributes>();
    dscaleTensor->set_dim({1, 64, 32, 32});
    dscaleTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dscale(dscaleTensor);
    const auto dbiasTensor = std::make_shared<TensorAttributes>();
    dbiasTensor->set_dim({1, 64, 32, 32});
    dbiasTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dbias(dbiasTensor);
    attrs.set_normalized_dim_count(3);

    // scale tensor is missing
    const GraphAttributes graphAttributes;
    const LayernormBackwardNode node(std::move(attrs), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestLayernormBackwardNode, PreValidateNodeMissingMeanTensor)
{
    LayernormBackwardAttributes attrs;

    // Set all required tensors except mean
    const auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({16, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dy(dyTensor);
    const auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({16, 64, 32, 32});
    xTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_x(xTensor);
    const auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 32, 32});
    scaleTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_scale(scaleTensor);
    const auto invVarianceTensor = std::make_shared<TensorAttributes>();
    invVarianceTensor->set_dim({16, 1, 1, 1});
    invVarianceTensor->set_stride({1, 1, 1, 1});
    attrs.set_inv_variance(invVarianceTensor);
    const auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_dim({16, 64, 32, 32});
    dxTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dx(dxTensor);
    const auto dscaleTensor = std::make_shared<TensorAttributes>();
    dscaleTensor->set_dim({1, 64, 32, 32});
    dscaleTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dscale(dscaleTensor);
    const auto dbiasTensor = std::make_shared<TensorAttributes>();
    dbiasTensor->set_dim({1, 64, 32, 32});
    dbiasTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dbias(dbiasTensor);
    attrs.set_normalized_dim_count(3);

    const GraphAttributes graphAttributes;
    const LayernormBackwardNode node(std::move(attrs), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestLayernormBackwardNode, PreValidateNodeMissingInverseVarianceTensor)
{
    LayernormBackwardAttributes attrs;

    // Set all required tensors except inverse variance
    const auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({16, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dy(dyTensor);
    const auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({16, 64, 32, 32});
    xTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_x(xTensor);
    const auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 32, 32});
    scaleTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_scale(scaleTensor);
    const auto meanTensor = std::make_shared<TensorAttributes>();
    meanTensor->set_dim({16, 1, 1, 1});
    meanTensor->set_stride({1, 1, 1, 1});
    attrs.set_mean(meanTensor);
    const auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_dim({16, 64, 32, 32});
    dxTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dx(dxTensor);
    const auto dscaleTensor = std::make_shared<TensorAttributes>();
    dscaleTensor->set_dim({1, 64, 32, 32});
    dscaleTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dscale(dscaleTensor);
    const auto dbiasTensor = std::make_shared<TensorAttributes>();
    dbiasTensor->set_dim({1, 64, 32, 32});
    dbiasTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dbias(dbiasTensor);
    attrs.set_normalized_dim_count(3);

    const GraphAttributes graphAttributes;
    const LayernormBackwardNode node(std::move(attrs), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestLayernormBackwardNode, PreValidateNodeMissingMeanAndInverseVarianceTensor)
{
    LayernormBackwardAttributes attrs;

    // Set all required tensors except inverse variance
    const auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({16, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dy(dyTensor);
    const auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({16, 64, 32, 32});
    xTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_x(xTensor);
    const auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 32, 32});
    scaleTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_scale(scaleTensor);
    const auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_dim({16, 64, 32, 32});
    dxTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dx(dxTensor);
    const auto dscaleTensor = std::make_shared<TensorAttributes>();
    dscaleTensor->set_dim({1, 64, 32, 32});
    dscaleTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dscale(dscaleTensor);
    const auto dbiasTensor = std::make_shared<TensorAttributes>();
    dbiasTensor->set_dim({1, 64, 32, 32});
    dbiasTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dbias(dbiasTensor);
    attrs.set_normalized_dim_count(3);

    const GraphAttributes graphAttributes;
    const LayernormBackwardNode node(std::move(attrs), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK);
}

TEST(TestLayernormBackwardNode, PreValidateNodeMissingDxTensor)
{
    LayernormBackwardAttributes attrs;

    // Set all required tensors except dx
    const auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({16, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dy(dyTensor);
    const auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({16, 64, 32, 32});
    xTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_x(xTensor);
    const auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 32, 32});
    scaleTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_scale(scaleTensor);
    const auto meanTensor = std::make_shared<TensorAttributes>();
    meanTensor->set_dim({16, 1, 1, 1});
    meanTensor->set_stride({1, 1, 1, 1});
    attrs.set_mean(meanTensor);
    const auto invVarianceTensor = std::make_shared<TensorAttributes>();
    invVarianceTensor->set_dim({16, 1, 1, 1});
    invVarianceTensor->set_stride({1, 1, 1, 1});
    attrs.set_inv_variance(invVarianceTensor);
    const auto dscaleTensor = std::make_shared<TensorAttributes>();
    dscaleTensor->set_dim({1, 64, 32, 32});
    dscaleTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dscale(dscaleTensor);
    const auto dbiasTensor = std::make_shared<TensorAttributes>();
    dbiasTensor->set_dim({1, 64, 32, 32});
    dbiasTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dbias(dbiasTensor);
    attrs.set_normalized_dim_count(3);

    // dx tensor is missing
    const GraphAttributes graphAttributes;
    const LayernormBackwardNode node(std::move(attrs), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestLayernormBackwardNode, PreValidateNodeMissingDscaleTensor)
{
    LayernormBackwardAttributes attrs;

    // Set all required tensors except dscale
    const auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({16, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dy(dyTensor);
    const auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({16, 64, 32, 32});
    xTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_x(xTensor);
    const auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 32, 32});
    scaleTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_scale(scaleTensor);
    const auto meanTensor = std::make_shared<TensorAttributes>();
    meanTensor->set_dim({16, 1, 1, 1});
    meanTensor->set_stride({1, 1, 1, 1});
    attrs.set_mean(meanTensor);
    const auto invVarianceTensor = std::make_shared<TensorAttributes>();
    invVarianceTensor->set_dim({16, 1, 1, 1});
    invVarianceTensor->set_stride({1, 1, 1, 1});
    attrs.set_inv_variance(invVarianceTensor);
    const auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_dim({16, 64, 32, 32});
    dxTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dx(dxTensor);
    const auto dbiasTensor = std::make_shared<TensorAttributes>();
    dbiasTensor->set_dim({1, 64, 32, 32});
    dbiasTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dbias(dbiasTensor);
    attrs.set_normalized_dim_count(3);

    // dscale tensor is missing
    const GraphAttributes graphAttributes;
    const LayernormBackwardNode node(std::move(attrs), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestLayernormBackwardNode, PreValidateNodeMissingDbiasTensor)
{
    LayernormBackwardAttributes attrs;

    // Set all required tensors except dbias
    const auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({16, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dy(dyTensor);
    const auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({16, 64, 32, 32});
    xTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_x(xTensor);
    const auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 32, 32});
    scaleTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_scale(scaleTensor);
    const auto meanTensor = std::make_shared<TensorAttributes>();
    meanTensor->set_dim({16, 1, 1, 1});
    meanTensor->set_stride({1, 1, 1, 1});
    attrs.set_mean(meanTensor);
    const auto invVarianceTensor = std::make_shared<TensorAttributes>();
    invVarianceTensor->set_dim({16, 1, 1, 1});
    invVarianceTensor->set_stride({1, 1, 1, 1});
    attrs.set_inv_variance(invVarianceTensor);
    const auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_dim({16, 64, 32, 32});
    dxTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dx(dxTensor);
    const auto dscaleTensor = std::make_shared<TensorAttributes>();
    dscaleTensor->set_dim({1, 64, 32, 32});
    dscaleTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dscale(dscaleTensor);
    attrs.set_normalized_dim_count(3);

    // dbias tensor is missing
    const GraphAttributes graphAttributes;
    const LayernormBackwardNode node(std::move(attrs), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestLayernormBackwardNode, PreValidateNodeAllValuesSet)
{
    auto attrs = createValidAttributes();

    const GraphAttributes graphAttributes;
    const LayernormBackwardNode node(std::move(attrs), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

// --- PreValidateNode: mismatched tensors ---

TEST(TestLayernormBackwardNode, PreValidateNodeMismatchingDyTensor)
{
    LayernormBackwardAttributes attrs;

    // Mismatch between dy and x
    const auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({32, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dy(dyTensor);
    const auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({16, 64, 32, 32});
    xTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_x(xTensor);
    const auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 32, 32});
    scaleTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_scale(scaleTensor);
    const auto meanTensor = std::make_shared<TensorAttributes>();
    meanTensor->set_dim({16, 1, 1, 1});
    meanTensor->set_stride({1, 1, 1, 1});
    attrs.set_mean(meanTensor);
    const auto invVarianceTensor = std::make_shared<TensorAttributes>();
    invVarianceTensor->set_dim({16, 1, 1, 1});
    invVarianceTensor->set_stride({1, 1, 1, 1});
    attrs.set_inv_variance(invVarianceTensor);
    const auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_dim({16, 64, 32, 32});
    dxTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dx(dxTensor);
    const auto dscaleTensor = std::make_shared<TensorAttributes>();
    dscaleTensor->set_dim({1, 64, 32, 32});
    dscaleTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dscale(dscaleTensor);
    const auto dbiasTensor = std::make_shared<TensorAttributes>();
    dbiasTensor->set_dim({1, 64, 32, 32});
    dbiasTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dbias(dbiasTensor);
    attrs.set_normalized_dim_count(3);

    const GraphAttributes graphAttributes;
    const LayernormBackwardNode node(std::move(attrs), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestLayernormBackwardNode, PreValidateNodeMismatchingDxTensor)
{
    LayernormBackwardAttributes attrs;

    // Mismatch between dx and x
    const auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({16, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dy(dyTensor);
    const auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({16, 64, 32, 32});
    xTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_x(xTensor);
    const auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 32, 32});
    scaleTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_scale(scaleTensor);
    const auto meanTensor = std::make_shared<TensorAttributes>();
    meanTensor->set_dim({16, 1, 1, 1});
    meanTensor->set_stride({1, 1, 1, 1});
    attrs.set_mean(meanTensor);
    const auto invVarianceTensor = std::make_shared<TensorAttributes>();
    invVarianceTensor->set_dim({16, 1, 1, 1});
    invVarianceTensor->set_stride({1, 1, 1, 1});
    attrs.set_inv_variance(invVarianceTensor);
    const auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_dim({32, 64, 32, 32});
    dxTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dx(dxTensor);
    const auto dscaleTensor = std::make_shared<TensorAttributes>();
    dscaleTensor->set_dim({1, 64, 32, 32});
    dscaleTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dscale(dscaleTensor);
    const auto dbiasTensor = std::make_shared<TensorAttributes>();
    dbiasTensor->set_dim({1, 64, 32, 32});
    dbiasTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dbias(dbiasTensor);
    attrs.set_normalized_dim_count(3);

    const GraphAttributes graphAttributes;
    const LayernormBackwardNode node(std::move(attrs), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestLayernormBackwardNode, PreValidateNodeMismatchingMeanAndInverseVarianceTensor)
{
    LayernormBackwardAttributes attrs;

    // Mismatch between mean and inverse variance
    const auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({16, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dy(dyTensor);
    const auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({16, 64, 32, 32});
    xTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_x(xTensor);
    const auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 32, 32});
    scaleTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_scale(scaleTensor);
    const auto meanTensor = std::make_shared<TensorAttributes>();
    meanTensor->set_dim({16, 1, 1, 1});
    meanTensor->set_stride({1, 1, 1, 1});
    attrs.set_mean(meanTensor);
    const auto invVarianceTensor = std::make_shared<TensorAttributes>();
    invVarianceTensor->set_dim({32, 1, 1, 1});
    invVarianceTensor->set_stride({1, 1, 1, 1});
    attrs.set_inv_variance(invVarianceTensor);
    const auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_dim({16, 64, 32, 32});
    dxTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dx(dxTensor);
    const auto dscaleTensor = std::make_shared<TensorAttributes>();
    dscaleTensor->set_dim({1, 64, 32, 32});
    dscaleTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dscale(dscaleTensor);
    const auto dbiasTensor = std::make_shared<TensorAttributes>();
    dbiasTensor->set_dim({1, 64, 32, 32});
    dbiasTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dbias(dbiasTensor);
    attrs.set_normalized_dim_count(3);

    const GraphAttributes graphAttributes;
    const LayernormBackwardNode node(std::move(attrs), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestLayernormBackwardNode, PreValidateNodeMismatchingDscaleTensor)
{
    LayernormBackwardAttributes attrs;

    // Mismatch between dscale and scale
    const auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({16, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dy(dyTensor);
    const auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({16, 64, 32, 32});
    xTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_x(xTensor);
    const auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 32, 32});
    scaleTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_scale(scaleTensor);
    const auto meanTensor = std::make_shared<TensorAttributes>();
    meanTensor->set_dim({16, 1, 1, 1});
    meanTensor->set_stride({1, 1, 1, 1});
    attrs.set_mean(meanTensor);
    const auto invVarianceTensor = std::make_shared<TensorAttributes>();
    invVarianceTensor->set_dim({16, 1, 1, 1});
    invVarianceTensor->set_stride({1, 1, 1, 1});
    attrs.set_inv_variance(invVarianceTensor);
    const auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_dim({16, 64, 32, 32});
    dxTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dx(dxTensor);
    const auto dscaleTensor = std::make_shared<TensorAttributes>();
    dscaleTensor->set_dim({1, 64, 64, 32});
    dscaleTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dscale(dscaleTensor);
    const auto dbiasTensor = std::make_shared<TensorAttributes>();
    dbiasTensor->set_dim({1, 64, 32, 32});
    dbiasTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dbias(dbiasTensor);
    attrs.set_normalized_dim_count(3);

    const GraphAttributes graphAttributes;
    const LayernormBackwardNode node(std::move(attrs), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestLayernormBackwardNode, PreValidateNodeMismatchingDbiasTensor)
{
    LayernormBackwardAttributes attrs;

    // Mismatch between dy and x
    const auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({16, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dy(dyTensor);
    const auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({16, 64, 32, 32});
    xTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_x(xTensor);
    const auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 32, 32});
    scaleTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_scale(scaleTensor);
    const auto meanTensor = std::make_shared<TensorAttributes>();
    meanTensor->set_dim({16, 1, 1, 1});
    meanTensor->set_stride({1, 1, 1, 1});
    attrs.set_mean(meanTensor);
    const auto invVarianceTensor = std::make_shared<TensorAttributes>();
    invVarianceTensor->set_dim({16, 1, 1, 1});
    invVarianceTensor->set_stride({1, 1, 1, 1});
    attrs.set_inv_variance(invVarianceTensor);
    const auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_dim({16, 64, 32, 32});
    dxTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dx(dxTensor);
    const auto dscaleTensor = std::make_shared<TensorAttributes>();
    dscaleTensor->set_dim({1, 64, 32, 32});
    dscaleTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dscale(dscaleTensor);
    const auto dbiasTensor = std::make_shared<TensorAttributes>();
    dbiasTensor->set_dim({1, 64, 64, 32});
    dbiasTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dbias(dbiasTensor);
    attrs.set_normalized_dim_count(3);

    const GraphAttributes graphAttributes;
    const LayernormBackwardNode node(std::move(attrs), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestLayernormBackwardNode, PreValidateNodeNegativeNormalizedDimCount)
{
    LayernormBackwardAttributes attrs;

    // Mismatch between dy and x
    const auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({16, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dy(dyTensor);
    const auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({16, 64, 32, 32});
    xTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_x(xTensor);
    const auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 32, 32});
    scaleTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_scale(scaleTensor);
    const auto meanTensor = std::make_shared<TensorAttributes>();
    meanTensor->set_dim({16, 1, 1, 1});
    meanTensor->set_stride({1, 1, 1, 1});
    attrs.set_mean(meanTensor);
    const auto invVarianceTensor = std::make_shared<TensorAttributes>();
    invVarianceTensor->set_dim({16, 1, 1, 1});
    invVarianceTensor->set_stride({1, 1, 1, 1});
    attrs.set_inv_variance(invVarianceTensor);
    const auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_dim({16, 64, 32, 32});
    dxTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dx(dxTensor);
    const auto dscaleTensor = std::make_shared<TensorAttributes>();
    dscaleTensor->set_dim({1, 64, 32, 32});
    dscaleTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dscale(dscaleTensor);
    const auto dbiasTensor = std::make_shared<TensorAttributes>();
    dbiasTensor->set_dim({1, 64, 32, 32});
    dbiasTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dbias(dbiasTensor);
    attrs.set_normalized_dim_count(-1);

    const GraphAttributes graphAttributes;
    const LayernormBackwardNode node(std::move(attrs), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestLayernormBackwardNode, PreValidateNodeTooLargeNormalizedDimCount)
{
    LayernormBackwardAttributes attrs;

    // Mismatch between dy and x
    const auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({16, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dy(dyTensor);
    const auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({16, 64, 32, 32});
    xTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_x(xTensor);
    const auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 32, 32});
    scaleTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_scale(scaleTensor);
    const auto meanTensor = std::make_shared<TensorAttributes>();
    meanTensor->set_dim({16, 1, 1, 1});
    meanTensor->set_stride({1, 1, 1, 1});
    attrs.set_mean(meanTensor);
    const auto invVarianceTensor = std::make_shared<TensorAttributes>();
    invVarianceTensor->set_dim({16, 1, 1, 1});
    invVarianceTensor->set_stride({1, 1, 1, 1});
    attrs.set_inv_variance(invVarianceTensor);
    const auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_dim({16, 64, 32, 32});
    dxTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dx(dxTensor);
    const auto dscaleTensor = std::make_shared<TensorAttributes>();
    dscaleTensor->set_dim({1, 64, 32, 32});
    dscaleTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dscale(dscaleTensor);
    const auto dbiasTensor = std::make_shared<TensorAttributes>();
    dbiasTensor->set_dim({1, 64, 32, 32});
    dbiasTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dbias(dbiasTensor);
    attrs.set_normalized_dim_count(5);

    const GraphAttributes graphAttributes;
    const LayernormBackwardNode node(std::move(attrs), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestLayernormBackwardNode, PreValidateNodeAcceptsOnePaddedScaleWithBatchOnlyStats)
{
    auto attrs = createShapedAttributes({16, 64, 32, 32}, {1, 64, 32, 32}, {16});
    attrs.set_normalized_dim_count(3);

    const GraphAttributes graphAttributes;
    const LayernormBackwardNode node(std::move(attrs), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

TEST(TestLayernormBackwardNode, PreValidateNodeInvalidOnePaddedBatchDims)
{
    LayernormBackwardAttributes attrs;

    // Mismatch between dy and x
    const auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({16, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dy(dyTensor);
    const auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({16, 64, 32, 32});
    xTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_x(xTensor);
    const auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 32, 32});
    scaleTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_scale(scaleTensor);
    const auto meanTensor = std::make_shared<TensorAttributes>();
    meanTensor->set_dim({16, 1, 1, 32});
    meanTensor->set_stride({32, 32, 32, 1});
    attrs.set_mean(meanTensor);
    const auto invVarianceTensor = std::make_shared<TensorAttributes>();
    invVarianceTensor->set_dim({16, 1, 1, 32});
    invVarianceTensor->set_stride({32, 32, 32, 1});
    attrs.set_inv_variance(invVarianceTensor);
    const auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_dim({16, 64, 32, 32});
    dxTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dx(dxTensor);
    const auto dscaleTensor = std::make_shared<TensorAttributes>();
    dscaleTensor->set_dim({1, 64, 32, 32});
    dscaleTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dscale(dscaleTensor);
    const auto dbiasTensor = std::make_shared<TensorAttributes>();
    dbiasTensor->set_dim({1, 64, 32, 32});
    dbiasTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dbias(dbiasTensor);
    attrs.set_normalized_dim_count(3);

    const GraphAttributes graphAttributes;
    const LayernormBackwardNode node(std::move(attrs), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestLayernormBackwardNode, PreValidateNodeInvalidOnePaddedNormalizedDims)
{
    LayernormBackwardAttributes attrs;

    // Mismatch between dy and x
    const auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({16, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dy(dyTensor);
    const auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({16, 64, 32, 32});
    xTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_x(xTensor);
    const auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({16, 64, 32, 32});
    scaleTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_scale(scaleTensor);
    const auto meanTensor = std::make_shared<TensorAttributes>();
    meanTensor->set_dim({16, 1, 1, 1});
    meanTensor->set_stride({1, 1, 1, 1});
    attrs.set_mean(meanTensor);
    const auto invVarianceTensor = std::make_shared<TensorAttributes>();
    invVarianceTensor->set_dim({16, 1, 1, 1});
    invVarianceTensor->set_stride({1, 1, 1, 1});
    attrs.set_inv_variance(invVarianceTensor);
    const auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_dim({16, 64, 32, 32});
    dxTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dx(dxTensor);
    const auto dscaleTensor = std::make_shared<TensorAttributes>();
    dscaleTensor->set_dim({16, 64, 32, 32});
    dscaleTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dscale(dscaleTensor);
    const auto dbiasTensor = std::make_shared<TensorAttributes>();
    dbiasTensor->set_dim({16, 64, 32, 32});
    dbiasTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dbias(dbiasTensor);
    attrs.set_normalized_dim_count(3);

    const GraphAttributes graphAttributes;
    const LayernormBackwardNode node(std::move(attrs), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestLayernormBackwardNode, PreValidateNodeInvalidOnePaddedBatchDimsLength)
{
    LayernormBackwardAttributes attrs;

    // Mismatch between dy and x
    const auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({16, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dy(dyTensor);
    const auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({16, 64, 32, 32});
    xTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_x(xTensor);
    const auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 32, 32});
    scaleTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_scale(scaleTensor);
    const auto meanTensor = std::make_shared<TensorAttributes>();
    meanTensor->set_dim({16, 1, 1});
    meanTensor->set_stride({1, 1, 1});
    attrs.set_mean(meanTensor);
    const auto invVarianceTensor = std::make_shared<TensorAttributes>();
    invVarianceTensor->set_dim({16, 1, 1});
    invVarianceTensor->set_stride({1, 1, 1});
    attrs.set_inv_variance(invVarianceTensor);
    const auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_dim({16, 64, 32, 32});
    dxTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dx(dxTensor);
    const auto dscaleTensor = std::make_shared<TensorAttributes>();
    dscaleTensor->set_dim({1, 64, 32, 32});
    dscaleTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dscale(dscaleTensor);
    const auto dbiasTensor = std::make_shared<TensorAttributes>();
    dbiasTensor->set_dim({1, 64, 32, 32});
    dbiasTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dbias(dbiasTensor);
    attrs.set_normalized_dim_count(3);

    const GraphAttributes graphAttributes;
    const LayernormBackwardNode node(std::move(attrs), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestLayernormBackwardNode, PreValidateNodeAcceptsRankReducedScaleWithOnePaddedStats)
{
    auto attrs = createShapedAttributes({16, 64, 32, 32}, {64, 32, 32}, {16, 1, 1, 1});
    attrs.set_normalized_dim_count(3);

    const GraphAttributes graphAttributes;
    const LayernormBackwardNode node(std::move(attrs), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

TEST(TestLayernormBackwardNode, PreValidateNodeAcceptsRankReducedScaleBelowExplicitCount)
{
    // x [2,3,5,7] with scale [1,5,7] (rank 3) and an explicit normalized_dim_count of 2: the
    // shared helper's affine check passes a scale rank between normalized_dim_count and rank(x),
    // and backward no longer requires the scale to hold exactly k dims on top of that.
    auto attrs = createShapedAttributes({2, 3, 5, 7}, {1, 5, 7}, {2, 3, 1, 1});
    attrs.set_normalized_dim_count(2);

    const GraphAttributes graphAttributes;
    const LayernormBackwardNode node(std::move(attrs), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

TEST(TestLayernormBackwardNode, PreValidateNodeAcceptsExplicitCountEqualToRankWithFullRankScale)
{
    // normalized_dim_count equal to rank(x) is the accepting boundary of
    // deriveLayernormNormalizedDimCount's explicitCount > rank check.
    auto attrs = createShapedAttributes({2, 3, 5, 7}, {2, 3, 5, 7}, {1, 1, 1, 1});
    attrs.set_normalized_dim_count(4);

    const GraphAttributes graphAttributes;
    const LayernormBackwardNode node(std::move(attrs), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

TEST(TestLayernormBackwardNode, PreValidateNodeRejectsStatsRankAboveX)
{
    // x [N, C, H, W] with scale [C, H, W]: statistics have rank 1 or 4
    const GraphAttributes graphAttributes;
    const LayernormBackwardNode node(
        createShapedAttributes({16, 64, 32, 32}, {64, 32, 32}, {16, 1, 1, 1, 1}), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE) << error.err_msg;
}

TEST(TestLayernormBackwardNode, PreValidateNodeRejectsStatsBatchMismatchWithoutCount)
{
    const GraphAttributes graphAttributes;
    const LayernormBackwardNode node(
        createShapedAttributes({4, 6, 8, 8}, {1, 6, 8, 8}, {3, 1, 1, 1}), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE) << error.err_msg;
}

TEST(TestLayernormBackwardNode, PreValidateNodeRejectsStatsBatchMismatchWithCount)
{
    auto attrs = createShapedAttributes({4, 6, 8, 8}, {1, 6, 8, 8}, {3, 1, 1, 1});
    attrs.set_normalized_dim_count(3);

    const GraphAttributes graphAttributes;
    const LayernormBackwardNode node(std::move(attrs), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE) << error.err_msg;
}

TEST(TestLayernormBackwardNode, PreValidateNodeRejectsScaleTrailingDimMismatch)
{
    const GraphAttributes graphAttributes;
    const LayernormBackwardNode node(
        createShapedAttributes({4, 6, 8, 8}, {1, 6, 8, 4}, {4, 1, 1, 1}), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE) << error.err_msg;
}

TEST(TestLayernormBackwardNode, PreValidateNodeRejectsAllOnesScaleOnNonUnitX)
{
    const GraphAttributes graphAttributes;
    const LayernormBackwardNode node(
        createShapedAttributes({4, 6, 8, 8}, {1, 1, 1, 1}, {4, 1, 1, 1}), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE) << error.err_msg;
}

TEST(TestLayernormBackwardNode, PreValidateNodeAcceptsRankReducedAllOnesScale)
{
    // x and scale all-ones, scale rank below rank(x): the scale equals x's own trailing
    // dims (both all 1s), so normalizedDimCount takes the scale's rank directly (rule 2 of
    // deriveLayernormNormalizedDimCount). Mirrors test_sdk's layer-norm backward graphs (x
    // [1,1,1,1], scale [1,1,1], mean/inv_variance [1]), which develop accepted via
    // k = scale rank = 3; the PR's right-aligned-only rule previously defaulted this case
    // to k = 1 and rejected the rank-1 saved-stats tensors ("Mean tensor rank must be 3
    // (batch dims of x) or 4 (rank of x), got 1").
    const GraphAttributes graphAttributes;
    const LayernormBackwardNode node(createShapedAttributes({1, 1, 1, 1}, {1, 1, 1}, {1}),
                                     graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

TEST(TestLayernormBackwardNode,
     PreValidateNodeAcceptsRankReducedScaleEqualToXTrailingDimsWithBatchStats)
{
    // x [2, 1, 14, 14] with scale/dscale/dbias [1, 14, 14]: the scale equals x's own
    // trailing 3 dims, so normalizedDimCount takes the scale's rank directly (rule 2),
    // k = 3, and the saved mean/inv_variance [2] (batch dims only) are valid. The
    // right-aligned-only rule previously inferred k = 2 (the first non-1 padded dim is at
    // index 2) and rejected these rank-1 saved statistics.
    const GraphAttributes graphAttributes;
    const LayernormBackwardNode node(createShapedAttributes({2, 1, 14, 14}, {1, 14, 14}, {2}),
                                     graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

TEST(TestLayernormBackwardNode, PreValidateNodeRejectsNormalizedDimCountConflictingWithScale)
{
    // A count of 1 leaves the scale's non-1 channel dim before the normalized dims
    auto attrs = createShapedAttributes({4, 8, 1, 1}, {1, 8, 1, 1}, {4, 1, 1, 1});
    attrs.set_normalized_dim_count(1);

    const GraphAttributes graphAttributes;
    const LayernormBackwardNode node(std::move(attrs), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE) << error.err_msg;
    EXPECT_NE(error.err_msg.find("normalized_dim_count 1"), std::string::npos) << error.err_msg;
}

TEST(TestLayernormBackwardNode, NormalizedDimCountSpansUnitDimsAfterFirstNormalizedDim)
{
    struct Case
    {
        std::vector<int64_t> xDims;
        std::vector<int64_t> scaleDims;
        std::vector<int64_t> statsDims;
    };
    const std::vector<Case> cases = {{{4, 16, 1, 1}, {1, 16, 1, 1}, {4, 1, 1, 1}},
                                     {{2, 3, 1, 129}, {1, 3, 1, 129}, {2, 1, 1, 1}}};

    for(const auto& testCase : cases)
    {
        const GraphAttributes graphAttributes;
        LayernormBackwardNode node(
            createShapedAttributes(testCase.xDims, testCase.scaleDims, testCase.statsDims),
            graphAttributes);

        auto error = node.pre_validate_node();
        ASSERT_EQ(error.code, error_code_t::OK) << error.err_msg;

        error = node.infer_properties_node();
        ASSERT_EQ(error.code, error_code_t::OK) << error.err_msg;
        EXPECT_EQ(node.attributes.get_normalized_dim_count(), 3);

        // A later validate (as in build_operation_graph) sees the derived count
        error = node.pre_validate_node();
        EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
    }
}

// --- InferPropertiesNode ---

TEST(TestLayernormBackwardNode, InferPropertiesNode)
{
    auto attrs = createValidAttributes();

    const GraphAttributes graphAttributes;
    LayernormBackwardNode node(std::move(attrs), graphAttributes);

    const auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
    EXPECT_EQ(node.attributes.get_normalized_dim_count(), 3);
}

// --- InferMinimalPropertiesNode ---
TEST(TestLayernormBackwardNode, InferMinimalPropertiesNode)
{
    auto attrs = createMinimalAttributes();

    const GraphAttributes graphAttributes;
    LayernormBackwardNode node(std::move(attrs), graphAttributes);

    const auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
    EXPECT_EQ(node.attributes.get_dx()->get_dim(), node.attributes.get_x()->get_dim());
    EXPECT_EQ(node.attributes.get_dx()->get_stride(), node.attributes.get_x()->get_stride());
    EXPECT_EQ(node.attributes.get_dscale()->get_dim(), node.attributes.get_scale()->get_dim());
    EXPECT_EQ(node.attributes.get_dscale()->get_stride(),
              node.attributes.get_scale()->get_stride());
    EXPECT_EQ(node.attributes.get_dbias()->get_dim(), node.attributes.get_scale()->get_dim());
    EXPECT_EQ(node.attributes.get_dbias()->get_stride(), node.attributes.get_scale()->get_stride());
    EXPECT_EQ(node.attributes.get_normalized_dim_count(), 3);
}

// --- GatherHipdnnTensors ---

TEST(TestLayernormBackwardNode, GatherHipdnnTensor)
{
    LayernormBackwardAttributes attrs;

    const auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_uid(10).set_name("DyTensor");
    attrs.set_dy(dyTensor);
    const auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(11).set_name("XTensor");
    attrs.set_x(xTensor);
    const auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_uid(12).set_name("ScaleTensor");
    attrs.set_scale(scaleTensor);
    const auto meanTensor = std::make_shared<TensorAttributes>();
    meanTensor->set_uid(13).set_name("MeanTensor");
    attrs.set_mean(meanTensor);
    const auto invVarianceTensor = std::make_shared<TensorAttributes>();
    invVarianceTensor->set_uid(14).set_name("InvVarianceTensor");
    attrs.set_inv_variance(invVarianceTensor);
    const auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_uid(16).set_name("DxTensor");
    attrs.set_dx(dxTensor);
    const auto dscaleTensor = std::make_shared<TensorAttributes>();
    dscaleTensor->set_uid(17).set_name("DscaleTensor");
    attrs.set_dscale(dscaleTensor);
    const auto dbiasTensor = std::make_shared<TensorAttributes>();
    dbiasTensor->set_uid(18).set_name("DbiasTensor");
    attrs.set_dbias(dbiasTensor);

    const GraphAttributes graphAttributes;
    const LayernormBackwardNode node(std::move(attrs), graphAttributes);

    hipdnn_frontend::detail::OrderedTensorSet allTensors;

    node.gather_hipdnn_tensors(allTensors);

    EXPECT_TRUE(allTensors.find(dyTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(xTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(scaleTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(meanTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(invVarianceTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(dxTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(dscaleTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(dbiasTensor) != allTensors.end());
    EXPECT_EQ(allTensors.size(), 8u);
}
