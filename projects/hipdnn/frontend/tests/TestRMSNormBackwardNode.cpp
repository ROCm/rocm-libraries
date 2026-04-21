// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/RMSNormBackwardAttributes.hpp>
#include <hipdnn_frontend/node/RMSNormBackwardNode.hpp>

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_data_sdk/data_objects/graph_generated.h>

#include <memory>
#include <unordered_set>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

// --- Helper: create fully configured attributes for a valid node ---
namespace
{

RMSNormBackwardAttributes createValidAttributes()
{
    RMSNormBackwardAttributes attrs;

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dy(dyTensor);
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 64, 32, 32});
    xTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_x(xTensor);
    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 1, 1});
    scaleTensor->set_stride({64, 1, 1, 1});
    attrs.set_scale(scaleTensor);
    auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_dim({1, 64, 32, 32});
    dxTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dx(dxTensor);
    auto dscaleTensor = std::make_shared<TensorAttributes>();
    dscaleTensor->set_dim({1, 64, 1, 1});
    dscaleTensor->set_stride({64, 1, 1, 1});
    attrs.set_dscale(dscaleTensor);

    return attrs;
}

} // namespace

// --- GetNodeType ---

TEST(TestRMSNormBackwardNode, GetNodeTypeReturns)
{
    const GraphAttributes graphAttrs;
    const RMSNormBackwardNode node(RMSNormBackwardAttributes{}, graphAttrs);
    EXPECT_EQ(node.getNodeType(), NodeType::RMS_NORM_BACKWARD);
}

// --- PreValidateNode (success case) ---

TEST(TestRMSNormBackwardNode, PreValidateNode)
{
    auto attrs = createValidAttributes();

    const GraphAttributes graphAttributes;
    const RMSNormBackwardNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

// --- PreValidateNode: missing required tensors ---

TEST(TestRMSNormBackwardNode, PreValidateNodeMissingDyTensor)
{
    RMSNormBackwardAttributes attrs;

    // Set all required tensors except dy
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 64, 32, 32});
    xTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_x(xTensor);
    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 1, 1});
    scaleTensor->set_stride({64, 1, 1, 1});
    attrs.set_scale(scaleTensor);
    auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_dim({1, 64, 32, 32});
    dxTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dx(dxTensor);
    auto dscaleTensor = std::make_shared<TensorAttributes>();
    dscaleTensor->set_dim({1, 64, 1, 1});
    dscaleTensor->set_stride({64, 1, 1, 1});
    attrs.set_dscale(dscaleTensor);

    // dy tensor is missing
    const GraphAttributes graphAttributes;
    const RMSNormBackwardNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestRMSNormBackwardNode, PreValidateNodeMissingXTensor)
{
    RMSNormBackwardAttributes attrs;

    // Set all required tensors except x
    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dy(dyTensor);
    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 1, 1});
    scaleTensor->set_stride({64, 1, 1, 1});
    attrs.set_scale(scaleTensor);
    auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_dim({1, 64, 32, 32});
    dxTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dx(dxTensor);
    auto dscaleTensor = std::make_shared<TensorAttributes>();
    dscaleTensor->set_dim({1, 64, 1, 1});
    dscaleTensor->set_stride({64, 1, 1, 1});
    attrs.set_dscale(dscaleTensor);

    // x tensor is missing
    const GraphAttributes graphAttributes;
    const RMSNormBackwardNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestRMSNormBackwardNode, PreValidateNodeMissingScaleTensor)
{
    RMSNormBackwardAttributes attrs;

    // Set all required tensors except scale
    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dy(dyTensor);
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 64, 32, 32});
    xTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_x(xTensor);
    auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_dim({1, 64, 32, 32});
    dxTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dx(dxTensor);
    auto dscaleTensor = std::make_shared<TensorAttributes>();
    dscaleTensor->set_dim({1, 64, 1, 1});
    dscaleTensor->set_stride({64, 1, 1, 1});
    attrs.set_dscale(dscaleTensor);

    // scale tensor is missing
    const GraphAttributes graphAttributes;
    const RMSNormBackwardNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestRMSNormBackwardNode, PreValidateNodeMissingDxTensor)
{
    RMSNormBackwardAttributes attrs;

    // Set all required tensors except dx
    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dy(dyTensor);
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 64, 32, 32});
    xTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_x(xTensor);
    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 1, 1});
    scaleTensor->set_stride({64, 1, 1, 1});
    attrs.set_scale(scaleTensor);
    auto dscaleTensor = std::make_shared<TensorAttributes>();
    dscaleTensor->set_dim({1, 64, 1, 1});
    dscaleTensor->set_stride({64, 1, 1, 1});
    attrs.set_dscale(dscaleTensor);

    // dx tensor is missing
    const GraphAttributes graphAttributes;
    const RMSNormBackwardNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestRMSNormBackwardNode, PreValidateNodeMissingDscaleTensor)
{
    RMSNormBackwardAttributes attrs;

    // Set all required tensors except dscale
    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dy(dyTensor);
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 64, 32, 32});
    xTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_x(xTensor);
    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 1, 1});
    scaleTensor->set_stride({64, 1, 1, 1});
    attrs.set_scale(scaleTensor);
    auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_dim({1, 64, 32, 32});
    dxTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dx(dxTensor);

    // dscale tensor is missing
    const GraphAttributes graphAttributes;
    const RMSNormBackwardNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestRMSNormBackwardNode, PreValidateNodeAllValuesSet)
{
    auto attrs = createValidAttributes();

    const GraphAttributes graphAttributes;
    const RMSNormBackwardNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

// --- InferPropertiesNode ---

TEST(TestRMSNormBackwardNode, InferPropertiesNode)
{
    RMSNormBackwardAttributes attrs;
    attrs.set_dy(std::make_shared<TensorAttributes>());
    attrs.set_x(std::make_shared<TensorAttributes>());
    attrs.set_scale(std::make_shared<TensorAttributes>());
    attrs.set_dx(std::make_shared<TensorAttributes>());
    attrs.set_dscale(std::make_shared<TensorAttributes>());
    attrs.set_dbias(std::make_shared<TensorAttributes>());

    auto dyTensor = attrs.get_dy();
    dyTensor->set_uid(1)
        .set_name("DyTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({24, 12, 4, 1}); // NCHW layout

    auto scaleTensor = attrs.get_scale();
    scaleTensor->set_uid(2)
        .set_name("ScaleTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 1, 1})
        .set_stride({2, 1, 1, 1});

    auto dxTensor = attrs.get_dx();
    dxTensor->set_uid(3).set_name("DxTensor");

    auto dscaleTensor = attrs.get_dscale();
    dscaleTensor->set_uid(4).set_name("DscaleTensor");

    auto dbiasTensor = attrs.get_dbias();
    dbiasTensor->set_uid(5).set_name("DbiasTensor");

    const GraphAttributes graphAttributes;
    RMSNormBackwardNode node(std::move(attrs), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    EXPECT_EQ(dxTensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(dxTensor->get_stride(), (std::vector<int64_t>{24, 12, 4, 1}));

    EXPECT_EQ(dscaleTensor->get_dim(), (std::vector<int64_t>{1, 2, 1, 1}));
    EXPECT_EQ(dscaleTensor->get_stride(),
              (std::vector<int64_t>{2, 1, 1, 1})); // Inherits NCHW layout

    EXPECT_EQ(dbiasTensor->get_dim(), (std::vector<int64_t>{1, 2, 1, 1}));
    EXPECT_EQ(dbiasTensor->get_stride(),
              (std::vector<int64_t>{2, 1, 1, 1})); // Inherits NCHW layout
}

// --- PackNode ---

TEST(TestRMSNormBackwardNode, PackNode)
{
    RMSNormBackwardAttributes attrs;
    attrs.set_name("RMSNormBackwardNode");

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_uid(70).set_name("DyTensor").set_data_type(DataType_t::FLOAT);
    dyTensor->set_dim({1, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dy(dyTensor);

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(71).set_name("XTensor").set_data_type(DataType_t::FLOAT);
    xTensor->set_dim({1, 64, 32, 32});
    xTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_x(xTensor);

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_uid(72).set_name("ScaleTensor").set_data_type(DataType_t::FLOAT);
    scaleTensor->set_dim({1, 64, 1, 1});
    scaleTensor->set_stride({64, 1, 1, 1});
    attrs.set_scale(scaleTensor);

    auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_uid(74).set_name("DxTensor").set_data_type(DataType_t::FLOAT);
    dxTensor->set_dim({1, 64, 32, 32});
    dxTensor->set_stride({65536, 1024, 32, 1});
    attrs.set_dx(dxTensor);

    auto dscaleTensor = std::make_shared<TensorAttributes>();
    dscaleTensor->set_uid(75).set_name("DscaleTensor").set_data_type(DataType_t::FLOAT);
    dscaleTensor->set_dim({1, 64, 1, 1});
    dscaleTensor->set_stride({64, 1, 1, 1});
    attrs.set_dscale(dscaleTensor);

    const GraphAttributes graphAttributes;
    const RMSNormBackwardNode node(std::move(attrs), graphAttributes);

    flatbuffers::FlatBufferBuilder builder;
    auto offset = node.pack_node(builder);
    EXPECT_NE(offset.o, 0);

    builder.Finish(offset);
    auto bufferPointer = builder.GetBufferPointer();
    auto nodeFlatbuffer = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::Node>(bufferPointer);

    EXPECT_STREQ(nodeFlatbuffer->name()->c_str(), "RMSNormBackwardNode");
    EXPECT_EQ(nodeFlatbuffer->attributes_type(),
              hipdnn_data_sdk::data_objects::NodeAttributes::RMSNormBackwardAttributes);

    auto packedAttributes = nodeFlatbuffer->attributes_as_RMSNormBackwardAttributes();
    ASSERT_NE(packedAttributes, nullptr);

    // Verify tensor UIDs
    EXPECT_EQ(packedAttributes->dy_tensor_uid(), dyTensor->get_uid());
    EXPECT_EQ(packedAttributes->x_tensor_uid(), xTensor->get_uid());
    EXPECT_EQ(packedAttributes->scale_tensor_uid(), scaleTensor->get_uid());
    EXPECT_EQ(packedAttributes->dx_tensor_uid(), dxTensor->get_uid());
    EXPECT_EQ(packedAttributes->dscale_tensor_uid(), dscaleTensor->get_uid());

    // Verify data fields
}

// --- GatherHipdnnTensors ---

TEST(TestRMSNormBackwardNode, GatherHipdnnTensor)
{
    RMSNormBackwardAttributes attrs;

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_uid(70).set_name("DyTensor");
    attrs.set_dy(dyTensor);
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(71).set_name("XTensor");
    attrs.set_x(xTensor);
    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_uid(72).set_name("ScaleTensor");
    attrs.set_scale(scaleTensor);
    auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_uid(74).set_name("DxTensor");
    attrs.set_dx(dxTensor);
    auto dscaleTensor = std::make_shared<TensorAttributes>();
    dscaleTensor->set_uid(75).set_name("DscaleTensor");
    attrs.set_dscale(dscaleTensor);

    const GraphAttributes graphAttributes;
    const RMSNormBackwardNode node(std::move(attrs), graphAttributes);

    std::unordered_set<std::shared_ptr<TensorAttributes>> allTensors;

    node.gather_hipdnn_tensors(allTensors);

    EXPECT_TRUE(allTensors.find(dyTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(xTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(scaleTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(dxTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(dscaleTensor) != allTensors.end());
    EXPECT_EQ(allTensors.size(), 5u);
}
