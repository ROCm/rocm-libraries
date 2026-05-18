// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/ResampleBwdAttributes.hpp>
#include <hipdnn_frontend/node/ResampleBwdNode.hpp>

#include <memory>
#include <unordered_set>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

// --- Helper: create fully configured attributes for a valid node ---
namespace
{

ResampleBwdAttributes createValidAttributes()
{
    ResampleBwdAttributes attrs;

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 3, 16, 16});
    dyTensor->set_stride({768, 256, 16, 1});
    attrs.set_dy(dyTensor);
    auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_dim({1, 3, 32, 32});
    dxTensor->set_stride({3072, 1024, 32, 1});
    attrs.set_dx(dxTensor);

    attrs.set_pre_padding({1, 1});
    attrs.set_post_padding({1, 1});
    attrs.set_stride({2, 2});
    attrs.set_window({3, 3});

    return attrs;
}

} // namespace

// --- GetNodeType ---

TEST(TestResampleBwdNode, GetNodeTypeReturnsResampleBwd)
{
    const GraphAttributes graphAttrs;
    const ResampleBwdNode node(ResampleBwdAttributes{}, graphAttrs);
    EXPECT_EQ(node.getNodeType(), NodeType::RESAMPLE_BWD);
}

// --- PreValidateNode (success case) ---

TEST(TestResampleBwdNode, PreValidateNode)
{
    auto attrs = createValidAttributes();

    const GraphAttributes graphAttributes;
    const ResampleBwdNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

// --- PreValidateNode: missing required tensors ---

TEST(TestResampleBwdNode, PreValidateNodeMissingDyTensor)
{
    ResampleBwdAttributes attrs;

    // Set all required tensors except dy
    auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_dim({1, 3, 32, 32});
    dxTensor->set_stride({3072, 1024, 32, 1});
    attrs.set_dx(dxTensor);

    attrs.set_pre_padding({1, 1});
    attrs.set_post_padding({1, 1});
    attrs.set_stride({2, 2});
    attrs.set_window({3, 3});

    // dy tensor is missing
    const GraphAttributes graphAttributes;
    const ResampleBwdNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestResampleBwdNode, PreValidateNodeMissingDxTensor)
{
    ResampleBwdAttributes attrs;

    // Set all required tensors except dx
    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 3, 16, 16});
    dyTensor->set_stride({768, 256, 16, 1});
    attrs.set_dy(dyTensor);

    attrs.set_pre_padding({1, 1});
    attrs.set_post_padding({1, 1});
    attrs.set_stride({2, 2});
    attrs.set_window({3, 3});

    // dx tensor is missing
    const GraphAttributes graphAttributes;
    const ResampleBwdNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestResampleBwdNode, PreValidateNodeMissingParameters)
{
    ResampleBwdAttributes attrs;

    // Set all required tensors but not data field parameters
    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 3, 16, 16});
    dyTensor->set_stride({768, 256, 16, 1});
    attrs.set_dy(dyTensor);
    auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_dim({1, 3, 32, 32});
    dxTensor->set_stride({3072, 1024, 32, 1});
    attrs.set_dx(dxTensor);

    // Required parameters are missing
    const GraphAttributes graphAttributes;
    const ResampleBwdNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestResampleBwdNode, PreValidateNodeAllValuesSet)
{
    auto attrs = createValidAttributes();

    const GraphAttributes graphAttributes;
    const ResampleBwdNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

// --- InferPropertiesNode ---

TEST(TestResampleBwdNode, InferPropertiesNode)
{
    auto attrs = createValidAttributes();

    const GraphAttributes graphAttributes;
    ResampleBwdNode node(std::move(attrs), graphAttributes);

    auto error = node.infer_properties_node();
    // Stub implementation: verify the method can be called without error
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

// --- GatherHipdnnTensors ---

TEST(TestResampleBwdNode, GatherHipdnnTensor)
{
    ResampleBwdAttributes attrs;

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_uid(50).set_name("DyTensor");
    attrs.set_dy(dyTensor);
    auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_uid(51).set_name("DxTensor");
    attrs.set_dx(dxTensor);

    attrs.set_pre_padding({1, 1});
    attrs.set_post_padding({1, 1});
    attrs.set_stride({2, 2});
    attrs.set_window({3, 3});

    const GraphAttributes graphAttributes;
    const ResampleBwdNode node(std::move(attrs), graphAttributes);

    std::unordered_set<std::shared_ptr<TensorAttributes>> allTensors;

    node.gather_hipdnn_tensors(allTensors);

    EXPECT_TRUE(allTensors.find(dyTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(dxTensor) != allTensors.end());
    EXPECT_EQ(allTensors.size(), 2u);
}
