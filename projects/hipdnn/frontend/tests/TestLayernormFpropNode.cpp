// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/LayernormFpropAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_frontend/node/LayernormFpropNode.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

namespace
{
// Helper: create a tensor with given dims
std::shared_ptr<TensorAttributes> makeTensor(const std::vector<int64_t>& dims)
{
    auto t = std::make_shared<TensorAttributes>();
    t->set_dim(dims);
    return t;
}

LayernormFpropAttributes makeMinimalAttrs(const std::shared_ptr<TensorAttributes>& x)
{
    LayernormFpropAttributes attrs;
    attrs.set_x(x);
    auto epsilon = std::make_shared<TensorAttributes>();
    epsilon->set_dim({1});
    epsilon->set_value(1e-5f);
    attrs.set_epsilon(epsilon);
    attrs.set_y(std::make_shared<TensorAttributes>());
    return attrs;
}
} // namespace

TEST(TestLayernormFpropNode, PreValidateSucceedsMinimal)
{
    // Simple 1D case: [N]
    auto x = makeTensor({10});
    auto attrs = makeMinimalAttrs(x);

    GraphAttributes graphAttrs;
    LayernormFpropNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;
}

TEST(TestLayernormFpropNode, PreValidateSucceeds2D)
{
    // 2D case: [Batch, Features]
    auto x = makeTensor({32, 512});
    auto attrs = makeMinimalAttrs(x);

    GraphAttributes graphAttrs;
    LayernormFpropNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;
}

TEST(TestLayernormFpropNode, PreValidateSucceeds4D)
{
    // 4D case: [Batch, Channels, Height, Width]
    auto x = makeTensor({2, 64, 28, 28});
    auto attrs = makeMinimalAttrs(x);

    GraphAttributes graphAttrs;
    LayernormFpropNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;
}

TEST(TestLayernormFpropNode, PreValidateSucceedsWithScale)
{
    auto x = makeTensor({32, 512});
    auto attrs = makeMinimalAttrs(x);

    // Scale matches the feature dimension
    auto scale = makeTensor({512});
    attrs.set_scale(scale);

    GraphAttributes graphAttrs;
    LayernormFpropNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;
}

TEST(TestLayernormFpropNode, PreValidateSucceedsWithBias)
{
    auto x = makeTensor({32, 512});
    auto attrs = makeMinimalAttrs(x);

    // Bias matches the feature dimension
    auto bias = makeTensor({512});
    attrs.set_bias(bias);

    GraphAttributes graphAttrs;
    LayernormFpropNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;
}

TEST(TestLayernormFpropNode, PreValidateSucceedsWithScaleAndBias)
{
    auto x = makeTensor({32, 512});
    auto attrs = makeMinimalAttrs(x);

    auto scale = makeTensor({512});
    auto bias = makeTensor({512});
    attrs.set_scale(scale);
    attrs.set_bias(bias);

    GraphAttributes graphAttrs;
    LayernormFpropNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;
}

TEST(TestLayernormFpropNode, PreValidateFailsMissingX)
{
    LayernormFpropAttributes attrs;
    auto epsilon = std::make_shared<TensorAttributes>();
    epsilon->set_dim({1});
    attrs.set_epsilon(epsilon);
    attrs.set_y(std::make_shared<TensorAttributes>());

    GraphAttributes graphAttrs;
    LayernormFpropNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestLayernormFpropNode, PreValidateFailsMissingY)
{
    LayernormFpropAttributes attrs;
    auto x = makeTensor({32, 512});
    auto epsilon = std::make_shared<TensorAttributes>();
    epsilon->set_dim({1});
    attrs.set_x(x);
    attrs.set_epsilon(epsilon);

    GraphAttributes graphAttrs;
    LayernormFpropNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestLayernormFpropNode, PreValidateFailsMissingEpsilon)
{
    LayernormFpropAttributes attrs;
    auto x = makeTensor({32, 512});
    attrs.set_x(x);
    attrs.set_y(std::make_shared<TensorAttributes>());

    GraphAttributes graphAttrs;
    LayernormFpropNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestLayernormFpropNode, PreValidateFailsScaleBiasMismatch)
{
    auto x = makeTensor({32, 512});
    auto attrs = makeMinimalAttrs(x);

    // Scale and bias have different shapes
    auto scale = makeTensor({512});
    auto bias = makeTensor({256}); // Mismatch
    attrs.set_scale(scale);
    attrs.set_bias(bias);

    GraphAttributes graphAttrs;
    LayernormFpropNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestLayernormFpropNode, InferPropertiesSetsOutputShape)
{
    auto x = makeTensor({32, 512});
    auto attrs = makeMinimalAttrs(x);
    auto y = attrs.get_y();

    GraphAttributes graphAttrs;
    LayernormFpropNode node(std::move(attrs), graphAttrs);
    auto err = node.infer_properties_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;

    auto dims = y->get_dim();
    ASSERT_EQ(dims.size(), 2u);
    EXPECT_EQ(dims[0], 32);
    EXPECT_EQ(dims[1], 512);
}

TEST(TestLayernormFpropNode, InferPropertiesSetsOutputStrides)
{
    auto x = makeTensor({32, 512});
    auto attrs = makeMinimalAttrs(x);
    auto y = attrs.get_y();

    GraphAttributes graphAttrs;
    LayernormFpropNode node(std::move(attrs), graphAttrs);
    auto err = node.infer_properties_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;

    auto strides = y->get_stride();
    ASSERT_EQ(strides.size(), 2u);
    // Row-major strides for [32, 512]: [512, 1]
    EXPECT_EQ(strides[1], 1);
    EXPECT_EQ(strides[0], 512);
}

TEST(TestLayernormFpropNode, InferPropertiesSetsMeanShape)
{
    auto x = makeTensor({32, 512});
    auto attrs = makeMinimalAttrs(x);

    auto mean = std::make_shared<TensorAttributes>();
    attrs.set_mean(mean);

    GraphAttributes graphAttrs;
    LayernormFpropNode node(std::move(attrs), graphAttrs);
    auto err = node.infer_properties_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;

    // Mean should be inferred (simplified to scalar)
    auto dims = mean->get_dim();
    EXPECT_FALSE(dims.empty());
}

TEST(TestLayernormFpropNode, InferPropertiesSetsRstdShape)
{
    auto x = makeTensor({32, 512});
    auto attrs = makeMinimalAttrs(x);

    auto rstd = std::make_shared<TensorAttributes>();
    attrs.set_rstd(rstd);

    GraphAttributes graphAttrs;
    LayernormFpropNode node(std::move(attrs), graphAttrs);
    auto err = node.infer_properties_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;

    // Rstd should be inferred (simplified to scalar)
    auto dims = rstd->get_dim();
    EXPECT_FALSE(dims.empty());
}

TEST(TestLayernormFpropNode, InferPropertiesPreservesExplicitOutputShape)
{
    // If Y dims are already set, they should not be overwritten
    auto x = makeTensor({32, 512});

    LayernormFpropAttributes attrs;
    attrs.set_x(x);
    auto epsilon = std::make_shared<TensorAttributes>();
    epsilon->set_dim({1});
    attrs.set_epsilon(epsilon);

    auto y = std::make_shared<TensorAttributes>();
    y->set_dim({32, 512});
    y->set_stride({512, 1});
    attrs.set_y(y);

    GraphAttributes graphAttrs;
    LayernormFpropNode node(std::move(attrs), graphAttrs);
    auto err = node.infer_properties_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;

    // Dims should remain unchanged
    EXPECT_EQ(y->get_dim(), (std::vector<int64_t>{32, 512}));
    // Strides should remain unchanged (they were already set)
    EXPECT_EQ(y->get_stride(), (std::vector<int64_t>{512, 1}));
}

TEST(TestLayernormFpropNode, PackNode)
{
    auto x = makeTensor({32, 512});
    auto attrs = makeMinimalAttrs(x);
    attrs.set_name("TestLayerNorm");

    GraphAttributes graphAttrs;
    LayernormFpropNode node(std::move(attrs), graphAttrs);

    flatbuffers::FlatBufferBuilder builder;
    auto packed = node.pack_node(builder);
    builder.Finish(packed);

    auto buf = builder.GetBufferPointer();
    auto fbNode = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::Node>(buf);

    EXPECT_EQ(fbNode->name()->str(), "TestLayerNorm");
    EXPECT_EQ(fbNode->attributes_type(),
              hipdnn_data_sdk::data_objects::NodeAttributes::LayernormFpropAttributes);
}
