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
// Helper: create a tensor with given dims and optional strides
std::shared_ptr<TensorAttributes> makeTensor(const std::vector<int64_t>& dims,
                                             const std::vector<int64_t>& strides = {})
{
    auto t = std::make_shared<TensorAttributes>();
    t->set_dim(dims);
    if(!strides.empty())
    {
        t->set_stride(strides);
    }
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

// ============================================================================
// Pre-validation: Dimension and Scalar Validation Tests
// ============================================================================

TEST(TestLayernormFpropNode, PreValidateFailsXWithNoDimensions)
{
    LayernormFpropAttributes attrs;
    attrs.set_x(std::make_shared<TensorAttributes>()); // No dimensions set

    auto epsilon = std::make_shared<TensorAttributes>();
    epsilon->set_dim({1});
    epsilon->set_value(1e-5f);
    attrs.set_epsilon(epsilon);
    attrs.set_y(std::make_shared<TensorAttributes>());

    GraphAttributes graphAttrs;
    LayernormFpropNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestLayernormFpropNode, PreValidateFailsEpsilonNotScalar)
{
    auto x = makeTensor({32, 512});
    LayernormFpropAttributes attrs;
    attrs.set_x(x);
    attrs.set_y(std::make_shared<TensorAttributes>());

    // Epsilon with more than one element
    auto epsilon = std::make_shared<TensorAttributes>();
    epsilon->set_dim({2});
    epsilon->set_value(1e-5f);
    attrs.set_epsilon(epsilon);

    GraphAttributes graphAttrs;
    LayernormFpropNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestLayernormFpropNode, PreValidateFailsEpsilonNotPassByValue)
{
    auto x = makeTensor({32, 512});
    LayernormFpropAttributes attrs;
    attrs.set_x(x);
    attrs.set_y(std::make_shared<TensorAttributes>());

    // Epsilon with correct dim but not pass-by-value (no set_value call)
    auto epsilon = std::make_shared<TensorAttributes>();
    epsilon->set_dim({1});
    attrs.set_epsilon(epsilon);

    GraphAttributes graphAttrs;
    LayernormFpropNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestLayernormFpropNode, PreValidateFailsEpsilonWithNoDimensions)
{
    auto x = makeTensor({32, 512});
    LayernormFpropAttributes attrs;
    attrs.set_x(x);
    attrs.set_y(std::make_shared<TensorAttributes>());

    // Epsilon set but no dimensions
    attrs.set_epsilon(std::make_shared<TensorAttributes>());

    GraphAttributes graphAttrs;
    LayernormFpropNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestLayernormFpropNode, PreValidateFailsXYShapeMismatch)
{
    auto x = makeTensor({32, 512});
    LayernormFpropAttributes attrs;
    attrs.set_x(x);

    auto epsilon = std::make_shared<TensorAttributes>();
    epsilon->set_dim({1});
    epsilon->set_value(1e-5f);
    attrs.set_epsilon(epsilon);

    // Y has different shape than X
    auto y = makeTensor({32, 256});
    attrs.set_y(y);

    GraphAttributes graphAttrs;
    LayernormFpropNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestLayernormFpropNode, PreValidateFailsScaleWithNoDimensions)
{
    auto x = makeTensor({32, 512});
    auto attrs = makeMinimalAttrs(x);
    attrs.set_scale(std::make_shared<TensorAttributes>()); // No dimensions

    GraphAttributes graphAttrs;
    LayernormFpropNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestLayernormFpropNode, PreValidateFailsBiasWithNoDimensions)
{
    auto x = makeTensor({32, 512});
    auto attrs = makeMinimalAttrs(x);
    attrs.set_bias(std::make_shared<TensorAttributes>()); // No dimensions

    GraphAttributes graphAttrs;
    LayernormFpropNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

// ============================================================================
// Infer Properties: Error and Edge-Case Tests
// ============================================================================

TEST(TestLayernormFpropNode, InferPropertiesFailsMissingX)
{
    LayernormFpropAttributes attrs;
    attrs.set_y(std::make_shared<TensorAttributes>());

    auto epsilon = std::make_shared<TensorAttributes>();
    epsilon->set_dim({1});
    epsilon->set_value(1e-5f);
    attrs.set_epsilon(epsilon);

    GraphAttributes graphAttrs;
    LayernormFpropNode node(std::move(attrs), graphAttrs);
    auto err = node.infer_properties_node();
    EXPECT_EQ(err.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestLayernormFpropNode, InferPropertiesFailsMissingY)
{
    auto x = makeTensor({32, 512});
    LayernormFpropAttributes attrs;
    attrs.set_x(x);

    auto epsilon = std::make_shared<TensorAttributes>();
    epsilon->set_dim({1});
    epsilon->set_value(1e-5f);
    attrs.set_epsilon(epsilon);

    GraphAttributes graphAttrs;
    LayernormFpropNode node(std::move(attrs), graphAttrs);
    auto err = node.infer_properties_node();
    EXPECT_EQ(err.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestLayernormFpropNode, InferPropertiesCopiesStridesFromX)
{
    // When x has strides set and y does not, y should get x's strides
    auto x = makeTensor({32, 512}, {512, 1});
    auto attrs = makeMinimalAttrs(x);
    auto y = attrs.get_y();

    GraphAttributes graphAttrs;
    LayernormFpropNode node(std::move(attrs), graphAttrs);
    auto err = node.infer_properties_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;

    EXPECT_EQ(y->get_dim(), (std::vector<int64_t>{32, 512}));
    EXPECT_EQ(y->get_stride(), (std::vector<int64_t>{512, 1}));
}

TEST(TestLayernormFpropNode, InferPropertiesMeanStrideFromXStrideOrder)
{
    // When x has strides, mean stride should be inferred from x's stride order
    auto x = makeTensor({32, 512}, {512, 1});
    auto attrs = makeMinimalAttrs(x);

    auto mean = std::make_shared<TensorAttributes>();
    attrs.set_mean(mean);

    GraphAttributes graphAttrs;
    LayernormFpropNode node(std::move(attrs), graphAttrs);
    auto err = node.infer_properties_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;

    EXPECT_EQ(mean->get_dim(), (std::vector<int64_t>{1}));
    EXPECT_FALSE(mean->get_stride().empty());
}

TEST(TestLayernormFpropNode, InferPropertiesRstdStrideFromXStrideOrder)
{
    // When x has strides, rstd stride should be inferred from x's stride order
    auto x = makeTensor({32, 512}, {512, 1});
    auto attrs = makeMinimalAttrs(x);

    auto rstd = std::make_shared<TensorAttributes>();
    attrs.set_rstd(rstd);

    GraphAttributes graphAttrs;
    LayernormFpropNode node(std::move(attrs), graphAttrs);
    auto err = node.infer_properties_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;

    EXPECT_EQ(rstd->get_dim(), (std::vector<int64_t>{1}));
    EXPECT_FALSE(rstd->get_stride().empty());
}

TEST(TestLayernormFpropNode, InferPropertiesPreservesExplicitMeanDims)
{
    // Mean with dims already set should not be overwritten
    auto x = makeTensor({32, 512});
    auto attrs = makeMinimalAttrs(x);

    auto mean = std::make_shared<TensorAttributes>();
    mean->set_dim({32});
    attrs.set_mean(mean);

    GraphAttributes graphAttrs;
    LayernormFpropNode node(std::move(attrs), graphAttrs);
    auto err = node.infer_properties_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;

    // Dims should remain as set by user
    EXPECT_EQ(mean->get_dim(), (std::vector<int64_t>{32}));
    // Strides should be inferred
    EXPECT_FALSE(mean->get_stride().empty());
}

TEST(TestLayernormFpropNode, InferPropertiesPreservesExplicitStatStrides)
{
    // Stats tensor with strides already set should not be overwritten
    auto x = makeTensor({32, 512});
    auto attrs = makeMinimalAttrs(x);

    auto mean = std::make_shared<TensorAttributes>();
    mean->set_dim({1});
    mean->set_stride({1});
    attrs.set_mean(mean);

    GraphAttributes graphAttrs;
    LayernormFpropNode node(std::move(attrs), graphAttrs);
    auto err = node.infer_properties_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;

    EXPECT_EQ(mean->get_dim(), (std::vector<int64_t>{1}));
    EXPECT_EQ(mean->get_stride(), (std::vector<int64_t>{1}));
}

TEST(TestLayernormFpropNode, InferPropertiesSetsBothMeanAndRstd)
{
    // Both mean and rstd should be inferred when set
    auto x = makeTensor({32, 512}, {512, 1});
    auto attrs = makeMinimalAttrs(x);

    auto mean = std::make_shared<TensorAttributes>();
    auto rstd = std::make_shared<TensorAttributes>();
    attrs.set_mean(mean);
    attrs.set_rstd(rstd);

    GraphAttributes graphAttrs;
    LayernormFpropNode node(std::move(attrs), graphAttrs);
    auto err = node.infer_properties_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;

    EXPECT_FALSE(mean->get_dim().empty());
    EXPECT_FALSE(mean->get_stride().empty());
    EXPECT_FALSE(rstd->get_dim().empty());
    EXPECT_FALSE(rstd->get_stride().empty());
}

// ============================================================================
// Gather Tensors Test
// ============================================================================

TEST(TestLayernormFpropNode, GatherHipdnnTensors)
{
    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1).set_name("X");

    auto y = std::make_shared<TensorAttributes>();
    y->set_uid(2).set_name("Y");

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_uid(3).set_name("Scale");

    auto bias = std::make_shared<TensorAttributes>();
    bias->set_uid(4).set_name("Bias");

    auto epsilon = std::make_shared<TensorAttributes>();
    epsilon->set_uid(5).set_name("Epsilon").set_value(1e-5f);

    auto mean = std::make_shared<TensorAttributes>();
    mean->set_uid(6).set_name("Mean");

    auto rstd = std::make_shared<TensorAttributes>();
    rstd->set_uid(7).set_name("Rstd");

    LayernormFpropAttributes attrs;
    attrs.set_x(x);
    attrs.set_y(y);
    attrs.set_scale(scale);
    attrs.set_bias(bias);
    attrs.set_epsilon(epsilon);
    attrs.set_mean(mean);
    attrs.set_rstd(rstd);

    GraphAttributes graphAttrs;
    LayernormFpropNode node(std::move(attrs), graphAttrs);

    std::unordered_set<std::shared_ptr<TensorAttributes>> allTensors;
    node.gather_hipdnn_tensors(allTensors);

    EXPECT_TRUE(allTensors.find(x) != allTensors.end());
    EXPECT_TRUE(allTensors.find(y) != allTensors.end());
    EXPECT_TRUE(allTensors.find(scale) != allTensors.end());
    EXPECT_TRUE(allTensors.find(bias) != allTensors.end());
    EXPECT_TRUE(allTensors.find(epsilon) != allTensors.end());
    EXPECT_TRUE(allTensors.find(mean) != allTensors.end());
    EXPECT_TRUE(allTensors.find(rstd) != allTensors.end());
    EXPECT_EQ(allTensors.size(), 7u);
}

TEST(TestLayernormFpropNode, GatherHipdnnTensorsMinimal)
{
    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1);

    auto y = std::make_shared<TensorAttributes>();
    y->set_uid(2);

    auto epsilon = std::make_shared<TensorAttributes>();
    epsilon->set_uid(3).set_value(1e-5f);

    LayernormFpropAttributes attrs;
    attrs.set_x(x);
    attrs.set_y(y);
    attrs.set_epsilon(epsilon);

    GraphAttributes graphAttrs;
    LayernormFpropNode node(std::move(attrs), graphAttrs);

    std::unordered_set<std::shared_ptr<TensorAttributes>> allTensors;
    node.gather_hipdnn_tensors(allTensors);

    EXPECT_TRUE(allTensors.find(x) != allTensors.end());
    EXPECT_TRUE(allTensors.find(y) != allTensors.end());
    EXPECT_TRUE(allTensors.find(epsilon) != allTensors.end());
    EXPECT_EQ(allTensors.size(), 3u);
}
