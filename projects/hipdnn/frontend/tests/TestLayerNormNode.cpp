// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "hipdnn_data_sdk/utilities/Constants.hpp"
#include <gtest/gtest.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/LayernormAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_frontend/node/LayerNormNode.hpp>

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

LayernormAttributes makeMinimalAttrs(const std::shared_ptr<TensorAttributes>& x)
{
    LayernormAttributes attrs;
    attrs.set_x(x);
    attrs.set_forward_phase(NormFwdPhase::INFERENCE);
    auto epsilon = std::make_shared<TensorAttributes>();
    epsilon->set_dim({1});
    epsilon->set_value(hipdnn_data_sdk::utilities::LAYERNORM_DEFAULT_EPSILON);
    attrs.set_epsilon(epsilon);
    attrs.set_y(std::make_shared<TensorAttributes>());

    // Scale and bias are required
    auto scale = makeTensor({x->get_dim().back()});
    auto bias = makeTensor({x->get_dim().back()});
    attrs.set_scale(scale);
    attrs.set_bias(bias);

    return attrs;
}
} // namespace

TEST(TestLayerNormNode, PreValidateSucceedsMinimal)
{
    // Simple 1D case: [N]
    auto x = makeTensor({10});
    auto attrs = makeMinimalAttrs(x);

    const GraphAttributes graphAttrs;
    const LayerNormNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;
}

TEST(TestLayerNormNode, PreValidateSucceeds2D)
{
    // 2D case: [Batch, Features]
    auto x = makeTensor({32, 512});
    auto attrs = makeMinimalAttrs(x);

    const GraphAttributes graphAttrs;
    const LayerNormNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;
}

TEST(TestLayerNormNode, PreValidateSucceeds4D)
{
    // 4D case: [Batch, Channels, Height, Width]
    auto x = makeTensor({2, 64, 28, 28});
    auto attrs = makeMinimalAttrs(x);

    const GraphAttributes graphAttrs;
    const LayerNormNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;
}

TEST(TestLayerNormNode, PreValidateSucceedsWithScaleAndBias)
{
    auto x = makeTensor({32, 512});
    auto attrs = makeMinimalAttrs(x);

    auto scale = makeTensor({512});
    auto bias = makeTensor({512});
    attrs.set_scale(scale);
    attrs.set_bias(bias);

    const GraphAttributes graphAttrs;
    const LayerNormNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;
}

TEST(TestLayerNormNode, PreValidateFailsForwardPhaseNotSet)
{
    auto x = makeTensor({32, 512});
    LayernormAttributes attrs;
    attrs.set_x(x);
    auto epsilon = std::make_shared<TensorAttributes>();
    epsilon->set_dim({1});
    epsilon->set_value(hipdnn_data_sdk::utilities::LAYERNORM_DEFAULT_EPSILON);
    attrs.set_epsilon(epsilon);
    attrs.set_y(std::make_shared<TensorAttributes>());
    attrs.set_scale(makeTensor({512}));
    attrs.set_bias(makeTensor({512}));
    // forward_phase intentionally NOT set

    const GraphAttributes graphAttrs;
    const LayerNormNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestLayerNormNode, PreValidateFailsMissingX)
{
    LayernormAttributes attrs;
    attrs.set_forward_phase(NormFwdPhase::INFERENCE);
    auto epsilon = std::make_shared<TensorAttributes>();
    epsilon->set_dim({1});
    epsilon->set_value(hipdnn_data_sdk::utilities::LAYERNORM_DEFAULT_EPSILON);
    attrs.set_epsilon(epsilon);
    attrs.set_y(std::make_shared<TensorAttributes>());
    attrs.set_scale(makeTensor({512}));
    attrs.set_bias(makeTensor({512}));

    const GraphAttributes graphAttrs;
    const LayerNormNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestLayerNormNode, PreValidateFailsMissingY)
{
    LayernormAttributes attrs;
    attrs.set_forward_phase(NormFwdPhase::INFERENCE);
    auto x = makeTensor({32, 512});
    auto epsilon = std::make_shared<TensorAttributes>();
    epsilon->set_dim({1});
    epsilon->set_value(hipdnn_data_sdk::utilities::LAYERNORM_DEFAULT_EPSILON);
    attrs.set_x(x);
    attrs.set_epsilon(epsilon);
    attrs.set_scale(makeTensor({512}));
    attrs.set_bias(makeTensor({512}));

    const GraphAttributes graphAttrs;
    const LayerNormNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestLayerNormNode, PreValidateFailsMissingEpsilon)
{
    LayernormAttributes attrs;
    attrs.set_forward_phase(NormFwdPhase::INFERENCE);
    auto x = makeTensor({32, 512});
    attrs.set_x(x);
    attrs.set_y(std::make_shared<TensorAttributes>());
    attrs.set_scale(makeTensor({512}));
    attrs.set_bias(makeTensor({512}));

    const GraphAttributes graphAttrs;
    const LayerNormNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestLayerNormNode, PreValidateFailsMissingScale)
{
    LayernormAttributes attrs;
    attrs.set_forward_phase(NormFwdPhase::INFERENCE);
    auto x = makeTensor({32, 512});
    auto epsilon = std::make_shared<TensorAttributes>();
    epsilon->set_dim({1});
    epsilon->set_value(hipdnn_data_sdk::utilities::LAYERNORM_DEFAULT_EPSILON);
    attrs.set_x(x);
    attrs.set_epsilon(epsilon);
    attrs.set_y(std::make_shared<TensorAttributes>());
    attrs.set_bias(makeTensor({512}));

    const GraphAttributes graphAttrs;
    const LayerNormNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestLayerNormNode, PreValidateFailsMissingBias)
{
    LayernormAttributes attrs;
    attrs.set_forward_phase(NormFwdPhase::INFERENCE);
    auto x = makeTensor({32, 512});
    auto epsilon = std::make_shared<TensorAttributes>();
    epsilon->set_dim({1});
    epsilon->set_value(hipdnn_data_sdk::utilities::LAYERNORM_DEFAULT_EPSILON);
    attrs.set_x(x);
    attrs.set_epsilon(epsilon);
    attrs.set_y(std::make_shared<TensorAttributes>());
    attrs.set_scale(makeTensor({512}));

    const GraphAttributes graphAttrs;
    const LayerNormNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestLayerNormNode, PreValidateFailsScaleBiasMismatch)
{
    auto x = makeTensor({32, 512});
    auto attrs = makeMinimalAttrs(x);

    // Scale and bias have different shapes
    auto scale = makeTensor({512});
    auto bias = makeTensor({256}); // Mismatch
    attrs.set_scale(scale);
    attrs.set_bias(bias);

    const GraphAttributes graphAttrs;
    const LayerNormNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestLayerNormNode, InferPropertiesSetsOutputShape)
{
    auto x = makeTensor({32, 512});
    auto attrs = makeMinimalAttrs(x);
    auto y = attrs.get_y();

    const GraphAttributes graphAttrs;
    LayerNormNode node(std::move(attrs), graphAttrs);
    auto err = node.infer_properties_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;

    auto dims = y->get_dim();
    ASSERT_EQ(dims.size(), 2u);
    EXPECT_EQ(dims[0], 32);
    EXPECT_EQ(dims[1], 512);
}

TEST(TestLayerNormNode, InferPropertiesSetsOutputStrides)
{
    auto x = makeTensor({32, 512});
    auto attrs = makeMinimalAttrs(x);
    auto y = attrs.get_y();

    const GraphAttributes graphAttrs;
    LayerNormNode node(std::move(attrs), graphAttrs);
    auto err = node.infer_properties_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;

    auto strides = y->get_stride();
    ASSERT_EQ(strides.size(), 2u);
    // Row-major strides for [32, 512]: [512, 1]
    EXPECT_EQ(strides[1], 1);
    EXPECT_EQ(strides[0], 512);
}

TEST(TestLayerNormNode, InferPropertiesInfersScaleDimsFromX)
{
    // If scale dims are empty, they should be inferred from X's normalized dims
    auto x = makeTensor({32, 512}, {512, 1});

    LayernormAttributes attrs;
    attrs.set_x(x);
    auto epsilon = std::make_shared<TensorAttributes>();
    epsilon->set_dim({1});
    epsilon->set_value(hipdnn_data_sdk::utilities::LAYERNORM_DEFAULT_EPSILON);
    attrs.set_epsilon(epsilon);
    attrs.set_y(std::make_shared<TensorAttributes>());

    auto scale = std::make_shared<TensorAttributes>(); // No dims set
    auto bias = std::make_shared<TensorAttributes>(); // No dims set
    attrs.set_scale(scale);
    attrs.set_bias(bias);

    const GraphAttributes graphAttrs;
    LayerNormNode node(std::move(attrs), graphAttrs);
    auto err = node.infer_properties_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;

    // Scale should have dims = [1, 512] (same rank as X, batch dim = 1)
    EXPECT_EQ(scale->get_dim(), (std::vector<int64_t>{1, 512}));
    EXPECT_FALSE(scale->get_stride().empty());

    // Bias should also have dims = [1, 512]
    EXPECT_EQ(bias->get_dim(), (std::vector<int64_t>{1, 512}));
    EXPECT_FALSE(bias->get_stride().empty());
}

TEST(TestLayerNormNode, InferPropertiesInfersScaleDimsFromX4D)
{
    // For 4D input [N, C, H, W], normalized dims should be [C, H, W]
    auto x = makeTensor({2, 64, 28, 28}, {50176, 784, 28, 1});

    LayernormAttributes attrs;
    attrs.set_x(x);
    auto epsilon = std::make_shared<TensorAttributes>();
    epsilon->set_dim({1});
    epsilon->set_value(hipdnn_data_sdk::utilities::LAYERNORM_DEFAULT_EPSILON);
    attrs.set_epsilon(epsilon);
    attrs.set_y(std::make_shared<TensorAttributes>());

    auto scale = std::make_shared<TensorAttributes>();
    auto bias = std::make_shared<TensorAttributes>();
    attrs.set_scale(scale);
    attrs.set_bias(bias);

    const GraphAttributes graphAttrs;
    LayerNormNode node(std::move(attrs), graphAttrs);
    auto err = node.infer_properties_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;

    // Scale should have dims = [1, 64, 28, 28] (same rank as X, batch dim = 1)
    EXPECT_EQ(scale->get_dim(), (std::vector<int64_t>{1, 64, 28, 28}));
    EXPECT_FALSE(scale->get_stride().empty());
    EXPECT_EQ(bias->get_dim(), (std::vector<int64_t>{1, 64, 28, 28}));
}

TEST(TestLayerNormNode, InferPropertiesPreservesNhwcStrideOrder)
{
    // NHWC layout: strides = {H*W*C, 1, W*C, C} = {50176, 1, 1792, 64}
    // Stride order: C is innermost (0), then W (1), H (2), N outermost (3)
    auto x = makeTensor({2, 64, 28, 28}, {50176, 1, 1792, 64});

    LayernormAttributes attrs;
    attrs.set_forward_phase(NormFwdPhase::TRAINING);
    attrs.set_x(x);
    auto epsilon = std::make_shared<TensorAttributes>();
    epsilon->set_dim({1});
    epsilon->set_value(hipdnn_data_sdk::utilities::LAYERNORM_DEFAULT_EPSILON);
    attrs.set_epsilon(epsilon);
    attrs.set_y(std::make_shared<TensorAttributes>());

    auto scale = std::make_shared<TensorAttributes>();
    auto bias = std::make_shared<TensorAttributes>();
    attrs.set_scale(scale);
    attrs.set_bias(bias);

    auto mean = std::make_shared<TensorAttributes>();
    auto invVariance = std::make_shared<TensorAttributes>();
    attrs.set_mean(mean);
    attrs.set_inv_variance(invVariance);

    const GraphAttributes graphAttrs;
    LayerNormNode node(std::move(attrs), graphAttrs);
    auto err = node.infer_properties_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;

    // Y should preserve NHWC strides from X
    EXPECT_EQ(node.attributes.get_y()->get_stride(), (std::vector<int64_t>{50176, 1, 1792, 64}));

    // Scale/bias dims: [1, 64, 28, 28], strides should preserve NHWC order
    EXPECT_EQ(scale->get_dim(), (std::vector<int64_t>{1, 64, 28, 28}));
    EXPECT_EQ(scale->get_stride(), (std::vector<int64_t>{50176, 1, 1792, 64}));
    EXPECT_EQ(bias->get_dim(), (std::vector<int64_t>{1, 64, 28, 28}));
    EXPECT_EQ(bias->get_stride(), (std::vector<int64_t>{50176, 1, 1792, 64}));

    // Stats dims: [2, 1, 1, 1] (batch from X, normalized dims set to 1)
    EXPECT_EQ(mean->get_dim(), (std::vector<int64_t>{2, 1, 1, 1}));
    EXPECT_EQ(invVariance->get_dim(), (std::vector<int64_t>{2, 1, 1, 1}));
}

TEST(TestLayerNormNode, InferPropertiesNhwcScaleStridesMatchX)
{
    // Verify stride order is preserved for scale when X is NHWC
    // For scale dim [1, 64, 28, 28] with NHWC order, the stride values should
    // reflect channels-last layout even though batch dim is 1
    auto x = makeTensor({4, 32, 16, 16}, {8192, 1, 512, 32}); // NHWC

    LayernormAttributes attrs;
    attrs.set_forward_phase(NormFwdPhase::INFERENCE);
    attrs.set_x(x);
    auto epsilon = std::make_shared<TensorAttributes>();
    epsilon->set_dim({1});
    epsilon->set_value(hipdnn_data_sdk::utilities::LAYERNORM_DEFAULT_EPSILON);
    attrs.set_epsilon(epsilon);
    attrs.set_y(std::make_shared<TensorAttributes>());

    auto scale = std::make_shared<TensorAttributes>();
    auto bias = std::make_shared<TensorAttributes>();
    attrs.set_scale(scale);
    attrs.set_bias(bias);

    const GraphAttributes graphAttrs;
    LayerNormNode node(std::move(attrs), graphAttrs);
    auto err = node.infer_properties_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;

    // Scale dims: [1, 32, 16, 16]
    EXPECT_EQ(scale->get_dim(), (std::vector<int64_t>{1, 32, 16, 16}));
    // Scale strides should preserve NHWC: {H*W*C, 1, W*C, C} = {8192, 1, 512, 32}
    EXPECT_EQ(scale->get_stride(), (std::vector<int64_t>{8192, 1, 512, 32}));
    EXPECT_EQ(bias->get_dim(), (std::vector<int64_t>{1, 32, 16, 16}));
    EXPECT_EQ(bias->get_stride(), (std::vector<int64_t>{8192, 1, 512, 32}));
}

TEST(TestLayerNormNode, InferPropertiesSetsNormalizedDimCount)
{
    auto x = makeTensor({2, 3, 5, 7}, {105, 35, 7, 1});

    // normalizedDimCount = 3
    auto scale = makeTensor({1, 3, 5, 7}, {105, 35, 7, 1});
    auto bias = makeTensor({1, 3, 5, 7}, {105, 35, 7, 1});

    auto attrs = makeMinimalAttrs(x);
    attrs.set_scale(scale);
    attrs.set_bias(bias);

    const GraphAttributes graphAttrs;
    LayerNormNode node(std::move(attrs), graphAttrs);
    auto err = node.infer_properties_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;

    EXPECT_EQ(node.attributes.get_normalized_dim_count(), 3);
}

TEST(TestLayerNormNode, InferPropertiesSetsAmbiguousNormalizedDimCount)
{
    auto x = makeTensor({2, 1, 5, 7}, {35, 35, 7, 1});

    // normalizedDimCount = 2
    auto scale = makeTensor({1, 1, 5, 7}, {35, 35, 7, 1});
    auto bias = makeTensor({1, 1, 5, 7}, {35, 35, 7, 1});

    auto attrs = makeMinimalAttrs(x);
    attrs.set_scale(scale);
    attrs.set_bias(bias);

    const GraphAttributes graphAttrs;
    LayerNormNode node(std::move(attrs), graphAttrs);
    auto err = node.infer_properties_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;

    EXPECT_EQ(node.attributes.get_normalized_dim_count(), 2);
}

TEST(TestLayerNormNode, InferPropertiesSetsNormalizedDimCountWithAlternativeLayoutConvention)
{
    auto x = makeTensor({2, 3, 5, 7}, {105, 35, 7, 1});

    // normalizedDimCount = 3
    auto scale = makeTensor({3, 5, 7}, {35, 7, 1});
    auto bias = makeTensor({3, 5, 7}, {35, 7, 1});

    auto attrs = makeMinimalAttrs(x);
    attrs.set_scale(scale);
    attrs.set_bias(bias);

    const GraphAttributes graphAttrs;
    LayerNormNode node(std::move(attrs), graphAttrs);
    auto err = node.infer_properties_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;

    EXPECT_EQ(node.attributes.get_normalized_dim_count(), 3);
}

TEST(TestLayerNormNode, InferPropertiesStatsSkippedInInferenceMode)
{
    // Stats should NOT be inferred during inference phase, even if tensors are provided
    auto x = makeTensor({32, 512}, {512, 1});
    auto attrs = makeMinimalAttrs(x); // Sets INFERENCE mode

    auto mean = std::make_shared<TensorAttributes>();
    auto invVariance = std::make_shared<TensorAttributes>();
    attrs.set_mean(mean);
    attrs.set_inv_variance(invVariance);

    const GraphAttributes graphAttrs;
    LayerNormNode node(std::move(attrs), graphAttrs);
    auto err = node.infer_properties_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;

    // Stats dims should remain empty — inference mode skips stats inference
    EXPECT_TRUE(mean->get_dim().empty());
    EXPECT_TRUE(invVariance->get_dim().empty());
}

TEST(TestLayerNormNode, InferPropertiesSetsMeanShape)
{
    auto x = makeTensor({32, 512});
    auto attrs = makeMinimalAttrs(x);
    attrs.set_forward_phase(NormFwdPhase::TRAINING);

    auto mean = std::make_shared<TensorAttributes>();
    attrs.set_mean(mean);

    const GraphAttributes graphAttrs;
    LayerNormNode node(std::move(attrs), graphAttrs);
    auto err = node.infer_properties_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;

    // Mean should be inferred (simplified to scalar)
    auto dims = mean->get_dim();
    EXPECT_FALSE(dims.empty());
}

TEST(TestLayerNormNode, InferPropertiesSetsInvVarianceShape)
{
    auto x = makeTensor({32, 512});
    auto attrs = makeMinimalAttrs(x);
    attrs.set_forward_phase(NormFwdPhase::TRAINING);

    auto invVariance = std::make_shared<TensorAttributes>();
    attrs.set_inv_variance(invVariance);

    const GraphAttributes graphAttrs;
    LayerNormNode node(std::move(attrs), graphAttrs);
    auto err = node.infer_properties_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;

    auto dims = invVariance->get_dim();
    EXPECT_FALSE(dims.empty());
}

TEST(TestLayerNormNode, InferPropertiesPreservesExplicitOutputShape)
{
    // If Y dims are already set, they should not be overwritten
    auto x = makeTensor({32, 512});

    LayernormAttributes attrs;
    attrs.set_x(x);
    auto epsilon = std::make_shared<TensorAttributes>();
    epsilon->set_dim({1});
    epsilon->set_value(hipdnn_data_sdk::utilities::LAYERNORM_DEFAULT_EPSILON);
    attrs.set_epsilon(epsilon);
    attrs.set_scale(makeTensor({512}));
    attrs.set_bias(makeTensor({512}));

    auto y = std::make_shared<TensorAttributes>();
    y->set_dim({32, 512});
    y->set_stride({512, 1});
    attrs.set_y(y);

    const GraphAttributes graphAttrs;
    LayerNormNode node(std::move(attrs), graphAttrs);
    auto err = node.infer_properties_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;

    // Dims should remain unchanged
    EXPECT_EQ(y->get_dim(), (std::vector<int64_t>{32, 512}));
    // Strides should remain unchanged (they were already set)
    EXPECT_EQ(y->get_stride(), (std::vector<int64_t>{512, 1}));
}

// ============================================================================
// Pre-validation: Dimension and Scalar Validation Tests
// ============================================================================

TEST(TestLayerNormNode, PreValidateFailsXWithNoDimensions)
{
    LayernormAttributes attrs;
    attrs.set_forward_phase(NormFwdPhase::INFERENCE);
    attrs.set_x(std::make_shared<TensorAttributes>()); // No dimensions set

    auto epsilon = std::make_shared<TensorAttributes>();
    epsilon->set_dim({1});
    epsilon->set_value(hipdnn_data_sdk::utilities::LAYERNORM_DEFAULT_EPSILON);
    attrs.set_epsilon(epsilon);
    attrs.set_y(std::make_shared<TensorAttributes>());
    attrs.set_scale(makeTensor({512}));
    attrs.set_bias(makeTensor({512}));

    const GraphAttributes graphAttrs;
    const LayerNormNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestLayerNormNode, PreValidateFailsEpsilonNotScalar)
{
    auto x = makeTensor({32, 512});
    LayernormAttributes attrs;
    attrs.set_forward_phase(NormFwdPhase::INFERENCE);
    attrs.set_x(x);
    attrs.set_y(std::make_shared<TensorAttributes>());
    attrs.set_scale(makeTensor({512}));
    attrs.set_bias(makeTensor({512}));

    // Epsilon with more than one element.
    // set_value must be called before set_dim because set_value resets dim to {1}.
    auto epsilon = std::make_shared<TensorAttributes>();
    epsilon->set_value(hipdnn_data_sdk::utilities::LAYERNORM_DEFAULT_EPSILON);
    epsilon->set_dim({2});
    attrs.set_epsilon(epsilon);

    const GraphAttributes graphAttrs;
    const LayerNormNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestLayerNormNode, PreValidateFailsEpsilonNotPassByValue)
{
    auto x = makeTensor({32, 512});
    LayernormAttributes attrs;
    attrs.set_forward_phase(NormFwdPhase::INFERENCE);
    attrs.set_x(x);
    attrs.set_y(std::make_shared<TensorAttributes>());
    attrs.set_scale(makeTensor({512}));
    attrs.set_bias(makeTensor({512}));

    // Epsilon with correct dim but not pass-by-value (no set_value call)
    auto epsilon = std::make_shared<TensorAttributes>();
    epsilon->set_dim({1});
    attrs.set_epsilon(epsilon);

    const GraphAttributes graphAttrs;
    const LayerNormNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestLayerNormNode, PreValidateFailsEpsilonWithNoDimensions)
{
    auto x = makeTensor({32, 512});
    LayernormAttributes attrs;
    attrs.set_forward_phase(NormFwdPhase::INFERENCE);
    attrs.set_x(x);
    attrs.set_y(std::make_shared<TensorAttributes>());
    attrs.set_scale(makeTensor({512}));
    attrs.set_bias(makeTensor({512}));

    // Epsilon set but no dimensions
    attrs.set_epsilon(std::make_shared<TensorAttributes>());

    const GraphAttributes graphAttrs;
    const LayerNormNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestLayerNormNode, PreValidateFailsXYShapeMismatch)
{
    auto x = makeTensor({32, 512});
    LayernormAttributes attrs;
    attrs.set_forward_phase(NormFwdPhase::INFERENCE);
    attrs.set_x(x);
    attrs.set_scale(makeTensor({512}));
    attrs.set_bias(makeTensor({512}));

    auto epsilon = std::make_shared<TensorAttributes>();
    epsilon->set_dim({1});
    epsilon->set_value(hipdnn_data_sdk::utilities::LAYERNORM_DEFAULT_EPSILON);
    attrs.set_epsilon(epsilon);

    // Y has different shape than X
    auto y = makeTensor({32, 256});
    attrs.set_y(y);

    const GraphAttributes graphAttrs;
    const LayerNormNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestLayerNormNode, PreValidateFailsScaleWithNoDimensions)
{
    auto x = makeTensor({32, 512});
    auto attrs = makeMinimalAttrs(x);
    attrs.set_scale(std::make_shared<TensorAttributes>()); // No dimensions

    const GraphAttributes graphAttrs;
    const LayerNormNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestLayerNormNode, PreValidateFailsBiasWithNoDimensions)
{
    auto x = makeTensor({32, 512});
    auto attrs = makeMinimalAttrs(x);
    attrs.set_bias(std::make_shared<TensorAttributes>()); // No dimensions

    const GraphAttributes graphAttrs;
    const LayerNormNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

// ============================================================================
// Infer Properties: Error and Edge-Case Tests
// ============================================================================

TEST(TestLayerNormNode, InferPropertiesFailsMissingX)
{
    LayernormAttributes attrs;
    attrs.set_y(std::make_shared<TensorAttributes>());

    auto epsilon = std::make_shared<TensorAttributes>();
    epsilon->set_dim({1});
    epsilon->set_value(hipdnn_data_sdk::utilities::LAYERNORM_DEFAULT_EPSILON);
    attrs.set_epsilon(epsilon);
    attrs.set_scale(makeTensor({512}));
    attrs.set_bias(makeTensor({512}));

    const GraphAttributes graphAttrs;
    LayerNormNode node(std::move(attrs), graphAttrs);
    auto err = node.infer_properties_node();
    EXPECT_EQ(err.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestLayerNormNode, InferPropertiesFailsMissingY)
{
    auto x = makeTensor({32, 512});
    LayernormAttributes attrs;
    attrs.set_x(x);

    auto epsilon = std::make_shared<TensorAttributes>();
    epsilon->set_dim({1});
    epsilon->set_value(hipdnn_data_sdk::utilities::LAYERNORM_DEFAULT_EPSILON);
    attrs.set_epsilon(epsilon);
    attrs.set_scale(makeTensor({512}));
    attrs.set_bias(makeTensor({512}));

    const GraphAttributes graphAttrs;
    LayerNormNode node(std::move(attrs), graphAttrs);
    auto err = node.infer_properties_node();
    EXPECT_EQ(err.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestLayerNormNode, InferPropertiesCopiesStridesFromX)
{
    // When x has strides set and y does not, y should get x's strides
    auto x = makeTensor({32, 512}, {512, 1});
    auto attrs = makeMinimalAttrs(x);
    auto y = attrs.get_y();

    const GraphAttributes graphAttrs;
    LayerNormNode node(std::move(attrs), graphAttrs);
    auto err = node.infer_properties_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;

    EXPECT_EQ(y->get_dim(), (std::vector<int64_t>{32, 512}));
    EXPECT_EQ(y->get_stride(), (std::vector<int64_t>{512, 1}));
}

TEST(TestLayerNormNode, InferPropertiesMeanStrideFromXStrideOrder)
{
    // When x has strides, mean stride should be inferred from x's stride order
    auto x = makeTensor({32, 512}, {512, 1});
    auto attrs = makeMinimalAttrs(x);
    attrs.set_forward_phase(NormFwdPhase::TRAINING);

    auto mean = std::make_shared<TensorAttributes>();
    attrs.set_mean(mean);

    const GraphAttributes graphAttrs;
    LayerNormNode node(std::move(attrs), graphAttrs);
    auto err = node.infer_properties_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;

    // Stats shape: batch dims from X, normalized dims set to 1
    // For X=[32,512] with scale=[512], stats=[32, 1]
    EXPECT_EQ(mean->get_dim(), (std::vector<int64_t>{32, 1}));
    EXPECT_FALSE(mean->get_stride().empty());
}

TEST(TestLayerNormNode, InferPropertiesInvVarianceStrideFromXStrideOrder)
{
    // When x has strides, inv_variance stride should be inferred from x's stride order
    auto x = makeTensor({32, 512}, {512, 1});
    auto attrs = makeMinimalAttrs(x);
    attrs.set_forward_phase(NormFwdPhase::TRAINING);

    auto invVariance = std::make_shared<TensorAttributes>();
    attrs.set_inv_variance(invVariance);

    const GraphAttributes graphAttrs;
    LayerNormNode node(std::move(attrs), graphAttrs);
    auto err = node.infer_properties_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;

    // Stats shape: batch dims from X, normalized dims set to 1
    // For X=[32,512] with scale=[512], stats=[32, 1]
    EXPECT_EQ(invVariance->get_dim(), (std::vector<int64_t>{32, 1}));
    EXPECT_FALSE(invVariance->get_stride().empty());
}

TEST(TestLayerNormNode, InferPropertiesPreservesExplicitMeanDims)
{
    // Mean with dims already set should not be overwritten
    auto x = makeTensor({32, 512});
    auto attrs = makeMinimalAttrs(x);
    attrs.set_forward_phase(NormFwdPhase::TRAINING);

    auto mean = std::make_shared<TensorAttributes>();
    mean->set_dim({32});
    attrs.set_mean(mean);

    const GraphAttributes graphAttrs;
    LayerNormNode node(std::move(attrs), graphAttrs);
    auto err = node.infer_properties_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;

    // Dims should remain as set by user
    EXPECT_EQ(mean->get_dim(), (std::vector<int64_t>{32}));
    // Strides should be inferred
    EXPECT_FALSE(mean->get_stride().empty());
}

TEST(TestLayerNormNode, InferPropertiesPreservesExplicitStatStrides)
{
    // Stats tensor with strides already set should not be overwritten
    auto x = makeTensor({32, 512});
    auto attrs = makeMinimalAttrs(x);
    attrs.set_forward_phase(NormFwdPhase::TRAINING);

    auto mean = std::make_shared<TensorAttributes>();
    mean->set_dim({1});
    mean->set_stride({1});
    attrs.set_mean(mean);

    const GraphAttributes graphAttrs;
    LayerNormNode node(std::move(attrs), graphAttrs);
    auto err = node.infer_properties_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;

    EXPECT_EQ(mean->get_dim(), (std::vector<int64_t>{1}));
    EXPECT_EQ(mean->get_stride(), (std::vector<int64_t>{1}));
}

TEST(TestLayerNormNode, InferPropertiesSetsBothMeanAndInvVariance)
{
    // Both mean and inv_variance should be inferred when set (training phase only)
    auto x = makeTensor({32, 512}, {512, 1});
    auto attrs = makeMinimalAttrs(x);
    attrs.set_forward_phase(NormFwdPhase::TRAINING);

    auto mean = std::make_shared<TensorAttributes>();
    auto invVariance = std::make_shared<TensorAttributes>();
    attrs.set_mean(mean);
    attrs.set_inv_variance(invVariance);

    const GraphAttributes graphAttrs;
    LayerNormNode node(std::move(attrs), graphAttrs);
    auto err = node.infer_properties_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;

    EXPECT_FALSE(mean->get_dim().empty());
    EXPECT_FALSE(mean->get_stride().empty());
    EXPECT_FALSE(invVariance->get_dim().empty());
    EXPECT_FALSE(invVariance->get_stride().empty());
}

// ============================================================================
// Gather Tensors Test
// ============================================================================

TEST(TestLayerNormNode, GatherHipdnnTensors)
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
    epsilon->set_uid(5).set_name("Epsilon").set_value(
        hipdnn_data_sdk::utilities::LAYERNORM_DEFAULT_EPSILON);

    auto mean = std::make_shared<TensorAttributes>();
    mean->set_uid(6).set_name("Mean");

    auto invVariance = std::make_shared<TensorAttributes>();
    invVariance->set_uid(7).set_name("InvVariance");

    LayernormAttributes attrs;
    attrs.set_x(x);
    attrs.set_y(y);
    attrs.set_scale(scale);
    attrs.set_bias(bias);
    attrs.set_epsilon(epsilon);
    attrs.set_mean(mean);
    attrs.set_inv_variance(invVariance);

    const GraphAttributes graphAttrs;
    const LayerNormNode node(std::move(attrs), graphAttrs);

    std::unordered_set<std::shared_ptr<TensorAttributes>> allTensors;
    node.gather_hipdnn_tensors(allTensors);

    EXPECT_TRUE(allTensors.find(x) != allTensors.end());
    EXPECT_TRUE(allTensors.find(y) != allTensors.end());
    EXPECT_TRUE(allTensors.find(scale) != allTensors.end());
    EXPECT_TRUE(allTensors.find(bias) != allTensors.end());
    EXPECT_TRUE(allTensors.find(epsilon) != allTensors.end());
    EXPECT_TRUE(allTensors.find(mean) != allTensors.end());
    EXPECT_TRUE(allTensors.find(invVariance) != allTensors.end());
    EXPECT_EQ(allTensors.size(), 7u);
}

TEST(TestLayerNormNode, GatherHipdnnTensorsRequired)
{
    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1);

    auto y = std::make_shared<TensorAttributes>();
    y->set_uid(2);

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_uid(3);

    auto bias = std::make_shared<TensorAttributes>();
    bias->set_uid(4);

    auto epsilon = std::make_shared<TensorAttributes>();
    epsilon->set_uid(5).set_value(hipdnn_data_sdk::utilities::LAYERNORM_DEFAULT_EPSILON);

    LayernormAttributes attrs;
    attrs.set_x(x);
    attrs.set_y(y);
    attrs.set_scale(scale);
    attrs.set_bias(bias);
    attrs.set_epsilon(epsilon);

    const GraphAttributes graphAttrs;
    const LayerNormNode node(std::move(attrs), graphAttrs);

    std::unordered_set<std::shared_ptr<TensorAttributes>> allTensors;
    node.gather_hipdnn_tensors(allTensors);

    EXPECT_TRUE(allTensors.find(x) != allTensors.end());
    EXPECT_TRUE(allTensors.find(y) != allTensors.end());
    EXPECT_TRUE(allTensors.find(scale) != allTensors.end());
    EXPECT_TRUE(allTensors.find(bias) != allTensors.end());
    EXPECT_TRUE(allTensors.find(epsilon) != allTensors.end());
    EXPECT_EQ(allTensors.size(), 5u);
}

TEST(TestLayerNormNode, GetNodeTypeReturnsLayerNorm)
{
    const GraphAttributes graphAttrs;
    const LayerNormNode node(LayernormAttributes{}, graphAttrs);
    EXPECT_EQ(node.getNodeType(), NodeType::LAYER_NORM);
}
