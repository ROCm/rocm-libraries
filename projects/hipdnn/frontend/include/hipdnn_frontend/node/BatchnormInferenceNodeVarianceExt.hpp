// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "Node.hpp"
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/BatchnormInferenceAttributesVarianceExt.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/detail/BatchnormInferenceNodeVarianceExtUnpacker.hpp>
#include <hipdnn_frontend/detail/BatchnormInferenceVarianceExtPacker.hpp>
#include <hipdnn_frontend/node/detail/Utilities.hpp>

namespace hipdnn_frontend::graph
{
class BatchnormInferenceNodeVarianceExt
    : public BaseNode<BatchnormInferenceNodeVarianceExt, NodeType::BATCHNORM_INFERENCE_VARIANCE_EXT>
{
public:
    BatchnormInferenceAttributesVarianceExt attributes;

    BatchnormInferenceNodeVarianceExt(BatchnormInferenceAttributesVarianceExt&& batchnormAttrs,
                                      const GraphAttributes& graphAttrs)
        : BaseNode(graphAttrs)
        , attributes(std::move(batchnormAttrs))
    {
    }

    Error pre_validate_node() const override
    {
        // ====================================================================
        // BATCH NORMALIZATION INFERENCE VARIANCE VALIDATION
        // (Spatial Mode: per-channel statistics over N×H×W for 4D, N×D×H×W for 5D)
        // ====================================================================
        // Algorithm Overview:
        // During inference, BN uses PRE-COMPUTED running statistics from training.
        // For each channel c, using saved running stats (runMean_c, runVar_c):
        //
        // Normalizes: x_normalized[n,c,h,w] = (x[n,c,h,w] - runMean_c) / sqrt(runVar_c + ε)
        // Transforms: y[n,c,h,w] = scale_c * x_normalized[n,c,h,w] + bias_c
        //
        // Key difference from standard BatchnormInference:
        // - Uses VARIANCE directly instead of INV_VARIANCE
        // - Computation will need to calculate 1/sqrt(var + eps) internally
        // ====================================================================

        // SECTION 1: Validate Required Tensor Pointers
        HIPDNN_RETURN_IF_FALSE(attributes.get_x(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "BatchnormInferenceNodeVarianceExt missing x for pre-validation");

        HIPDNN_RETURN_IF_FALSE(
            attributes.get_scale(),
            ErrorCode::ATTRIBUTE_NOT_SET,
            "BatchnormInferenceNodeVarianceExt missing scale for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_bias(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "BatchnormInferenceNodeVarianceExt missing bias for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_y(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "BatchnormInferenceNodeVarianceExt missing y for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_mean(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "BatchnormInferenceNodeVarianceExt missing mean for pre-validation");

        HIPDNN_RETURN_IF_FALSE(
            attributes.get_variance(),
            ErrorCode::ATTRIBUTE_NOT_SET,
            "BatchnormInferenceNodeVarianceExt missing variance for pre-validation");

        HIPDNN_RETURN_IF_FALSE(
            attributes.get_epsilon(),
            ErrorCode::ATTRIBUTE_NOT_SET,
            "BatchnormInferenceNodeVarianceExt missing epsilon for pre-validation");

        // Epsilon (ε) provides numerical stability: x_normalized = (x - mean) / sqrt(var + ε)
        // Without ε, division by zero occurs when var ≈ 0. Must be a scalar.
        HIPDNN_CHECK_ERROR(detail::validateScalarParameter(attributes.get_epsilon(), "Epsilon"));

        // Get tensor references
        auto x = attributes.get_x();
        auto y = attributes.get_y();
        auto scale = attributes.get_scale();
        auto bias = attributes.get_bias();
        auto mean = attributes.get_mean();
        auto variance = attributes.get_variance();

        // SECTION 2: Validate Required Tensor Dimensions
        // Why: All required tensors must have dimensions set by user - they are never inferred.
        // For inference: x, scale, bias, mean, variance are all required user parameters.
        HIPDNN_CHECK_ERROR(detail::validateMinimumTensorDimensions(x, 2, "Input tensor (x)"));
        HIPDNN_CHECK_ERROR(detail::validateMinimumTensorDimensions(scale, 2, "Scale tensor"));
        HIPDNN_CHECK_ERROR(detail::validateMinimumTensorDimensions(bias, 2, "Bias tensor"));
        HIPDNN_CHECK_ERROR(detail::validateMinimumTensorDimensions(mean, 2, "Mean tensor"));
        HIPDNN_CHECK_ERROR(detail::validateMinimumTensorDimensions(variance, 2, "Variance tensor"));

        // SECTION 3: Validate Output Tensor Shape Consistency
        // Why: BN preserves tensor shape during inference just as in training.
        // Output y[n,c,h,w] has same shape as input x[n,c,h,w].
        HIPDNN_CHECK_ERROR(
            detail::validateTensorShapesMatchIfSet(x, y, "Input tensor (x)", "Output tensor (y)"));

        // SECTION 4: Validate Channel Dimensions and Parameter Tensor Shapes
        // Why: All parameters are per-channel with shape [1, C, 1, 1]:
        // - mean_c and var_c: Running statistics saved from training
        // - scale and bias: Learned parameters from training
        // These are the same parameters used during training, now fixed for inference.

        // Extract channel count - safe to access xDims[1] after SECTION 2 validation
        auto& xDims = x->get_dim();
        const int64_t channels = xDims[1];

        // Validate scale has correct channel-only shape (required user parameter)
        HIPDNN_CHECK_ERROR(detail::validateChannelOnlyTensorShape(scale, channels, "Scale tensor"));

        // Validate bias has correct channel-only shape (required user parameter)
        HIPDNN_CHECK_ERROR(detail::validateChannelOnlyTensorShape(bias, channels, "Bias tensor"));

        // Validate mean has correct channel-only shape (required user parameter for inference)
        HIPDNN_CHECK_ERROR(detail::validateChannelOnlyTensorShape(mean, channels, "Mean tensor"));

        // Validate variance has correct channel-only shape (required user parameter for inference)
        HIPDNN_CHECK_ERROR(
            detail::validateChannelOnlyTensorShape(variance, channels, "Variance tensor"));

        // NOTE: Unlike training, inference does NOT require m > 1 (where m = N*H*W for 4D
        // or m = N*D*H*W for 5D) since it uses pre-computed statistics rather than
        // computing batch statistics.

        return {ErrorCode::OK, ""};
    }

    Error infer_properties_node() override
    {
        auto x = attributes.get_x();
        auto y = attributes.get_y();

        if(!x)
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET,
                    "BatchnormInferenceNodeVarianceExt missing x for setting properties"};
        }

        if(!y)
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET,
                    "BatchnormInferenceNodeVarianceExt missing y for setting properties"};
        }

        HIPDNN_CHECK_ERROR(attributes.fill_from_context(graph_attributes));

        if(y->get_dim().empty())
        {
            y->set_dim(x->get_dim());
        }

        if(y->get_stride().empty())
        {
            y->set_stride(x->get_stride());
        }

        return {};
    }

    Error create_operation(
        std::unordered_map<int64_t, detail::ScopedHipdnnBackendDescriptor>& tensorDescs,
        std::vector<detail::ScopedHipdnnBackendDescriptor>& operations) const override
    {
        return detail::createBatchnormInferenceVarianceExtOperation(
            attributes, tensorDescs, operations);
    }

    Error unpack_from_descriptor(
        hipdnnBackendDescriptor_t opDesc,
        std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>>& tensorMap) override
    {
        BatchnormInferenceAttributesVarianceExt attrs;
        HIPDNN_CHECK_ERROR(
            detail::unpackBatchnormInferenceVarianceExtOperation(opDesc, tensorMap, attrs));
        attributes = std::move(attrs);
        return {};
    }
};
}
