// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "Node.hpp"
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/BatchnormInferenceAttributes.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/detail/BatchnormInferencePacker.hpp>
#include <hipdnn_frontend/detail/BatchnormInferenceUnpacker.hpp>
#include <hipdnn_frontend/node/detail/Utilities.hpp>

namespace hipdnn_frontend::graph
{
class BatchnormInferenceNode
    : public BaseNode<BatchnormInferenceNode, NodeType::BATCHNORM_INFERENCE>
{
public:
    BatchnormInferenceAttributes attributes;

    BatchnormInferenceNode(BatchnormInferenceAttributes&& batchnormAttrs,
                           const GraphAttributes& graphAttrs)
        : BaseNode(graphAttrs)
        , attributes(std::move(batchnormAttrs))
    {
    }

    Error pre_validate_node() const override
    {
        // ====================================================================
        // BATCH NORMALIZATION INFERENCE VALIDATION
        // (Spatial Mode: per-channel statistics over N×H×W for 4D, N×D×H×W for 5D)
        // ====================================================================
        // Algorithm Overview:
        // During inference, BN uses PRE-COMPUTED running statistics from training.
        // For each channel c, using saved running stats (runMean_c, runVar_c):
        //
        // Normalizes: xhat[n,c,h,w] = (x[n,c,h,w] - runMean_c) / sqrt(runVar_c + ε)
        // Transforms: y[n,c,h,w] = scale_c * xhat[n,c,h,w] + bias_c
        //
        // Key difference from training:
        // - NO batch statistics computed (no dependence on current batch)
        // - Uses saved runMean_c and runVar_c (estimated from training)
        // - NO running stats updates (read-only operation)
        // ====================================================================

        // SECTION 1: Validate Required Tensor Pointers
        HIPDNN_RETURN_IF_FALSE(attributes.get_x(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "BatchnormInferenceNode missing x for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_scale(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "BatchnormInferenceNode missing scale for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_bias(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "BatchnormInferenceNode missing bias for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_y(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "BatchnormInferenceNode missing y for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_mean(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "BatchnormInferenceNode missing mean for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_inv_variance(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "BatchnormInferenceNode missing inv_variance for pre-validation");

        // Get tensor references
        auto x = attributes.get_x();
        auto y = attributes.get_y();
        auto scale = attributes.get_scale();
        auto bias = attributes.get_bias();
        auto mean = attributes.get_mean();
        auto invVar = attributes.get_inv_variance();

        // SECTION 2: Validate Required Tensor Dimensions
        // Why: All required tensors must have dimensions set by user - they are never inferred.
        // For inference: x, scale, bias, mean, invVar are all required user parameters.
        HIPDNN_CHECK_ERROR(detail::validateMinimumTensorDimensions(x, 2, "Input tensor (x)"));
        HIPDNN_CHECK_ERROR(detail::validateMinimumTensorDimensions(scale, 2, "Scale tensor"));
        HIPDNN_CHECK_ERROR(detail::validateMinimumTensorDimensions(bias, 2, "Bias tensor"));
        HIPDNN_CHECK_ERROR(detail::validateMinimumTensorDimensions(mean, 2, "Mean tensor"));
        HIPDNN_CHECK_ERROR(
            detail::validateMinimumTensorDimensions(invVar, 2, "Inverse variance tensor"));

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

        // Validate inv_variance has correct channel-only shape (required user parameter for inference)
        HIPDNN_CHECK_ERROR(
            detail::validateChannelOnlyTensorShape(invVar, channels, "Inverse variance tensor"));

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
                    "BatchnormInferenceNode missing x for setting properties"};
        }

        if(!y)
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET,
                    "BatchnormInferenceNode missing y for setting properties"};
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
        return detail::createBatchnormInferenceOperation(attributes, tensorDescs, operations);
    }

    Error unpack_from_descriptor(
        hipdnnBackendDescriptor_t opDesc,
        std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>>& tensorMap) override
    {
        BatchnormInferenceAttributes attrs;
        HIPDNN_CHECK_ERROR(detail::unpackBatchnormInferenceOperation(opDesc, tensorMap, attrs));
        attributes = std::move(attrs);
        return {};
    }
};
}
