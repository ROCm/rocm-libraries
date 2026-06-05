// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "Node.hpp"
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/BatchnormBackwardAttributes.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/detail/BatchnormBackwardPacker.hpp>
#include <hipdnn_frontend/detail/BatchnormBackwardUnpacker.hpp>
#include <hipdnn_frontend/node/detail/Utilities.hpp>

namespace hipdnn_frontend::graph
{

class BatchnormBackwardNode : public BaseNode<BatchnormBackwardNode, NodeType::BATCHNORM_BACKWARD>
{
public:
    BatchnormBackwardAttributes attributes;

    BatchnormBackwardNode(BatchnormBackwardAttributes&& batchnormAttrs,
                          const GraphAttributes& graphAttrs)
        : BaseNode(graphAttrs)
        , attributes(std::move(batchnormAttrs))
    {
    }

    Error unpack_from_descriptor(
        hipdnnBackendDescriptor_t opDesc,
        std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>>& tensorMap) override
    {
        BatchnormBackwardAttributes attrs;
        HIPDNN_CHECK_ERROR(detail::unpackBatchnormBackwardOperation(opDesc, tensorMap, attrs));
        attributes = std::move(attrs);
        return {};
    }

    Error pre_validate_node() const override
    {
        // ====================================================================
        // BATCH NORMALIZATION BACKWARD VALIDATION
        // (Spatial Mode: per-channel statistics over N×H×W for 4D, N×D×H×W for 5D)
        // ====================================================================
        // Algorithm Overview:
        // Given dy (gradient of loss w.r.t. y), compute gradients w.r.t. inputs:
        //
        // INPUTS: x, dy, scale, mean_c, invStd_c (saved from forward pass)
        // OUTPUTS: dx, dscale, dbias
        //
        // For each channel c, using saved forward stats (mean_c, invStd_c):
        //   xhat[n,c,h,w] = (x[n,c,h,w] - mean_c) * invStd_c
        //
        // Compute parameter gradients (accumulated over N,H,W):
        //   dscale_c += xhat[n,c,h,w] * dy[n,c,h,w]  // gradient of scale
        //   dbias_c  += dy[n,c,h,w]                   // gradient of bias
        //
        // Compute input gradient (where m = N*H*W per channel):
        //   dx_i = (scale_c * invStd_c / m) * (m*dy_i - dbias_c - xhat_i*dscale_c)
        //
        // This chain rule derivative accounts for batch statistics coupling.
        // ====================================================================

        // SECTION 1: Validate Required Tensor Pointers
        HIPDNN_RETURN_IF_FALSE(attributes.get_dy(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "BatchnormBackwardNode missing dy for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_x(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "BatchnormBackwardNode missing x for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_scale(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "BatchnormBackwardNode missing scale for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_dx(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "BatchnormBackwardNode missing dx for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_dscale(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "BatchnormBackwardNode missing dscale for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_dbias(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "BatchnormBackwardNode missing dbias for pre-validation");

        // Get tensor references
        auto dy = attributes.get_dy();
        auto x = attributes.get_x();
        auto dx = attributes.get_dx();
        auto scale = attributes.get_scale();
        auto dscale = attributes.get_dscale();
        auto dbias = attributes.get_dbias();

        // SECTION 2: Validate Required Tensor Dimensions
        // Why: All required tensors (x, dy, scale) must have dimensions set by user.
        // Outputs (dx, dscale, dbias) are inferred, so not validated here.
        HIPDNN_CHECK_ERROR(detail::validateMinimumTensorDimensions(x, 2, "Input tensor (x)"));
        HIPDNN_CHECK_ERROR(
            detail::validateMinimumTensorDimensions(dy, 2, "Gradient input tensor (dy)"));
        HIPDNN_CHECK_ERROR(detail::validateMinimumTensorDimensions(scale, 2, "Scale tensor"));

        // SECTION 3: Validate Tensor Shape Consistency
        // Why: Gradients flow through the same computational graph as forward pass.
        // - dy has same shape as y (and therefore x) from forward pass
        // - dx has same shape as x (gradient w.r.t. input)
        // All gradient tensors must match the data tensor shapes they correspond to.

        // Both x and dy validated in SECTION 2, can call directly
        HIPDNN_CHECK_ERROR(detail::validateTensorShapesMatch(
            x, dy, "Input tensor (x)", "Gradient input tensor (dy)"));

        // dx may not have dimensions set yet (will be inferred)
        HIPDNN_CHECK_ERROR(detail::validateTensorShapesMatchIfSet(
            x, dx, "Input tensor (x)", "Gradient output tensor (dx)"));

        // SECTION 4: Validate Channel Dimensions and Parameter Tensor Shapes
        // Why: Parameter gradients (dscale, dbias) are accumulated per-channel over (N,H,W):
        //   dscale_c = Σ_{n,h,w} xhat[n,c,h,w] * dy[n,c,h,w]  -> shape [1, C, 1, 1]
        //   dbias_c  = Σ_{n,h,w} dy[n,c,h,w]                   -> shape [1, C, 1, 1]
        // scale from forward pass is also per-channel [1, C, 1, 1].
        // Saved statistics (mean_c, invStd_c) are per-channel for backward computation.

        // Extract optional tensors once for reuse in both shape validation and consistency check
        auto mean = attributes.get_mean();
        auto invVar = attributes.get_inv_variance();

        // Extract channel count - safe to access xDims[1] after SECTION 2 validation
        auto& xDims = x->get_dim();
        const int64_t channels = xDims[1];

        // Validate scale has correct channel-only shape (required user parameter)
        HIPDNN_CHECK_ERROR(detail::validateChannelOnlyTensorShape(scale, channels, "Scale tensor"));

        // Validate gradient outputs (only if dimensions set, will be inferred otherwise)
        HIPDNN_CHECK_ERROR(detail::validateChannelOnlyShapeIfSet(
            dscale, channels, "Scale gradient tensor (dscale)"));
        HIPDNN_CHECK_ERROR(
            detail::validateChannelOnlyShapeIfSet(dbias, channels, "Bias gradient tensor (dbias)"));

        // Validate optional saved statistics from forward pass (only if dimensions set)
        HIPDNN_CHECK_ERROR(detail::validateChannelOnlyShapeIfSet(mean, channels, "Mean tensor"));
        HIPDNN_CHECK_ERROR(
            detail::validateChannelOnlyShapeIfSet(invVar, channels, "Inverse variance tensor"));

        // SECTION 5: Validate Mean and Inverse Variance Consistency
        // Why: Backward computation uses saved statistics (mean_c, invStd_c) from forward pass.
        // These must be provided together (both or neither). If neither is provided, they will
        // be recomputed during backward pass (less efficient but valid).
        const bool hasMean = (mean != nullptr);
        const bool hasInvVariance = (invVar != nullptr);
        if(hasMean != hasInvVariance)
        {
            return {ErrorCode::INVALID_VALUE,
                    "BatchnormBackwardNode requires both mean and inv_variance to be set, or "
                    "neither"};
        }

        // SECTION 6: Validate Spatial Mode Constraints
        HIPDNN_CHECK_ERROR(detail::validateBatchNormTrainingSpatialDimensions(
            x, scale, "Batch normalization backward"));

        return {ErrorCode::OK, ""};
    }

    Error infer_properties_node() override
    {
        auto x = attributes.get_x();
        auto dx = attributes.get_dx();
        auto dscale = attributes.get_dscale();
        auto dbias = attributes.get_dbias();

        if(!x)
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET,
                    "BatchnormBackwardNode missing x for setting properties"};
        }
        if(!dx)
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET,
                    "BatchnormBackwardNode missing dx for setting properties"};
        }
        if(!dscale)
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET,
                    "BatchnormBackwardNode missing dscale for setting properties"};
        }
        if(!dbias)
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET,
                    "BatchnormBackwardNode missing dbias for setting properties"};
        }

        HIPDNN_CHECK_ERROR(attributes.fill_from_context(graph_attributes));

        if(dx->get_dim().empty())
        {
            dx->set_dim(x->get_dim());
        }

        if(dx->get_stride().empty())
        {
            dx->set_stride(x->get_stride());
        }

        auto inferCTensor = [&](std::shared_ptr<TensorAttributes>& tensorToInfer) {
            if(tensorToInfer->get_dim().empty())
            {
                std::vector<int64_t> tensorDims(x->get_dim().size(), 1);
                tensorDims[1] = x->get_dim()[1];
                tensorToInfer->set_dim(tensorDims);
            }

            if(tensorToInfer->get_stride().empty())
            {
                auto strideOrder = hipdnn_data_sdk::utilities::extractStrideOrder(x->get_stride());
                tensorToInfer->set_stride(hipdnn_data_sdk::utilities::generateStrides(
                    tensorToInfer->get_dim(), strideOrder));
            }
        };

        inferCTensor(dscale);
        inferCTensor(dbias);

        return {};
    }

    void gather_hipdnn_tensors(
        std::unordered_set<std::shared_ptr<TensorAttributes>>& allTensors) const override
    {
        BaseNode<BatchnormBackwardNode, NodeType::BATCHNORM_BACKWARD>::gather_hipdnn_tensors(
            allTensors);

        for(auto& tensor : attributes.peer_stats)
        {
            if(tensor)
            {
                allTensors.insert(tensor);
            }
        }
    }

    Error create_operation(
        std::unordered_map<int64_t, detail::ScopedHipdnnBackendDescriptor>& tensorDescs,
        std::vector<detail::ScopedHipdnnBackendDescriptor>& operations) const override
    {
        return detail::createBatchnormBackwardOperation(attributes, tensorDescs, operations);
    }
};

typedef BatchnormBackwardNode DBNNode;
}
