// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "Node.hpp"
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/BatchnormAttributes.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/detail/BatchnormPacker.hpp>
#include <hipdnn_frontend/detail/BatchnormUnpacker.hpp>
#include <hipdnn_frontend/node/detail/Utilities.hpp>

namespace hipdnn_frontend::graph
{
class BatchnormNode : public BaseNode<BatchnormNode, NodeType::BATCHNORM>
{
public:
    BatchnormAttributes attributes;

    BatchnormNode(BatchnormAttributes&& batchnormAttrs, const GraphAttributes& graphAttrs)
        : BaseNode(graphAttrs)
        , attributes(std::move(batchnormAttrs))
    {
    }

    Error unpack_from_descriptor(
        hipdnnBackendDescriptor_t opDesc,
        std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>>& tensorMap) override
    {
        BatchnormAttributes attrs;
        HIPDNN_CHECK_ERROR(detail::unpackBatchnormOperation(opDesc, tensorMap, attrs));
        attributes = std::move(attrs);
        return {};
    }

    Error pre_validate_node() const override
    {
        // ====================================================================
        // BATCH NORMALIZATION FORWARD TRAINING VALIDATION
        // (Spatial Mode: per-channel statistics over N×H×W for 4D, N×D×H×W for 5D)
        // ====================================================================
        // Algorithm Overview:
        // For each channel c, BN computes batch statistics over (N,H,W):
        //   mean_c = (1/m) * Σ_{n,h,w} x[n,c,h,w]  where m = N*H*W
        //   var_c  = (1/m) * Σ_{n,h,w} (x[n,c,h,w] - mean_c)²
        //
        // Normalizes: xhat[n,c,h,w] = (x[n,c,h,w] - mean_c) / sqrt(var_c + ε)
        // Transforms: y[n,c,h,w] = scale_c * xhat[n,c,h,w] + bias_c
        //
        // Optionally outputs mean_c and invStd_c to device buffers
        // (consumed by backward pass for gradient computation)
        //
        // Updates running stats: runMean_c = (1-momentum)*runMean_c + momentum*mean_c
        //                       runVar_c  = (1-momentum)*runVar_c  + momentum*var_c
        // ====================================================================

        // SECTION 1: Validate Required Tensor Pointers
        HIPDNN_RETURN_IF_FALSE(attributes.get_x(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "BatchnormNode missing x for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_scale(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "BatchnormNode missing scale for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_bias(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "BatchnormNode missing bias for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_y(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "BatchnormNode missing y for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_epsilon(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "BatchnormNode missing epsilon for pre-validation");

        // Get tensor references
        auto x = attributes.get_x();
        auto y = attributes.get_y();
        auto scale = attributes.get_scale();
        auto bias = attributes.get_bias();
        auto epsilon = attributes.get_epsilon();

        // SECTION 2: Validate Required Parameter Dimensions
        // Why: All required parameters (x, scale, bias, epsilon) must have dimensions
        // set by user. Validate them upfront before proceeding with shape checks.
        HIPDNN_CHECK_ERROR(detail::validateMinimumTensorDimensions(x, 2, "Input tensor (x)"));
        HIPDNN_CHECK_ERROR(detail::validateMinimumTensorDimensions(scale, 2, "Scale tensor"));
        HIPDNN_CHECK_ERROR(detail::validateMinimumTensorDimensions(bias, 2, "Bias tensor"));

        // Epsilon (ε) provides numerical stability: xhat = (x - mean) / sqrt(var + ε)
        // Without ε, division by zero occurs when var ≈ 0. Must be a scalar.
        HIPDNN_CHECK_ERROR(detail::validateScalarParameter(epsilon, "Epsilon"));

        // SECTION 3: Validate Output Tensor Shape Consistency
        // Why: BN preserves tensor shape - it only transforms values, not dimensions.
        // Output y[n,c,h,w] has same shape as input x[n,c,h,w].
        HIPDNN_CHECK_ERROR(
            detail::validateTensorShapesMatchIfSet(x, y, "Input tensor (x)", "Output tensor (y)"));

        // SECTION 4: Validate Channel Dimensions and Parameter Tensor Shapes
        // Why: All BN parameters (scale, bias, mean, variance) are per-channel with
        // shape [1, C, 1, 1, ...]. This is because:
        // - Each channel c has its own statistics: mean_c, var_c
        // - Each channel c has its own learnable parameters: scale_c, bias_c
        //   - scale_c controls feature importance/gain after normalization
        //   - bias_c controls activation threshold (e.g., for ReLU: active when scale_c*xhat + bias_c > 0)

        // Extract channel count - safe to access xDims[1] after SECTION 2 validation
        auto& xDims = x->get_dim();
        const int64_t channels = xDims[1];

        // Validate scale has correct channel-only shape (required user parameter)
        HIPDNN_CHECK_ERROR(detail::validateChannelOnlyTensorShape(scale, channels, "Scale tensor"));

        // Validate bias has correct channel-only shape (required user parameter)
        HIPDNN_CHECK_ERROR(detail::validateChannelOnlyTensorShape(bias, channels, "Bias tensor"));

        // Validate optional mean and inv_variance tensors (only if dimensions set)
        HIPDNN_CHECK_ERROR(
            detail::validateChannelOnlyShapeIfSet(attributes.get_mean(), channels, "Mean tensor"));
        HIPDNN_CHECK_ERROR(detail::validateChannelOnlyShapeIfSet(
            attributes.get_inv_variance(), channels, "Inverse variance tensor"));

        // SECTION 5: Validate Running Stats Consistency
        // Why: Running statistics are updated together during training:
        //   nextRunMean_c = (1-momentum)*prevRunMean_c + momentum*batchMean_c
        //   nextRunVar_c  = (1-momentum)*prevRunVar_c  + momentum*batchVar_c
        // These are used for inference after training completes. If any are provided,
        // all must be provided to ensure consistent exponential moving average updates.
        auto prevRunningMean = attributes.get_prev_running_mean();
        auto prevRunningVar = attributes.get_prev_running_variance();
        auto nextRunningMean = attributes.get_next_running_mean();
        auto nextRunningVar = attributes.get_next_running_variance();

        // If any running stat is provided, all must be provided
        const bool hasPrevRunningMean = prevRunningMean != nullptr;
        const bool hasPrevRunningVar = prevRunningVar != nullptr;
        const bool hasNextRunningMean = nextRunningMean != nullptr;
        const bool hasNextRunningVar = nextRunningVar != nullptr;

        if(hasPrevRunningMean || hasPrevRunningVar || hasNextRunningMean || hasNextRunningVar)
        {
            HIPDNN_RETURN_IF_FALSE(
                hasPrevRunningMean && hasPrevRunningVar && hasNextRunningMean && hasNextRunningVar,
                ErrorCode::INVALID_VALUE,
                "BatchnormNode: If any running statistics are provided, all running "
                "statistics "
                "(prev_running_mean, prev_running_variance, next_running_mean, "
                "next_running_variance) must be provided");

            // Validate running stats have correct shapes (only if dimensions set)
            HIPDNN_CHECK_ERROR(detail::validateChannelOnlyShapeIfSet(
                prevRunningMean, channels, "Previous running mean tensor"));
            HIPDNN_CHECK_ERROR(detail::validateChannelOnlyShapeIfSet(
                prevRunningVar, channels, "Previous running variance tensor"));
            HIPDNN_CHECK_ERROR(detail::validateChannelOnlyShapeIfSet(
                nextRunningMean, channels, "Next running mean tensor"));
            HIPDNN_CHECK_ERROR(detail::validateChannelOnlyShapeIfSet(
                nextRunningVar, channels, "Next running variance tensor"));
        }

        // SECTION 6: Validate Spatial Mode Constraints
        // Why: For spatial BN, statistics are computed over N*H*W elements per channel.
        // We need N*H*W > 1 to compute meaningful statistics (mean and variance).
        // With only 1 element, variance is undefined and normalization degenerates.
        HIPDNN_CHECK_ERROR(detail::validateBatchNormTrainingSpatialDimensions(
            x, scale, "Batch normalization training"));

        return {ErrorCode::OK, ""};
    }

    Error infer_properties_node() override
    {
        auto x = attributes.get_x();
        auto y = attributes.get_y();

        if(!x)
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET, "BatchnormNode missing x for setting properties"};
        }

        if(!y)
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET, "BatchnormNode missing y for setting properties"};
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

        auto mean = attributes.get_mean();
        auto invVar = attributes.get_inv_variance();

        if(mean)
        {
            inferCTensor(mean);
        }

        if(invVar)
        {
            inferCTensor(invVar);
        }

        auto prevRunningMean = attributes.get_prev_running_mean();
        auto prevRunningVar = attributes.get_prev_running_variance();

        auto nextRunningMean = attributes.get_next_running_mean();
        auto nextRunningVar = attributes.get_next_running_variance();

        if(prevRunningMean && prevRunningVar && nextRunningMean && nextRunningVar)
        {
            inferCTensor(nextRunningMean);
            inferCTensor(nextRunningVar);
        }

        return {};
    }

    void gather_hipdnn_tensors(
        std::unordered_set<std::shared_ptr<TensorAttributes>>& allTensors) const override
    {
        BaseNode<BatchnormNode, NodeType::BATCHNORM>::gather_hipdnn_tensors(allTensors);

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
        return detail::createBatchnormOperation(attributes, tensorDescs, operations);
    }
};

typedef BatchnormNode BatchNormNode;
}
