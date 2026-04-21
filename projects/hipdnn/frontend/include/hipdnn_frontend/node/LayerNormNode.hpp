// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file LayerNormNode.hpp
 * @brief Graph node for layer normalization operations
 */

#pragma once

#include "Node.hpp"
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/LayernormAttributes.hpp>
#include <hipdnn_frontend/detail/LayerNormPacker.hpp>
#include <hipdnn_frontend/detail/LayerNormUnpacker.hpp>
#include <hipdnn_frontend/node/detail/Utilities.hpp>

namespace hipdnn_frontend::graph
{
/**
 * @class LayerNormNode
 * @brief Graph node that performs layer normalization
 *
 * Validates input tensors, infers output shapes, and serializes the
 * layer normalization operation to a backend descriptor.
 *
 * @see LayernormAttributes, Graph::layernorm()
 */
class LayerNormNode : public BaseNode<LayerNormNode, NodeType::LAYER_NORM>
{
public:
    LayernormAttributes attributes;

    LayerNormNode(LayernormAttributes&& layernormAttrs, const GraphAttributes& graphAttrs)
        : BaseNode(graphAttrs)
        , attributes(std::move(layernormAttrs))
    {
    }

    Error unpack_from_descriptor(
        hipdnnBackendDescriptor_t opDesc,
        std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>>& tensorMap) override
    {
        LayernormAttributes lnAttr;
        HIPDNN_CHECK_ERROR(detail::unpackLayernormOperation(opDesc, tensorMap, lnAttr));
        attributes = std::move(lnAttr);
        return {};
    }

    Error pre_validate_node() const override
    {
        // ====================================================================
        // LAYER NORMALIZATION FORWARD VALIDATION
        // ====================================================================
        // Algorithm Overview:
        // LayerNorm computes statistics over the feature dimensions (last normalized_shape dims):
        //   For input shape [N, ..., D1, D2, ..., Dk] where last k dims are normalized:
        //   mean = (1/m) * sum x[..., i] over normalized dims, where m = D1*D2*...*Dk
        //   var  = (1/m) * sum (x[..., i] - mean)^2 over normalized dims
        //
        // Normalizes: xhat = (x - mean) / sqrt(var + epsilon)
        // Transforms: y = scale * xhat + bias (scale and bias have shape of normalized dims)
        //
        // Outputs:
        // - Y: normalized output (same shape as X)
        // - Mean: computed mean per sample (optional, shape: batch dims only)
        // - InvVariance: inverse standard deviation (optional, shape: batch dims only)
        // ====================================================================

        // SECTION 1: Validate Forward Phase
        HIPDNN_RETURN_IF_FALSE(attributes.get_forward_phase() != NormFwdPhase::NOT_SET,
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "Forward phase not set of layernorm node.");

        // SECTION 2: Validate Required Tensor Pointers
        HIPDNN_RETURN_IF_FALSE(attributes.get_x(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "LayerNormNode missing x for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_y(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "LayerNormNode missing y for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_epsilon(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "LayerNormNode missing epsilon for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_scale(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "LayerNormNode missing scale for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_bias(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "LayerNormNode missing bias for pre-validation");

        // Get tensor references
        auto x = attributes.get_x();
        auto y = attributes.get_y();
        auto scale = attributes.get_scale();
        auto bias = attributes.get_bias();
        auto epsilon = attributes.get_epsilon();

        // SECTION 3: Validate Required Parameter Dimensions
        HIPDNN_CHECK_ERROR(detail::validateMinimumTensorDimensions(x, 1, "Input tensor (x)"));

        // Epsilon provides numerical stability: xhat = (x - mean) / sqrt(var + epsilon)
        HIPDNN_CHECK_ERROR(detail::validateScalarParameter(epsilon, "Epsilon"));

        // SECTION 4: Validate Output Tensor Shape Consistency
        // LayerNorm preserves tensor shape - output has same shape as input
        HIPDNN_CHECK_ERROR(
            detail::validateTensorShapesMatchIfSet(x, y, "Input tensor (x)", "Output tensor (y)"));

        // SECTION 5: Validate Scale and Bias Tensors
        // Scale and bias are per-feature parameters matching the normalized dimensions.
        // For input shape [N, C, H, W] normalized over last 3 dims, scale/bias shape is [C, H, W]
        HIPDNN_CHECK_ERROR(detail::validateMinimumTensorDimensions(scale, 1, "Scale tensor"));
        HIPDNN_CHECK_ERROR(detail::validateMinimumTensorDimensions(bias, 1, "Bias tensor"));

        // Scale and bias should have matching shapes
        HIPDNN_CHECK_ERROR(
            detail::validateTensorShapesMatchIfSet(scale, bias, "Scale tensor", "Bias tensor"));

        return {ErrorCode::OK, ""};
    }

    Error infer_properties_node() override
    {
        auto x = attributes.get_x();
        auto y = attributes.get_y();

        if(!x)
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET, "LayerNormNode missing x for setting properties"};
        }

        if(!y)
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET, "LayerNormNode missing y for setting properties"};
        }

        HIPDNN_CHECK_ERROR(attributes.fill_from_context(graph_attributes));

        // Output Y has the same shape as input X
        if(y->get_dim().empty())
        {
            y->set_dim(x->get_dim());
        }

        if(y->get_stride().empty())
        {
            if(!x->get_stride().empty())
            {
                y->set_stride(x->get_stride());
            }
            else
            {
                y->set_stride(hipdnn_data_sdk::utilities::generateStrides(y->get_dim()));
            }
        }

        // Infer scale and bias dims and strides from X's normalized dimensions
        // For input [N, C, H, W], scale/bias dim = [1, C, H, W] (batch dim set to 1)
        auto scale = attributes.get_scale();
        auto bias = attributes.get_bias();

        if(scale && !x->get_dim().empty())
        {
            if(scale->get_dim().empty())
            {
                auto scaleBiasDim = x->get_dim();
                scaleBiasDim[0] = 1;
                scale->set_dim(scaleBiasDim);
            }
            if(scale->get_stride().empty())
            {
                const auto& scaleDim = scale->get_dim();
                if(!x->get_stride().empty() && x->get_stride().size() == scaleDim.size())
                {
                    auto strideOrder
                        = hipdnn_data_sdk::utilities::extractStrideOrder(x->get_stride());
                    scale->set_stride(
                        hipdnn_data_sdk::utilities::generateStrides(scaleDim, strideOrder));
                }
                else
                {
                    scale->set_stride(hipdnn_data_sdk::utilities::generateStrides(scaleDim));
                }
            }
        }

        if(bias && !x->get_dim().empty())
        {
            if(bias->get_dim().empty())
            {
                auto scaleBiasDim = x->get_dim();
                scaleBiasDim[0] = 1;
                bias->set_dim(scaleBiasDim);
            }
            if(bias->get_stride().empty())
            {
                const auto& biasDim = bias->get_dim();
                if(!x->get_stride().empty() && x->get_stride().size() == biasDim.size())
                {
                    auto strideOrder
                        = hipdnn_data_sdk::utilities::extractStrideOrder(x->get_stride());
                    bias->set_stride(
                        hipdnn_data_sdk::utilities::generateStrides(biasDim, strideOrder));
                }
                else
                {
                    bias->set_stride(hipdnn_data_sdk::utilities::generateStrides(biasDim));
                }
            }
        }

        // Infer mean and inv_variance shapes (training phase outputs)
        // Mean and inv_variance have shape of the batch dimensions (everything except normalized dims)
        auto mean = attributes.get_mean();
        auto invVariance = attributes.get_inv_variance();

        auto inferStatsTensor = [&](std::shared_ptr<TensorAttributes>& tensorToInfer) {
            if(tensorToInfer && tensorToInfer->get_dim().empty() && scale
               && !scale->get_dim().empty())
            {
                // Stats dims mirror X's shape, but where scale has dim 1 (batch dims),
                // the stats dim retains X's value; where scale has dim > 1 (normalized
                // dims), the stats dim becomes 1.
                const auto& xDim = x->get_dim();
                const auto& scaleDim = scale->get_dim();
                if(scaleDim.size() == xDim.size())
                {
                    std::vector<int64_t> statsDims(xDim.size());
                    for(size_t i = 0; i < xDim.size(); i++)
                    {
                        statsDims[i] = (scaleDim[i] == 1) ? xDim[i] : 1;
                    }
                    tensorToInfer->set_dim(statsDims);
                }
                else
                {
                    // Fallback: batch dim from X, rest set to 1
                    std::vector<int64_t> statsDims(xDim.size(), 1);
                    statsDims[0] = xDim[0];
                    tensorToInfer->set_dim(statsDims);
                }
            }

            if(tensorToInfer && tensorToInfer->get_stride().empty())
            {
                if(!x->get_stride().empty()
                   && x->get_stride().size() == tensorToInfer->get_dim().size())
                {
                    auto strideOrder
                        = hipdnn_data_sdk::utilities::extractStrideOrder(x->get_stride());
                    tensorToInfer->set_stride(hipdnn_data_sdk::utilities::generateStrides(
                        tensorToInfer->get_dim(), strideOrder));
                }
                else
                {
                    tensorToInfer->set_stride(
                        hipdnn_data_sdk::utilities::generateStrides(tensorToInfer->get_dim()));
                }
            }
        };

        if(attributes.get_forward_phase() != NormFwdPhase::INFERENCE)
        {
            if(mean)
            {
                inferStatsTensor(mean);
            }

            if(invVariance)
            {
                inferStatsTensor(invVariance);
            }
        }

        // Infer normalized dimension count if not available
        if(attributes.get_normalized_dim_count() <= 0)
        {
            if(attributes.get_x()->get_dim().size()
               == attributes.get_scale()
                      ->get_dim()
                      .size()) // Dimensions not used by scale have been set to 1
            {
                int64_t normalizedDimCount = 1;
                for(int64_t i = static_cast<int64_t>(attributes.get_scale()->get_dim().size()) - 1;
                    i >= 0;
                    --i)
                {
                    if(attributes.get_scale()->get_dim()[static_cast<size_t>(i)] == 1)
                    {
                        break;
                    }
                    normalizedDimCount
                        = static_cast<int64_t>(attributes.get_scale()->get_dim().size()) - i;
                }
                attributes.set_normalized_dim_count(normalizedDimCount);
            }
            else // Dimensions not used by scale have been omitted
            {
                attributes.set_normalized_dim_count(
                    static_cast<int64_t>(attributes.get_scale()->get_dim().size()));
            }
        }

        return {};
    }

    Error create_operation(
        std::unordered_map<int64_t, detail::ScopedHipdnnBackendDescriptor>& tensorDescs,
        std::vector<detail::ScopedHipdnnBackendDescriptor>& operations) const override
    {
        return detail::createLayernormOperation(attributes, tensorDescs, operations);
    }
};
} // namespace hipdnn_frontend::graph
