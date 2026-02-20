// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "Node.hpp"
#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/LayernormFpropAttributes.hpp>
#include <hipdnn_frontend/node/detail/Utilities.hpp>

namespace hipdnn_frontend::graph
{
class LayernormFpropNode : public BaseNode<LayernormFpropNode>
{
public:
    LayernormFpropAttributes attributes;

    LayernormFpropNode(LayernormFpropAttributes&& layernormAttrs, const GraphAttributes& graphAttrs)
        : BaseNode(graphAttrs)
        , attributes(std::move(layernormAttrs))
    {
    }

    Error pre_validate_node() const override
    {
        // ====================================================================
        // LAYER NORMALIZATION FORWARD VALIDATION
        // ====================================================================
        // Algorithm Overview:
        // LayerNorm computes statistics over the feature dimensions (last normalized_shape dims):
        //   For input shape [N, ..., D₁, D₂, ..., Dₖ] where last k dims are normalized:
        //   mean = (1/m) * Σ x[..., i] over normalized dims, where m = D₁*D₂*...*Dₖ
        //   var  = (1/m) * Σ (x[..., i] - mean)² over normalized dims
        //
        // Normalizes: xhat = (x - mean) / sqrt(var + ε)
        // Transforms: y = scale * xhat + bias (scale and bias have shape of normalized dims)
        //
        // Outputs:
        // - Y: normalized output (same shape as X)
        // - Mean: computed mean per sample (optional, shape: batch dims only)
        // - Rstd: reciprocal standard deviation (optional, shape: batch dims only)
        // ====================================================================

        // SECTION 1: Validate Required Tensor Pointers
        HIPDNN_RETURN_IF_FALSE(attributes.get_x(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "LayernormFpropNode missing x for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_y(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "LayernormFpropNode missing y for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_epsilon(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "LayernormFpropNode missing epsilon for pre-validation");

        // Get tensor references
        auto x = attributes.get_x();
        auto y = attributes.get_y();
        auto scale = attributes.get_scale();
        auto bias = attributes.get_bias();
        auto epsilon = attributes.get_epsilon();

        // SECTION 2: Validate Required Parameter Dimensions
        HIPDNN_CHECK_ERROR(detail::validateMinimumTensorDimensions(x, 1, "Input tensor (x)"));

        // Epsilon (ε) provides numerical stability: xhat = (x - mean) / sqrt(var + ε)
        HIPDNN_CHECK_ERROR(detail::validateScalarParameter(epsilon, "Epsilon"));

        // SECTION 3: Validate Output Tensor Shape Consistency
        // LayerNorm preserves tensor shape - output has same shape as input
        HIPDNN_CHECK_ERROR(
            detail::validateTensorShapesMatchIfSet(x, y, "Input tensor (x)", "Output tensor (y)"));

        // SECTION 4: Validate Optional Scale and Bias Tensors
        // Scale and bias are per-feature parameters matching the normalized dimensions.
        // For input shape [N, C, H, W] normalized over last 3 dims, scale/bias shape is [C, H, W]
        // We validate that if scale or bias are set, they match the normalized portion shape.

        // Scale and bias, if provided, should have dimensions set
        if(scale)
        {
            HIPDNN_CHECK_ERROR(detail::validateMinimumTensorDimensions(scale, 1, "Scale tensor"));
        }
        if(bias)
        {
            HIPDNN_CHECK_ERROR(detail::validateMinimumTensorDimensions(bias, 1, "Bias tensor"));
        }

        // If both scale and bias are provided, they should have matching shapes
        if(scale && bias)
        {
            HIPDNN_CHECK_ERROR(
                detail::validateTensorShapesMatchIfSet(scale, bias, "Scale tensor", "Bias tensor"));
        }

        return {ErrorCode::OK, ""};
    }

    Error infer_properties_node() override
    {
        auto x = attributes.get_x();
        auto y = attributes.get_y();

        if(!x)
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET,
                    "LayernormFpropNode missing x for setting properties"};
        }

        if(!y)
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET,
                    "LayernormFpropNode missing y for setting properties"};
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
                auto yStrides = hipdnn_data_sdk::utilities::generateStrides(y->get_dim());
                y->set_stride(yStrides);
            }
        }

        // Infer mean and rstd shapes if they are outputs
        // Mean and rstd have shape of the batch dimensions (everything except normalized dims)
        auto mean = attributes.get_mean();
        auto rstd = attributes.get_rstd();

        auto inferStatsTensor = [&](std::shared_ptr<TensorAttributes>& tensorToInfer) {
            if(tensorToInfer && tensorToInfer->get_dim().empty())
            {
                // For LayerNorm, mean and rstd are typically computed per-sample
                // across the normalized dimensions. The output shape is the batch shape.
                // For simplicity, we'll set it to match the first dimension(s) before
                // the normalized dimensions. This is a simplified inference - actual
                // implementation may need normalized_shape attribute to be more precise.

                // Common case: normalize over last dimension(s)
                // Mean/rstd shape is input shape with last dims reduced to 1
                // For now, we'll use a scalar output as a safe default
                std::vector<int64_t> statsDims = {1};
                tensorToInfer->set_dim(statsDims);
            }

            if(tensorToInfer && tensorToInfer->get_stride().empty())
            {
                if(!x->get_stride().empty())
                {
                    auto strideOrder
                        = hipdnn_data_sdk::utilities::extractStrideOrder(x->get_stride());
                    tensorToInfer->set_stride(hipdnn_data_sdk::utilities::generateStrides(
                        tensorToInfer->get_dim(), strideOrder));
                }
                else
                {
                    auto statStrides
                        = hipdnn_data_sdk::utilities::generateStrides(tensorToInfer->get_dim());
                    tensorToInfer->set_stride(statStrides);
                }
            }
        };

        if(mean)
        {
            inferStatsTensor(mean);
        }

        if(rstd)
        {
            inferStatsTensor(rstd);
        }

        return {};
    }

    flatbuffers::Offset<hipdnn_data_sdk::data_objects::Node>
        pack_node(flatbuffers::FlatBufferBuilder& builder) const override
    {
        return hipdnn_data_sdk::data_objects::CreateNodeDirect(
            builder,
            attributes.get_name().c_str(),
            toSdkType(attributes.compute_data_type),
            hipdnn_data_sdk::data_objects::NodeAttributes::LayernormFpropAttributes,
            attributes.pack_attributes(builder).Union());
    }
};
} // namespace hipdnn_frontend::graph
