// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "Node.hpp"
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/RMSNormAttributes.hpp>
#include <hipdnn_frontend/detail/RMSNormPacker.hpp>
#include <hipdnn_frontend/detail/RMSNormUnpacker.hpp>
#include <hipdnn_frontend/node/detail/Utilities.hpp>

namespace hipdnn_frontend::graph
{
class RMSNormNode : public BaseNode<RMSNormNode, NodeType::RMS_NORM>
{
public:
    RMSNormAttributes attributes;

    RMSNormNode(RMSNormAttributes&& rmsnormAttrs, const GraphAttributes& graphAttrs)
        : BaseNode(graphAttrs)
        , attributes(std::move(rmsnormAttrs))
    {
    }

    Error unpack_from_descriptor(
        hipdnnBackendDescriptor_t opDesc,
        std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>>& tensorMap) override
    {
        RMSNormAttributes attrs;
        HIPDNN_CHECK_ERROR(detail::unpackRMSNormOperation(opDesc, tensorMap, attrs));
        attributes = std::move(attrs);
        return {};
    }

    Error pre_validate_node() const override
    {
        // Algorithm Overview:
        //   normalized_shape = trailing-suffix dims where scale[i] == input[i]; leading
        //   dims (incl. batch) are preserved. For each leading position:
        //     inv_rms = 1 / sqrt(mean(x^2 over normalized_shape) + epsilon)
        //     y       = scale * x * inv_rms (+ bias)
        //   inv_rms shape: leading dims preserved, normalized dims collapsed to 1.

        // SECTION 1: Validate Required Tensor Pointers
        HIPDNN_RETURN_IF_FALSE(attributes.get_x(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "RMSNormNode missing x for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_scale(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "RMSNormNode missing scale for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_y(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "RMSNormNode missing y for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_epsilon(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "RMSNormNode missing epsilon for pre-validation");

        // Get tensor references
        auto x = attributes.get_x();
        auto y = attributes.get_y();
        auto scale = attributes.get_scale();
        auto epsilon = attributes.get_epsilon();

        // SECTION 2: Validate Required Parameter Dimensions
        HIPDNN_CHECK_ERROR(detail::validateMinimumTensorDimensions(x, 2, "Input tensor (x)"));
        HIPDNN_CHECK_ERROR(detail::validateMinimumTensorDimensions(scale, 2, "Scale tensor"));

        // Epsilon provides numerical stability in the denominator: rms = sqrt(mean(x^2) + eps)
        HIPDNN_CHECK_ERROR(detail::validateScalarParameter(epsilon, "Epsilon"));

        // SECTION 3: Validate Output Tensor Shape Consistency
        // RMSNorm preserves tensor shape - it only transforms values, not dimensions.
        HIPDNN_CHECK_ERROR(
            detail::validateTensorShapesMatchIfSet(x, y, "Input tensor (x)", "Output tensor (y)"));

        // SECTION 4: Validate Scale Tensor Shape (encodes normalized_shape)
        HIPDNN_CHECK_ERROR(detail::validateScaleNormalizedShape(scale, x, "Scale tensor"));

        // Bias must have the same shape as scale (scale already validated above).
        auto bias = attributes.get_bias();
        if(bias)
        {
            HIPDNN_CHECK_ERROR(
                detail::validateTensorShapesMatchIfSet(scale, bias, "Scale tensor", "Bias tensor"));
        }

        // Validate forward_phase is set
        HIPDNN_RETURN_IF_EQ(attributes.get_forward_phase(),
                            NormFwdPhase::NOT_SET,
                            ErrorCode::ATTRIBUTE_NOT_SET,
                            "RMSNormNode forward_phase must be set to TRAINING or INFERENCE");

        // Validate inv_rms tensor based on forward_phase
        // Stats shape is derived from scale: where scale is non-1, stats must be 1
        if(attributes.get_forward_phase() == NormFwdPhase::TRAINING)
        {
            HIPDNN_CHECK_ERROR(detail::validateNormStatsShapeIfSet(
                attributes.get_inv_rms(), x, scale, "Inverse RMS tensor"));
        }

        return {ErrorCode::OK, ""};
    }

    Error infer_properties_node() override
    {
        auto x = attributes.get_x();
        auto y = attributes.get_y();
        auto scale = attributes.get_scale();

        if(!x)
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET, "RMSNormNode missing x for setting properties"};
        }

        if(!y)
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET, "RMSNormNode missing y for setting properties"};
        }

        if(!scale)
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET,
                    "RMSNormNode missing scale for setting properties"};
        }

        HIPDNN_CHECK_ERROR(attributes.fill_from_context(graph_attributes));

        if(y->get_dim().empty())
        {
            y->set_dim(x->get_dim());
        }

        if(y->get_stride().empty() && !x->get_stride().empty())
        {
            y->set_stride(x->get_stride());
        }

        // Scale strides inherit x's stride order.
        if(scale->get_stride().empty() && !scale->get_dim().empty())
        {
            if(!x->get_stride().empty() && x->get_stride().size() == scale->get_dim().size())
            {
                auto strideOrder = hipdnn_data_sdk::utilities::extractStrideOrder(x->get_stride());
                scale->set_stride(
                    hipdnn_data_sdk::utilities::generateStrides(scale->get_dim(), strideOrder));
            }
            else
            {
                scale->set_stride(hipdnn_data_sdk::utilities::generateStrides(scale->get_dim()));
            }
        }

        if(attributes.get_forward_phase() == NormFwdPhase::TRAINING)
        {
            auto invRms = attributes.get_inv_rms();
            if(invRms)
            {
                // Derive inv_rms dims from input and scale:
                // Where scale has a non-1 dim, inv_rms gets 1 (normalized dimension collapses).
                // Where scale has dim 1, inv_rms keeps the input dim.
                if(invRms->get_dim().empty())
                {
                    auto invRmsDims = x->get_dim();
                    const auto& scaleDims = scale->get_dim();
                    for(size_t i = 0; i < invRmsDims.size(); ++i)
                    {
                        if(scaleDims[i] != 1)
                        {
                            invRmsDims[i] = 1;
                        }
                    }
                    invRms->set_dim(invRmsDims);
                }

                if(invRms->get_stride().empty())
                {
                    if(!x->get_stride().empty())
                    {
                        auto strideOrder
                            = hipdnn_data_sdk::utilities::extractStrideOrder(x->get_stride());
                        invRms->set_stride(hipdnn_data_sdk::utilities::generateStrides(
                            invRms->get_dim(), strideOrder));
                    }
                    else
                    {
                        invRms->set_stride(
                            hipdnn_data_sdk::utilities::generateStrides(invRms->get_dim()));
                    }
                }
            }
        }

        // Bias inherits scale's shape and stride layout (validator enforces dim match).
        auto bias = attributes.get_bias();
        if(bias)
        {
            if(bias->get_dim().empty())
            {
                bias->set_dim(scale->get_dim());
            }
            if(bias->get_stride().empty())
            {
                bias->set_stride(scale->get_stride());
            }
        }

        return {};
    }

    Error create_operation(
        std::unordered_map<int64_t, detail::ScopedHipdnnBackendDescriptor>& tensorDescs,
        std::vector<detail::ScopedHipdnnBackendDescriptor>& operations) const override
    {
        return detail::createRMSNormOperation(attributes, tensorDescs, operations);
    }
};
}
