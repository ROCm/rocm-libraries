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
        // ====================================================================
        // RMS NORMALIZATION FORWARD VALIDATION
        // (Normalization across channels without mean subtraction)
        // ====================================================================
        // Algorithm Overview:
        // For each (batch, spatial) position, RMSNorm computes:
        //   rms[n,h,w]  = sqrt((1/C) * sum_c x[n,c,h,w]^2 + epsilon)
        //   y[n,c,h,w]  = scale[c] * (x[n,c,h,w] / rms[n,h,w]) + bias[c]
        // ====================================================================

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

        // SECTION 4: Validate Channel Dimensions and Scale Tensor Shape
        // Scale is per-channel with shape [1, C, D, H, ...]
        HIPDNN_CHECK_ERROR(detail::validateNonBatchShapeMatch(scale, x, "Scale tensor"));

        // Validate optional bias tensor (per-channel with shape [1, C, D, H, ...])
        auto bias = attributes.get_bias();
        if(bias)
        {
            HIPDNN_CHECK_ERROR(detail::validateNonBatchShapeMatch(bias, x, "Bias tensor"));
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

        if(!x)
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET, "RMSNormNode missing x for setting properties"};
        }

        if(!y)
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET, "RMSNormNode missing y for setting properties"};
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

        auto inferCTensor = [&](std::shared_ptr<TensorAttributes>& tensorToInfer) {
            if(tensorToInfer->get_dim().empty())
            {
                std::vector<int64_t> tensorDims(x->get_dim().size(), 1);
                tensorDims[1] = x->get_dim()[1];
                tensorToInfer->set_dim(tensorDims);
            }

            if(tensorToInfer->get_stride().empty())
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
                    tensorToInfer->set_stride(
                        hipdnn_data_sdk::utilities::generateStrides(tensorToInfer->get_dim()));
                }
            }
        };

        if(attributes.get_forward_phase() == NormFwdPhase::TRAINING)
        {
            auto invRms = attributes.get_inv_rms();
            if(invRms)
            {
                // Derive inv_rms dims from input and scale:
                // Where scale has a non-1 dim, inv_rms gets 1 (normalized dimension collapses).
                // Where scale has dim 1, inv_rms keeps the input dim.
                // Fallback (no scale dims): all dims except batch become 1 → [N, 1, 1, 1].
                if(invRms->get_dim().empty())
                {
                    auto invRmsDims = x->get_dim();
                    auto scale = attributes.get_scale();
                    if(scale && !scale->get_dim().empty())
                    {
                        const auto& scaleDims = scale->get_dim();
                        for(size_t i = 0; i < invRmsDims.size(); ++i)
                        {
                            if(scaleDims[i] != 1)
                            {
                                invRmsDims[i] = 1;
                            }
                        }
                    }
                    else
                    {
                        for(size_t i = 1; i < invRmsDims.size(); ++i)
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

        auto bias = attributes.get_bias();
        if(bias)
        {
            inferCTensor(bias);
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
