// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "Node.hpp"
#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/RMSNormAttributes.hpp>
#include <hipdnn_frontend/detail/RMSNormPacker.hpp>
#include <hipdnn_frontend/node/detail/Utilities.hpp>

namespace hipdnn_frontend::graph
{
class RMSNormNode : public BaseNode<RMSNormNode>
{
public:
    RMSNormAttributes attributes;

    RMSNormNode(RMSNormAttributes&& rmsnormAttrs, const GraphAttributes& graphAttrs)
        : BaseNode(graphAttrs)
        , attributes(std::move(rmsnormAttrs))
    {
    }

    Error pre_validate_node() const override
    {
        // ====================================================================
        // RMS NORMALIZATION FORWARD VALIDATION
        // (Per-channel normalization without mean subtraction)
        // ====================================================================
        // Algorithm Overview:
        // For each channel c, RMSNorm computes the root mean square:
        //   rms_c = sqrt((1/m) * sum_{n,h,w} x[n,c,h,w]^2 + epsilon)
        //
        //   y[n,c,h,w] = (x[n,c,h,w] / rms_c) * scale_c + bias_c
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
        // Scale is per-channel with shape [1, C, 1, 1, ...]
        auto& xDims = x->get_dim();
        int64_t channels = xDims[1];

        HIPDNN_CHECK_ERROR(detail::validateChannelOnlyTensorShape(scale, channels, "Scale tensor"));

        // Validate optional bias tensor (per-channel with shape [1, C, 1, 1, ...])
        HIPDNN_CHECK_ERROR(
            detail::validateChannelOnlyShapeIfSet(attributes.get_bias(), channels, "Bias tensor"));

        // Validate forward_phase is set
        HIPDNN_RETURN_IF_EQ(attributes.get_forward_phase(),
                            NormFwdPhase::NOT_SET,
                            ErrorCode::ATTRIBUTE_NOT_SET,
                            "RMSNormNode forward_phase must be set to TRAINING or INFERENCE");

        // Validate inv_rms tensor based on forward_phase
        if(attributes.get_forward_phase() == NormFwdPhase::TRAINING)
        {
            HIPDNN_CHECK_ERROR(detail::validateChannelOnlyShapeIfSet(
                attributes.get_inv_rms(), channels, "Inverse RMS tensor"));
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
                inferCTensor(invRms);
            }
        }

        auto bias = attributes.get_bias();
        if(bias)
        {
            inferCTensor(bias);
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
            hipdnn_data_sdk::data_objects::NodeAttributes::RMSNormAttributes,
            attributes.pack_attributes(builder).Union());
    }

    Error create_operation(
        std::unordered_map<int64_t, detail::ScopedHipdnnBackendDescriptor>& tensorDescs,
        std::vector<detail::ScopedHipdnnBackendDescriptor>& operations) const override
    {
        return detail::createRMSNormOperation(attributes, tensorDescs, operations);
    }
};
}
