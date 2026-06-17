// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "Node.hpp"
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/RMSNormBackwardAttributes.hpp>
#include <hipdnn_frontend/detail/RMSNormBackwardPacker.hpp>
#include <hipdnn_frontend/detail/RMSNormBackwardUnpacker.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>
#include <hipdnn_frontend/node/detail/Utilities.hpp>

namespace hipdnn_frontend::graph
{
class RMSNormBackwardNode : public BaseNode<RMSNormBackwardNode, NodeType::RMS_NORM_BACKWARD>
{
public:
    RMSNormBackwardAttributes attributes;

    RMSNormBackwardNode(RMSNormBackwardAttributes&& attrs, const GraphAttributes& graphAttrs)
        : BaseNode(graphAttrs)
        , attributes(std::move(attrs))
    {
    }

    Error unpack_from_descriptor(
        hipdnnBackendDescriptor_t opDesc,
        std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>>& tensorMap) override
    {
        RMSNormBackwardAttributes attrs;
        HIPDNN_CHECK_ERROR(detail::unpackRMSNormBackwardOperation(opDesc, tensorMap, attrs));
        attributes = std::move(attrs);
        return {};
    }

    Error pre_validate_node() const override
    {
        // Validate required tensor pointers
        HIPDNN_RETURN_IF_FALSE(attributes.get_dy(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "RMSNormBackwardNode missing dy (input) for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_x(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "RMSNormBackwardNode missing x (input) for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_scale(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "RMSNormBackwardNode missing scale (input) for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_inv_rms(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "RMSNormBackwardNode missing inv_rms (input) for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_dx(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "RMSNormBackwardNode missing dx (output) for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_dscale(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "RMSNormBackwardNode missing dscale (output) for pre-validation");

        auto dyTensor = attributes.get_dy();
        auto xTensor = attributes.get_x();
        auto scaleTensor = attributes.get_scale();
        auto dxTensor = attributes.get_dx();
        auto dscaleTensor = attributes.get_dscale();
        auto invRmsTensor = attributes.get_inv_rms();

        HIPDNN_CHECK_ERROR(detail::validateMinimumTensorDimensions(xTensor, 2, "Input tensor (x)"));
        HIPDNN_CHECK_ERROR(
            detail::validateMinimumTensorDimensions(dyTensor, 2, "Gradient input tensor (dy)"));
        HIPDNN_CHECK_ERROR(detail::validateMinimumTensorDimensions(scaleTensor, 2, "Scale tensor"));

        // RMSNorm preserves tensor shape - it only transforms values, not dimensions.
        HIPDNN_CHECK_ERROR(detail::validateTensorShapesMatch(
            xTensor, dyTensor, "Input tensor (x)", "Gradient input tensor (dy)"));

        // dx may not have dimensions set yet (will be inferred)
        HIPDNN_CHECK_ERROR(detail::validateTensorShapesMatchIfSet(
            xTensor, dxTensor, "Input tensor (x)", "Gradient output tensor (dx)"));

        // Validate Scale Tensor Shape (encodes normalized_shape)
        HIPDNN_CHECK_ERROR(
            detail::validateScaleNormalizedShape(scaleTensor, xTensor, "Scale tensor"));
        HIPDNN_CHECK_ERROR(detail::validateTensorShapesMatchIfSet(
            scaleTensor, dscaleTensor, "Input tensor (scale)", "Gradient output tensor (dscale)"));

        // Bias must have the same shape as scale (scale already validated above).
        auto dbiasTensor = attributes.get_dbias();
        if(dbiasTensor)
        {
            HIPDNN_CHECK_ERROR(
                detail::validateTensorShapesMatchIfSet(scaleTensor,
                                                       dbiasTensor,
                                                       "Input tensor (scale)",
                                                       "Gradient output tensor (dbias)"));
        }

        // Validate invrms shape
        // NOLINTNEXTLINE(readability-suspicious-call-argument)
        HIPDNN_CHECK_ERROR(detail::validateNormStatsShapeIfSet(
            invRmsTensor, xTensor, scaleTensor, "Inverse RMS tensor"));
        return {};
    }

    Error infer_properties_node() override
    {
        // Validate required tensor pointers
        HIPDNN_RETURN_IF_FALSE(attributes.get_dy(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "RMSNormBackwardNode missing dy for setting properties");

        HIPDNN_RETURN_IF_FALSE(attributes.get_x(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "RMSNormBackwardNode missing x for setting properties");

        HIPDNN_RETURN_IF_FALSE(attributes.get_scale(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "RMSNormBackwardNode missing scale for setting properties");

        HIPDNN_RETURN_IF_FALSE(attributes.get_dx(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "RMSNormBackwardNode missing dx for setting properties");

        HIPDNN_RETURN_IF_FALSE(attributes.get_dscale(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "RMSNormBackwardNode missing dscale for setting properties");

        HIPDNN_CHECK_ERROR(attributes.fill_from_context(graph_attributes));

        auto dxTensor = attributes.get_dx();
        auto dyTensor = attributes.get_dy();
        auto dscaleTensor = attributes.get_dscale();
        auto scaleTensor = attributes.get_scale();

        // Infer dx dimensions and strides from dy (same shape)
        if(dxTensor->get_dim().empty())
        {
            dxTensor->set_dim(dyTensor->get_dim());
        }

        if(dxTensor->get_stride().empty())
        {
            dxTensor->set_stride(dyTensor->get_stride());
        }

        // Infer dscale dimensions and strides from scale (same shape)
        if(dscaleTensor->get_dim().empty())
        {
            dscaleTensor->set_dim(scaleTensor->get_dim());
        }

        if(dscaleTensor->get_stride().empty())
        {
            auto strideOrder
                = hipdnn_data_sdk::utilities::extractStrideOrder(scaleTensor->get_stride());
            dscaleTensor->set_stride(
                hipdnn_data_sdk::utilities::generateStrides(dscaleTensor->get_dim(), strideOrder));
        }

        // Infer dbias dimensions and strides from scale (same shape) if dbias is set
        auto dbiasTensor = attributes.get_dbias();
        if(dbiasTensor)
        {
            if(dbiasTensor->get_dim().empty())
            {
                dbiasTensor->set_dim(scaleTensor->get_dim());
            }

            if(dbiasTensor->get_stride().empty())
            {
                auto strideOrder
                    = hipdnn_data_sdk::utilities::extractStrideOrder(scaleTensor->get_stride());
                dbiasTensor->set_stride(hipdnn_data_sdk::utilities::generateStrides(
                    dbiasTensor->get_dim(), strideOrder));
            }
        }

        return {};
    }

    Error create_operation(
        std::unordered_map<int64_t, detail::ScopedHipdnnBackendDescriptor>& tensorDescs,
        std::vector<detail::ScopedHipdnnBackendDescriptor>& operations) const override
    {
        return detail::createRMSNormBackwardOperation(attributes, tensorDescs, operations);
    }
};
}
