// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "Node.hpp"
#include <algorithm>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/ConvolutionDgradAttributes.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/utilities/ShapeUtilities.hpp>
#include <numeric>

namespace hipdnn_frontend::graph
{
class ConvolutionDgradNode : public BaseNode<ConvolutionDgradNode>
{
public:
    ConvDgradAttributes attributes;

    ConvolutionDgradNode(ConvDgradAttributes&& convAttrs, const GraphAttributes& graphAttrs)
        : BaseNode(graphAttrs)
        , attributes(std::move(convAttrs))
    {
    }

    Error pre_validate_node() const override
    {
        HIPDNN_RETURN_IF_FALSE(
            attributes.get_dy(),
            ErrorCode::ATTRIBUTE_NOT_SET,
            "ConvolutionDgradNode missing dy (gradient of output) for pre-validation");

        HIPDNN_RETURN_IF_FALSE(attributes.get_w(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "ConvolutionDgradNode missing w (weights) for pre-validation");

        HIPDNN_RETURN_IF_FALSE(
            attributes.get_dx(),
            ErrorCode::ATTRIBUTE_NOT_SET,
            "ConvolutionDgradNode missing dx (gradient of input) for pre-validation");

        HIPDNN_RETURN_IF_TRUE(attributes.get_pre_padding().empty(),
                              ErrorCode::ATTRIBUTE_NOT_SET,
                              "ConvolutionDgradNode missing pre_padding for pre-validation");

        HIPDNN_RETURN_IF_TRUE(attributes.get_post_padding().empty(),
                              ErrorCode::ATTRIBUTE_NOT_SET,
                              "ConvolutionDgradNode missing post_padding for pre-validation");

        HIPDNN_RETURN_IF_TRUE(attributes.get_stride().empty(),
                              ErrorCode::ATTRIBUTE_NOT_SET,
                              "ConvolutionDgradNode missing stride for pre-validation");

        HIPDNN_RETURN_IF_TRUE(attributes.get_dilation().empty(),
                              ErrorCode::ATTRIBUTE_NOT_SET,
                              "ConvolutionDgradNode missing dilation for pre-validation");

        auto dy = attributes.get_dy();
        auto w = attributes.get_w();
        auto dx = attributes.get_dx();

        auto& dyDims = dy->get_dim();

        HIPDNN_RETURN_IF_FALSE(
            dy->validate_dims_and_strides_set_and_positive(),
            ErrorCode::INVALID_VALUE,
            "ConvolutionDgradNode: dy tensor dimensions and strides must be set and positive");

        HIPDNN_RETURN_IF_LT(
            dyDims.size(),
            3,
            ErrorCode::INVALID_VALUE,
            "ConvolutionDgradNode: dy tensor must have at least 3 dimensions (N, C, spatial)");

        auto& wDims = w->get_dim();

        HIPDNN_RETURN_IF_FALSE(
            w->validate_dims_and_strides_set_and_positive(),
            ErrorCode::INVALID_VALUE,
            "ConvolutionDgradNode: Weight tensor dimensions and strides must be set and positive");

        HIPDNN_RETURN_IF_NE(
            wDims.size(),
            dyDims.size(),
            ErrorCode::INVALID_VALUE,
            "ConvolutionDgradNode: Weight tensor dimension count must match dy tensor "
            "dimension count");

        // Validate output channels match between dy and w tensors
        HIPDNN_RETURN_IF_NE(
            dyDims[1],
            wDims[0],
            ErrorCode::INVALID_VALUE,
            "ConvolutionDgradNode: dy tensor channels must match weight tensor output channels");

        auto& dxDims = dx->get_dim();
        auto& dxStrides = dx->get_stride();

        if(!dxDims.empty())
        {
            HIPDNN_RETURN_IF_NE(
                dxDims.size(),
                dyDims.size(),
                ErrorCode::INVALID_VALUE,
                "ConvolutionDgradNode: dx tensor dimension count must match dy tensor "
                "dimension count");

            // Validate batch size matches
            HIPDNN_RETURN_IF_NE(dxDims[0],
                                dyDims[0],
                                ErrorCode::INVALID_VALUE,
                                "ConvolutionDgradNode: dx tensor batch size must match dy "
                                "tensor batch size");

            HIPDNN_RETURN_IF_NE(
                dxDims[1] % wDims[1],
                0,
                ErrorCode::INVALID_VALUE,
                "ConvolutionDgradNode: dx tensor channels must be divisible by weight "
                "tensor input channels");

            // dxDims[1] / wDims[1] is group count
            // weightChannels = inputChannels / groups
            // groups = inputChannels / weightChannels
            auto groupCount = dxDims[1] / wDims[1];
            HIPDNN_RETURN_IF_NE(
                wDims[0] % groupCount,
                0,
                ErrorCode::INVALID_VALUE,
                "ConvolutionDgradNode: Weight tensor output channels must be divisible by "
                "the number of groups");

            HIPDNN_RETURN_IF_FALSE(
                dx->validate_dims_set_and_positive(),
                ErrorCode::INVALID_VALUE,
                "ConvolutionDgradNode: dx tensor dimensions must be set and positive");
        }

        if(!dxStrides.empty())
        {
            HIPDNN_RETURN_IF_FALSE(dx->validate_dims_and_strides_set_and_positive(),
                                   ErrorCode::INVALID_VALUE,
                                   "ConvolutionDgradNode: dx tensor dimensions and strides "
                                   "must be set and positive");
        }

        // Validate spatial parameter counts match spatial dimensions
        auto spatialDims = dyDims.size() - 2; // Skip N and C dimensions
        auto& prePadding = attributes.get_pre_padding();
        auto& postPadding = attributes.get_post_padding();
        auto& stride = attributes.get_stride();
        auto& dilation = attributes.get_dilation();

        HIPDNN_RETURN_IF_NE(
            prePadding.size(),
            spatialDims,
            ErrorCode::INVALID_VALUE,
            "ConvolutionDgradNode: pre_padding parameter count must match spatial dimension count");

        HIPDNN_RETURN_IF_NE(postPadding.size(),
                            spatialDims,
                            ErrorCode::INVALID_VALUE,
                            "ConvolutionDgradNode: post_padding parameter count must match spatial "
                            "dimension count");

        HIPDNN_RETURN_IF_NE(
            stride.size(),
            spatialDims,
            ErrorCode::INVALID_VALUE,
            "ConvolutionDgradNode: stride parameter count must match spatial dimension count");

        HIPDNN_RETURN_IF_NE(
            dilation.size(),
            spatialDims,
            ErrorCode::INVALID_VALUE,
            "ConvolutionDgradNode: dilation parameter count must match spatial dimension count");

        // Check spatial parameters for each dimension
        for(size_t i = 0; i < spatialDims; ++i)
        {
            auto prePad = prePadding[i];
            auto postPad = postPadding[i];
            auto strideVal = stride[i];
            auto dilationVal = dilation[i];

            HIPDNN_RETURN_IF_LT(
                strideVal, 1, ErrorCode::INVALID_VALUE, "ConvolutionDgradNode: Stride must be > 0");

            HIPDNN_RETURN_IF_LT(dilationVal,
                                1,
                                ErrorCode::INVALID_VALUE,
                                "ConvolutionDgradNode: Dilation must > 0");

            HIPDNN_RETURN_IF_LT(prePad,
                                0,
                                ErrorCode::INVALID_VALUE,
                                "ConvolutionDgradNode: Pre-padding must be non-negative");

            HIPDNN_RETURN_IF_LT(postPad,
                                0,
                                ErrorCode::INVALID_VALUE,
                                "ConvolutionDgradNode: Post-padding must be non-negative");
        }

        return {};
    }

    Error infer_properties_node() override
    {
        auto dy = attributes.get_dy();
        auto w = attributes.get_w();
        auto dx = attributes.get_dx();

        HIPDNN_RETURN_IF_FALSE(dy,
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "ConvolutionDgradNode missing dy for setting properties");

        HIPDNN_RETURN_IF_FALSE(w,
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "ConvolutionDgradNode missing w for setting properties");

        HIPDNN_RETURN_IF_FALSE(dx,
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "ConvolutionDgradNode missing dx for setting properties");

        HIPDNN_CHECK_ERROR(attributes.fill_from_context(graph_attributes));

        auto dxDims = dx->get_dim();

        if(dxDims.empty())
        {
            auto& dyDims = dy->get_dim();
            auto& wDims = w->get_dim();

            dxDims.resize(dyDims.size());

            auto& prePadding = attributes.get_pre_padding();
            auto& postPadding = attributes.get_post_padding();
            auto& stride = attributes.get_stride();
            auto& dilation = attributes.get_dilation();

            dxDims[0] = dyDims[0]; // N (batch) matches dy

            // Impossible to infer group count without dx dimensions.
            // Therefore, assume groups = 1.
            dxDims[1] = wDims[1]; // C (input channels)

            // We calculate spatial dimensions (i_2, ..., i_n)
            for(size_t i = 2; i < dyDims.size(); ++i)
            {
                auto spatialIdx = i - 2;

                HIPDNN_RETURN_IF_TRUE(
                    spatialIdx >= prePadding.size() || spatialIdx >= postPadding.size()
                        || spatialIdx >= stride.size() || spatialIdx >= dilation.size(),
                    ErrorCode::INVALID_VALUE,
                    "ConvolutionDgradNode: Insufficient padding/stride/dilation parameters for "
                    "spatial dimensions");

                auto dySize = dyDims[i];
                auto kernelSize = wDims[i];
                auto prePad = prePadding[spatialIdx];
                auto postPad = postPadding[spatialIdx];
                auto strideVal = stride[spatialIdx];
                auto dilationVal = dilation[spatialIdx];

                HIPDNN_RETURN_IF_LT(strideVal,
                                    1,
                                    ErrorCode::INVALID_VALUE,
                                    "ConvolutionDgradNode: Stride must be positive");

                HIPDNN_RETURN_IF_LT(dilationVal,
                                    1,
                                    ErrorCode::INVALID_VALUE,
                                    "ConvolutionDgradNode: Dilation must be positive");

                HIPDNN_RETURN_IF_LT(prePad,
                                    0,
                                    ErrorCode::INVALID_VALUE,
                                    "ConvolutionDgradNode: Pre-padding must be non-negative");

                HIPDNN_RETURN_IF_LT(postPad,
                                    0,
                                    ErrorCode::INVALID_VALUE,
                                    "ConvolutionDgradNode: Post-padding must be non-negative");

                // Conv fwd output spatial dim size:
                // out_i = (in_i + pre_padding + post_padding - dilation * (kernel_size - 1) - 1) / stride + 1

                // Solve for input spatial dim size:
                // in_i = stride * (out_i - 1) - pre_padding - post_padding + dilation * (kernel_size - 1) + 1)
                auto dilatedKernelSize = (dilationVal * (kernelSize - 1)) + 1;

                dxDims[i] = strideVal * (dySize - 1) + dilatedKernelSize - prePad - postPad;
            }

            dx->set_dim(dxDims);
        }

        if(dx->get_stride().empty())
        {
            auto& dyStrides = dy->get_stride();
            auto& dxDimsFinal = dx->get_dim();

            HIPDNN_RETURN_IF_TRUE(
                dyStrides.empty(),
                ErrorCode::ATTRIBUTE_NOT_SET,
                "ConvolutionDgradNode: Cannot infer dx strides - missing dy strides");

            HIPDNN_RETURN_IF_NE(
                dyStrides.size(),
                dxDimsFinal.size(),
                ErrorCode::ATTRIBUTE_NOT_SET,
                "ConvolutionDgradNode: Stride dimension mismatch between dy and dx tensors");

            // Extract stride order from dy tensor and apply to dx tensor
            auto strideOrder = hipdnn_sdk::utilities::extractStrideOrder(dyStrides);

            // Generate dx strides using the extracted stride order and dx dimensions
            auto dxStrides = hipdnn_sdk::utilities::generateStrides(dxDimsFinal, strideOrder);

            dx->set_stride(dxStrides);
        }

        return {};
    }

    flatbuffers::Offset<hipdnn_sdk::data_objects::Node>
        pack_node(flatbuffers::FlatBufferBuilder& builder) const override
    {
        return hipdnn_sdk::data_objects::CreateNodeDirect(
            builder,
            attributes.get_name().c_str(),
            hipdnn_sdk::data_objects::NodeAttributes::ConvolutionBwdAttributes,
            attributes.pack_attributes(builder).Union());
    }
};

typedef ConvolutionDgradNode DgradNode;
}
