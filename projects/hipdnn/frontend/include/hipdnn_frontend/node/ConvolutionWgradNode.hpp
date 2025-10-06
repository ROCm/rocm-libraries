// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "Node.hpp"
#include <algorithm>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/ConvolutionWgradAttributes.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/utilities/ShapeUtilities.hpp>
#include <numeric>

namespace hipdnn_frontend::graph
{
class ConvolutionWgradNode : public BaseNode<ConvolutionWgradNode>
{
public:
    ConvWgradAttributes attributes;

    ConvolutionWgradNode(ConvWgradAttributes&& convAttrs, const GraphAttributes& graphAttrs)
        : BaseNode(graphAttrs)
        , attributes(std::move(convAttrs))
    {
    }

    Error pre_validate_node() const override
    {
        auto x = attributes.get_x();
        auto dy = attributes.get_dy();
        auto dw = attributes.get_dw();

        auto& xDims = x->get_dim();
        auto& dyDims = dy->get_dim();
        auto& dwDims = dw->get_dim();
        auto& dwStrides = dw->get_stride();

        auto spatialDims = dyDims.size() - 2; // N & C dimensions aren't spatial
        auto& prePadding = attributes.get_pre_padding();
        auto& postPadding = attributes.get_post_padding();
        auto& stride = attributes.get_stride();
        auto& dilation = attributes.get_dilation();

        HIPDNN_RETURN_IF_FALSE(x,
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "ConvolutionWgradNode missing x (input) for pre-validation");

        HIPDNN_RETURN_IF_FALSE(
            dy,
            ErrorCode::ATTRIBUTE_NOT_SET,
            "ConvolutionWgradNode missing dy (gradient of output) for pre-validation");

        HIPDNN_RETURN_IF_FALSE(
            dw,
            ErrorCode::ATTRIBUTE_NOT_SET,
            "ConvolutionWgradNode missing dw (gradient of weights) for pre-validation");

        HIPDNN_RETURN_IF_TRUE(attributes.get_pre_padding().empty(),
                              ErrorCode::ATTRIBUTE_NOT_SET,
                              "ConvolutionWgradNode missing pre_padding for pre-validation");

        HIPDNN_RETURN_IF_TRUE(attributes.get_post_padding().empty(),
                              ErrorCode::ATTRIBUTE_NOT_SET,
                              "ConvolutionWgradNode missing post_padding for pre-validation");

        HIPDNN_RETURN_IF_TRUE(attributes.get_stride().empty(),
                              ErrorCode::ATTRIBUTE_NOT_SET,
                              "ConvolutionWgradNode missing stride for pre-validation");

        HIPDNN_RETURN_IF_TRUE(attributes.get_dilation().empty(),
                              ErrorCode::ATTRIBUTE_NOT_SET,
                              "ConvolutionWgradNode missing dilation for pre-validation");

        HIPDNN_RETURN_IF_FALSE(
            x->validate_dims_and_strides_set_and_positive(),
            ErrorCode::INVALID_VALUE,
            "ConvolutionWgradNode: x tensor dimensions and strides must be set and positive");

        // dy implicitly checked here too since they must be equal
        HIPDNN_RETURN_IF_LT(
            xDims.size(),
            3,
            ErrorCode::INVALID_VALUE,
            "ConvolutionWgradNode: x tensor must have at least 3 dimensions (N, C, spatial)");

        HIPDNN_RETURN_IF_FALSE(
            dy->validate_dims_and_strides_set_and_positive(),
            ErrorCode::INVALID_VALUE,
            "ConvolutionWgradNode: dy tensor dimensions and strides must be set and positive");

        HIPDNN_RETURN_IF_NE(dyDims.size(),
                            xDims.size(),
                            ErrorCode::INVALID_VALUE,
                            "ConvolutionWgradNode: dy tensor dimension count must match x tensor "
                            "dimension count");

        HIPDNN_RETURN_IF_NE(
            xDims[0],
            dyDims[0],
            ErrorCode::INVALID_VALUE,
            "ConvolutionWgradNode: x tensor batch size must match dy tensor batch size");

        if(!dwDims.empty())
        {
            HIPDNN_RETURN_IF_NE(
                dwDims.size(),
                dyDims.size(),
                ErrorCode::INVALID_VALUE,
                "ConvolutionWgradNode: dw tensor dimension count must match dy tensor "
                "dimension count");

            // Validate output channels match between dy and dw tensors
            HIPDNN_RETURN_IF_NE(
                dyDims[1],
                dwDims[0],
                ErrorCode::INVALID_VALUE,
                "ConvolutionWgradNode: dy tensor channels must match dw tensor output channels");

            HIPDNN_RETURN_IF_NE(xDims[1] % dwDims[1],
                                0,
                                ErrorCode::INVALID_VALUE,
                                "ConvolutionWgradNode: x tensor channels must be divisible by dw "
                                "tensor input channels");

            // xDims[1] / dwDims[1] is group count
            auto groupCount = xDims[1] / dwDims[1];
            HIPDNN_RETURN_IF_NE(
                dwDims[0] % groupCount,
                0,
                ErrorCode::INVALID_VALUE,
                "ConvolutionWgradNode: dw tensor output channels must be divisible by "
                "the number of groups");

            HIPDNN_RETURN_IF_FALSE(
                dw->validate_dims_set_and_positive(),
                ErrorCode::INVALID_VALUE,
                "ConvolutionWgradNode: dw tensor dimensions must be set and positive");

            // Verifies that spatial dimensions are compatible
            for(size_t i = 0; i < spatialDims; ++i)
            {
                auto spatialIdx = i + 2;
                int64_t xDim = xDims[spatialIdx];
                int64_t dyDim = dyDims[spatialIdx];
                int64_t kernelDim = dwDims[spatialIdx];

                int64_t kernelSize = (dilation[i] * (kernelDim - 1)) + 1;
                int64_t expectedDyDim
                    = ((xDim + prePadding[i] + postPadding[i] - kernelSize) / stride[i]) + 1;

                // Verifying dy implicitly verifies dw and x
                HIPDNN_RETURN_IF_NE(
                    dyDim,
                    expectedDyDim,
                    ErrorCode::INVALID_VALUE,
                    "ConvolutionWgradNode: dy tensor spatial dimension at index "
                        + std::to_string(i) + " (" + std::to_string(dyDim)
                        + ") does not match expected dimension (" + std::to_string(expectedDyDim)
                        + ") given x dimensions, kernel size, padding, stride, and dilation");
            }
        }

        if(!dwStrides.empty())
        {
            HIPDNN_RETURN_IF_FALSE(dw->validate_dims_and_strides_set_and_positive(),
                                   ErrorCode::INVALID_VALUE,
                                   "ConvolutionWgradNode: dw tensor dimensions and strides "
                                   "must be set and positive");
        }

        HIPDNN_RETURN_IF_NE(
            prePadding.size(),
            spatialDims,
            ErrorCode::INVALID_VALUE,
            "ConvolutionWgradNode: pre_padding parameter count must match spatial dimension count");

        HIPDNN_RETURN_IF_NE(postPadding.size(),
                            spatialDims,
                            ErrorCode::INVALID_VALUE,
                            "ConvolutionWgradNode: post_padding parameter count must match spatial "
                            "dimension count");

        HIPDNN_RETURN_IF_NE(
            stride.size(),
            spatialDims,
            ErrorCode::INVALID_VALUE,
            "ConvolutionWgradNode: stride parameter count must match spatial dimension count");

        HIPDNN_RETURN_IF_NE(
            dilation.size(),
            spatialDims,
            ErrorCode::INVALID_VALUE,
            "ConvolutionWgradNode: dilation parameter count must match spatial dimension count");

        // Check spatial parameters for each dimension
        for(size_t i = 0; i < spatialDims; ++i)
        {
            auto prePad = prePadding[i];
            auto postPad = postPadding[i];
            auto strideVal = stride[i];
            auto dilationVal = dilation[i];

            HIPDNN_RETURN_IF_LT(
                strideVal, 1, ErrorCode::INVALID_VALUE, "ConvolutionWgradNode: Stride must be > 0");

            HIPDNN_RETURN_IF_LT(dilationVal,
                                1,
                                ErrorCode::INVALID_VALUE,
                                "ConvolutionWgradNode: Dilation must > 0");

            HIPDNN_RETURN_IF_LT(prePad,
                                0,
                                ErrorCode::INVALID_VALUE,
                                "ConvolutionWgradNode: Pre-padding must be non-negative");

            HIPDNN_RETURN_IF_LT(postPad,
                                0,
                                ErrorCode::INVALID_VALUE,
                                "ConvolutionWgradNode: Post-padding must be non-negative");
        }

        return {};
    }

    Error infer_properties_node() override
    {
        auto x = attributes.get_x();
        auto dy = attributes.get_dy();
        auto dw = attributes.get_dw();

        // Repeated checks from pre_validate_node for cases where this is called standalone
        HIPDNN_RETURN_IF_FALSE(x,
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "ConvolutionWgradNode missing x for setting properties");

        HIPDNN_RETURN_IF_FALSE(dy,
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "ConvolutionWgradNode missing dy for setting properties");

        HIPDNN_RETURN_IF_FALSE(dw,
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "ConvolutionWgradNode missing dw for setting properties");

        HIPDNN_CHECK_ERROR(attributes.fill_from_context(graph_attributes));

        auto dwDims = dw->get_dim();

        if(dwDims.empty())
        {
            auto& xDims = x->get_dim();
            auto& dyDims = dy->get_dim();

            dwDims.resize(dyDims.size());

            auto& prePadding = attributes.get_pre_padding();
            auto& postPadding = attributes.get_post_padding();
            auto& stride = attributes.get_stride();
            auto& dilation = attributes.get_dilation();

            dwDims[0] = dyDims[1]; // Output channels match dy channels

            // Impossible to infer group count without dw dimensions.
            // Therefore, assume groups = 1.
            dwDims[1] = xDims[1]; // Input channels (per group)

            // Calculate kernel spatial dimensions (k_2, ..., k_n)
            for(size_t i = 2; i < dyDims.size(); ++i)
            {
                auto spatialIdx = i - 2;

                HIPDNN_RETURN_IF_TRUE(
                    spatialIdx >= prePadding.size() || spatialIdx >= postPadding.size()
                        || spatialIdx >= stride.size() || spatialIdx >= dilation.size(),
                    ErrorCode::INVALID_VALUE,
                    "ConvolutionWgradNode: Insufficient padding/stride/dilation parameters for "
                    "spatial dimensions");

                auto xSize = xDims[i];
                auto dySize = dyDims[i];
                auto prePad = prePadding[spatialIdx];
                auto postPad = postPadding[spatialIdx];
                auto strideVal = stride[spatialIdx];
                auto dilationVal = dilation[spatialIdx];

                HIPDNN_RETURN_IF_LT(strideVal,
                                    1,
                                    ErrorCode::INVALID_VALUE,
                                    "ConvolutionWgradNode: Stride must be positive");

                HIPDNN_RETURN_IF_LT(dilationVal,
                                    1,
                                    ErrorCode::INVALID_VALUE,
                                    "ConvolutionWgradNode: Dilation must be positive");

                HIPDNN_RETURN_IF_LT(prePad,
                                    0,
                                    ErrorCode::INVALID_VALUE,
                                    "ConvolutionWgradNode: Pre-padding must be non-negative");

                HIPDNN_RETURN_IF_LT(postPad,
                                    0,
                                    ErrorCode::INVALID_VALUE,
                                    "ConvolutionWgradNode: Post-padding must be non-negative");

                // Conv fwd output spatial dim size:
                // out_i = (in_i + pre_padding + post_padding - dilation * (kernel_size - 1) - 1) / stride + 1

                // Solve for kernel size:
                // kernel_size = ((in_i + pre_padding + post_padding - stride * (out_i - 1) - 1) / dilation) + 1
                auto numerator = xSize + prePad + postPad - (strideVal * (dySize - 1)) - 1;

                HIPDNN_RETURN_IF_LT(
                    numerator,
                    0,
                    ErrorCode::INVALID_VALUE,
                    "ConvolutionWgradNode: Invalid spatial dimensions for kernel size inference");

                HIPDNN_RETURN_IF_NE(numerator % dilationVal,
                                    0,
                                    ErrorCode::INVALID_VALUE,
                                    "ConvolutionWgradNode: Spatial dimensions incompatible with "
                                    "dilation parameter");

                dwDims[i] = (numerator / dilationVal) + 1;
            }

            dw->set_dim(dwDims);
        }

        if(dw->get_stride().empty())
        {
            auto& xStrides = x->get_stride();
            auto& dwDimsFinal = dw->get_dim();

            HIPDNN_RETURN_IF_TRUE(
                xStrides.empty(),
                ErrorCode::ATTRIBUTE_NOT_SET,
                "ConvolutionWgradNode: Cannot infer dw strides - missing x strides");

            HIPDNN_RETURN_IF_NE(
                xStrides.size(),
                dwDimsFinal.size(),
                ErrorCode::ATTRIBUTE_NOT_SET,
                "ConvolutionWgradNode: Stride dimension mismatch between x and dw tensors");

            // Extract stride order from x tensor and apply to dw tensor
            auto strideOrder = hipdnn_sdk::utilities::extractStrideOrder(xStrides);

            // Generate dw strides using the extracted stride order and dw dimensions
            auto dwStrides = hipdnn_sdk::utilities::generateStrides(dwDimsFinal, strideOrder);

            dw->set_stride(dwStrides);
        }

        return {};
    }

    flatbuffers::Offset<hipdnn_sdk::data_objects::Node>
        pack_node(flatbuffers::FlatBufferBuilder& builder) const override
    {
        return hipdnn_sdk::data_objects::CreateNodeDirect(
            builder,
            attributes.get_name().c_str(),
            hipdnn_sdk::data_objects::NodeAttributes::ConvolutionWrwAttributes,
            attributes.pack_attributes(builder).Union());
    }
};

typedef ConvolutionWgradNode WgradNode;
}
