// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "Node.hpp"
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/ConvolutionWgradAttributes.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/detail/ConvolutionWgradPacker.hpp>
#include <hipdnn_frontend/detail/ConvolutionWgradUnpacker.hpp>

namespace hipdnn_frontend::graph
{
class ConvolutionWgradNode : public BaseNode<ConvolutionWgradNode, NodeType::CONVOLUTION_WGRAD>
{
public:
    ConvWgradAttributes attributes;

    ConvolutionWgradNode(ConvWgradAttributes&& convAttrs, const GraphAttributes& graphAttrs)
        : BaseNode(graphAttrs)
        , attributes(std::move(convAttrs))
    {
    }

    Error unpack_from_descriptor(
        hipdnnBackendDescriptor_t opDesc,
        std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>>& tensorMap) override
    {
        ConvWgradAttributes attrs;
        HIPDNN_CHECK_ERROR(detail::unpackConvWgradOperation(opDesc, tensorMap, attrs));
        attributes = std::move(attrs);
        return {};
    }

    Error pre_validate_node() const override
    {
        auto x = attributes.get_x();
        auto dy = attributes.get_dy();
        auto dw = attributes.get_dw();

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

        auto& xDims = x->get_dim();
        auto& dyDims = dy->get_dim();
        auto& dwDims = dw->get_dim();

        auto& prePadding = attributes.get_pre_padding();
        auto& postPadding = attributes.get_post_padding();
        auto& stride = attributes.get_stride();
        auto& dilation = attributes.get_dilation();

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

        // dy implicitly checked here too since they must be equal
        HIPDNN_RETURN_IF_LT(
            xDims.size(),
            3,
            ErrorCode::INVALID_VALUE,
            "ConvolutionWgradNode: x tensor must have at least 3 dimensions (N, C, spatial)");

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
        auto spatialDims = dyDims.size() - 2; // N & C dimensions aren't spatial

        HIPDNN_RETURN_IF_TRUE(dwDims.empty(),
                              ErrorCode::ATTRIBUTE_NOT_SET,
                              "ConvolutionWgradNode: output dimension inference is not possible; "
                              "set dw dimensions explicitly.");

        HIPDNN_RETURN_IF_NE(dwDims.size(),
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
        HIPDNN_RETURN_IF_NE(dwDims[0] % groupCount,
                            0,
                            ErrorCode::INVALID_VALUE,
                            "ConvolutionWgradNode: dw tensor output channels must be divisible by "
                            "the number of groups");

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

        // Check spatial parameters and verify spatial dimensions are compatible.
        // Parameter positivity (stride/dilation > 0) is validated before the
        // stride division below to avoid a divide-by-zero on malformed input.
        for(size_t i = 0; i < spatialDims; ++i)
        {
            const int64_t prePad = prePadding[i];
            const int64_t postPad = postPadding[i];
            const int64_t strideVal = stride[i];
            const int64_t dilationVal = dilation[i];

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

            auto spatialIdx = i + 2;
            const int64_t xDim = xDims[spatialIdx];
            const int64_t dyDim = dyDims[spatialIdx];
            const int64_t kernelDim = dwDims[spatialIdx];

            const int64_t kernelSize = (dilationVal * (kernelDim - 1)) + 1;
            auto numerator = xDim + prePad + postPad - kernelSize;

            HIPDNN_RETURN_IF_LT(numerator,
                                0,
                                ErrorCode::INVALID_VALUE,
                                "ConvolutionWgradNode: Input spatial dimension at index "
                                    + std::to_string(i) + " (" + std::to_string(xDim)
                                    + ") is too small for the kernel size ("
                                    + std::to_string(kernelDim) + ") and dilation ("
                                    + std::to_string(dilationVal) + ")");

            const int64_t expectedDyDim = (numerator / strideVal) + 1;

            // Verifying dy implicitly verifies dw and x
            HIPDNN_RETURN_IF_NE(
                dyDim,
                expectedDyDim,
                ErrorCode::INVALID_VALUE,
                "ConvolutionWgradNode: dy tensor spatial dimension at index " + std::to_string(i)
                    + " (" + std::to_string(dyDim) + ") does not match expected dimension ("
                    + std::to_string(expectedDyDim)
                    + ") given x dimensions, kernel size, padding, stride, and dilation");
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

        auto& dwDims = dw->get_dim();
        HIPDNN_RETURN_IF_TRUE(dwDims.empty(),
                              ErrorCode::ATTRIBUTE_NOT_SET,
                              "ConvolutionWgradNode: output dimension inference is not possible; "
                              "set dw dimensions explicitly.");

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
            auto strideOrder = hipdnn_data_sdk::utilities::extractStrideOrder(xStrides);

            // Generate dw strides using the extracted stride order and dw dimensions
            auto dwStrides = hipdnn_data_sdk::utilities::generateStrides(dwDimsFinal, strideOrder);

            dw->set_stride(dwStrides);
        }

        return {};
    }

    Error create_operation(
        std::unordered_map<int64_t, detail::ScopedHipdnnBackendDescriptor>& tensorDescs,
        std::vector<detail::ScopedHipdnnBackendDescriptor>& operations) const override
    {
        return detail::createConvWgradOperation(attributes, tensorDescs, operations);
    }
};

typedef ConvolutionWgradNode WgradNode;
}
