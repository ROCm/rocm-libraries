// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "Node.hpp"
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/ResampleBwdAttributes.hpp>
#include <hipdnn_frontend/detail/ResampleBwdPacker.hpp>
#include <hipdnn_frontend/detail/ResampleBwdUnpacker.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>
#include <hipdnn_frontend/node/detail/CheckedShapeMath.hpp>
#include <hipdnn_frontend/node/detail/ResampleParamsValidation.hpp>

#include <string>

namespace hipdnn_frontend::graph
{
class ResampleBwdNode : public BaseNode<ResampleBwdNode, NodeType::RESAMPLE_BWD>
{
public:
    ResampleBwdAttributes attributes;

    ResampleBwdNode(ResampleBwdAttributes&& attrs, const GraphAttributes& graphAttrs)
        : BaseNode(graphAttrs)
        , attributes(std::move(attrs))
    {
    }

    Error unpack_from_descriptor(
        hipdnnBackendDescriptor_t opDesc,
        std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>>& tensorMap) override
    {
        ResampleBwdAttributes attrs;
        HIPDNN_CHECK_ERROR(detail::unpackResampleBwdOperation(opDesc, tensorMap, attrs));
        attributes = std::move(attrs);
        return {};
    }

    Error pre_validate_node() const override
    {
        // Validate required input tensors
        HIPDNN_RETURN_IF_FALSE(attributes.get_dy(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "ResampleBwdNode missing dy (input) for pre-validation");
        if(attributes.get_resample_mode() == ResampleMode::MAXPOOL && !attributes.get_index())
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET,
                    "ResampleBwdNode missing Index input for MAXPOOL pre-validation"};
        }

        // Validate required output tensors
        HIPDNN_RETURN_IF_FALSE(attributes.get_dx(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "ResampleBwdNode missing dx (output) for pre-validation");

        // The modes must be ones the lowering can express; padding_mode may stay unset
        HIPDNN_RETURN_IF_FALSE(
            toBackendResampleMode(attributes.get_resample_mode()).has_value(),
            ErrorCode::INVALID_VALUE,
            "ResampleBwdNode resample mode "
                + std::to_string(static_cast<int>(attributes.get_resample_mode()))
                + " is not supported by hipDNN lowering");
        HIPDNN_RETURN_IF_FALSE(toBackendPaddingMode(attributes.get_padding_mode()).has_value(),
                               ErrorCode::INVALID_VALUE,
                               "ResampleBwdNode padding mode "
                                   + std::to_string(static_cast<int>(attributes.get_padding_mode()))
                                   + " is not supported by hipDNN lowering");

        // Validate required tensor dimensions
        HIPDNN_RETURN_IF_TRUE(attributes.get_dy()->get_dim().empty(),
                              ErrorCode::ATTRIBUTE_NOT_SET,
                              "ResampleBwdNode missing dy dimensions for pre-validation");

        // Validate required resample parameters
        HIPDNN_RETURN_IF_TRUE(attributes.get_pre_padding().empty(),
                              ErrorCode::ATTRIBUTE_NOT_SET,
                              "ResampleBwdNode missing pre_padding for pre-validation");
        HIPDNN_RETURN_IF_TRUE(attributes.get_post_padding().empty(),
                              ErrorCode::ATTRIBUTE_NOT_SET,
                              "ResampleBwdNode missing post_padding for pre-validation");
        HIPDNN_RETURN_IF_TRUE(attributes.get_stride().empty(),
                              ErrorCode::ATTRIBUTE_NOT_SET,
                              "ResampleBwdNode missing stride for pre-validation");
        HIPDNN_RETURN_IF_TRUE(attributes.get_window().empty(),
                              ErrorCode::ATTRIBUTE_NOT_SET,
                              "ResampleBwdNode missing window for pre-validation");

        const auto dy = attributes.get_dy();
        if(!dy->get_dim().empty() && dy->get_dim().size() < 3)
        {
            return {ErrorCode::INVALID_VALUE,
                    "ResampleBwdNode input DY must have at least rank 3 (N, C, spatial...)"};
        }

        HIPDNN_CHECK_ERROR(detail::validateResampleParams("ResampleBwdNode",
                                                          dy->get_dim().size() - 2,
                                                          attributes.get_window(),
                                                          attributes.get_stride(),
                                                          attributes.get_pre_padding(),
                                                          attributes.get_post_padding()));

        const auto dx = attributes.get_dx();
        if(!dx->get_dim().empty() && dx->get_dim().size() < 3)
        {
            return {ErrorCode::INVALID_VALUE,
                    "ResampleBwdNode output DX must have at least rank 3 (N, C, spatial...)"};
        }

        if(const auto index = attributes.get_index())
        {
            if(!index->get_dim().empty() && index->get_dim().size() < 3)
            {
                return {ErrorCode::INVALID_VALUE,
                        "ResampleBwdNode input Index must have at least rank 3 (N, C, spatial...)"};
            }

            switch(index->get_data_type())
            {
            case DataType::INT8:
            case DataType::INT32:
                break;
            default:
                return {ErrorCode::INVALID_VALUE,
                        "ResampleBwdNode input Index type must be Int8 or Int32"};
            }
        }

        return {};
    }

    Error infer_properties_node() override
    {
        // Validate required tensor pointers
        HIPDNN_RETURN_IF_FALSE(attributes.get_dy(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "ResampleBwdNode missing dy for setting properties");

        HIPDNN_RETURN_IF_FALSE(attributes.get_dx(),
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               "ResampleBwdNode missing dx for setting properties");

        HIPDNN_CHECK_ERROR(attributes.fill_from_context(graph_attributes));

        const auto dxTensor = attributes.get_dx();
        const auto dyTensor = attributes.get_dy();

        // Infer output dimensions if not set
        auto dxDims = dxTensor->get_dim();
        if(dxDims.empty())
        {
            auto& dyDims = dyTensor->get_dim();
            HIPDNN_RETURN_IF_TRUE(dyDims.empty(),
                                  ErrorCode::ATTRIBUTE_NOT_SET,
                                  "ResampleBwdNode missing dy dimensions for inferring dx "
                                  "dimensions");
            HIPDNN_RETURN_IF_LT(
                dyDims.size(),
                3,
                ErrorCode::INVALID_VALUE,
                "ResampleBwdNode input DY must have at least rank 3 (N, C, spatial...)");

            auto& prePadding = attributes.get_pre_padding();
            auto& postPadding = attributes.get_post_padding();
            auto& strideVec = attributes.get_stride();
            auto& windowVec = attributes.get_window();

            HIPDNN_CHECK_ERROR(detail::requireResampleParams("ResampleBwdNode",
                                                             dyDims.size() - 2,
                                                             windowVec,
                                                             strideVec,
                                                             prePadding,
                                                             postPadding));

            dxDims.resize(dyDims.size());
            dxDims[0] = dyDims[0]; // N (batch)
            dxDims[1] = dyDims[1]; // C (channels preserved)

            // Calculate spatial dimensions (inverse of forward)
            // forward: y = (x + pre + post - window) / stride + 1
            // inverse: x = (y - 1) * stride - pre - post + window
            for(size_t i = 2; i < dyDims.size(); ++i)
            {
                const size_t spatialIdx = i - 2;
                HIPDNN_CHECK_ERROR(detail::checkedTransposedExtent("ResampleBwdNode",
                                                                   spatialIdx,
                                                                   dyDims[i],
                                                                   strideVec[spatialIdx],
                                                                   prePadding[spatialIdx],
                                                                   postPadding[spatialIdx],
                                                                   windowVec[spatialIdx],
                                                                   dxDims[i]));

                HIPDNN_RETURN_IF_LT(dxDims[i],
                                    1,
                                    ErrorCode::INVALID_VALUE,
                                    "ResampleBwdNode: inferred dx extent ("
                                        + std::to_string(dxDims[i]) + ") at spatial index "
                                        + std::to_string(spatialIdx)
                                        + " is not positive for the given dy, window, stride "
                                          "and padding");
            }

            dxTensor->set_dim(dxDims);
        }

        // Infer output strides if not set
        if(dxTensor->get_stride().empty())
        {
            auto& dyStrides = dyTensor->get_stride();
            auto& currentDxDims = dxTensor->get_dim();

            HIPDNN_RETURN_IF_TRUE(
                dyStrides.empty(),
                ErrorCode::ATTRIBUTE_NOT_SET,
                "ResampleBwdNode: Cannot infer output strides - missing input strides");

            HIPDNN_RETURN_IF_TRUE(
                currentDxDims.empty(),
                ErrorCode::ATTRIBUTE_NOT_SET,
                "ResampleBwdNode: Cannot infer output strides - missing output dimensions");

            HIPDNN_RETURN_IF_NE(
                dyStrides.size(),
                currentDxDims.size(),
                ErrorCode::ATTRIBUTE_NOT_SET,
                "ResampleBwdNode: Stride dimension mismatch between input and output tensors");

            const auto strideOrder = hipdnn_data_sdk::utilities::extractStrideOrder(dyStrides);
            const auto dxStrides
                = hipdnn_data_sdk::utilities::generateStrides(currentDxDims, strideOrder);
            dxTensor->set_stride(dxStrides);
        }

        const auto index = attributes.get_index();
        if(index)
        {
            if(index->get_dim().empty())
            {
                index->set_dim(dyTensor->get_dim());
            }
            if(index->get_stride().empty())
            {
                index->set_stride(dyTensor->get_stride());
            }
        }

        return {};
    }

    Error create_operation(
        std::unordered_map<int64_t, detail::ScopedHipdnnBackendDescriptor>& tensorDescs,
        std::vector<detail::ScopedHipdnnBackendDescriptor>& operations) const override
    {
        return detail::createResampleBwdOperation(attributes, tensorDescs, operations);
    }
};
}
