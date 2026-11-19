// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "Node.hpp"
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/ResampleFwdAttributes.hpp>
#include <hipdnn_frontend/detail/ResampleFwdPacker.hpp>
#include <hipdnn_frontend/detail/ResampleFwdUnpacker.hpp>
#include <hipdnn_frontend/node/detail/CheckedShapeMath.hpp>
#include <hipdnn_frontend/node/detail/ResampleParamsValidation.hpp>

#include <string>

namespace hipdnn_frontend::graph
{
class ResampleFwdNode : public BaseNode<ResampleFwdNode, NodeType::RESAMPLE_FWD>
{
public:
    ResampleFwdAttributes attributes;

    ResampleFwdNode(ResampleFwdAttributes&& resampleFwdAttributes,
                    const GraphAttributes& graphAttrs)
        : BaseNode(graphAttrs)
        , attributes(std::move(resampleFwdAttributes))
    {
    }

    Error pre_validate_node() const override
    {
        if(!attributes.get_x())
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET,
                    "ResampleFwdNode missing input X for pre-validation"};
        }
        if(!attributes.get_y())
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET,
                    "ResampleFwdNode missing output Y for pre-validation"};
        }

        // The modes must be ones the lowering can express; padding_mode has no default
        HIPDNN_RETURN_IF_TRUE(attributes.get_padding_mode() == PaddingMode::NOT_SET,
                              ErrorCode::ATTRIBUTE_NOT_SET,
                              "ResampleFwdNode missing padding_mode for pre-validation");
        HIPDNN_RETURN_IF_FALSE(
            toBackendResampleMode(attributes.get_resample_mode()).has_value(),
            ErrorCode::INVALID_VALUE,
            "ResampleFwdNode resample mode "
                + std::to_string(static_cast<int>(attributes.get_resample_mode()))
                + " is not supported by hipDNN lowering");
        HIPDNN_RETURN_IF_FALSE(toBackendPaddingMode(attributes.get_padding_mode()).has_value(),
                               ErrorCode::INVALID_VALUE,
                               "ResampleFwdNode padding mode "
                                   + std::to_string(static_cast<int>(attributes.get_padding_mode()))
                                   + " is not supported by hipDNN lowering");

        // generate_index is ignored for modes other than MAXPOOL, as in cuDNN
        const bool generateIndex = attributes.get_generate_index().value_or(false)
                                   && attributes.get_resample_mode() == ResampleMode::MAXPOOL;
        if(generateIndex && !attributes.get_index())
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET,
                    "ResampleFwdNode missing output Index when index generation requested"};
        }

        // Validate tensor ranks if dimensions are already set
        auto x = attributes.get_x();
        if(!x->get_dim().empty() && x->get_dim().size() < 3)
        {
            return {ErrorCode::INVALID_VALUE,
                    "ResampleFwdNode input X must have at least rank 3 (N, C, spatial...)"};
        }
        auto y = attributes.get_y();
        if(!y->get_dim().empty() && y->get_dim().size() < 3)
        {
            return {ErrorCode::INVALID_VALUE,
                    "ResampleFwdNode output Y must have at least rank 3 (N, C, spatial...)"};
        }

        auto& xDims = x->get_dim();
        if(!xDims.empty())
        {
            HIPDNN_CHECK_ERROR(detail::validateResampleParams("ResampleFwdNode",
                                                              xDims.size() - 2,
                                                              attributes.get_window(),
                                                              attributes.get_stride(),
                                                              attributes.get_pre_padding(),
                                                              attributes.get_post_padding()));
        }

        if(auto index = attributes.get_index())
        {
            switch(index->get_data_type())
            {
            case DataType::INT8:
            case DataType::INT32:
                break;
            default:
                return {ErrorCode::INVALID_VALUE,
                        "ResampleFwdNode output Index type must be Int8 or Int32"};
            }
        }

        return {};
    }

    Error infer_properties_node() override
    {
        auto x = attributes.get_x();
        auto y = attributes.get_y();

        HIPDNN_RETURN_IF_FALSE(
            x, ErrorCode::ATTRIBUTE_NOT_SET, "ResampleFwdNode missing x for setting properties");
        HIPDNN_RETURN_IF_FALSE(
            y, ErrorCode::ATTRIBUTE_NOT_SET, "ResampleFwdNode missing y for setting properties");

        HIPDNN_CHECK_ERROR(attributes.fill_from_context(graph_attributes));

        auto yDims = y->get_dim();

        // Infer output dimensions if not set
        if(yDims.empty())
        {
            auto& xDims = x->get_dim();
            HIPDNN_RETURN_IF_TRUE(xDims.empty(),
                                  ErrorCode::ATTRIBUTE_NOT_SET,
                                  "ResampleFwdNode missing input X dimensions for inferring "
                                  "output Y dimensions");
            HIPDNN_RETURN_IF_LT(
                xDims.size(),
                3,
                ErrorCode::INVALID_VALUE,
                "ResampleFwdNode input X must have at least rank 3 (N, C, spatial...)");

            auto& prePadding = attributes.get_pre_padding();
            auto& postPadding = attributes.get_post_padding();
            auto& strideVec = attributes.get_stride();
            auto& windowVec = attributes.get_window();

            HIPDNN_CHECK_ERROR(detail::requireResampleParams("ResampleFwdNode",
                                                             xDims.size() - 2,
                                                             windowVec,
                                                             strideVec,
                                                             prePadding,
                                                             postPadding));

            yDims.resize(xDims.size());
            yDims[0] = xDims[0]; // N (batch)
            yDims[1] = xDims[1]; // C (channels preserved)

            // Calculate spatial dimensions with truncating division, as cuDNN does: a window
            // wider than the padded input by less than the stride still yields one element
            for(size_t i = 2; i < xDims.size(); ++i)
            {
                auto spatialIdx = i - 2;
                auto inputSize = xDims[i];
                auto prePad = prePadding[spatialIdx];
                auto postPad = postPadding[spatialIdx];
                auto strideVal = strideVec[spatialIdx];
                auto windowVal = windowVec[spatialIdx];
                int64_t numerator = 0;
                HIPDNN_CHECK_ERROR(detail::checkedPaddedNumerator("ResampleFwdNode",
                                                                  spatialIdx,
                                                                  inputSize,
                                                                  prePad,
                                                                  postPad,
                                                                  windowVal,
                                                                  numerator));
                HIPDNN_CHECK_ERROR(detail::checkedStridedExtent(
                    "ResampleFwdNode", spatialIdx, numerator, strideVal, yDims[i]));

                HIPDNN_RETURN_IF_LT(
                    yDims[i],
                    1,
                    ErrorCode::INVALID_VALUE,
                    "ResampleFwdNode: inferred y extent (" + std::to_string(yDims[i])
                        + ") at spatial index " + std::to_string(spatialIdx)
                        + " is below 1 for x extent " + std::to_string(inputSize) + ", pre_padding "
                        + std::to_string(prePad) + ", post_padding " + std::to_string(postPad)
                        + ", window " + std::to_string(windowVal) + " and stride "
                        + std::to_string(strideVal));
            }

            y->set_dim(yDims);
        }

        // Infer output strides if not set
        if(y->get_stride().empty())
        {
            auto& xStrides = x->get_stride();
            if(!xStrides.empty())
            {
                auto strideOrder = hipdnn_data_sdk::utilities::extractStrideOrder(xStrides);
                y->set_stride(hipdnn_data_sdk::utilities::generateStrides(yDims, strideOrder));
            }
            else
            {
                y->set_stride(hipdnn_data_sdk::utilities::generateStrides(
                    yDims, hipdnn_data_sdk::utilities::strideOrderNhwc(yDims.size())));
            }
        }

        auto index = attributes.get_index();
        if(index)
        {
            if(index->get_dim().empty())
            {
                index->set_dim(y->get_dim());
            }
            if(index->get_stride().empty())
            {
                index->set_stride(y->get_stride());
            }
        }

        return {};
    }

    Error create_operation(
        std::unordered_map<int64_t, detail::ScopedHipdnnBackendDescriptor>& tensorDescs,
        std::vector<detail::ScopedHipdnnBackendDescriptor>& operations) const override
    {
        if(attributes.get_resample_mode() == ResampleMode::MAXPOOL
           || !attributes.get_generate_index().value_or(false))
        {
            return detail::createResampleFwdOperation(attributes, tensorDescs, operations);
        }

        // generate_index is ignored for modes other than MAXPOOL, so engines never see the
        // flag without an index tensor.
        auto loweredAttributes = attributes;
        loweredAttributes.set_generate_index(false);
        return detail::createResampleFwdOperation(loweredAttributes, tensorDescs, operations);
    }

    Error unpack_from_descriptor(
        hipdnnBackendDescriptor_t opDesc,
        std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>>& tensorMap) override
    {
        ResampleFwdAttributes attrs;
        HIPDNN_CHECK_ERROR(detail::unpackResampleFwdOperation(opDesc, tensorMap, attrs));
        attributes = std::move(attrs);
        return {};
    }
};
} // namespace hipdnn_frontend::graph
