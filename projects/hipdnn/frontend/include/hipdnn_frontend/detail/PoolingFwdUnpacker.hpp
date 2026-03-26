// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/attributes/PoolingFwdAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorUnpackHelpers.hpp>
#include <memory>
#include <optional>
#include <unordered_map>

namespace hipdnn_frontend::detail
{

[[nodiscard]] inline Error unpackPoolingFwdOperation(
    hipdnnBackendDescriptor_t opDesc,
    std::unordered_map<int64_t, std::shared_ptr<graph::TensorAttributes>>& tensorMap,
    graph::PoolingFwdAttributes& attributes)
{
    // Unpack x tensor
    std::shared_ptr<graph::TensorAttributes> xTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_POOLING_FORWARD_X_EXT,
                                               tensorMap,
                                               xTensor,
                                               "pooling X tensor"));
    attributes.set_x(xTensor);

    // Unpack y tensor
    std::shared_ptr<graph::TensorAttributes> yTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_POOLING_FORWARD_Y_EXT,
                                               tensorMap,
                                               yTensor,
                                               "pooling Y tensor"));
    attributes.set_y(yTensor);

    // Unpack index tensor
    std::shared_ptr<graph::TensorAttributes> indexTensor;
    HIPDNN_CHECK_ERROR(unpackOptionalTensor(opDesc,
                                            HIPDNN_ATTR_OPERATION_POOLING_FORWARD_INDEX_EXT,
                                            tensorMap,
                                            indexTensor,
                                            "pooling INDEX tensor"));
    if(indexTensor)
    {
        attributes.set_index(indexTensor);
    }

    // Unpack pre_padding
    std::vector<int64_t> prePadding;
    HIPDNN_CHECK_ERROR(getDescriptorAttrVec(
        opDesc, HIPDNN_ATTR_POOLING_PRE_PADDINGS_EXT, prePadding, "pooling pre_padding"));
    attributes.set_pre_padding(std::move(prePadding));

    // Unpack post_padding
    std::vector<int64_t> postPadding;
    HIPDNN_CHECK_ERROR(getDescriptorAttrVec(
        opDesc, HIPDNN_ATTR_POOLING_POST_PADDINGS_EXT, postPadding, "pooling post_padding"));
    attributes.set_post_padding(std::move(postPadding));

    // Unpack stride
    std::vector<int64_t> stride;
    HIPDNN_CHECK_ERROR(
        getDescriptorAttrVec(opDesc, HIPDNN_ATTR_POOLING_STRIDES_EXT, stride, "pooling stride"));
    attributes.set_stride(std::move(stride));

    // Unpack window
    std::vector<int64_t> window;
    HIPDNN_CHECK_ERROR(
        getDescriptorAttrVec(opDesc, HIPDNN_ATTR_POOLING_WINDOW_EXT, window, "pooling window"));
    attributes.set_window(std::move(window));

    // Unpack pooling_mode
    hipdnnPoolingMode_t poolingMode{};
    HIPDNN_CHECK_ERROR(getDescriptorAttrScalar(opDesc,
                                               HIPDNN_ATTR_POOLING_MODE_EXT,
                                               HIPDNN_TYPE_POOLING_MODE,
                                               poolingMode,
                                               "pooling pooling_mode"));
    auto [poolingModeResult, poolingModeErr] = fromHipdnnPoolingMode(poolingMode);
    if(poolingModeErr.is_bad())
    {
        return poolingModeErr;
    }
    attributes.set_pooling_mode(poolingModeResult);

    // Unpack padding_mode
    hipdnnPaddingMode_t paddingMode{};
    HIPDNN_CHECK_ERROR(getDescriptorAttrScalar(opDesc,
                                               HIPDNN_ATTR_POOLING_PADDING_MODE_EXT,
                                               HIPDNN_TYPE_PADDING_MODE,
                                               paddingMode,
                                               "pooling padding_mode"));
    auto [paddingModeResult, paddingModeErr] = fromHipdnnPaddingMode(paddingMode);
    if(paddingModeErr.is_bad())
    {
        return paddingModeErr;
    }
    attributes.set_padding_mode(paddingModeResult);

    // Unpack generate_index (optional)
    {
        std::optional<bool> generateIndex;
        HIPDNN_CHECK_ERROR(getDescriptorAttrOptionalScalar(opDesc,
                                                           HIPDNN_ATTR_POOLING_GENERATE_INDEX_EXT,
                                                           HIPDNN_TYPE_BOOLEAN,
                                                           generateIndex,
                                                           "pooling generate_index"));
        if(generateIndex.has_value())
        {
            attributes.set_generate_index(*generateIndex);
        }
    }

    // Unpack operation name
    std::string opName;
    HIPDNN_CHECK_ERROR(
        getDescriptorAttrString(opDesc, HIPDNN_ATTR_OPERATION_NAME_EXT, opName, "operation name"));
    attributes.set_name(opName);

    return {};
}

} // namespace hipdnn_frontend::detail
