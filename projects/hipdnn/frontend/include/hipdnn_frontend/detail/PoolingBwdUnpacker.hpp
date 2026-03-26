// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/attributes/PoolingBwdAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorUnpackHelpers.hpp>
#include <memory>
#include <optional>
#include <unordered_map>

namespace hipdnn_frontend::detail
{

[[nodiscard]] inline Error unpackPoolingBwdOperation(
    hipdnnBackendDescriptor_t opDesc,
    std::unordered_map<int64_t, std::shared_ptr<graph::TensorAttributes>>& tensorMap,
    graph::PoolingBwdAttributes& attributes)
{
    // Unpack dy tensor
    std::shared_ptr<graph::TensorAttributes> dyTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(
        opDesc, HIPDNN_ATTR_OPERATION_POOLING_BACKWARD_DY, tensorMap, dyTensor, "pooling DY tensor"));
    attributes.set_dy(dyTensor);

    // Unpack dx tensor
    std::shared_ptr<graph::TensorAttributes> dxTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(
        opDesc, HIPDNN_ATTR_OPERATION_POOLING_BACKWARD_DX, tensorMap, dxTensor, "pooling DX tensor"));
    attributes.set_dx(dxTensor);

    // Unpack pre_padding
    std::vector<int64_t> prePadding;
    HIPDNN_CHECK_ERROR(getDescriptorAttrVec(
        opDesc, HIPDNN_ATTR_POOLING_PRE_PADDINGS, prePadding, "pooling pre_padding"));
    attributes.set_pre_padding(std::move(prePadding));

    // Unpack post_padding
    std::vector<int64_t> postPadding;
    HIPDNN_CHECK_ERROR(getDescriptorAttrVec(
        opDesc, HIPDNN_ATTR_POOLING_POST_PADDINGS, postPadding, "pooling post_padding"));
    attributes.set_post_padding(std::move(postPadding));

    // Unpack stride
    std::vector<int64_t> stride;
    HIPDNN_CHECK_ERROR(getDescriptorAttrVec(
        opDesc, HIPDNN_ATTR_POOLING_STRIDES, stride, "pooling stride"));
    attributes.set_stride(std::move(stride));

    // Unpack window_size
    std::vector<int64_t> windowSize;
    HIPDNN_CHECK_ERROR(getDescriptorAttrVec(
        opDesc, HIPDNN_ATTR_POOLING_WINDOW_SIZE, windowSize, "pooling window_size"));
    attributes.set_window_size(std::move(windowSize));

    // Unpack pooling_mode
    hipdnnPoolingMode_t poolingMode{};
    HIPDNN_CHECK_ERROR(getDescriptorAttrScalar(opDesc,
                                               HIPDNN_ATTR_POOLING_MODE,
                                               HIPDNN_TYPE_POOLING_MODE,
                                               poolingMode,
                                               "pooling pooling_mode"));
    auto [poolingModeResult, poolingModeErr] = fromHipdnnPoolingMode(poolingMode);
    if(poolingModeErr.is_bad())
    {
        return poolingModeErr;
    }
    attributes.set_pooling_mode(poolingModeResult);

    // Unpack compute data type
    auto [dt, dtErr]
        = unpackGraphDataType(opDesc, HIPDNN_ATTR_POOLING_COMP_TYPE, "pooling compute data type");
    if(dtErr.is_bad())
    {
        return dtErr;
    }
    attributes.set_compute_data_type(dt);

    // Unpack operation name
    std::string opName;
    HIPDNN_CHECK_ERROR(
        getDescriptorAttrString(opDesc, HIPDNN_ATTR_OPERATION_NAME_EXT, opName, "operation name"));
    attributes.set_name(opName);

    return {};
}

} // namespace hipdnn_frontend::detail
