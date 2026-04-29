// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/attributes/ResampleFwdAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorUnpackHelpers.hpp>
#include <memory>
#include <optional>
#include <unordered_map>

namespace hipdnn_frontend::detail
{

[[nodiscard]] inline Error unpackResampleFwdOperation(
    hipdnnBackendDescriptor_t opDesc,
    std::unordered_map<int64_t, std::shared_ptr<graph::TensorAttributes>>& tensorMap,
    graph::ResampleFwdAttributes& attributes)
{
    // Unpack x tensor
    std::shared_ptr<graph::TensorAttributes> xTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(
        opDesc, HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC, tensorMap, xTensor, "resample X tensor"));
    attributes.set_x(xTensor);

    // Unpack y tensor
    std::shared_ptr<graph::TensorAttributes> yTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(
        opDesc, HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC, tensorMap, yTensor, "resample Y tensor"));
    attributes.set_y(yTensor);

    // Unpack index tensor
    std::shared_ptr<graph::TensorAttributes> indexTensor;
    HIPDNN_CHECK_ERROR(unpackOptionalTensor(opDesc,
                                            HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC,
                                            tensorMap,
                                            indexTensor,
                                            "resample INDEX tensor"));
    if(indexTensor)
    {
        attributes.set_index(indexTensor);
    }

    // Unpack pre_padding
    std::vector<int64_t> prePadding;
    HIPDNN_CHECK_ERROR(getDescriptorAttrVec(
        opDesc, HIPDNN_ATTR_RESAMPLE_PRE_PADDINGS, prePadding, "resample pre_padding"));
    attributes.set_pre_padding(std::move(prePadding));

    // Unpack post_padding
    std::vector<int64_t> postPadding;
    HIPDNN_CHECK_ERROR(getDescriptorAttrVec(
        opDesc, HIPDNN_ATTR_RESAMPLE_POST_PADDINGS, postPadding, "resample post_padding"));
    attributes.set_post_padding(std::move(postPadding));

    // Unpack stride
    std::vector<int64_t> stride;
    HIPDNN_CHECK_ERROR(
        getDescriptorAttrVec(opDesc, HIPDNN_ATTR_RESAMPLE_STRIDES, stride, "resample stride"));
    attributes.set_stride(std::move(stride));

    // Unpack window
    std::vector<int64_t> window;
    HIPDNN_CHECK_ERROR(
        getDescriptorAttrVec(opDesc, HIPDNN_ATTR_RESAMPLE_WINDOW_DIMS, window, "resample window"));
    attributes.set_window(std::move(window));

    // Unpack resample_mode
    hipdnnResampleMode_t resampleMode{};
    HIPDNN_CHECK_ERROR(getDescriptorAttrScalar(opDesc,
                                               HIPDNN_ATTR_RESAMPLE_MODE,
                                               HIPDNN_TYPE_RESAMPLE_MODE,
                                               resampleMode,
                                               "resample resample_mode"));
    auto [resampleModeResult, resampleModeErr] = fromHipdnnResampleMode(resampleMode);
    if(resampleModeErr.is_bad())
    {
        return resampleModeErr;
    }
    attributes.set_resample_mode(resampleModeResult);

    // Unpack padding_mode
    hipdnnPaddingMode_t paddingMode{};
    HIPDNN_CHECK_ERROR(getDescriptorAttrScalar(opDesc,
                                               HIPDNN_ATTR_RESAMPLE_PADDING_MODE,
                                               HIPDNN_TYPE_PADDING_MODE,
                                               paddingMode,
                                               "resample padding_mode"));
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
                                                           HIPDNN_ATTR_RESAMPLE_GENERATE_INDEX_EXT,
                                                           HIPDNN_TYPE_BOOLEAN,
                                                           generateIndex,
                                                           "resample generate_index"));
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

    // Unpack compute data type
    auto [dt, dtErr]
        = unpackGraphDataType(opDesc, HIPDNN_ATTR_RESAMPLE_COMP_TYPE, "resample compute data type");
    if(dtErr.is_bad())
    {
        return dtErr;
    }
    attributes.set_compute_data_type(dt);

    return {};
}

} // namespace hipdnn_frontend::detail
