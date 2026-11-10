// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/attributes/RopeAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorUnpackHelpers.hpp>
#include <memory>
#include <optional>
#include <unordered_map>

namespace hipdnn_frontend::detail
{

[[nodiscard]] inline Error unpackRopeOperation(
    hipdnnBackendDescriptor_t opDesc,
    std::unordered_map<int64_t, std::shared_ptr<graph::TensorAttributes>>& tensorMap,
    graph::RopeAttributes& attributes)
{
    // Unpack x tensor
    std::shared_ptr<graph::TensorAttributes> xTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(
        opDesc, HIPDNN_ATTR_OPERATION_ROPE_FWD_XDESC, tensorMap, xTensor, "RoPE XDESC tensor"));
    attributes.set_x(xTensor);

    // Unpack freqs tensor
    std::shared_ptr<graph::TensorAttributes> freqsTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_ROPE_FWD_FREQSDESC,
                                               tensorMap,
                                               freqsTensor,
                                               "RoPE FREQSDESC tensor"));
    attributes.set_freqs(freqsTensor);

    // Unpack y tensor
    std::shared_ptr<graph::TensorAttributes> yTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(
        opDesc, HIPDNN_ATTR_OPERATION_ROPE_FWD_YDESC, tensorMap, yTensor, "RoPE YDESC tensor"));
    attributes.set_y(yTensor);

    // Unpack output_scale
    float outputScale = 0.0f;
    HIPDNN_CHECK_ERROR(getDescriptorAttrScalar(opDesc,
                                               HIPDNN_ATTR_OPERATION_ROPE_FWD_OUTPUT_SCALE,
                                               HIPDNN_TYPE_FLOAT,
                                               outputScale,
                                               "RoPE output_scale"));
    attributes.set_output_scale(outputScale);

    // Unpack rope_dim
    int64_t ropeDim = 0;
    HIPDNN_CHECK_ERROR(getDescriptorAttrScalar(opDesc,
                                               HIPDNN_ATTR_OPERATION_ROPE_FWD_ROPE_DIM,
                                               HIPDNN_TYPE_INT64,
                                               ropeDim,
                                               "RoPE rope_dim"));
    attributes.set_rope_dim(ropeDim);

    // Unpack rope_offset (optional)
    {
        std::optional<int64_t> ropeOffset;
        HIPDNN_CHECK_ERROR(
            getDescriptorAttrOptionalScalar(opDesc,
                                            HIPDNN_ATTR_OPERATION_ROPE_FWD_ROPE_OFFSET_EXT,
                                            HIPDNN_TYPE_INT64,
                                            ropeOffset,
                                            "RoPE rope_offset"));
        if(ropeOffset.has_value())
        {
            attributes.set_rope_offset(*ropeOffset);
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
