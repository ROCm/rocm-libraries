// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/attributes/SoftmaxAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorUnpackHelpers.hpp>
#include <memory>
#include <optional>
#include <unordered_map>

namespace hipdnn_frontend::detail
{

[[nodiscard]] inline Error unpackSoftmaxOperation(
    hipdnnBackendDescriptor_t opDesc,
    std::unordered_map<int64_t, std::shared_ptr<graph::TensorAttributes>>& tensorMap,
    graph::SoftmaxAttributes& attributes)
{
    // Unpack p tensor
    std::shared_ptr<graph::TensorAttributes> pTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(
        opDesc, HIPDNN_ATTR_OPERATION_SOFTMAX_XDESC, tensorMap, pTensor, "softmax XDESC tensor"));
    attributes.set_p(pTensor);

    // Unpack s tensor
    std::shared_ptr<graph::TensorAttributes> sTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(
        opDesc, HIPDNN_ATTR_OPERATION_SOFTMAX_YDESC, tensorMap, sTensor, "softmax YDESC tensor"));
    attributes.set_s(sTensor);

    // Unpack stats tensor
    std::shared_ptr<graph::TensorAttributes> statsTensor;
    HIPDNN_CHECK_ERROR(unpackOptionalTensor(opDesc,
                                            HIPDNN_ATTR_OPERATION_SOFTMAX_STATS_DESC,
                                            tensorMap,
                                            statsTensor,
                                            "softmax STATS_DESC tensor"));
    if(statsTensor)
    {
        attributes.set_stats(statsTensor);
        attributes.set_generate_stats(true);
    }

    // Unpack stats_use_log2
    bool statsUseLog2 = false;
    HIPDNN_CHECK_ERROR(getDescriptorAttrScalar(opDesc,
                                               HIPDNN_ATTR_OPERATION_SOFTMAX_STATS_LOG2,
                                               HIPDNN_TYPE_BOOLEAN,
                                               statsUseLog2,
                                               "softmax stats_use_log2"));
    attributes.set_stats_use_log2(statsUseLog2);

    // Unpack compute data type
    auto [dt, dtErr] = unpackGraphDataType(
        opDesc, HIPDNN_ATTR_SOFTMAX_COMP_TYPE_EXT, "softmax compute data type");
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
