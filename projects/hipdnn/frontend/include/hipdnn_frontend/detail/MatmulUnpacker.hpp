// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/attributes/MatmulAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorUnpackHelpers.hpp>
#include <memory>
#include <unordered_map>

namespace hipdnn_frontend::detail
{

/// Unpacks a matmul operation from a backend operation descriptor.
/// Populates the MatmulAttributes with tensors (using tensorMap for sharing)
/// and the compute data type.
[[nodiscard]] inline Error unpackMatmulOperation(
    hipdnnBackendDescriptor_t opDesc,
    std::unordered_map<int64_t, std::shared_ptr<graph::TensorAttributes>>& tensorMap,
    graph::MatmulAttributes& attributes)
{
    // Unpack A tensor
    std::shared_ptr<graph::TensorAttributes> aTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(
        opDesc, HIPDNN_ATTR_OPERATION_MATMUL_ADESC, tensorMap, aTensor, "matmul A tensor"));
    attributes.set_a(aTensor);

    // Unpack B tensor
    std::shared_ptr<graph::TensorAttributes> bTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(
        opDesc, HIPDNN_ATTR_OPERATION_MATMUL_BDESC, tensorMap, bTensor, "matmul B tensor"));
    attributes.set_b(bTensor);

    // Unpack C tensor
    std::shared_ptr<graph::TensorAttributes> cTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(
        opDesc, HIPDNN_ATTR_OPERATION_MATMUL_CDESC, tensorMap, cTensor, "matmul C tensor"));
    attributes.set_c(cTensor);

    // Unpack compute data type
    auto [dt, dtErr]
        = unpackGraphDataType(opDesc, HIPDNN_ATTR_MATMUL_COMP_TYPE, "matmul compute data type");
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
