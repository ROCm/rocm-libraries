// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <hip/hip_runtime_api.h>
#include <hipdnn-gpu-ref/detail/GpuRefHipError.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>

namespace hipdnn_integration_tests::gpu_graph_executor::detail
{

// A scalar operand is stored one of three ways:
//  - baked: the value lives in the graph, and the variant pack may omit it.
//  - runtime pass-by-value: the variant-pack slot holds a host pointer.
//  - device-resident: an ordinary tensor whose variant-pack slot holds a device pointer.
// The first two are host scalars and must never reach a kernel as a device pointer.
inline bool
    isHostScalarOperand(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT& operand)
{
    return operand.value.value != nullptr || operand.is_runtime_pass_by_value;
}

inline int64_t elementCount(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT& operand)
{
    return std::accumulate(
        operand.dims.begin(), operand.dims.end(), int64_t{1}, std::multiplies<>());
}

// Resolves a scalar operand to a host float. A device-resident operand must be a single FLOAT
// element and is read back with one device-to-host copy.
inline float
    resolveScalarOperand(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT& operand,
                         const std::unordered_map<int64_t, void*>& variantPack,
                         const char* name)
{
    if(isHostScalarOperand(operand))
    {
        return hipdnn_flatbuffers_sdk::utilities::resolveScalarFromVariantPack<float>(
            operand, variantPack, name);
    }
    if(operand.data_type != hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT
       || elementCount(operand) != 1)
    {
        throw std::invalid_argument(std::string(name)
                                    + " device tensor must be a single FLOAT element");
    }
    float value = 0.0f;
    hipdnn_gpu_ref::detail::throwOnHipError(
        hipMemcpy(&value, variantPack.at(operand.uid), sizeof(float), hipMemcpyDeviceToHost),
        "failed to read a device scalar operand");
    return value;
}

} // namespace hipdnn_integration_tests::gpu_graph_executor::detail
