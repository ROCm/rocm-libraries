// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <hip/hip_runtime_api.h>
#include <hipdnn-gpu-ref/detail/GpuRefHipError.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>

namespace hipdnn_integration_tests::gpu_graph_executor::detail
{

// Resolves a scalar operand to a host float. A pass-by-value scalar (baked or runtime) is a host
// value, converted from its data type, and must never reach a kernel as a device pointer. A
// device-resident operand must be a single FLOAT element and is read back with one device-to-host
// copy.
inline float
    resolveScalarOperand(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT& operand,
                         const std::unordered_map<int64_t, void*>& variantPack,
                         const char* name)
{
    if(hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(operand))
    {
        return hipdnn_flatbuffers_sdk::utilities::resolveScalarFromVariantPack<float>(
            operand, variantPack, name);
    }
    if(operand.data_type != hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT
       || hipdnn_test_sdk::detail::elementCount(operand) != 1)
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
