// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>

#include <hipdnn-gpu-ref/ShallowGpuTensor.hpp>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>

namespace hipdnn_integration_tests::gpu_graph_executor::detail
{

template <typename T>
struct SdpaSinkTypeTag
{
    using type = T;
};

// Binds the optional SDPA sink_token operand (device-resident [1, H, 1, 1]) and calls
// fn(sinkTensorOrNull, SdpaSinkTypeTag<SinkDataType>{}). A FLOAT sink binds as float; a sink in
// the Q data type binds as QDataType when that is half or bfloat16. Without a sink, fn gets a
// null float tensor. isApplicable (isSdpaSinkSupported) has already rejected any other sink.
template <typename QDataType, typename Fn>
inline void withSdpaSinkOperand(
    const std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT>& sink,
    const std::unordered_map<int64_t, void*>& variantPack,
    const Fn& fn)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    if(!sink.has_value())
    {
        fn(static_cast<hipdnn_gpu_ref::ShallowGpuTensor<float>*>(nullptr),
           SdpaSinkTypeTag<float>{});
        return;
    }
    if(sink->data_type == DataType::FLOAT)
    {
        hipdnn_gpu_ref::ShallowGpuTensor<float> tensor(
            variantPack.at(sink->uid), sink->dims, sink->strides);
        fn(&tensor, SdpaSinkTypeTag<float>{});
        return;
    }
    if constexpr(std::is_same_v<QDataType, hipdnn_data_sdk::types::half>
                 || std::is_same_v<QDataType, hipdnn_data_sdk::types::bfloat16>)
    {
        hipdnn_gpu_ref::ShallowGpuTensor<QDataType> tensor(
            variantPack.at(sink->uid), sink->dims, sink->strides);
        fn(&tensor, SdpaSinkTypeTag<QDataType>{});
    }
    else
    {
        throw std::invalid_argument(
            "SDPA sink_token must be FLOAT, or the Q data type when that is HALF or BFLOAT16");
    }
}

} // namespace hipdnn_integration_tests::gpu_graph_executor::detail
