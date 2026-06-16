// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <optional>
#include <unordered_map>

#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>

namespace hip_kernel_provider::layernorm
{

size_t getMinNormalizedDimFromAffine(
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* ioAttr,
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* affineAttr);

size_t getMinNormalizedDimFromAffine(
    int64_t ioTensorId,
    int64_t affineTensorId,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap);

size_t getMaxNormalizedDimFromAffine(
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* ioAttr,
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* affineAttr);

size_t getMaxNormalizedDimFromAffine(
    int64_t ioTensorId,
    int64_t affineTensorId,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap);

size_t getMinNormalizedDimFromStat(
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* ioAttr,
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* statAttr);

size_t getMinNormalizedDimFromStat(
    int64_t ioTensorId,
    int64_t statTensorId,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap);

size_t getMaxNormalizedDimFromStat(
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* ioAttr,
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* statAttr);

size_t getMaxNormalizedDimFromStat(
    int64_t ioTensorId,
    int64_t statTensorId,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap);

size_t guessNormalizedDim(
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* ioAttr,
    std::optional<const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*> affineAttr,
    std::optional<const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*> statAttr);

size_t guessNormalizedDim(
    int64_t ioTensorId,
    std::optional<int64_t> affineTensorId,
    std::optional<int64_t> statTensorId,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap);

} // namespace hip_kernel_provider::layernorm
