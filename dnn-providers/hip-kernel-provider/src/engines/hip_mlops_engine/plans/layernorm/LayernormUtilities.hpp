// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <optional>
#include <unordered_map>

#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>

namespace hip_kernel_provider::layernorm
{

enum class Direction
{
    FORWARD,
    BACKWARD
};

class ProblemDescription
{
public:
    ProblemDescription(
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* ioAttr,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* affineAttr,
        std::optional<const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*> statAttr,
        Direction direction);

    Direction direction() const
    {
        return _direction;
    }
    size_t normalizedDim() const
    {
        return _normalizedDim;
    }
    int64_t outerSize() const
    {
        return _outerSize;
    }
    int64_t innerSize() const
    {
        return _innerSize;
    }
    int64_t stride() const
    {
        return _stride;
    }

private:
    Direction _direction;
    size_t _normalizedDim;
    int64_t _outerSize;
    int64_t _innerSize;
    int64_t _stride;
};

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

} // namespace hip_kernel_provider::layernorm
