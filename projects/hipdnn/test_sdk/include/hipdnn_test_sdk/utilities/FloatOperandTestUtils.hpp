// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <vector>

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

namespace hipdnn_test_sdk::utilities
{

// How a FLOAT operand (attention scale, fp8 descale) is stored.
enum class OperandStorage
{
    // An ordinary tensor. Its variant-pack slot holds the executor's buffer (device memory on
    // the GPU reference, host memory on the CPU reference).
    TENSOR,
    BAKED, // value lives in the graph; the variant pack may omit it
    RUNTIME_PASS_BY_VALUE, // host pointer in the variant pack
};

// BAKED and RUNTIME_PASS_BY_VALUE operands must be scalars ([1]). TENSOR operands may be larger,
// e.g. per-KV-head descales [B, H_kv, 1, 1].
struct FloatOperandSpec
{
    int64_t uid = 0;
    OperandStorage storage = OperandStorage::TENSOR;
    float bakedValue = 0.0f; // BAKED only
    std::vector<int64_t> dims = {1};
};

// Appends a FLOAT operand tensor in the storage mode its spec asks for.
inline void addFloatOperandTensor(
    flatbuffers::FlatBufferBuilder& builder,
    std::vector<flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>&
        tensors,
    const FloatOperandSpec& spec,
    const char* name)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    const auto strides = hipdnn_data_sdk::utilities::generateStrides(spec.dims);
    const bool baked = spec.storage == OperandStorage::BAKED;
    const Float32Value bakedValue(spec.bakedValue);
    const flatbuffers::Offset<void> value
        = baked ? builder.CreateStruct(bakedValue).Union() : flatbuffers::Offset<void>();
    tensors.push_back(CreateTensorAttributesDirect(
        builder,
        spec.uid,
        name,
        DataType::FLOAT,
        &strides,
        &spec.dims,
        /*virtual_=*/false,
        baked ? TensorValue::Float32Value : TensorValue::NONE,
        value,
        /*is_runtime_pass_by_value=*/spec.storage == OperandStorage::RUNTIME_PASS_BY_VALUE));
}

} // namespace hipdnn_test_sdk::utilities
