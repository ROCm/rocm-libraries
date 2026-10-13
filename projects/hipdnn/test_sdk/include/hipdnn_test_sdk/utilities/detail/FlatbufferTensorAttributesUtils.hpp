// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "hipdnn_data_sdk/utilities/RaggedTensor.hpp"
#include "hipdnn_data_sdk/utilities/Tensor.hpp"
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <hipdnn_data_sdk/utilities/PackedFp4Tensor.hpp>
#include <hipdnn_data_sdk/utilities/PackedFp6Tensor.hpp>
#include <hipdnn_data_sdk/utilities/ShallowTensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>

namespace hipdnn_test_sdk::detail
{

inline hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT unpackTensorAttributes(
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& tensorAttributes)
{
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT tensorAttributesT;
    tensorAttributes.UnPackTo(&tensorAttributesT);
    return tensorAttributesT;
}

/// Folds the two mutually-exclusive SDPA scale sources into a single optional
/// scalar operand: a real scale tensor if present, else a synthesized baked
/// FLOAT scalar carrying attn_scale_value, else nullopt (1.0, no scaling).
/// The frontend (SdpaFwdNode/SdpaBwdNode) enforces that at most one source is
/// set, so this never has to reconcile a conflict.
inline std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT>
    foldSdpaScale(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* scaleTensor,
                  std::optional<float> attnScaleValue)
{
    if(scaleTensor != nullptr)
    {
        return unpackTensorAttributes(*scaleTensor);
    }
    if(attnScaleValue.has_value())
    {
        hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT baked;
        baked.data_type = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT;
        baked.dims = {1};
        baked.strides = {1};
        baked.is_runtime_pass_by_value = false;
        baked.value.Set(hipdnn_flatbuffers_sdk::data_objects::Float32Value(attnScaleValue.value()));
        return baked;
    }
    return std::nullopt;
}

template <typename T>
inline std::unique_ptr<hipdnn_data_sdk::utilities::ShallowTensor<T>> createShallowTensor(
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT& tensorDetails, void* ptr)
{
    return std::make_unique<hipdnn_data_sdk::utilities::ShallowTensor<T>>(
        ptr, tensorDetails.dims, tensorDetails.strides);
}

/// Binds a required tensor from the variant pack. Throws std::out_of_range
/// (via unordered_map::at) when the UID is absent, matching existing plans.
template <typename T>
inline std::unique_ptr<hipdnn_data_sdk::utilities::ShallowTensor<T>>
    bindShallowTensor(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT& tensorDetails,
                      const std::unordered_map<int64_t, void*>& variantPack)
{
    return createShallowTensor<T>(tensorDetails, variantPack.at(tensorDetails.uid));
}

/// Binds an optional tensor; returns nullptr when the operand is not present.
template <typename T>
inline std::unique_ptr<hipdnn_data_sdk::utilities::ShallowTensor<T>> bindOptionalShallowTensor(
    const std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT>& tensorDetails,
    const std::unordered_map<int64_t, void*>& variantPack)
{
    return tensorDetails.has_value() ? bindShallowTensor<T>(*tensorDetails, variantPack) : nullptr;
}

template <typename T>
struct TypeTag
{
    using type = T;
};

/// Invokes `visitor` with the TypeTag of the host element type `dataType` maps to. Sub-byte
/// types map to their unpacked one-element-per-byte types, with INT4 stored as uint8_t.
template <typename Visitor>
inline std::unique_ptr<hipdnn_data_sdk::utilities::ITensor>
    visitNativeType(hipdnn_flatbuffers_sdk::data_objects::DataType dataType, Visitor&& visitor)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    using namespace hipdnn_data_sdk::types;
    switch(dataType)
    {
    case DataType::FLOAT:
        return visitor(TypeTag<float>{});
    case DataType::HALF:
        return visitor(TypeTag<half>{});
    case DataType::BFLOAT16:
        return visitor(TypeTag<bfloat16>{});
    case DataType::DOUBLE:
        return visitor(TypeTag<double>{});
    case DataType::UINT8:
        return visitor(TypeTag<uint8_t>{});
    case DataType::INT32:
        return visitor(TypeTag<int32_t>{});
    case DataType::INT8:
        return visitor(TypeTag<int8_t>{});
    case DataType::FP8_E4M3:
        return visitor(TypeTag<fp8_e4m3>{});
    case DataType::FP8_E5M2:
        return visitor(TypeTag<fp8_e5m2>{});
    case DataType::INT64:
        return visitor(TypeTag<int64_t>{});
    case DataType::FP8_E8M0:
        return visitor(TypeTag<fp8_e8m0>{});
    case DataType::FP4_E2M1:
        return visitor(TypeTag<fp4_e2m1>{});
    case DataType::INT4:
        return visitor(TypeTag<uint8_t>{});
    case DataType::FP6_E2M3:
        return visitor(TypeTag<fp6_e2m3>{});
    case DataType::FP6_E3M2:
        return visitor(TypeTag<fp6_e3m2>{});
    case DataType::BOOLEAN:
        return visitor(TypeTag<bool>{});
    default:
        throw std::runtime_error("Unsupported data type for tensor");
    }
}

inline std::unique_ptr<hipdnn_data_sdk::utilities::ITensor>
    createTensor(hipdnn_flatbuffers_sdk::data_objects::DataType dataType,
                 const std::vector<int64_t>& dims,
                 const std::vector<int64_t>& strides,
                 bool packSubByteElements = false)
{
    using namespace hipdnn_data_sdk::utilities;
    using namespace hipdnn_data_sdk::types;
    if(packSubByteElements)
    {
        switch(dataType)
        {
        case hipdnn_flatbuffers_sdk::data_objects::DataType::FP4_E2M1:
            return std::make_unique<PackedFp4Tensor>(dims, strides);
        case hipdnn_flatbuffers_sdk::data_objects::DataType::INT4:
            throw std::runtime_error("createTensor: packed layout not implemented for INT4");
        case hipdnn_flatbuffers_sdk::data_objects::DataType::FP6_E2M3:
            return std::make_unique<PackedFp6Tensor<fp6_e2m3>>(dims, strides);
        case hipdnn_flatbuffers_sdk::data_objects::DataType::FP6_E3M2:
            return std::make_unique<PackedFp6Tensor<fp6_e3m2>>(dims, strides);
        default:
            break;
        }
    }

    return visitNativeType(dataType, [&](auto tag) -> std::unique_ptr<ITensor> {
        using T = typename decltype(tag)::type;
        return std::make_unique<Tensor<T>>(dims, strides);
    });
}

/// Ragged graph tensors are SDPA operands, which keep logical dims [B, H, S, D] with the
/// sequence axis fixed at SDPA_SEQ_AXIS.
inline std::unique_ptr<hipdnn_data_sdk::utilities::ITensor>
    createRaggedTensor(hipdnn_flatbuffers_sdk::data_objects::DataType dataType,
                       const std::vector<int64_t>& dims,
                       const std::vector<int64_t>& strides,
                       std::shared_ptr<hipdnn_data_sdk::utilities::ITensor> offsets,
                       int64_t raggedOffsetMultiplier = 1)
{
    using namespace hipdnn_data_sdk::utilities;
    if(dims.size() != 4)
    {
        throw std::invalid_argument(
            "ragged tensors must be rank 4 (logical [B, H, S, D]); got rank "
            + std::to_string(dims.size()));
    }

    return visitNativeType(dataType, [&](auto tag) -> std::unique_ptr<ITensor> {
        using T = typename decltype(tag)::type;
        return std::make_unique<RaggedTensor<T>>(
            dims, strides, SDPA_SEQ_AXIS, std::move(offsets), std::nullopt, raggedOffsetMultiplier);
    });
}

inline std::shared_ptr<hipdnn_data_sdk::utilities::ITensor> createTensorFromAttribute(
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& attribute,
    bool packSubByteElements = false)
{
    auto dims
        = hipdnn_flatbuffers_sdk::utilities::convertFlatBufferVectorToStdVector(attribute.dims());
    auto strides = hipdnn_flatbuffers_sdk::utilities::convertFlatBufferVectorToStdVector(
        attribute.strides());

    return createTensor(attribute.data_type(), dims, strides, packSubByteElements);
}

inline bool isSubByteDataType(hipdnn_flatbuffers_sdk::data_objects::DataType dataType)
{
    return dataType == hipdnn_flatbuffers_sdk::data_objects::DataType::FP4_E2M1
           || dataType == hipdnn_flatbuffers_sdk::data_objects::DataType::FP6_E2M3
           || dataType == hipdnn_flatbuffers_sdk::data_objects::DataType::FP6_E3M2;
}

inline std::shared_ptr<hipdnn_data_sdk::utilities::ITensor>
    createRaggedTensorFromAttributeAndOffset(
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& attribute,
        std::shared_ptr<hipdnn_data_sdk::utilities::ITensor> raggedOffset)
{
    auto dims
        = hipdnn_flatbuffers_sdk::utilities::convertFlatBufferVectorToStdVector(attribute.dims());
    auto strides = hipdnn_flatbuffers_sdk::utilities::convertFlatBufferVectorToStdVector(
        attribute.strides());

    return createRaggedTensor(attribute.data_type(),
                              dims,
                              strides,
                              std::move(raggedOffset),
                              attribute.ragged_offset_multiplier());
}

} // namespace hipdnn_test_sdk::detail
