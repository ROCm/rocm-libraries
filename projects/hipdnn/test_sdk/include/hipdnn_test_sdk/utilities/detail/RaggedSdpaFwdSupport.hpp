// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <optional>
#include <unordered_map>

#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/RaggedTokenBoundaries.hpp>

// Support shared by the CPU and GPU reference plans for ragged SDPA forward (RFC-0014). Both
// references use the same routing, the same applicability gate and the same parameters. Each
// registry lists its own Q/K/V/O type tuples; while those lists match, a ragged SDPA node is
// accepted by both references or by neither.
namespace hipdnn_test_sdk::detail
{

/// True if any of an SDPA node's Q/K/V/O tensors has a ragged_offset. The reference graph
/// executors key such a node as ragged. The ragged plan needs all four tensors ragged, and the
/// dense plan rejects any ragged tensor, so a partly ragged node is declined by both. Uids missing
/// from the map are ignored.
inline bool hasRaggedSdpaPrimary(
    const hipdnn_flatbuffers_sdk::data_objects::SdpaAttributes& attributes,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    for(const auto uid : {attributes.q_tensor_uid(),
                          attributes.k_tensor_uid(),
                          attributes.v_tensor_uid(),
                          attributes.o_tensor_uid()})
    {
        const auto it = tensorMap.find(uid);
        if(it != tensorMap.end() && it->second != nullptr
           && it->second->ragged_offset_tensor_uid().has_value())
        {
            return true;
        }
    }
    return false;
}

/// RFC-0014 requires ragged_offset_multiplier >= 1, and logical [B, H, S, D] dims with
/// token-major (BSHD) strides (see isTokenMajorRaggedLayout).
inline bool isSupportedRaggedSdpaLayout(
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& attributes)
{
    return attributes.ragged_offset_multiplier() >= 1
           && isTokenMajorRaggedLayout(
               hipdnn_flatbuffers_sdk::utilities::convertFlatBufferVectorToStdVector(
                   attributes.dims()),
               hipdnn_flatbuffers_sdk::utilities::convertFlatBufferVectorToStdVector(
                   attributes.strides()));
}

namespace ragged_sdpa_detail
{

inline const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* findTensor(
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap,
    int64_t uid)
{
    const auto it = tensorMap.find(uid);
    return it != tensorMap.end() ? it->second : nullptr;
}

inline bool hasTensorOfType(
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap,
    int64_t uid,
    hipdnn_flatbuffers_sdk::data_objects::DataType dataType)
{
    const auto* tensor = findTensor(tensorMap, uid);
    return tensor != nullptr && tensor->data_type() == dataType;
}

// A ragged tensor needs an INT32 ragged_offset aux in the map and a supported layout.
inline bool isSupportedRaggedTensor(
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& tensor,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    const auto raggedOffsetUid = tensor.ragged_offset_tensor_uid();
    return raggedOffsetUid.has_value()
           && hasTensorOfType(tensorMap,
                              raggedOffsetUid.value(),
                              hipdnn_flatbuffers_sdk::data_objects::DataType::INT32)
           && isSupportedRaggedSdpaLayout(tensor);
}

inline std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> unpackOptional(
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap,
    ::flatbuffers::Optional<int64_t> uid)
{
    if(!uid.has_value())
    {
        return std::nullopt;
    }
    return unpackTensorAttributes(*tensorMap.at(uid.value()));
}

// The ragged_offset aux of a primary that isSupportedRaggedSdpaFwdNode accepted.
inline hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT unpackRaggedOffset(
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT& primary,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    return unpackTensorAttributes(*tensorMap.at(primary.ragged_offset_tensor_uid.value()));
}

} // namespace ragged_sdpa_detail

/// The ragged SDPA forward gate shared by the CPU and GPU plan builders, except the Q/K/V/O data
/// type checks, which depend on each builder's registered types. It never throws, and it returns
/// false for a node that breaks any rule below. Shape consistency (batch and head counts, per-batch
/// lengths, LSE and descale dims) is not checked here; the references throw on it at execute time.
///  - Q/K/V/O (and a packed LSE) must each have an INT32 ragged_offset aux with a multiplier >= 1
///    and use logical [B, H, S, D] dims with token-major (BSHD) strides.
///  - seq_len_q/kv must be absent. Lengths come from ragged_offset alone.
///  - No alibi or padding mask, additive attention bias, dropout, paged KV cache or block-sparse
///    attention. The ragged references have no inputs for them.
///  - Only Q/K/V descales, which must be FLOAT. A pass-by-value descale must be one element.
///  - The scale must be one element. An ordinary scale tensor must be FLOAT.
///  - No max / sum-exp outputs. An optional LSE must be FLOAT; it is dense [B, H, Sq, 1] or packed.
inline bool isSupportedRaggedSdpaFwdNode(
    const hipdnn_flatbuffers_sdk::data_objects::SdpaAttributes& attributes,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    using ragged_sdpa_detail::findTensor;
    using ragged_sdpa_detail::hasTensorOfType;

    for(const auto primaryUid : {attributes.q_tensor_uid(),
                                 attributes.k_tensor_uid(),
                                 attributes.v_tensor_uid(),
                                 attributes.o_tensor_uid()})
    {
        const auto* primary = findTensor(tensorMap, primaryUid);
        if(primary == nullptr || !ragged_sdpa_detail::isSupportedRaggedTensor(*primary, tensorMap))
        {
            return false;
        }
    }

    if(attributes.seq_len_q_tensor_uid().has_value()
       || attributes.seq_len_kv_tensor_uid().has_value())
    {
        return false;
    }

    if(attributes.alibi_mask() || attributes.padding_mask()
       || attributes.attn_mask_tensor_uid().has_value())
    {
        return false;
    }

    if(attributes.dropout_probability().has_value() || attributes.seed_tensor_uid().has_value()
       || attributes.offset_tensor_uid().has_value()
       || attributes.dropout_mask_tensor_uid().has_value()
       || attributes.dropout_scale_tensor_uid().has_value()
       || attributes.rng_dump_tensor_uid().has_value())
    {
        return false;
    }

    if(attributes.page_table_k_tensor_uid().has_value()
       || attributes.page_table_v_tensor_uid().has_value()
       || attributes.block_mask_tensor_uid().has_value()
       || attributes.sink_token_tensor_uid().has_value())
    {
        return false;
    }

    // Q/K/V descales only: no softmax descale, softmax/output scales or amax outputs.
    if(attributes.descale_s_tensor_uid().has_value() || attributes.scale_s_tensor_uid().has_value()
       || attributes.scale_o_tensor_uid().has_value() || attributes.amax_s_tensor_uid().has_value()
       || attributes.amax_o_tensor_uid().has_value())
    {
        return false;
    }
    for(const auto descaleUid : {attributes.descale_q_tensor_uid(),
                                 attributes.descale_k_tensor_uid(),
                                 attributes.descale_v_tensor_uid()})
    {
        if(!descaleUid.has_value())
        {
            continue;
        }
        if(!hasTensorOfType(tensorMap, descaleUid.value(), DataType::FLOAT))
        {
            return false;
        }
        const auto descale = unpackTensorAttributes(*tensorMap.at(descaleUid.value()));
        if(hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(descale)
           && elementCount(descale) != 1)
        {
            return false;
        }
    }

    if(attributes.scale_tensor_uid().has_value())
    {
        const auto* scalePtr = findTensor(tensorMap, attributes.scale_tensor_uid().value());
        if(scalePtr == nullptr)
        {
            return false;
        }
        const auto scale = unpackTensorAttributes(*scalePtr);
        if(elementCount(scale) != 1
           || (!hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(scale)
               && scale.data_type != DataType::FLOAT))
        {
            return false;
        }
    }

    if(attributes.max_tensor_uid().has_value() || attributes.sum_exp_tensor_uid().has_value())
    {
        return false;
    }

    if(attributes.stats_tensor_uid().has_value())
    {
        if(!hasTensorOfType(tensorMap, attributes.stats_tensor_uid().value(), DataType::FLOAT))
        {
            return false;
        }
        const auto* stats = tensorMap.at(attributes.stats_tensor_uid().value());
        if(stats->ragged_offset_tensor_uid().has_value()
           && !ragged_sdpa_detail::isSupportedRaggedTensor(*stats, tensorMap))
        {
            return false;
        }
    }

    return true;
}

/// Unpacked tensors and resolved parameters of a ragged SDPA forward node. Q/K/V/O are logical
/// [B, H, S, D] tensors packed token by token. Each has an INT32 ragged_offset aux [B + 1, 1, 1, 1]
/// whose entries, times the tensor's ragged_offset_multiplier, are element offsets of each batch.
/// The optional LSE [B, H, Sq, 1] is packed if it has its own ragged_offset aux, else dense.
struct RaggedSdpaFwdParams
{
    /// Precondition: isSupportedRaggedSdpaFwdNode(attributes, tensorMap) is true.
    RaggedSdpaFwdParams(
        const hipdnn_flatbuffers_sdk::data_objects::SdpaAttributes& attributes,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap,
        int64_t leftBound,
        int64_t rightBound,
        bool topLeftAlignment)
        : qTensor(unpackTensorAttributes(*tensorMap.at(attributes.q_tensor_uid())))
        , kTensor(unpackTensorAttributes(*tensorMap.at(attributes.k_tensor_uid())))
        , vTensor(unpackTensorAttributes(*tensorMap.at(attributes.v_tensor_uid())))
        , oTensor(unpackTensorAttributes(*tensorMap.at(attributes.o_tensor_uid())))
        , raggedOffsetQTensor(ragged_sdpa_detail::unpackRaggedOffset(qTensor, tensorMap))
        , raggedOffsetKTensor(ragged_sdpa_detail::unpackRaggedOffset(kTensor, tensorMap))
        , raggedOffsetVTensor(ragged_sdpa_detail::unpackRaggedOffset(vTensor, tensorMap))
        , raggedOffsetOTensor(ragged_sdpa_detail::unpackRaggedOffset(oTensor, tensorMap))
        , scaleTensor(foldSdpaScale(attributes.scale_tensor_uid().has_value()
                                        ? tensorMap.at(attributes.scale_tensor_uid().value())
                                        : nullptr,
                                    attributes.attn_scale_value().has_value()
                                        ? std::make_optional(attributes.attn_scale_value().value())
                                        : std::nullopt))
        , leftBound(leftBound)
        , rightBound(rightBound)
        , topLeftAlignment(topLeftAlignment)
        , lseTensor(ragged_sdpa_detail::unpackOptional(tensorMap, attributes.stats_tensor_uid()))
        , raggedOffsetLseTensor(lseTensor.has_value()
                                    ? ragged_sdpa_detail::unpackOptional(
                                          tensorMap, lseTensor->ragged_offset_tensor_uid)
                                    : std::nullopt)
        , descaleQTensor(
              ragged_sdpa_detail::unpackOptional(tensorMap, attributes.descale_q_tensor_uid()))
        , descaleKTensor(
              ragged_sdpa_detail::unpackOptional(tensorMap, attributes.descale_k_tensor_uid()))
        , descaleVTensor(
              ragged_sdpa_detail::unpackOptional(tensorMap, attributes.descale_v_tensor_uid()))
    {
    }

    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT qTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT kTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT vTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT oTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT raggedOffsetQTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT raggedOffsetKTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT raggedOffsetVTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT raggedOffsetOTensor;
    // Folded scale operand (scale tensor, else baked attn_scale_value), resolved at execute time.
    // Absent means 1.0 (no scaling), as in cuDNN.
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> scaleTensor;
    int64_t leftBound;
    int64_t rightBound;
    bool topLeftAlignment;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> lseTensor;
    // Set only for a packed LSE. Absent means a dense [B, H, Sq, 1] LSE.
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> raggedOffsetLseTensor;
    // Optional FLOAT Q/K/V descales: scalar [1] or per KV head [B, H_kv, 1, 1].
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> descaleQTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> descaleKTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> descaleVTensor;
};

} // namespace hipdnn_test_sdk::detail
