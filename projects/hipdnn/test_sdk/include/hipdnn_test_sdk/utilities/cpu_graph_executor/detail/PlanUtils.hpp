// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/ApplicabilityUtils.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>

/// Result of extracting diagonal band mask parameters from SDPA node attributes.
struct DiagonalBandParams
{
    int64_t leftBound;
    int64_t rightBound;
    bool topLeftAlignment;
};

/// Extracts diagonal band mask parameters from SDPA attributes (forward or backward).
/// A deprecated causal_mask / causal_mask_bottom_right flag is merged the way cuDNN's setters
/// compose: it sets right_bound=0 (and BOTTOM_RIGHT alignment for causal_mask_bottom_right), then
/// any left_bound, right_bound other than -1, and (for causal_mask) diagonal_alignment apply on
/// top. Setting both flags is rejected.
/// Works with both SdpaAttributes and SdpaBackwardAttributes FlatBuffers types
/// (they expose identical accessor signatures for mask fields).
template <typename SdpaAttributesType>
DiagonalBandParams extractDiagonalBandParams(const SdpaAttributesType& nodeAttributes,
                                             const char* planName)
{
    const int64_t leftBound
        = nodeAttributes.left_bound().has_value() ? nodeAttributes.left_bound().value() : -1;
    const int64_t rightBound
        = nodeAttributes.right_bound().has_value() ? nodeAttributes.right_bound().value() : -1;

    if(leftBound < -1 || rightBound < -1)
    {
        throw std::invalid_argument(
            std::string(planName) + ": left_bound and right_bound must be >= -1 (got left_bound="
            + std::to_string(leftBound) + ", right_bound=" + std::to_string(rightBound) + ")");
    }
    // cuDNN's left_bound counts the diagonal key, so 0 would mask every key (cuDNN rejects it).
    if(leftBound == 0)
    {
        throw std::invalid_argument(std::string(planName)
                                    + ": left_bound must be positive or -1 (unbounded); 0 keeps "
                                      "no keys");
    }

    const bool isTopLeft = nodeAttributes.diagonal_alignment()
                           == hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment::TOP_LEFT;
    const bool causalDeprecated = nodeAttributes.causal_mask();
    const bool bottomRightDeprecated = nodeAttributes.causal_mask_bottom_right();

    if(causalDeprecated && bottomRightDeprecated)
    {
        throw std::invalid_argument(std::string(planName)
                                    + ": cannot set both causal_mask and causal_mask_bottom_right. "
                                      "Use diagonal_alignment={TOP_LEFT|BOTTOM_RIGHT} with "
                                      "left_bound=-1, right_bound=0 instead.");
    }

    if(causalDeprecated || bottomRightDeprecated)
    {
        return {leftBound, rightBound >= 0 ? rightBound : 0, causalDeprecated && isTopLeft};
    }

    return {leftBound, rightBound, isTopLeft};
}

/// cuDNN ALiBi is supported only as cuDNN defines it: with top-left causal masking (right bound 0
/// once the deprecated flags are resolved, top-left alignment), SdpaSemanticsRegister SEM-021.
/// Works for forward and backward attributes; never throws.
template <typename SdpaAttributesType>
bool isSupportedSdpaAlibi(const SdpaAttributesType& nodeAttributes)
{
    if(!nodeAttributes.alibi_mask())
    {
        return true;
    }
    try
    {
        const auto band = extractDiagonalBandParams(nodeAttributes, "SDPA ALiBi");
        return band.rightBound == 0 && band.topLeftAlignment;
    }
    catch(const std::invalid_argument&)
    {
        return false;
    }
}

/// The raw pointer an options setter takes, or nullptr. (Calling get() inline inside the
/// templated options chains trips clang-tidy-20's readability-redundant-smartptr-get, a false
/// positive in dependent calls.)
template <typename T>
T* rawPointer(const std::unique_ptr<T>& owner)
{
    return owner.get();
}

/// Whether the SDPA node's padding attributes are ones the references implement. Padding is
/// given by both seq_len_q and seq_len_kv, each INT32 with one value per batch ([B, 1, 1, 1]).
/// padding_mask without both tensors, or only one tensor, is declined (cuDNN requires both). The
/// tensors without padding_mask are accepted as padding, because hipDNN graphs omit the flag
/// (SdpaSemanticsRegister.md SEM-008). Works for forward and backward attributes.
template <typename SdpaAttributesType, typename TensorMap>
bool isSupportedSdpaPadding(const SdpaAttributesType& nodeAttributes, const TensorMap& tensorMap)
{
    const auto seqLenQ = nodeAttributes.seq_len_q_tensor_uid();
    const auto seqLenKv = nodeAttributes.seq_len_kv_tensor_uid();
    if(!seqLenQ.has_value() && !seqLenKv.has_value())
    {
        return !nodeAttributes.padding_mask();
    }
    if(!seqLenQ.has_value() || !seqLenKv.has_value())
    {
        return false;
    }
    const auto qIt = tensorMap.find(nodeAttributes.q_tensor_uid());
    if(qIt == tensorMap.end() || qIt->second->dims() == nullptr || qIt->second->dims()->size() == 0)
    {
        return false;
    }
    const auto batch = qIt->second->dims()->Get(0);
    for(const auto uid : {seqLenQ.value(), seqLenKv.value()})
    {
        const auto it = tensorMap.find(uid);
        if(it == tensorMap.end()
           || it->second->data_type() != hipdnn_flatbuffers_sdk::data_objects::DataType::INT32
           || it->second->dims() == nullptr || it->second->dims()->size() == 0
           || it->second->dims()->Get(0) != batch)
        {
            return false;
        }
        int64_t count = 1;
        for(const auto dim : *it->second->dims())
        {
            count *= dim;
        }
        if(count != batch)
        {
            return false;
        }
    }
    return true;
}

/// Logical key extent of a paged SDPA forward node, inferred as cuDNN does
/// (scaled_dot_product_flash_attention.h): max_seq_len_kv when set; else a non-paged K's (then
/// V's) sequence length; else the bias's last dim; else rng_dump's last dim; else the smaller
/// tableSize * blockSize of the two sides. Returns -1 when the node is
/// not a paging layout the references implement:
/// - paging needs padding (both seq_len tensors), as cuDNN requires;
/// - each page table is INT32 [B, 1, tableSize, 1] over a rank-4 [blocks, H, blockSize, D]
///   container;
/// - max_seq_len_kv must fit every page table, whose size must be exactly
///   ceil(max_seq_len_kv / blockSize) as in cuDNN;
/// - a non-paged K or V must already be dense [B, H, maxSeqKv, D];
/// - a bias's last dim must be maxSeqKv or 1.
/// Returns 0 for a non-paged node, which must not set max_seq_len_kv (cuDNN rejects it).
template <typename TensorMap>
int64_t sdpaPagedMaxSeqKv(const hipdnn_flatbuffers_sdk::data_objects::SdpaAttributes& attrs,
                          const TensorMap& tensorMap)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    const auto pageK = attrs.page_table_k_tensor_uid();
    const auto pageV = attrs.page_table_v_tensor_uid();
    if(!pageK.has_value() && !pageV.has_value())
    {
        return attrs.max_seq_len_kv().has_value() ? -1 : 0;
    }
    if(!attrs.seq_len_q_tensor_uid().has_value() || !attrs.seq_len_kv_tensor_uid().has_value())
    {
        return -1;
    }
    const auto dimsOf = [&tensorMap](int64_t uid) -> std::vector<int64_t> {
        const auto it = tensorMap.find(uid);
        if(it == tensorMap.end() || it->second->dims() == nullptr)
        {
            return {};
        }
        return {it->second->dims()->begin(), it->second->dims()->end()};
    };
    const auto qDims = dimsOf(attrs.q_tensor_uid());
    if(qDims.size() != 4)
    {
        return -1;
    }
    const auto batch = qDims[0];

    // Capacity (tableSize * blockSize) of one paged side, or -1 if its layout is unsupported.
    const auto pagedCapacity = [&](int64_t tableUid, int64_t containerUid) -> int64_t {
        const auto it = tensorMap.find(tableUid);
        const auto tableDims = dimsOf(tableUid);
        const auto containerDims = dimsOf(containerUid);
        if(it == tensorMap.end() || it->second->data_type() != DataType::INT32
           || tableDims.size() != 4 || tableDims[0] != batch || tableDims[1] != 1
           || tableDims[3] != 1 || containerDims.size() != 4)
        {
            return -1;
        }
        return tableDims[2] * containerDims[2];
    };
    int64_t capacityK = -1;
    int64_t capacityV = -1;
    if(pageK.has_value())
    {
        capacityK = pagedCapacity(pageK.value(), attrs.k_tensor_uid());
        if(capacityK < 0)
        {
            return -1;
        }
    }
    if(pageV.has_value())
    {
        capacityV = pagedCapacity(pageV.value(), attrs.v_tensor_uid());
        if(capacityV < 0)
        {
            return -1;
        }
    }

    const auto lastDimOf = [&dimsOf](std::optional<int64_t> uid) -> int64_t {
        const auto dims = uid.has_value() ? dimsOf(uid.value()) : std::vector<int64_t>{};
        return dims.size() == 4 ? dims[3] : -1;
    };
    int64_t maxSeqKv = -1;
    if(attrs.max_seq_len_kv().has_value())
    {
        maxSeqKv = static_cast<int64_t>(attrs.max_seq_len_kv().value());
    }
    else if(!pageK.has_value() || !pageV.has_value())
    {
        const auto dense = dimsOf(pageK.has_value() ? attrs.v_tensor_uid() : attrs.k_tensor_uid());
        maxSeqKv = dense.size() == 4 ? dense[2] : -1;
    }
    else if(attrs.attn_mask_tensor_uid().has_value())
    {
        maxSeqKv = lastDimOf(attrs.attn_mask_tensor_uid());
    }
    else if(attrs.rng_dump_tensor_uid().has_value())
    {
        maxSeqKv = lastDimOf(attrs.rng_dump_tensor_uid());
    }
    else
    {
        maxSeqKv = std::min(capacityK, capacityV);
    }
    if(maxSeqKv <= 0 || (pageK.has_value() && maxSeqKv > capacityK)
       || (pageV.has_value() && maxSeqKv > capacityV))
    {
        return -1;
    }
    for(const auto& [paged, uid] : {std::pair{pageK.has_value(), attrs.k_tensor_uid()},
                                    std::pair{pageV.has_value(), attrs.v_tensor_uid()}})
    {
        const auto dims = dimsOf(uid);
        if(!paged && (dims.size() != 4 || dims[0] != batch || dims[2] != maxSeqKv))
        {
            return -1;
        }
    }
    // cuDNN's paged cache load takes a table of exactly ceil(maxSeqKv / blockSize) entries per batch
    // (cudnn-frontend paged_cache_load.h); a wider one is declined.
    for(const auto& [table, container] :
        {std::pair{pageK, attrs.k_tensor_uid()}, std::pair{pageV, attrs.v_tensor_uid()}})
    {
        if(table.has_value())
        {
            const auto blockSize = dimsOf(container)[2];
            if(blockSize <= 0 || dimsOf(table.value())[2] != (maxSeqKv + blockSize - 1) / blockSize)
            {
                return -1;
            }
        }
    }
    if(attrs.attn_mask_tensor_uid().has_value())
    {
        const auto maskDims = dimsOf(attrs.attn_mask_tensor_uid().value());
        if(maskDims.empty() || (maskDims.back() != maxSeqKv && maskDims.back() != 1))
        {
            return -1;
        }
    }
    return maxSeqKv;
}

/// Whether an SDPA forward node's attention sink is one the references implement: absent, or a
/// FLOAT tensor [1, H, 1, 1] with H the query head count (cuDNN's shape).
template <typename TensorMap>
bool isSupportedSdpaSink(const hipdnn_flatbuffers_sdk::data_objects::SdpaAttributes& attrs,
                         const TensorMap& tensorMap)
{
    if(!attrs.sink_token_tensor_uid().has_value())
    {
        return true;
    }
    const auto sinkIt = tensorMap.find(attrs.sink_token_tensor_uid().value());
    const auto qIt = tensorMap.find(attrs.q_tensor_uid());
    if(sinkIt == tensorMap.end() || qIt == tensorMap.end() || sinkIt->second->dims() == nullptr
       || qIt->second->dims() == nullptr || qIt->second->dims()->size() != 4
       || sinkIt->second->data_type() != hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT)
    {
        return false;
    }
    const std::vector<int64_t> dims(sinkIt->second->dims()->begin(), sinkIt->second->dims()->end());
    return dims == std::vector<int64_t>{1, qIt->second->dims()->Get(1), 1, 1};
}

/// Whether an SDPA node's dropout is one the references implement: none, or cuDNN's custom
/// dropout with an explicit FLOAT mask [B or 1, H or 1, Sq, Skv] and a single-element FLOAT
/// dropout_scale. Philox dropout (dropout_probability / seed / offset) is declined: the
/// references cannot reproduce a kernel's random stream. Works for forward and backward
/// attributes; the backward's dropout_scale_inv is checked by the caller.
template <typename SdpaAttributesType, typename TensorMap>
bool isSupportedSdpaDropout(const SdpaAttributesType& attrs, const TensorMap& tensorMap)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    if(attrs.dropout_probability().has_value() || attrs.seed_tensor_uid().has_value()
       || attrs.offset_tensor_uid().has_value())
    {
        return false;
    }
    const auto maskUid = attrs.dropout_mask_tensor_uid();
    const auto scaleUid = attrs.dropout_scale_tensor_uid();
    if(!maskUid.has_value())
    {
        return !scaleUid.has_value();
    }
    if(!scaleUid.has_value())
    {
        return false;
    }
    const auto maskIt = tensorMap.find(maskUid.value());
    const auto scaleIt = tensorMap.find(scaleUid.value());
    const auto qIt = tensorMap.find(attrs.q_tensor_uid());
    const auto kIt = tensorMap.find(attrs.k_tensor_uid());
    if(maskIt == tensorMap.end() || scaleIt == tensorMap.end() || qIt == tensorMap.end()
       || kIt == tensorMap.end() || maskIt->second->dims() == nullptr
       || scaleIt->second->dims() == nullptr || qIt->second->dims() == nullptr
       || kIt->second->dims() == nullptr || maskIt->second->data_type() != DataType::FLOAT
       || scaleIt->second->data_type() != DataType::FLOAT)
    {
        return false;
    }
    int64_t scaleCount = 1;
    for(const auto dim : *scaleIt->second->dims())
    {
        scaleCount *= dim;
    }
    const auto* m = maskIt->second->dims();
    const auto* q = qIt->second->dims();
    const auto* k = kIt->second->dims();
    return scaleCount == 1 && m->size() == 4 && q->size() == 4 && k->size() == 4
           && (m->Get(0) == q->Get(0) || m->Get(0) == 1)
           && (m->Get(1) == q->Get(1) || m->Get(1) == 1) && m->Get(2) == q->Get(2)
           && m->Get(3) == k->Get(2);
}

/// cuDNN's block-sparse mask as the references implement it (SEM-025): UINT8
/// [B, H_q, ceil(Sq / 128), ceil(ceil(Skv / 128) / 8)], where Skv is the paged key extent when K/V
/// are paged. Absent is supported.
template <typename TensorMap>
bool isSupportedSdpaBlockMask(const hipdnn_flatbuffers_sdk::data_objects::SdpaAttributes& attrs,
                              const TensorMap& tensorMap)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    const auto uid = attrs.block_mask_tensor_uid();
    if(!uid.has_value())
    {
        return true;
    }
    const auto maskIt = tensorMap.find(uid.value());
    const auto qIt = tensorMap.find(attrs.q_tensor_uid());
    const auto kIt = tensorMap.find(attrs.k_tensor_uid());
    if(maskIt == tensorMap.end() || qIt == tensorMap.end() || kIt == tensorMap.end()
       || maskIt->second->data_type() != DataType::UINT8 || maskIt->second->dims() == nullptr
       || qIt->second->dims() == nullptr || kIt->second->dims() == nullptr
       || maskIt->second->dims()->size() != 4 || qIt->second->dims()->size() != 4
       || kIt->second->dims()->size() != 4)
    {
        return false;
    }
    constexpr int64_t TILE = 128;
    const auto paged = sdpaPagedMaxSeqKv(attrs, tensorMap);
    const int64_t seqKv = paged > 0 ? paged : kIt->second->dims()->Get(2);
    const auto* q = qIt->second->dims();
    const auto* m = maskIt->second->dims();
    const auto tilesKv = (seqKv + TILE - 1) / TILE;
    return m->Get(0) == q->Get(0) && m->Get(1) == q->Get(1)
           && m->Get(2) == (q->Get(2) + TILE - 1) / TILE && m->Get(3) == (tilesKv + 7) / 8;
}

/// The cuDNN sdpa_fp8 softmax / output scaling the references implement (SEM-024): with fp8
/// inputs, scale_s and descale_s together, scale_o, and the amax_s / amax_o outputs, each a
/// one-element FLOAT tensor; none of them for other inputs. Dropout does not combine with them.
template <typename TensorMap>
bool isSupportedSdpaFp8Scaling(const hipdnn_flatbuffers_sdk::data_objects::SdpaAttributes& attrs,
                               const TensorMap& tensorMap,
                               bool fp8Input)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    bool any = false;
    for(const auto uid : {attrs.scale_s_tensor_uid(),
                          attrs.descale_s_tensor_uid(),
                          attrs.scale_o_tensor_uid(),
                          attrs.amax_s_tensor_uid(),
                          attrs.amax_o_tensor_uid()})
    {
        if(!uid.has_value())
        {
            continue;
        }
        any = true;
        const auto it = tensorMap.find(uid.value());
        if(it == tensorMap.end() || it->second->data_type() != DataType::FLOAT
           || it->second->dims() == nullptr)
        {
            return false;
        }
        int64_t count = 1;
        for(const auto dim : *it->second->dims())
        {
            count *= dim;
        }
        if(count != 1)
        {
            return false;
        }
    }
    if(!any)
    {
        return true;
    }
    const bool dropout
        = attrs.dropout_mask_tensor_uid().has_value() || attrs.dropout_probability().has_value();
    return fp8Input && !dropout
           && attrs.scale_s_tensor_uid().has_value() == attrs.descale_s_tensor_uid().has_value();
}

/// A one-element FLOAT operand of a CPU plan: a pass-by-value or baked scalar, or an ordinary
/// tensor whose host pointer is in the variant pack (the CPU executor binds host memory).
inline float
    resolveHostScalarOperand(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT& attrs,
                             const std::unordered_map<int64_t, void*>& variantPack,
                             const char* name)
{
    if(attrs.is_runtime_pass_by_value || attrs.value.value != nullptr)
    {
        return hipdnn_flatbuffers_sdk::utilities::resolveScalarFromVariantPack<float>(
            attrs, variantPack, name);
    }
    const auto it = variantPack.find(attrs.uid);
    if(it == variantPack.end() || it->second == nullptr)
    {
        throw std::runtime_error(std::string(name) + " tensor missing from the variant pack");
    }
    return *static_cast<const float*>(it->second);
}

/// Whether an SDPA forward node is Philox dropout that the references can check through
/// RNG_DUMP: dropout_probability in [0, 1), seed and offset given, no explicit mask, and an
/// rng_dump output FLOAT [B, H_q, Sq, Skv]. The engine writes its keep mask to rng_dump; the
/// references read it back as the explicit mask with scale 1 / (1 - p) (cuDNN's composition).
template <typename TensorMap>
bool isSdpaPhiloxDropoutWithDump(const hipdnn_flatbuffers_sdk::data_objects::SdpaAttributes& attrs,
                                 const TensorMap& tensorMap)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    if(!attrs.dropout_probability().has_value() || !attrs.seed_tensor_uid().has_value()
       || !attrs.offset_tensor_uid().has_value() || !attrs.rng_dump_tensor_uid().has_value()
       || attrs.dropout_mask_tensor_uid().has_value()
       || attrs.dropout_scale_tensor_uid().has_value())
    {
        return false;
    }
    const auto p = attrs.dropout_probability().value();
    if(std::isnan(p) || p < 0.0F || p >= 1.0F)
    {
        return false;
    }
    const auto dumpIt = tensorMap.find(attrs.rng_dump_tensor_uid().value());
    const auto qIt = tensorMap.find(attrs.q_tensor_uid());
    const auto kIt = tensorMap.find(attrs.k_tensor_uid());
    if(dumpIt == tensorMap.end() || qIt == tensorMap.end() || kIt == tensorMap.end()
       || dumpIt->second->dims() == nullptr || qIt->second->dims() == nullptr
       || kIt->second->dims() == nullptr || dumpIt->second->data_type() != DataType::FLOAT)
    {
        return false;
    }
    const auto* d = dumpIt->second->dims();
    const auto* q = qIt->second->dims();
    const auto* k = kIt->second->dims();
    return d->size() == 4 && q->size() == 4 && k->size() == 4 && d->Get(0) == q->Get(0)
           && d->Get(1) == q->Get(1) && d->Get(2) == q->Get(2) && d->Get(3) == k->Get(2);
}

#define CHECK_TENSOR_EXISTS(tensor_map, tensor_uid) \
    do                                              \
    {                                               \
        auto it = (tensor_map).find((tensor_uid));  \
        if(it == (tensor_map).end())                \
        {                                           \
            return false;                           \
        }                                           \
    } while(0)

#define CHECK_TENSOR_TYPE(tensor_map, tensor_uid, datatype_enum) \
    do                                                           \
    {                                                            \
        auto tensor = (tensor_map).at((tensor_uid));             \
        if(tensor->data_type() != (datatype_enum))               \
        {                                                        \
            return false;                                        \
        }                                                        \
    } while(0)

#define CHECK_OPTIONAL_TENSOR_EXISTS(tensor_map, optional_tensor_uid) \
    do                                                                \
    {                                                                 \
        if(!(optional_tensor_uid).has_value())                        \
        {                                                             \
            return false;                                             \
        }                                                             \
        CHECK_TENSOR_EXISTS(tensor_map, *(optional_tensor_uid));      \
    } while(0)

#define CHECK_OPTIONAL_TENSOR_TYPE(tensor_map, optional_tensor_uid, datatype_enum) \
    CHECK_TENSOR_TYPE(tensor_map, *(optional_tensor_uid), datatype_enum)

// For reference plans that do not support ragged tensors: a graph containing one makes the plan
// builder report not applicable.
#define CHECK_NO_RAGGED_TENSORS(tensor_map)                                        \
    do                                                                             \
    {                                                                              \
        if(!hipdnn_flatbuffers_sdk::utilities::hasNoRaggedTensorIds((tensor_map))) \
        {                                                                          \
            return false;                                                          \
        }                                                                          \
    } while(0)
