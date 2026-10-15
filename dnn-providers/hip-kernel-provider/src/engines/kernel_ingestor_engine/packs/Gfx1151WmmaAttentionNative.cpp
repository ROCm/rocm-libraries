// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/PluginDeviceBuffers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>
#include <hipdnn_plugin_sdk/ingestor/SymbolScope.hpp>

#include "compilation/KpackKernelLoader.hpp"
#include "compilation/KpackModuleCache.hpp"
#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

namespace
{

constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.gfx1151_wmma_attention.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.gfx1151_wmma_attention.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.gfx1151_wmma_attention.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.gfx1151_wmma_attention.dispatch";

constexpr std::string_view DTYPE_FIELD = "dtype";
constexpr std::string_view HEAD_SIZE_FIELD = "head_size";
constexpr std::string_view MASK_MODE_FIELD = "mask_mode";
constexpr std::string_view TRANSPOSED_QK_FIELD = "transposed_qk";
constexpr std::string_view QUERY_TAIL_FIELD = "query_tail";
constexpr std::string_view KV_TAIL_FIELD = "kv_tail";
constexpr std::string_view BLOCK_N_FIELD = "block_n";
constexpr std::string_view NUM_WAVES_FIELD = "num_waves";

constexpr std::string_view Q_TOKEN = "gfx1151_wmma_attention.q.uid";
constexpr std::string_view K_TOKEN = "gfx1151_wmma_attention.k.uid";
constexpr std::string_view V_TOKEN = "gfx1151_wmma_attention.v.uid";
constexpr std::string_view O_TOKEN = "gfx1151_wmma_attention.o.uid";
constexpr std::string_view LSE_TOKEN = "gfx1151_wmma_attention.lse.uid";
constexpr std::string_view WRITE_LSE_TOKEN = "gfx1151_wmma_attention.write_lse";
constexpr std::string_view MASK_TOKEN = "gfx1151_wmma_attention.mask";
constexpr std::string_view BOTTOM_RIGHT_TOKEN = "gfx1151_wmma_attention.bottom_right";
constexpr std::string_view WINDOW_LEFT_TOKEN = "gfx1151_wmma_attention.window_left";
constexpr std::string_view WINDOW_RIGHT_TOKEN = "gfx1151_wmma_attention.window_right";
constexpr std::string_view SCALE_BITS_TOKEN = "gfx1151_wmma_attention.scale_bits";

constexpr uint32_t BATCH_AXIS = 0;
constexpr uint32_t HEAD_AXIS = 1;
constexpr uint32_t SEQ_AXIS = 2;
constexpr uint32_t HEAD_SIZE_AXIS = 3;
constexpr uint32_t SDPA_RANK = 4;
constexpr int64_t UNBOUNDED = -1;

const data_objects::TensorAttributes* findTensor(const MatchContext& context, int64_t uid)
{
    const auto& tensors = context.graph.getTensorMap();
    const auto it = tensors.find(uid);
    return it == tensors.end() ? nullptr : it->second;
}

const data_objects::SdpaAttributes* sdpaNode(const MatchContext& context)
{
    if(context.graph.nodeCount() != 1)
    {
        return nullptr;
    }
    const auto& node = context.graph.getNodeWrapper(0);
    if(node.attributesType() != data_objects::NodeAttributes::SdpaAttributes)
    {
        return nullptr;
    }
    return &node.attributesAs<data_objects::SdpaAttributes>();
}

bool fitsI32(int64_t value)
{
    return value >= std::numeric_limits<int32_t>::min()
           && value <= std::numeric_limits<int32_t>::max();
}

bool isWellFormedOperand(const data_objects::TensorAttributes& tensor)
{
    const auto* dims = tensor.dims();
    const auto* strides = tensor.strides();
    // Ragged (THD) operands address each batch from a device offset table; this
    // engine binds only dense strides, so it must decline them rather than read
    // the buffer as dense.
    if(dims == nullptr || strides == nullptr || dims->size() != SDPA_RANK
       || strides->size() != SDPA_RANK || tensor.virtual_()
       || tensor.ragged_offset_tensor_uid().has_value()
       || hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(&tensor))
    {
        return false;
    }
    for(uint32_t i = 0; i < SDPA_RANK; ++i)
    {
        if(dims->Get(i) <= 0 || strides->Get(i) < 0 || !fitsI32(strides->Get(i)))
        {
            return false;
        }
    }
    if(dims->Get(HEAD_SIZE_AXIS) > 1 && strides->Get(HEAD_SIZE_AXIS) != 1)
    {
        return false;
    }

    std::array<std::pair<int64_t, int64_t>, SDPA_RANK> active{};
    size_t count = 0;
    for(uint32_t i = 0; i < SDPA_RANK; ++i)
    {
        if(dims->Get(i) > 1)
        {
            active[count++] = {strides->Get(i), dims->Get(i)};
        }
    }
    std::sort(active.begin(), active.begin() + static_cast<std::ptrdiff_t>(count));
    int64_t span = 1;
    for(size_t i = 0; i < count; ++i)
    {
        if(active[i].first < span)
        {
            return false;
        }
        int64_t term = 0;
        int64_t next = 0;
        if(__builtin_mul_overflow(active[i].first, active[i].second - 1, &term)
           || __builtin_add_overflow(span, term, &next))
        {
            return false;
        }
        span = next;
    }
    return true;
}

bool addressFitsI32(const data_objects::TensorAttributes& tensor, int64_t elementBytes)
{
    int64_t maximum = 0;
    for(uint32_t i = 0; i < SDPA_RANK; ++i)
    {
        int64_t term = 0;
        if(__builtin_mul_overflow(tensor.dims()->Get(i) - 1, tensor.strides()->Get(i), &term)
           || __builtin_add_overflow(maximum, term, &maximum))
        {
            return false;
        }
    }
    int64_t elements = 0;
    int64_t bytes = 0;
    return !__builtin_add_overflow(maximum, int64_t{1}, &elements)
           && !__builtin_mul_overflow(elements, elementBytes, &bytes) && fitsI32(bytes);
}

std::optional<std::string> supportedDataTypeName(data_objects::DataType dataType)
{
    if(dataType == data_objects::DataType::BFLOAT16)
    {
        return std::string("BF16");
    }
    if(dataType == data_objects::DataType::HALF)
    {
        return std::string("FP16");
    }
    return std::nullopt;
}

enum class MaskKind : int64_t
{
    NONE = 0,
    CAUSAL = 1,
    WINDOW = 2
};

struct MaskParameters
{
    MaskKind kind = MaskKind::NONE;
    int32_t bottomRight = 0;
    int32_t left = -1;
    int32_t right = -1;
};

std::optional<MaskParameters> maskFor(const data_objects::SdpaAttributes& attributes)
{
    const bool topLeftDeprecated = attributes.causal_mask();
    const bool bottomRightDeprecated = attributes.causal_mask_bottom_right();
    if(topLeftDeprecated && bottomRightDeprecated)
    {
        return std::nullopt;
    }

    const int64_t left
        = attributes.left_bound().has_value() ? attributes.left_bound().value() : UNBOUNDED;
    const int64_t right
        = attributes.right_bound().has_value() ? attributes.right_bound().value() : UNBOUNDED;
    if(left < UNBOUNDED || right < UNBOUNDED || !fitsI32(left) || !fitsI32(right))
    {
        return std::nullopt;
    }

    MaskParameters result;
    // The deprecated causal flags override any bounds and alignment, exactly as
    // the hipDNN reference resolves them (extractDiagonalBandParams).
    if(topLeftDeprecated || bottomRightDeprecated)
    {
        result.kind = MaskKind::CAUSAL;
        result.right = 0;
        result.bottomRight = static_cast<int32_t>(bottomRightDeprecated);
        return result;
    }
    result.bottomRight = static_cast<int32_t>(attributes.diagonal_alignment()
                                              == data_objects::DiagonalAlignment::BOTTOM_RIGHT);
    result.left = static_cast<int32_t>(left);
    result.right = static_cast<int32_t>(right);
    if(left != UNBOUNDED || right != UNBOUNDED)
    {
        result.kind = left == UNBOUNDED && right == 0 ? MaskKind::CAUSAL : MaskKind::WINDOW;
    }
    return result;
}

struct Problem
{
    int64_t batch = 0;
    int64_t queryHeads = 0;
    int64_t kvHeads = 0;
    int64_t seqLenQ = 0;
    int64_t seqLenK = 0;
    int64_t headSize = 0;
    data_objects::DataType dataType = data_objects::DataType::UNSET;
};

Problem problemFor(const data_objects::TensorAttributes& q, const data_objects::TensorAttributes& k)
{
    return {q.dims()->Get(BATCH_AXIS),
            q.dims()->Get(HEAD_AXIS),
            k.dims()->Get(HEAD_AXIS),
            q.dims()->Get(SEQ_AXIS),
            k.dims()->Get(SEQ_AXIS),
            q.dims()->Get(HEAD_SIZE_AXIS),
            q.data_type()};
}

struct Binding
{
    int64_t q = 0;
    int64_t k = 0;
    int64_t v = 0;
    int64_t o = 0;
    int64_t lse = 0;
    int32_t writeLse = 0;
    MaskParameters mask;
    float scale = 0.0F;
};

std::optional<BoundTokens> gfx1151WmmaAttentionGraphMatches(const MatchContext& context)
{
    const auto* attributesPtr = sdpaNode(context);
    if(attributesPtr == nullptr)
    {
        return std::nullopt;
    }
    const auto& attributes = *attributesPtr;
    const auto* q = findTensor(context, attributes.q_tensor_uid());
    const auto* k = findTensor(context, attributes.k_tensor_uid());
    const auto* v = findTensor(context, attributes.v_tensor_uid());
    const auto* o = findTensor(context, attributes.o_tensor_uid());
    if(q == nullptr || k == nullptr || v == nullptr || o == nullptr || !isWellFormedOperand(*q)
       || !isWellFormedOperand(*k) || !isWellFormedOperand(*v) || !isWellFormedOperand(*o))
    {
        return std::nullopt;
    }

    const auto p = problemFor(*q, *k);
    if(!fitsI32(p.batch) || !fitsI32(p.queryHeads) || !fitsI32(p.kvHeads) || !fitsI32(p.seqLenQ)
       || !fitsI32(p.seqLenK) || !fitsI32(p.headSize) || p.kvHeads <= 0
       || p.queryHeads % p.kvHeads != 0
       || (p.headSize != 64 && p.headSize != 96 && p.headSize != 128 && p.headSize != 256))
    {
        return std::nullopt;
    }
    if(k->dims()->Get(BATCH_AXIS) != p.batch || v->dims()->Get(BATCH_AXIS) != p.batch
       || o->dims()->Get(BATCH_AXIS) != p.batch || k->dims()->Get(SEQ_AXIS) != p.seqLenK
       || v->dims()->Get(SEQ_AXIS) != p.seqLenK || v->dims()->Get(HEAD_AXIS) != p.kvHeads
       || o->dims()->Get(HEAD_AXIS) != p.queryHeads || o->dims()->Get(SEQ_AXIS) != p.seqLenQ
       || k->dims()->Get(HEAD_SIZE_AXIS) != p.headSize
       || v->dims()->Get(HEAD_SIZE_AXIS) != p.headSize
       || o->dims()->Get(HEAD_SIZE_AXIS) != p.headSize)
    {
        return std::nullopt;
    }
    if(k->data_type() != p.dataType || v->data_type() != p.dataType || o->data_type() != p.dataType
       || !supportedDataTypeName(p.dataType).has_value())
    {
        return std::nullopt;
    }
    for(const auto* tensor : {q, k, v, o})
    {
        if(!addressFitsI32(*tensor, 2))
        {
            return std::nullopt;
        }
    }

    const bool writeLse = attributes.generate_stats().value_or(false);
    const data_objects::TensorAttributes* lse = o;
    int64_t lseUid = attributes.o_tensor_uid();
    if(writeLse)
    {
        if(!attributes.stats_tensor_uid().has_value())
        {
            return std::nullopt;
        }
        lseUid = attributes.stats_tensor_uid().value();
        lse = findTensor(context, lseUid);
        if(lse == nullptr || !isWellFormedOperand(*lse)
           || lse->data_type() != data_objects::DataType::FLOAT
           || lse->dims()->Get(BATCH_AXIS) != p.batch || lse->dims()->Get(HEAD_AXIS) != p.queryHeads
           || lse->dims()->Get(SEQ_AXIS) != p.seqLenQ || lse->dims()->Get(HEAD_SIZE_AXIS) != 1
           || !addressFitsI32(*lse, 4))
        {
            return std::nullopt;
        }
    }
    else if(attributes.stats_tensor_uid().has_value())
    {
        return std::nullopt;
    }

    const auto mask = maskFor(attributes);
    if(!mask.has_value())
    {
        return std::nullopt;
    }
    if(attributes.attn_mask_tensor_uid().has_value() || attributes.scale_tensor_uid().has_value()
       || attributes.seq_len_q_tensor_uid().has_value()
       || attributes.seq_len_kv_tensor_uid().has_value() || attributes.seed_tensor_uid().has_value()
       || attributes.offset_tensor_uid().has_value()
       || attributes.dropout_mask_tensor_uid().has_value()
       || attributes.dropout_scale_tensor_uid().has_value()
       || (attributes.dropout_probability().has_value()
           && attributes.dropout_probability().value() != 0.0F)
       || attributes.page_table_k_tensor_uid().has_value()
       || attributes.page_table_v_tensor_uid().has_value()
       || attributes.max_seq_len_kv().has_value() || attributes.block_mask_tensor_uid().has_value()
       || attributes.sink_token_tensor_uid().has_value()
       || attributes.descale_q_tensor_uid().has_value()
       || attributes.descale_k_tensor_uid().has_value()
       || attributes.descale_v_tensor_uid().has_value()
       || attributes.descale_s_tensor_uid().has_value()
       || attributes.scale_s_tensor_uid().has_value() || attributes.scale_o_tensor_uid().has_value()
       || attributes.amax_s_tensor_uid().has_value() || attributes.amax_o_tensor_uid().has_value()
       || attributes.max_tensor_uid().has_value() || attributes.sum_exp_tensor_uid().has_value()
       || attributes.rng_dump_tensor_uid().has_value() || attributes.alibi_mask()
       || attributes.padding_mask()
       || attributes.implementation() != data_objects::AttentionImplementation::AUTO
       || !attributes.attn_scale_value().has_value()
       || !std::isfinite(attributes.attn_scale_value().value()))
    {
        return std::nullopt;
    }
    // The kernel always accumulates in FP32. UNSET and FLOAT therefore describe
    // the same execution; HALF/BFLOAT16 are accepted for cuDNN-compat graphs.
    const auto mmaCoreMode = attributes.mma_core_mode();
    if(mmaCoreMode != data_objects::DataType::UNSET && mmaCoreMode != data_objects::DataType::FLOAT
       && mmaCoreMode != data_objects::DataType::HALF
       && mmaCoreMode != data_objects::DataType::BFLOAT16)
    {
        return std::nullopt;
    }

    BoundTokens bound;
    bound[std::string(Q_TOKEN)] = attributes.q_tensor_uid();
    bound[std::string(K_TOKEN)] = attributes.k_tensor_uid();
    bound[std::string(V_TOKEN)] = attributes.v_tensor_uid();
    bound[std::string(O_TOKEN)] = attributes.o_tensor_uid();
    bound[std::string(LSE_TOKEN)] = lseUid;
    bound[std::string(WRITE_LSE_TOKEN)] = static_cast<int64_t>(writeLse);
    bound[std::string(MASK_TOKEN)] = static_cast<int64_t>(mask->kind);
    bound[std::string(BOTTOM_RIGHT_TOKEN)] = mask->bottomRight;
    bound[std::string(WINDOW_LEFT_TOKEN)] = mask->left;
    bound[std::string(WINDOW_RIGHT_TOKEN)] = mask->right;
    int32_t scaleBits = 0;
    const float scale = attributes.attn_scale_value().value();
    std::memcpy(&scaleBits, &scale, sizeof(scale));
    bound[std::string(SCALE_BITS_TOKEN)] = static_cast<int64_t>(scaleBits);
    return bound;
}

int64_t readBound(const BoundTokens& bound, std::string_view token)
{
    const auto value = hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, token);
    if(!value.has_value())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "gfx1151 WMMA attention dispatch is missing bound token '" + std::string(token) + "'");
    }
    return *value;
}

Binding bindingFor(const BoundTokens& bound)
{
    Binding binding;
    binding.q = readBound(bound, Q_TOKEN);
    binding.k = readBound(bound, K_TOKEN);
    binding.v = readBound(bound, V_TOKEN);
    binding.o = readBound(bound, O_TOKEN);
    binding.lse = readBound(bound, LSE_TOKEN);
    binding.writeLse = static_cast<int32_t>(readBound(bound, WRITE_LSE_TOKEN));
    binding.mask.kind = static_cast<MaskKind>(readBound(bound, MASK_TOKEN));
    binding.mask.bottomRight = static_cast<int32_t>(readBound(bound, BOTTOM_RIGHT_TOKEN));
    binding.mask.left = static_cast<int32_t>(readBound(bound, WINDOW_LEFT_TOKEN));
    binding.mask.right = static_cast<int32_t>(readBound(bound, WINDOW_RIGHT_TOKEN));
    const auto scaleBits = static_cast<int32_t>(readBound(bound, SCALE_BITS_TOKEN));
    std::memcpy(&binding.scale, &scaleBits, sizeof(binding.scale));
    return binding;
}

std::optional<int64_t> integerMetadata(const KernelDefinition& kernel, std::string_view field)
{
    const auto it = kernel.metadata.find(std::string(field));
    if(it == kernel.metadata.end())
    {
        return std::nullopt;
    }
    const auto* value = std::get_if<int64_t>(&it->second);
    return value == nullptr ? std::nullopt : std::optional<int64_t>(*value);
}

std::optional<std::string> stringMetadata(const KernelDefinition& kernel, std::string_view field)
{
    const auto it = kernel.metadata.find(std::string(field));
    if(it == kernel.metadata.end())
    {
        return std::nullopt;
    }
    const auto* value = std::get_if<std::string>(&it->second);
    return value == nullptr ? std::nullopt : std::optional<std::string>(*value);
}

std::string maskName(MaskKind kind)
{
    switch(kind)
    {
    case MaskKind::NONE:
        return "none";
    case MaskKind::CAUSAL:
        return "causal";
    case MaskKind::WINDOW:
        return "window";
    default:
        return {};
    }
}

bool kernelMatches(const MatchContext& context,
                   const BoundTokens& bound,
                   const KernelDefinition& kernel)
{
    const auto* attributes = sdpaNode(context);
    if(attributes == nullptr)
    {
        return false;
    }
    const auto* q = findTensor(context, attributes->q_tensor_uid());
    const auto* k = findTensor(context, attributes->k_tensor_uid());
    if(q == nullptr || k == nullptr)
    {
        return false;
    }
    const auto p = problemFor(*q, *k);
    const auto dtype = supportedDataTypeName(p.dataType);
    const auto headSize = integerMetadata(kernel, HEAD_SIZE_FIELD);
    const auto maskMode = stringMetadata(kernel, MASK_MODE_FIELD);
    const auto transposed = integerMetadata(kernel, TRANSPOSED_QK_FIELD);
    const auto queryTail = integerMetadata(kernel, QUERY_TAIL_FIELD);
    const auto kvTail = integerMetadata(kernel, KV_TAIL_FIELD);
    const auto blockN = integerMetadata(kernel, BLOCK_N_FIELD);
    const auto numWaves = integerMetadata(kernel, NUM_WAVES_FIELD);
    if(!dtype.has_value() || !headSize.has_value() || !maskMode.has_value()
       || !transposed.has_value() || !queryTail.has_value() || !kvTail.has_value()
       || !blockN.has_value() || !numWaves.has_value() || *headSize != p.headSize
       || kernel.getStringMetadata(std::string(DTYPE_FIELD)) != *dtype)
    {
        return false;
    }
    if((*transposed != 0 && *transposed != 1) || (*queryTail != 0 && *queryTail != 1)
       || (*kvTail != 0 && *kvTail != 1))
    {
        return false;
    }
    const auto requestedMask = static_cast<MaskKind>(readBound(bound, MASK_TOKEN));
    if(*maskMode != maskName(requestedMask))
    {
        return false;
    }
    if(*transposed != 0)
    {
        const bool validGeometry
            = (*blockN == 32 && *numWaves == 1) || (*blockN == 64 && *numWaves == 2);
        if(requestedMask == MaskKind::WINDOW || (p.headSize != 64 && p.headSize != 128)
           || *queryTail != 0 || *kvTail != 0 || !validGeometry || p.seqLenQ % (16 * *numWaves) != 0
           || p.seqLenK % *blockN != 0)
        {
            return false;
        }
    }
    else if(*blockN != 32 || *numWaves != 1)
    {
        return false;
    }
    if(*queryTail == 0 && p.seqLenQ % 16 != 0)
    {
        return false;
    }
    if(*kvTail == 0 && p.seqLenK % 16 != 0)
    {
        return false;
    }
    return true;
}

double scoreKernel(const MatchContext& context,
                   const BoundTokens& /*bound*/,
                   const KernelDefinition& kernel)
{
    const auto* attributes = sdpaNode(context);
    const auto* q
        = attributes == nullptr ? nullptr : findTensor(context, attributes->q_tensor_uid());
    const auto* k
        = attributes == nullptr ? nullptr : findTensor(context, attributes->k_tensor_uid());
    const auto transposed = integerMetadata(kernel, TRANSPOSED_QK_FIELD);
    const auto blockN = integerMetadata(kernel, BLOCK_N_FIELD);
    const auto numWaves = integerMetadata(kernel, NUM_WAVES_FIELD);
    if(q == nullptr || k == nullptr || !transposed.has_value() || !blockN.has_value()
       || !numWaves.has_value())
    {
        return 0.0;
    }
    if(*transposed == 0)
    {
        return 10.0;
    }
    const int64_t sq = q->dims()->Get(SEQ_AXIS);
    const int64_t sk = k->dims()->Get(SEQ_AXIS);
    const bool wantWide = sq >= 512 && sq % 32 == 0 && sk % 64 == 0;
    const bool isWide = *blockN == 64 && *numWaves == 2;
    return wantWide == isWide ? 100.0 : 50.0;
}

std::vector<KernelArgument> kernelSignature()
{
    constexpr auto PTR = static_cast<uint32_t>(sizeof(void*));
    constexpr auto I32 = static_cast<uint32_t>(sizeof(int32_t));
    constexpr auto F32 = static_cast<uint32_t>(sizeof(float));
    std::vector<KernelArgument> signature;
    uint32_t offset = 0;
    const auto add = [&](const char* kind, uint32_t size, const char* name) {
        signature.emplace_back(KernelArgument{kind, size, offset, name});
        offset += size;
    };
    for(const char* name : {"Q", "K", "V", "O", "LSE"})
    {
        add("global_buffer", PTR, name);
    }
    add("by_value", F32, "scale_log2");
    for(const char* name :
        {"seqlen_q",         "seqlen_k",         "num_query_heads", "num_kv_heads",
         "bottom_right",     "window_left",      "window_right",    "write_lse",
         "stride_q_batch",   "stride_q_token",   "stride_q_head",   "stride_k_batch",
         "stride_k_token",   "stride_k_head",    "stride_v_batch",  "stride_v_token",
         "stride_v_head",    "stride_o_batch",   "stride_o_token",  "stride_o_head",
         "stride_lse_batch", "stride_lse_token", "stride_lse_head"})
    {
        add("by_value", I32, name);
    }
    return signature;
}

struct RuntimeArguments
{
    int32_t batch;
    int32_t queryHeads;
    int32_t kvHeads;
    int32_t seqLenQ;
    int32_t seqLenK;
    int32_t bottomRight;
    int32_t windowLeft;
    int32_t windowRight;
    int32_t writeLse;
    std::array<int32_t, 3> qStrides;
    std::array<int32_t, 3> kStrides;
    std::array<int32_t, 3> vStrides;
    std::array<int32_t, 3> oStrides;
    std::array<int32_t, 3> lseStrides;
};

class PreparedGfx1151WmmaAttention : public PreparedDispatch
{
public:
    PreparedGfx1151WmmaAttention(IngestorKernelCode code,
                                 Binding binding,
                                 RuntimeArguments arguments)
        : _code(std::move(code))
        , _binding(binding)
        , _arguments(arguments)
    {
    }

    compilation::IRunnableKernel& kernelForStream(hipStream_t stream) const
    {
        return _code.kernelForStream(stream);
    }

    const Binding& binding() const
    {
        return _binding;
    }

    const RuntimeArguments& arguments() const
    {
        return _arguments;
    }

private:
    IngestorKernelCode _code;
    Binding _binding;
    RuntimeArguments _arguments;
};

RuntimeArguments runtimeArguments(const data_objects::TensorAttributes& q,
                                  const data_objects::TensorAttributes& k,
                                  const data_objects::TensorAttributes& v,
                                  const data_objects::TensorAttributes& o,
                                  const data_objects::TensorAttributes& lse,
                                  const Problem& problem,
                                  const Binding& binding)
{
    const auto strides = [](const data_objects::TensorAttributes& tensor) {
        return std::array<int32_t, 3>{static_cast<int32_t>(tensor.strides()->Get(BATCH_AXIS)),
                                      static_cast<int32_t>(tensor.strides()->Get(SEQ_AXIS)),
                                      static_cast<int32_t>(tensor.strides()->Get(HEAD_AXIS))};
    };
    RuntimeArguments args;
    args.batch = static_cast<int32_t>(problem.batch);
    args.queryHeads = static_cast<int32_t>(problem.queryHeads);
    args.kvHeads = static_cast<int32_t>(problem.kvHeads);
    args.seqLenQ = static_cast<int32_t>(problem.seqLenQ);
    args.seqLenK = static_cast<int32_t>(problem.seqLenK);
    args.bottomRight = binding.mask.bottomRight;
    args.windowLeft = binding.mask.left;
    args.windowRight = binding.mask.right;
    args.writeLse = binding.writeLse;
    args.qStrides = strides(q);
    args.kStrides = strides(k);
    args.vStrides = strides(v);
    args.oStrides = strides(o);
    args.lseStrides = binding.writeLse != 0 ? strides(lse) : std::array<int32_t, 3>{0, 0, 0};
    return args;
}

class Gfx1151WmmaAttentionDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    explicit Gfx1151WmmaAttentionDispatchHandler(const compilation::KpackKernelLoader& kpackLoader)
        : _kpackLoader(kpackLoader)
    {
    }

    size_t workspaceBytes(const MatchContext& /*context*/,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& /*kernel*/) const override
    {
        return 0;
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& bound,
                                              const KernelDefinition& kernel) const override
    {
        const auto binding = bindingFor(bound);
        const auto* q = findTensor(context, binding.q);
        const auto* k = findTensor(context, binding.k);
        const auto* v = findTensor(context, binding.v);
        const auto* o = findTensor(context, binding.o);
        const auto* lse = findTensor(context, binding.lse);
        if(q == nullptr || k == nullptr || v == nullptr || o == nullptr || lse == nullptr)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "gfx1151 WMMA attention lost a bound tensor before preparation");
        }
        auto code = buildIngestorKernelCode(_kpackLoader, context, kernel, kernelSignature());
        const auto problem = problemFor(*q, *k);
        const auto transposed = kernel.getIntMetadata(std::string(TRANSPOSED_QK_FIELD));
        const auto numWaves = kernel.getIntMetadata(std::string(NUM_WAVES_FIELD));
        const int64_t qRows = 16 * (transposed != 0 ? numWaves : 1);
        const int64_t gridX = (problem.seqLenQ + qRows - 1) / qRows;
        const int64_t gridZ = problem.batch;
        if(gridX <= 0 || gridX > std::numeric_limits<uint32_t>::max() || problem.queryHeads <= 0
           || problem.queryHeads > std::numeric_limits<uint32_t>::max() || gridZ <= 0
           || gridZ > std::numeric_limits<uint32_t>::max())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "gfx1151 WMMA attention launch geometry does not fit uint32");
        }
        code.setBlockSize(static_cast<uint32_t>(32 * (transposed != 0 ? numWaves : 1)), 1, 1);
        code.setGridSize(static_cast<uint32_t>(gridX),
                         static_cast<uint32_t>(problem.queryHeads),
                         static_cast<uint32_t>(gridZ));
        return std::make_unique<PreparedGfx1151WmmaAttention>(
            std::move(code), binding, runtimeArguments(*q, *k, *v, *o, *lse, problem, binding));
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* /*workspace*/) const override
    {
        const auto& state = dynamic_cast<const PreparedGfx1151WmmaAttention&>(prepared);
        const auto& binding = state.binding();
        const auto& p = state.arguments();
        const auto q
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.q, deviceBuffers, numDeviceBuffers);
        const auto k
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.k, deviceBuffers, numDeviceBuffers);
        const auto v
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.v, deviceBuffers, numDeviceBuffers);
        const auto o
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.o, deviceBuffers, numDeviceBuffers);
        const auto lse
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.lse, deviceBuffers, numDeviceBuffers);
        const float scaleLog2 = binding.scale * 1.4426950408889634F;
        state.kernelForStream(handle.getStream())
            .launch(handle.getStream(),
                    q.ptr,
                    k.ptr,
                    v.ptr,
                    o.ptr,
                    lse.ptr,
                    scaleLog2,
                    p.seqLenQ,
                    p.seqLenK,
                    p.queryHeads,
                    p.kvHeads,
                    p.bottomRight,
                    p.windowLeft,
                    p.windowRight,
                    p.writeLse,
                    p.qStrides[0],
                    p.qStrides[1],
                    p.qStrides[2],
                    p.kStrides[0],
                    p.kStrides[1],
                    p.kStrides[2],
                    p.vStrides[0],
                    p.vStrides[1],
                    p.vStrides[2],
                    p.oStrides[0],
                    p.oStrides[1],
                    p.oStrides[2],
                    p.lseStrides[0],
                    p.lseStrides[1],
                    p.lseStrides[2]);
    }

private:
    const compilation::KpackKernelLoader& _kpackLoader;
};

compilation::KpackModuleCache& gfx1151WmmaAttentionKpackModuleCache()
{
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

} // namespace

void resetGfx1151WmmaAttentionModuleCache()
{
    gfx1151WmmaAttentionKpackModuleCache().clear();
}

namespace
{

const Gfx1151WmmaAttentionDispatchHandler& gfx1151WmmaAttentionDispatchHandler()
{
    static const compilation::KpackKernelLoader s_loader(gfx1151WmmaAttentionKpackModuleCache());
    static const Gfx1151WmmaAttentionDispatchHandler s_handler(s_loader);
    return s_handler;
}

} // namespace

void registerGfx1151WmmaAttentionSymbols(SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &gfx1151WmmaAttentionGraphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &kernelMatches);
    scope.add(std::string(SCORE_SYMBOL), &scoreKernel);
    scope.add(std::string(DISPATCH_SYMBOL), &gfx1151WmmaAttentionDispatchHandler());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
