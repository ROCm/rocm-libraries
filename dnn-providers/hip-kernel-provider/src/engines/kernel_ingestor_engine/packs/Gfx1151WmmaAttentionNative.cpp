// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <hip/hip_runtime_api.h>

#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/PluginDeviceBuffers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/RuntimePassByValue.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/DeviceProperties.hpp>
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
#include "engines/kernel_ingestor_engine/LaunchKernelDefinition.hpp"

/**
 * @file Gfx1151WmmaAttentionNative.cpp
 * @brief hipkernel:Gfx1151WmmaAttention and hipkernel:Gfx1151WmmaAttentionDecode: the rocKE
 *        RDNA (gfx11 / gfx12) SDPA forward, as two engines over one shared matcher core.
 *
 * The catalog (IngestorGenerator configs/gfx1151_wmma_attention.yaml and
 * configs/gfx1151_wmma_attention_decode.yaml) ships two kernels, one per engine, because
 * one engine has one launch set (RFC 0017 section 15.1):
 *
 * - "prefill", engine hipkernel:Gfx1151WmmaAttention: rocKE
 *   kernels/gfx1151/wmma_fmha_fwd_v2.py, one multi-wave object per dtype x (Q/K, V/O)
 *   head-width bucket x bias kind, all of the "causal" class whose runtime bounds also
 *   express no mask and sliding windows. Flat 1-D grid; one launch, read from
 *   `kernel_source`.
 * - "decode", engine hipkernel:Gfx1151WmmaAttentionDecode: rocKE
 *   kernels/gfx1151/wmma_fmha_decode.py, the split-KV GQA-packed decode, one object per
 *   dtype x bucket x bias kind. Its kernels carry two launches in a `sources` map: "decode"
 *   and "merge", the kernel merging the splits, which share the object's code object
 *   (LaunchKernelDefinition.hpp).
 *
 * Which of the two serves a graph, the decode object's bucket and the decode split count
 * mirror the rocKE dispatcher (library/dispatch/attention/gfx1151.py
 * rdna_attention_use_decode and decode_launch_plan, kernels/gfx1151/wmma_fmha_decode.py
 * host helpers). The decode engine serves exactly the problems the dispatcher sends to
 * decode; the prefill engine declines those, so hipDNN's choice between the two engines is
 * never the ranking's. The prefill object is not the one the Python route builds: that
 * route JIT-builds the exact (Q/K, V/O) bucket pair, and a mask "none" object for an
 * unmasked request, while hipDNN runs the shipped catalog object of the tightest bucket
 * pair that holds the widths, of the "causal" class whose runtime bounds express every
 * mask. The decode launch differs in one way: more than one split always merges in the
 * merge launch, never in the kernel's fused merge, whose WS_CNT counters hipDNN's
 * stateless workspace would need cleared every launch; the workspace therefore holds only
 * the split partials.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

namespace
{

constexpr std::string_view PREFILL_GRAPH_MATCHER_SYMBOL
    = "hipkernel.gfx1151_wmma_attention.graph_match";
constexpr std::string_view PREFILL_KERNEL_MATCHER_SYMBOL
    = "hipkernel.gfx1151_wmma_attention.kernel_match";
constexpr std::string_view PREFILL_SCORE_SYMBOL = "hipkernel.gfx1151_wmma_attention.score";
constexpr std::string_view PREFILL_DISPATCH_SYMBOL = "hipkernel.gfx1151_wmma_attention.dispatch";
constexpr std::string_view DECODE_GRAPH_MATCHER_SYMBOL
    = "hipkernel.gfx1151_wmma_attention_decode.graph_match";
constexpr std::string_view DECODE_KERNEL_MATCHER_SYMBOL
    = "hipkernel.gfx1151_wmma_attention_decode.kernel_match";
constexpr std::string_view DECODE_SCORE_SYMBOL = "hipkernel.gfx1151_wmma_attention_decode.score";
constexpr std::string_view DECODE_DISPATCH_SYMBOL
    = "hipkernel.gfx1151_wmma_attention_decode.dispatch";

constexpr std::string_view DTYPE_FIELD = "dtype";
constexpr std::string_view KIND_FIELD = "kind";
constexpr std::string_view HEAD_SIZE_FIELD = "head_size";
constexpr std::string_view HEAD_SIZE_V_FIELD = "head_size_v";
constexpr std::string_view MASK_MODE_FIELD = "mask_mode";
constexpr std::string_view BIAS_FIELD = "bias";
constexpr std::string_view BLOCK_SIZE_FIELD = "block_size";
constexpr std::string_view BLOCK_M_FIELD = "block_m";
constexpr std::string_view DV_SPLIT_FIELD = "dv_split";
// The decode engine's launch names: the `sources` keys of each of its kernels.
constexpr const char* DECODE_LAUNCH = "decode";
constexpr const char* MERGE_LAUNCH = "merge";

constexpr std::string_view KIND_PREFILL = "prefill";
constexpr std::string_view KIND_DECODE = "decode";

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
// Present only for a runtime pass-by-value scale tensor read at launch.
constexpr std::string_view SCALE_UID_TOKEN = "gfx1151_wmma_attention.scale.uid";
constexpr std::string_view SCALE_DTYPE_TOKEN = "gfx1151_wmma_attention.scale.dtype";
constexpr std::string_view BIAS_TOKEN = "gfx1151_wmma_attention.bias";
// Present only when the graph carries an additive bias (attn_mask).
constexpr std::string_view BIAS_UID_TOKEN = "gfx1151_wmma_attention.bias.uid";
constexpr std::string_view BIAS_STRIDE_BATCH_TOKEN = "gfx1151_wmma_attention.bias.stride_batch";
constexpr std::string_view BIAS_STRIDE_HEAD_TOKEN = "gfx1151_wmma_attention.bias.stride_head";
constexpr std::string_view BIAS_STRIDE_QUERY_TOKEN = "gfx1151_wmma_attention.bias.stride_query";

constexpr uint32_t BATCH_AXIS = 0;
constexpr uint32_t HEAD_AXIS = 1;
constexpr uint32_t SEQ_AXIS = 2;
constexpr uint32_t HEAD_SIZE_AXIS = 3;
constexpr uint32_t SDPA_RANK = 4;
constexpr int64_t UNBOUNDED = -1;
// Both kernels take runtime head widths in multiples of 8 up to 256 and load whole
// 16-bit rows as 16-byte vectors, so every outer stride is a multiple of 8 elements.
constexpr int64_t HEAD_DIM_GRANULE = 8;
constexpr int64_t MAX_HEAD_DIM = 256;
constexpr int64_t ROW_ALIGN_ELEMENTS = 8;
constexpr uintptr_t ROW_ALIGN_BYTES = 16;
constexpr int64_t I32_MAX = std::numeric_limits<int32_t>::max();

// ---- the rocKE decode host helpers (kernels/gfx1151/wmma_fmha_decode.py), mirrored ----

constexpr int64_t DECODE_ROWS = 16; // packed query rows per WMMA tile
constexpr int64_t DECODE_SPLIT_KEYS = 32; // key granularity of the split heuristic
constexpr int64_t DECODE_MAX_SPLITS = 64;
constexpr int64_t DECODE_WG_PER_CU = 2;
constexpr int64_t DECODE_BLOCK_SIZE = 128; // threads per decode workgroup, every shipped object
constexpr int64_t DECODE_MIN_TILES_PER_WAVE = 4;
constexpr uint32_t MERGE_THREADS = 64;
// The rocKE dispatcher's compute-unit count when the device's is unknown (gfx1151.py
// _DEFAULT_NUM_CUS); HIP reports RDNA multiprocessors as WGPs of two CUs each.
constexpr int64_t DEFAULT_NUM_CUS = 40;

/// wmma_fmha_decode_bucket: the smallest of 64/128/256 holding both widths whose
/// smallest served Q/K width (16/72/136; narrower K chunks load unmasked) is at most
/// head_dim_q.
std::optional<int64_t> decodeBucket(int64_t headDimQ, int64_t headDimV)
{
    if(headDimQ <= 0 || headDimV <= 0 || headDimQ % 8 != 0 || headDimV % 8 != 0)
    {
        return std::nullopt;
    }
    constexpr std::array<std::pair<int64_t, int64_t>, 3> BUCKETS{{{64, 16}, {128, 72}, {256, 136}}};
    for(const auto& [bucket, minimum] : BUCKETS)
    {
        if(std::max(headDimQ, headDimV) <= bucket)
        {
            return headDimQ >= minimum ? std::optional<int64_t>(bucket) : std::nullopt;
        }
    }
    return std::nullopt;
}

/// wmma_fmha_decode_kv_range: the keys any query row can see, [lo, hi).
std::pair<int64_t, int64_t> decodeKvRange(
    int64_t seqLenQ, int64_t seqLenK, bool bottomRight, int64_t windowLeft, int64_t windowRight)
{
    const int64_t ctx = bottomRight ? seqLenK - seqLenQ : 0;
    const int64_t lo = windowLeft >= 0 ? std::max<int64_t>(ctx - windowLeft, 0) : 0;
    const int64_t hi
        = windowRight >= 0
              ? std::max<int64_t>(std::min<int64_t>(ctx + windowRight + seqLenQ, seqLenK), 0)
              : seqLenK;
    return {lo, hi};
}

/// The rocKE dispatcher's prefill-vs-decode rule (gfx1151.py rdna_attention_use_decode),
/// a pure function of the problem. The decode kernel packs the g = Hq / Hkv query heads
/// of a K/V head and their positions into 16-row tiles; it wins while g * Sq fills few
/// tiles. Past 64 query rows the prefill wins unless its 128-row CTAs would leave most
/// of the device idle. @p visibleKeys is the key range any query row sees (hi - lo of
/// decodeKvRange); @p numCus counts CUs, as the dispatcher's request does (twice HIP's
/// RDNA multiProcessorCount).
bool rdnaAttentionUseDecode(int64_t seqLenQ,
                            int64_t visibleKeys,
                            int64_t queryHeads,
                            int64_t kvHeads,
                            int64_t batch,
                            bool gfx12,
                            int64_t numCus)
{
    constexpr int64_t PREFILL_BLOCK_M = 128;
    const int64_t groups = std::max<int64_t>(1, queryHeads / std::max<int64_t>(kvHeads, 1));
    if(seqLenQ > 64)
    {
        const int64_t prefillCtas
            = batch * queryHeads * ((seqLenQ + PREFILL_BLOCK_M - 1) / PREFILL_BLOCK_M);
        return 4 * prefillCtas < numCus;
    }
    if(gfx12)
    {
        return seqLenQ <= 32 || groups == 1 || visibleKeys >= 2048;
    }
    return groups * seqLenQ < 512 || visibleKeys < 64 * numCus
           || batch * visibleKeys < 512 * numCus;
}

bool isGfx12(const DeviceProperties& device)
{
    return device.gcnArchName.rfind("gfx12", 0) == 0;
}

/// HIP's multiProcessorCount (RDNA WGPs), or the dispatcher's default when unknown.
int64_t deviceMultiprocessors(const DeviceProperties& device)
{
    return device.multiProcessorCount > 0 ? device.multiProcessorCount : DEFAULT_NUM_CUS / 2;
}

const data_objects::TensorAttributes* findTensor(const MatchContext& context, int64_t uid)
{
    const auto& tensors = context.graph.getTensorMap();
    const auto it = tensors.find(uid);
    return it == tensors.end() ? nullptr : it->second;
}

const data_objects::SdpaAttributes* sdpaNode(const MatchContext& context)
{
    // The engine declares graph schema 1.2.0 for runtime pass-by-value scales,
    // which also admits override-shape graphs (1.1.0). Launch arguments are
    // bound from the graph's shapes at prepare, so those are declined.
    if(context.graph.nodeCount() != 1 || context.graph.getGraph().is_override_shape_enabled())
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
    return value >= std::numeric_limits<int32_t>::min() && value <= I32_MAX;
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

/// Q/K/V/O rows load as 16-byte vectors: every stride of an extent above 1 other
/// than the unit head-dim stride is a multiple of 8 16-bit elements.
bool hasAlignedRows(const data_objects::TensorAttributes& tensor)
{
    for(const uint32_t axis : {BATCH_AXIS, HEAD_AXIS, SEQ_AXIS})
    {
        if(tensor.dims()->Get(axis) > 1 && tensor.strides()->Get(axis) % ROW_ALIGN_ELEMENTS != 0)
        {
            return false;
        }
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
    int64_t vHeads = 0;
    int64_t seqLenQ = 0;
    int64_t seqLenK = 0;
    int64_t headDimQ = 0;
    int64_t headDimV = 0;
    data_objects::DataType dataType = data_objects::DataType::UNSET;
};

Problem problemFor(const data_objects::TensorAttributes& q,
                   const data_objects::TensorAttributes& k,
                   const data_objects::TensorAttributes& v)
{
    return {q.dims()->Get(BATCH_AXIS),
            q.dims()->Get(HEAD_AXIS),
            k.dims()->Get(HEAD_AXIS),
            v.dims()->Get(HEAD_AXIS),
            q.dims()->Get(SEQ_AXIS),
            k.dims()->Get(SEQ_AXIS),
            q.dims()->Get(HEAD_SIZE_AXIS),
            v.dims()->Get(HEAD_SIZE_AXIS),
            q.data_type()};
}

/// The problem of the graph @p context holds; nullopt when it is not one SDPA node.
std::optional<Problem> problemFor(const MatchContext& context)
{
    const auto* attributes = sdpaNode(context);
    if(attributes == nullptr)
    {
        return std::nullopt;
    }
    const auto* q = findTensor(context, attributes->q_tensor_uid());
    const auto* k = findTensor(context, attributes->k_tensor_uid());
    const auto* v = findTensor(context, attributes->v_tensor_uid());
    if(q == nullptr || k == nullptr || v == nullptr || q->dims() == nullptr || k->dims() == nullptr
       || v->dims() == nullptr || q->dims()->size() != SDPA_RANK || k->dims()->size() != SDPA_RANK
       || v->dims()->size() != SDPA_RANK)
    {
        return std::nullopt;
    }
    return problemFor(*q, *k, *v);
}

bool isServedHeadDim(int64_t headDim)
{
    return headDim >= HEAD_DIM_GRANULE && headDim <= MAX_HEAD_DIM
           && headDim % HEAD_DIM_GRANULE == 0;
}

// The additive bias element type: FP32, or the Q dtype.
enum class BiasKind : int64_t
{
    NONE = 0,
    F32 = 1,
    Q = 2
};

struct BiasParameters
{
    BiasKind kind = BiasKind::NONE;
    int64_t uid = 0;
    // Element strides of the batch, query-head and query-row axes; 0 broadcasts.
    std::array<int32_t, 3> strides{0, 0, 0};
};

// hipDNN's attn_mask is an additive bias on the scaled scores. The kernels read
// it as [B|1, Hq|1, Sq|1, Sk] with a unit-stride key axis, indexed by the query
// head, and take one runtime stride per leading axis, where an extent-1 axis
// broadcasts through stride 0. Masks of rank below 4, a broadcast key axis and
// element types other than FP32 or the Q dtype are declined.
std::optional<BiasParameters> biasFor(const MatchContext& context,
                                      const data_objects::SdpaAttributes& attributes,
                                      const Problem& problem)
{
    BiasParameters result;
    const auto uid = attributes.attn_mask_tensor_uid();
    if(!uid.has_value())
    {
        return result;
    }
    const auto* tensor = findTensor(context, *uid);
    if(tensor == nullptr || tensor->virtual_() || tensor->ragged_offset_tensor_uid().has_value()
       || hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(tensor))
    {
        return std::nullopt;
    }
    const auto* dims = tensor->dims();
    const auto* strides = tensor->strides();
    if(dims == nullptr || strides == nullptr || dims->size() != SDPA_RANK
       || strides->size() != SDPA_RANK || dims->Get(HEAD_SIZE_AXIS) != problem.seqLenK
       || (problem.seqLenK > 1 && strides->Get(HEAD_SIZE_AXIS) != 1))
    {
        return std::nullopt;
    }
    const std::array<int64_t, 3> extents{problem.batch, problem.queryHeads, problem.seqLenQ};
    for(uint32_t axis = 0; axis < extents.size(); ++axis)
    {
        const int64_t extent = dims->Get(axis);
        const int64_t stride = strides->Get(axis);
        if((extent != 1 && extent != extents[axis]) || stride < 0 || !fitsI32(stride))
        {
            return std::nullopt;
        }
        result.strides[axis] = extent == 1 ? 0 : static_cast<int32_t>(stride);
    }

    int64_t elementBytes = 0;
    if(tensor->data_type() == data_objects::DataType::FLOAT)
    {
        result.kind = BiasKind::F32;
        elementBytes = 4;
    }
    else if(tensor->data_type() == problem.dataType)
    {
        result.kind = BiasKind::Q;
        elementBytes = 2;
    }
    else
    {
        return std::nullopt;
    }
    if(!addressFitsI32(*tensor, elementBytes))
    {
        return std::nullopt;
    }
    result.uid = *uid;
    return result;
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
    // Set for a runtime pass-by-value scale tensor; `scale` is then unused.
    std::optional<hipdnn_plugin_sdk::ScalarOperand> runtimeScale;
    BiasParameters bias;
};

bool isFloatScalarType(data_objects::DataType type)
{
    return type == data_objects::DataType::FLOAT || type == data_objects::DataType::HALF
           || type == data_objects::DataType::BFLOAT16 || type == data_objects::DataType::DOUBLE;
}

struct ScaleSource
{
    float value = 1.0F;
    std::optional<int64_t> runtimeUid;
    data_objects::DataType runtimeType = data_objects::DataType::UNSET;
};

// hipDNN scale semantics: attn_scale_value, or a pass-by-value scale tensor
// (a compile-time constant, or a runtime host scalar read at launch), or
// neither, which means 1.0 (not 1/sqrt(d)). Device-resident scale tensors and
// constants that are not finite and positive are declined (the rocKE kernels
// take log2(e) * scale and require it > 0, as the Python route does); a runtime
// scale is checked the same way at launch.
std::optional<ScaleSource> scaleFor(const MatchContext& context,
                                    const data_objects::SdpaAttributes& attributes)
{
    ScaleSource result;
    const auto scaleUid = attributes.scale_tensor_uid();
    if(attributes.attn_scale_value().has_value())
    {
        if(scaleUid.has_value())
        {
            return std::nullopt;
        }
        result.value = attributes.attn_scale_value().value();
    }
    else if(scaleUid.has_value())
    {
        const auto* tensor = findTensor(context, *scaleUid);
        if(tensor == nullptr || !hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(tensor)
           || !isFloatScalarType(tensor->data_type()))
        {
            return std::nullopt;
        }
        hipdnn_plugin_sdk::ScalarOperand operand;
        try
        {
            operand = hipdnn_plugin_sdk::makeScalarOperand(
                context.graph.getTensorMap(), *scaleUid, "attn_scale");
        }
        catch(const hipdnn_plugin_sdk::HipdnnPluginException&)
        {
            return std::nullopt;
        }
        if(operand.isRuntimeUserSupplied)
        {
            result.runtimeUid = scaleUid;
            result.runtimeType = tensor->data_type();
            return result;
        }
        result.value = static_cast<float>(hipdnn_plugin_sdk::toDouble(operand.bakedDefault));
    }
    if(!std::isfinite(result.value) || !(result.value > 0.0F))
    {
        return std::nullopt;
    }
    return result;
}

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

    // Q/K and V/O head widths are independent multiples of 8, and K and V may each
    // carry any head count that divides the query heads.
    const auto p = problemFor(*q, *k, *v);
    if(!fitsI32(p.batch) || !fitsI32(p.queryHeads) || !fitsI32(p.kvHeads) || !fitsI32(p.vHeads)
       || !fitsI32(p.seqLenQ) || !fitsI32(p.seqLenK) || p.queryHeads % p.kvHeads != 0
       || p.queryHeads % p.vHeads != 0 || !isServedHeadDim(p.headDimQ)
       || !isServedHeadDim(p.headDimV))
    {
        return std::nullopt;
    }
    if(k->dims()->Get(BATCH_AXIS) != p.batch || v->dims()->Get(BATCH_AXIS) != p.batch
       || o->dims()->Get(BATCH_AXIS) != p.batch || v->dims()->Get(SEQ_AXIS) != p.seqLenK
       || o->dims()->Get(HEAD_AXIS) != p.queryHeads || o->dims()->Get(SEQ_AXIS) != p.seqLenQ
       || k->dims()->Get(HEAD_SIZE_AXIS) != p.headDimQ
       || o->dims()->Get(HEAD_SIZE_AXIS) != p.headDimV)
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
        if(!addressFitsI32(*tensor, 2) || !hasAlignedRows(*tensor))
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
    // hipDNN graph features the rocKE kernels serve but this engine cannot bind (the
    // kernels' sinks, ALiBi slopes, paged and ragged layouts and FP8 K/V take operands
    // of their own; see the catalog config) are declined, as is everything else.
    if(attributes.seq_len_q_tensor_uid().has_value()
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
       || attributes.implementation() != data_objects::AttentionImplementation::AUTO)
    {
        return std::nullopt;
    }
    const auto scale = scaleFor(context, attributes);
    if(!scale.has_value())
    {
        return std::nullopt;
    }
    // The kernels always accumulate in FP32. UNSET and FLOAT therefore describe
    // the same execution; HALF/BFLOAT16 are accepted for cuDNN-compat graphs.
    const auto mmaCoreMode = attributes.mma_core_mode();
    if(mmaCoreMode != data_objects::DataType::UNSET && mmaCoreMode != data_objects::DataType::FLOAT
       && mmaCoreMode != data_objects::DataType::HALF
       && mmaCoreMode != data_objects::DataType::BFLOAT16)
    {
        return std::nullopt;
    }
    const auto bias = biasFor(context, attributes, p);
    if(!bias.has_value())
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
    std::memcpy(&scaleBits, &scale->value, sizeof(scale->value));
    bound[std::string(SCALE_BITS_TOKEN)] = static_cast<int64_t>(scaleBits);
    if(scale->runtimeUid.has_value())
    {
        bound[std::string(SCALE_UID_TOKEN)] = *scale->runtimeUid;
        bound[std::string(SCALE_DTYPE_TOKEN)] = static_cast<int64_t>(scale->runtimeType);
    }
    bound[std::string(BIAS_TOKEN)] = static_cast<int64_t>(bias->kind);
    if(bias->kind != BiasKind::NONE)
    {
        bound[std::string(BIAS_UID_TOKEN)] = bias->uid;
        bound[std::string(BIAS_STRIDE_BATCH_TOKEN)] = bias->strides[0];
        bound[std::string(BIAS_STRIDE_HEAD_TOKEN)] = bias->strides[1];
        bound[std::string(BIAS_STRIDE_QUERY_TOKEN)] = bias->strides[2];
    }
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

MaskParameters boundMask(const BoundTokens& bound)
{
    MaskParameters mask;
    mask.kind = static_cast<MaskKind>(readBound(bound, MASK_TOKEN));
    mask.bottomRight = static_cast<int32_t>(readBound(bound, BOTTOM_RIGHT_TOKEN));
    mask.left = static_cast<int32_t>(readBound(bound, WINDOW_LEFT_TOKEN));
    mask.right = static_cast<int32_t>(readBound(bound, WINDOW_RIGHT_TOKEN));
    return mask;
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
    binding.mask = boundMask(bound);
    const auto scaleBits = static_cast<int32_t>(readBound(bound, SCALE_BITS_TOKEN));
    std::memcpy(&binding.scale, &scaleBits, sizeof(binding.scale));
    const auto scaleUid = hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, SCALE_UID_TOKEN);
    if(scaleUid.has_value())
    {
        binding.runtimeScale = hipdnn_plugin_sdk::ScalarOperand{
            *scaleUid,
            static_cast<data_objects::DataType>(readBound(bound, SCALE_DTYPE_TOKEN)),
            true,
            0.0};
    }
    binding.bias.kind = static_cast<BiasKind>(readBound(bound, BIAS_TOKEN));
    if(binding.bias.kind != BiasKind::NONE)
    {
        binding.bias.uid = readBound(bound, BIAS_UID_TOKEN);
        binding.bias.strides = {static_cast<int32_t>(readBound(bound, BIAS_STRIDE_BATCH_TOKEN)),
                                static_cast<int32_t>(readBound(bound, BIAS_STRIDE_HEAD_TOKEN)),
                                static_cast<int32_t>(readBound(bound, BIAS_STRIDE_QUERY_TOKEN))};
    }
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

/// What the catalog metadata says one object is. Every field is required.
struct ObjectTraits
{
    bool decode = false;
    std::string dtype;
    int64_t headSize = 0;
    int64_t headSizeV = 0;
    std::string maskMode;
    BiasKind bias = BiasKind::NONE;
    int64_t blockSize = 0;
    int64_t blockM = 0;
    int64_t dvSplit = 0;
};

std::optional<BiasKind> biasKindNamed(std::string_view name)
{
    if(name == "none")
    {
        return BiasKind::NONE;
    }
    if(name == "f32")
    {
        return BiasKind::F32;
    }
    if(name == "q")
    {
        return BiasKind::Q;
    }
    return std::nullopt;
}

std::optional<ObjectTraits> traitsOf(const KernelDefinition& kernel)
{
    const auto kind = stringMetadata(kernel, KIND_FIELD);
    const auto dtype = stringMetadata(kernel, DTYPE_FIELD);
    const auto headSize = integerMetadata(kernel, HEAD_SIZE_FIELD);
    const auto headSizeV = integerMetadata(kernel, HEAD_SIZE_V_FIELD);
    const auto maskMode = stringMetadata(kernel, MASK_MODE_FIELD);
    const auto biasName = stringMetadata(kernel, BIAS_FIELD);
    const auto blockSize = integerMetadata(kernel, BLOCK_SIZE_FIELD);
    const auto blockM = integerMetadata(kernel, BLOCK_M_FIELD);
    const auto dvSplit = integerMetadata(kernel, DV_SPLIT_FIELD);
    if(!kind.has_value() || (*kind != KIND_PREFILL && *kind != KIND_DECODE) || !dtype.has_value()
       || !headSize.has_value() || !headSizeV.has_value() || !maskMode.has_value()
       || !biasName.has_value() || !blockSize.has_value() || !blockM.has_value()
       || !dvSplit.has_value())
    {
        return std::nullopt;
    }
    const auto bias = biasKindNamed(*biasName);
    if(!bias.has_value() || *headSize <= 0 || *headSize > MAX_HEAD_DIM || *headSizeV <= 0
       || *headSizeV > MAX_HEAD_DIM || *blockSize <= 0 || *blockSize > 1024 || *blockM <= 0
       || *dvSplit <= 0 || *headSizeV % *dvSplit != 0)
    {
        return std::nullopt;
    }
    return ObjectTraits{*kind == KIND_DECODE,
                        *dtype,
                        *headSize,
                        *headSizeV,
                        *maskMode,
                        *bias,
                        *blockSize,
                        *blockM,
                        *dvSplit};
}

ObjectTraits requireTraits(const KernelDefinition& kernel)
{
    auto traits = traitsOf(kernel);
    if(!traits.has_value())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
            "gfx1151 WMMA attention " + describeDescriptor("kernel", kernel.name, kernel.kernelId)
                + " carries incomplete or unknown catalog metadata");
    }
    return *traits;
}

/// A "causal" prefill object applies runtime diagonal and window bounds, which also
/// express no mask (-1/-1), so it serves every request; a "none" object only unmasked
/// ones. The decode kernel's masks are all runtime. No shipped object is "none" today;
/// the branch (and the score's no-mask bonus) stay because the catalog generator's
/// mask_mode metadata (yaml catalog_mask) still permits one, and without them such an
/// object would serve masked graphs.
bool maskModeServes(std::string_view maskMode, MaskKind requested)
{
    if(maskMode == "causal" || maskMode == "runtime")
    {
        return true;
    }
    return maskMode == "none" && requested == MaskKind::NONE;
}

/// One decode launch, decode_launch_plan of the rocKE dispatcher, except that more than
/// one split always merges in the merge companion: the kernel's fused merge needs its
/// WS_CNT arrival counters zero on entry, which hipDNN's stateless workspace would cost a
/// memset dispatch per launch (slower than the merge kernel) to guarantee.
struct DecodePlan
{
    int64_t numSplits = 1;
    int64_t grid = 0;
    int64_t mergeGrid = 0;
    // wmma_fmha_decode_workspace byte offsets without the counter region (WS_CNT is
    // passed null); all zero with one split.
    int64_t wsLse = 0;
    int64_t wsBytes = 0;
};

/// The decode launch of @p problem on @p device, or nullopt when the decode kernel cannot
/// take it: head counts that do not pack (one of Hkv / Hv must divide the other) or a
/// grid / workspace past the kernels' I32 arithmetic.
std::optional<DecodePlan> decodePlanFor(const Problem& problem,
                                        const MaskParameters& mask,
                                        const ObjectTraits& traits,
                                        const DeviceProperties& device)
{
    const int64_t hq = problem.queryHeads;
    const int64_t hk = problem.kvHeads;
    const int64_t hv = problem.vHeads;
    if(std::min({hq, hk, hv}) <= 0 || hq % hk != 0 || hq % hv != 0
       || std::max(hk, hv) % std::min(hk, hv) != 0 || traits.blockSize % 32 != 0)
    {
        return std::nullopt;
    }
    // wmma_fmha_decode_pack_groups / _row_tiles
    const int64_t groups = std::max(hk, hv);
    const int64_t rowTiles = ((hq / groups) * problem.seqLenQ + DECODE_ROWS - 1) / DECODE_ROWS;
    const auto [lo, hi] = decodeKvRange(
        problem.seqLenQ, problem.seqLenK, mask.bottomRight != 0, mask.left, mask.right);
    // wmma_fmha_decode_num_splits(..., num_cus=multiProcessorCount, wg_per_cu=2,
    // min_tiles_per_wave=4): decode_launch_plan hands the helper num_cus // 2 of the
    // dispatcher's CU count, i.e. HIP's RDNA multiProcessorCount (WGPs), whereas
    // rdnaAttentionUseDecode takes the CU count itself (usesDecode doubles it).
    const int64_t numWaves = traits.blockSize / 32;
    const int64_t workGroups
        = std::max<int64_t>(1, problem.batch * groups * rowTiles * traits.dvSplit);
    const int64_t tiles
        = (std::max<int64_t>(hi - lo, 0) + DECODE_SPLIT_KEYS - 1) / DECODE_SPLIT_KEYS;
    const int64_t byWork = std::max<int64_t>(1, tiles / (numWaves * DECODE_MIN_TILES_PER_WAVE));
    const int64_t byDevice
        = std::max<int64_t>(1, (deviceMultiprocessors(device) * DECODE_WG_PER_CU) / workGroups);

    DecodePlan plan;
    plan.numSplits = std::max<int64_t>(1, std::min({byWork, byDevice, DECODE_MAX_SPLITS}));
    // wmma_fmha_decode_grid / _merge_grid
    plan.grid = problem.batch * plan.numSplits * groups * traits.dvSplit * rowTiles;
    plan.mergeGrid = problem.batch * hq * problem.seqLenQ;
    if(plan.grid <= 0 || plan.grid > I32_MAX || plan.mergeGrid <= 0 || plan.mergeGrid > I32_MAX)
    {
        return std::nullopt;
    }
    // wmma_fmha_decode_workspace
    if(plan.numSplits > 1)
    {
        const int64_t rows = problem.batch * hq * problem.seqLenQ * plan.numSplits;
        if(rows * (problem.headDimV + 1) > I32_MAX)
        {
            return std::nullopt;
        }
        plan.wsLse = rows * problem.headDimV * 4;
        plan.wsBytes = plan.wsLse + rows * 4;
    }
    return plan;
}

/// Whether the rocKE dispatcher routes @p problem to the decode kernel on @p device.
bool usesDecode(const Problem& problem, const MaskParameters& mask, const DeviceProperties& device)
{
    const auto [lo, hi] = decodeKvRange(
        problem.seqLenQ, problem.seqLenK, mask.bottomRight != 0, mask.left, mask.right);
    return rdnaAttentionUseDecode(problem.seqLenQ,
                                  hi - lo,
                                  problem.queryHeads,
                                  problem.kvHeads,
                                  problem.batch,
                                  isGfx12(device),
                                  2 * deviceMultiprocessors(device))
           && decodeBucket(problem.headDimQ, problem.headDimV).has_value();
}

/// The decode launch the shipped catalog makes for @p problem, or nullopt when the
/// dispatcher does not route it to decode or the catalog's decode object cannot take it.
/// This is what the decode engine's matcher admits, so the prefill engine declines exactly
/// these. The object's block size and dv_split are the catalog's (the D256 objects split
/// the O columns over 2 workgroups on gfx11 and 4 on gfx12; the others are unsplit), not
/// read from a kernel: the prefill engine has no decode kernel in hand.
std::optional<DecodePlan> catalogDecodePlan(const Problem& problem,
                                            const MaskParameters& mask,
                                            const DeviceProperties& device)
{
    if(!usesDecode(problem, mask, device))
    {
        return std::nullopt;
    }
    ObjectTraits traits;
    traits.decode = true;
    traits.blockSize = DECODE_BLOCK_SIZE;
    traits.dvSplit = 1;
    if(decodeBucket(problem.headDimQ, problem.headDimV) == MAX_HEAD_DIM)
    {
        traits.dvSplit = isGfx12(device) ? 4 : 2;
    }
    return decodePlanFor(problem, mask, traits, device);
}

/// The kernel matcher of one engine: @p decodeEngine selects which of the two kernels
/// it admits.
bool kernelMatchesFor(bool decodeEngine,
                      const MatchContext& context,
                      const BoundTokens& bound,
                      const KernelDefinition& kernel)
{
    const auto problem = problemFor(context);
    const auto traits = traitsOf(kernel);
    if(!problem.has_value() || !traits.has_value() || traits->decode != decodeEngine)
    {
        return false;
    }
    const auto dtype = supportedDataTypeName(problem->dataType);
    if(!dtype.has_value() || traits->dtype != *dtype
       || traits->bias != static_cast<BiasKind>(readBound(bound, BIAS_TOKEN)))
    {
        return false;
    }
    const auto mask = boundMask(bound);
    if(!maskModeServes(traits->maskMode, mask.kind))
    {
        return false;
    }
    if(decodeEngine)
    {
        // Exactly the dispatcher's object: decode by its rule, in the bucket
        // wmma_fmha_decode_bucket picks (which also bounds Q/K below).
        return usesDecode(*problem, mask, context.deviceProperties)
               && decodeBucket(problem->headDimQ, problem->headDimV) == traits->headSize
               && decodePlanFor(*problem, mask, *traits, context.deviceProperties).has_value();
    }
    // What the decode engine serves is not the prefill engine's.
    if(catalogDecodePlan(*problem, mask, context.deviceProperties).has_value())
    {
        return false;
    }
    // A prefill object serves any widths up to its buckets.
    const int64_t queryTiles = (problem->seqLenQ + traits->blockM - 1) / traits->blockM;
    return problem->headDimQ <= traits->headSize && problem->headDimV <= traits->headSizeV
           && problem->batch * queryTiles * problem->queryHeads <= I32_MAX;
}

bool prefillKernelMatches(const MatchContext& context,
                          const BoundTokens& bound,
                          const KernelDefinition& kernel)
{
    return kernelMatchesFor(false, context, bound, kernel);
}

bool decodeKernelMatches(const MatchContext& context,
                         const BoundTokens& bound,
                         const KernelDefinition& kernel)
{
    return kernelMatchesFor(true, context, bound, kernel);
}

double scoreKernel(const MatchContext& context,
                   const BoundTokens& bound,
                   const KernelDefinition& kernel)
{
    const auto traits = traitsOf(kernel);
    if(!traits.has_value())
    {
        return 0.0;
    }
    // kernelMatches admits a decode object only where the dispatcher picks decode, and
    // then it outranks every prefill object.
    if(traits->decode)
    {
        return 1000.0;
    }
    // Head-dim tiles past the request still execute, so the tightest bucket pair
    // wins (2 per 8-wide step of Q/K + V/O). At equal buckets an object with no
    // mask compiled in beats the causal class for an unmasked request (1).
    const auto steps = static_cast<double>(2 * MAX_HEAD_DIM - traits->headSize - traits->headSizeV)
                       / static_cast<double>(HEAD_DIM_GRANULE);
    const bool unmasked = static_cast<MaskKind>(readBound(bound, MASK_TOKEN)) == MaskKind::NONE;
    const double specializedMask = unmasked && traits->maskMode == "none" ? 1.0 : 0.0;
    static_cast<void>(context); // the device does not change the ranking
    return 10.0 + 2.0 * steps + specializedMask;
}

// ---- kernel ABIs (wmma_fmha_fwd_v2._declare_params, wmma_fmha_decode._declare_params
// and _declare_merge_params) ----

/// An argument list in kernarg order, each argument at its natural alignment.
class SignatureBuilder
{
public:
    void pointer(std::string name)
    {
        add("global_buffer", static_cast<uint32_t>(sizeof(void*)), std::move(name));
    }
    void f32(std::string name)
    {
        add("by_value", static_cast<uint32_t>(sizeof(float)), std::move(name));
    }
    void i32(std::string name)
    {
        add("by_value", static_cast<uint32_t>(sizeof(int32_t)), std::move(name));
    }
    /// stride_<tensor>_{batch,token,head} per tensor, in elements.
    void strides(std::initializer_list<const char*> tensors)
    {
        for(const char* tensor : tensors)
        {
            for(const char* axis : {"batch", "token", "head"})
            {
                i32(std::string("stride_") + tensor + "_" + axis);
            }
        }
    }
    std::vector<KernelArgument> take()
    {
        return std::move(_signature);
    }

private:
    void add(const char* kind, uint32_t size, std::string name)
    {
        _offset = (_offset + size - 1) / size * size;
        _signature.emplace_back(KernelArgument{kind, size, _offset, std::move(name)});
        _offset += size;
    }

    std::vector<KernelArgument> _signature;
    uint32_t _offset = 0;
};

void addBiasArguments(SignatureBuilder& builder)
{
    builder.pointer("attn_bias_ptr");
    for(const char* name : {"bias_stride_b", "bias_stride_h", "bias_stride_q"})
    {
        builder.i32(name);
    }
}

std::vector<KernelArgument> prefillSignature(const ObjectTraits& traits)
{
    SignatureBuilder builder;
    for(const char* name : {"Q", "K", "V", "O", "LSE"})
    {
        builder.pointer(name);
    }
    builder.f32("scale_log2");
    for(const char* name : {"batch",
                            "seqlen_q",
                            "seqlen_k",
                            "num_query_heads",
                            "num_kv_heads",
                            "num_v_heads",
                            "head_dim_q",
                            "head_dim_v",
                            "bottom_right",
                            "window_left",
                            "window_right",
                            "write_lse"})
    {
        builder.i32(name);
    }
    builder.strides({"q", "k", "v", "o", "lse"});
    if(traits.bias != BiasKind::NONE)
    {
        addBiasArguments(builder);
    }
    return builder.take();
}

std::vector<KernelArgument> decodeSignature(const ObjectTraits& traits)
{
    SignatureBuilder builder;
    for(const char* name : {"Q", "K", "V", "O", "LSE", "WS_O", "WS_LSE", "WS_CNT"})
    {
        builder.pointer(name);
    }
    builder.f32("scale_log2");
    for(const char* name : {"seqlen_q",
                            "seqlen_k",
                            "num_query_heads",
                            "num_kv_heads",
                            "num_v_heads",
                            "head_dim_q",
                            "head_dim_v",
                            "bottom_right",
                            "window_left",
                            "window_right",
                            "write_lse",
                            "num_splits"})
    {
        builder.i32(name);
    }
    builder.strides({"q", "k", "v", "o", "lse"});
    if(traits.bias != BiasKind::NONE)
    {
        addBiasArguments(builder);
    }
    return builder.take();
}

std::vector<KernelArgument> mergeSignature()
{
    SignatureBuilder builder;
    for(const char* name : {"O", "LSE", "WS_O", "WS_LSE"})
    {
        builder.pointer(name);
    }
    for(const char* name : {"seqlen_q", "num_query_heads", "head_dim_v", "num_splits", "write_lse"})
    {
        builder.i32(name);
    }
    builder.strides({"o", "lse"});
    return builder.take();
}

struct RuntimeArguments
{
    int32_t batch;
    int32_t queryHeads;
    int32_t kvHeads;
    int32_t vHeads;
    int32_t seqLenQ;
    int32_t seqLenK;
    int32_t headDimQ;
    int32_t headDimV;
    int32_t bottomRight;
    int32_t windowLeft;
    int32_t windowRight;
    int32_t writeLse;
    std::array<int32_t, 3> qStrides;
    std::array<int32_t, 3> kStrides;
    std::array<int32_t, 3> vStrides;
    std::array<int32_t, 3> oStrides;
    std::array<int32_t, 3> lseStrides;
    // Passed only to a bias object.
    bool attnBias;
    std::array<int32_t, 3> biasStrides;
};

RuntimeArguments runtimeArguments(const data_objects::TensorAttributes& q,
                                  const data_objects::TensorAttributes& k,
                                  const data_objects::TensorAttributes& v,
                                  const data_objects::TensorAttributes& o,
                                  const data_objects::TensorAttributes& lse,
                                  const Problem& problem,
                                  const Binding& binding,
                                  const ObjectTraits& traits)
{
    // Element strides of the (batch, token, head) axes.
    const auto strides = [](const data_objects::TensorAttributes& tensor) {
        return std::array<int32_t, 3>{static_cast<int32_t>(tensor.strides()->Get(BATCH_AXIS)),
                                      static_cast<int32_t>(tensor.strides()->Get(SEQ_AXIS)),
                                      static_cast<int32_t>(tensor.strides()->Get(HEAD_AXIS))};
    };
    RuntimeArguments args;
    args.batch = static_cast<int32_t>(problem.batch);
    args.queryHeads = static_cast<int32_t>(problem.queryHeads);
    args.kvHeads = static_cast<int32_t>(problem.kvHeads);
    args.vHeads = static_cast<int32_t>(problem.vHeads);
    args.seqLenQ = static_cast<int32_t>(problem.seqLenQ);
    args.seqLenK = static_cast<int32_t>(problem.seqLenK);
    args.headDimQ = static_cast<int32_t>(problem.headDimQ);
    args.headDimV = static_cast<int32_t>(problem.headDimV);
    args.bottomRight = binding.mask.bottomRight;
    args.windowLeft = binding.mask.left;
    args.windowRight = binding.mask.right;
    args.writeLse = binding.writeLse;
    args.qStrides = strides(q);
    args.kStrides = strides(k);
    args.vStrides = strides(v);
    args.oStrides = strides(o);
    args.lseStrides = binding.writeLse != 0 ? strides(lse) : std::array<int32_t, 3>{0, 0, 0};
    args.attnBias = traits.bias != BiasKind::NONE;
    args.biasStrides = binding.bias.strides;
    return args;
}

class PreparedGfx1151WmmaAttention : public PreparedDispatch
{
public:
    PreparedGfx1151WmmaAttention(IngestorKernelCode code,
                                 std::optional<IngestorKernelCode> merge,
                                 bool decode,
                                 DecodePlan plan,
                                 Binding binding,
                                 RuntimeArguments arguments)
        : _code(std::move(code))
        , _merge(std::move(merge))
        , _decode(decode)
        , _plan(plan)
        , _binding(binding)
        , _arguments(arguments)
    {
    }

    compilation::IRunnableKernel& kernelForStream(hipStream_t stream) const
    {
        return _code.kernelForStream(stream);
    }

    compilation::IRunnableKernel& mergeForStream(hipStream_t stream) const
    {
        return _merge->kernelForStream(stream);
    }

    bool decode() const
    {
        return _decode;
    }

    const DecodePlan& plan() const
    {
        return _plan;
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
    // The decode object's merge companion; absent for prefill.
    std::optional<IngestorKernelCode> _merge;
    bool _decode;
    DecodePlan _plan;
    Binding _binding;
    RuntimeArguments _arguments;
};

void requireLaunchRange(int64_t value, int64_t maximum, const char* what)
{
    if(value <= 0 || value > maximum)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       std::string("gfx1151 WMMA attention ") + what
                                                           + " does not fit its launch range");
    }
}

void requireRowAligned(const void* pointer, const char* name)
{
    if(reinterpret_cast<uintptr_t>(pointer) % ROW_ALIGN_BYTES != 0)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            std::string("gfx1151 WMMA attention ") + name
                + " must be 16-byte aligned: the kernels load whole rows as 16-byte vectors");
    }
}

class Gfx1151WmmaAttentionDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    /// @p decode selects the engine: the decode engine's kernels carry the "decode" and
    /// "merge" launches, the prefill engine's one `kernel_source`.
    Gfx1151WmmaAttentionDispatchHandler(const compilation::KpackKernelLoader& kpackLoader,
                                        bool decode)
        : _kpackLoader(kpackLoader)
        , _decode(decode)
    {
    }

    std::set<std::string> launchNames() const override
    {
        return _decode ? std::set<std::string>{DECODE_LAUNCH, MERGE_LAUNCH}
                       : std::set<std::string>{};
    }

    /// The decode split partials (wmma_fmha_decode_workspace); prefill needs none.
    size_t workspaceBytes(const MatchContext& context,
                          const BoundTokens& bound,
                          const KernelDefinition& kernel) const override
    {
        const auto traits = requireEngineTraits(kernel);
        if(!traits.decode)
        {
            return 0;
        }
        return static_cast<size_t>(decodePlan(context, bound, traits).wsBytes);
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
        const auto traits = requireEngineTraits(kernel);
        const auto problem = problemFor(*q, *k, *v);
        const auto arguments = runtimeArguments(*q, *k, *v, *o, *lse, problem, binding, traits);
        requireLaunchRange(traits.blockSize, 1024, "block size");

        if(!traits.decode)
        {
            auto code
                = buildIngestorKernelCode(_kpackLoader, context, kernel, prefillSignature(traits));
            // wmma_fmha_fwd_v2_grid: heads fastest, then query tile and batch.
            const int64_t queryTiles = (problem.seqLenQ + traits.blockM - 1) / traits.blockM;
            const int64_t grid = problem.batch * queryTiles * problem.queryHeads;
            requireLaunchRange(grid, I32_MAX, "prefill grid");
            code.setBlockSize(static_cast<uint32_t>(traits.blockSize), 1, 1);
            code.setGridSize(static_cast<uint32_t>(grid), 1, 1);
            return std::make_unique<PreparedGfx1151WmmaAttention>(
                std::move(code), std::nullopt, false, DecodePlan{}, binding, arguments);
        }

        const auto plan = decodePlan(context, bound, traits);
        // Both launches are resolved and the merge ABI checked before either archive
        // load, so a descriptor disagreeing with this pack on the decode or the merge
        // argument list fails on the host, whatever the state of the code object.
        const auto decodeKernel = launchKernelDefinition(kernel, DECODE_LAUNCH);
        const auto mergeKernel = launchKernelDefinition(kernel, MERGE_LAUNCH);
        requireSignatureMatch(mergeKernel.source.signature,
                              mergeSignature(),
                              mergeKernel.source.symbol,
                              describeDescriptor("kernel", kernel.name, kernel.kernelId));
        auto code
            = buildIngestorKernelCode(_kpackLoader, context, decodeKernel, decodeSignature(traits));
        code.setBlockSize(static_cast<uint32_t>(traits.blockSize), 1, 1);
        code.setGridSize(static_cast<uint32_t>(plan.grid), 1, 1);
        // The merge kernel resolves out of the decode object's own code object.
        auto merge = buildIngestorKernelCode(_kpackLoader, context, mergeKernel, mergeSignature());
        merge.setBlockSize(MERGE_THREADS, 1, 1);
        merge.setGridSize(static_cast<uint32_t>(plan.mergeGrid), 1, 1);
        return std::make_unique<PreparedGfx1151WmmaAttention>(
            std::move(code), std::move(merge), true, plan, binding, arguments);
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* workspace) const override
    {
        const auto& state = dynamic_cast<const PreparedGfx1151WmmaAttention&>(prepared);
        const auto& binding = state.binding();
        const auto& p = state.arguments();
        const auto find = [&](int64_t uid) {
            return hipdnn_plugin_sdk::findDeviceBuffer(uid, deviceBuffers, numDeviceBuffers).ptr;
        };
        void* q = find(binding.q);
        void* k = find(binding.k);
        void* v = find(binding.v);
        void* o = find(binding.o);
        void* lse = find(binding.lse);
        requireRowAligned(q, "Q");
        requireRowAligned(k, "K");
        requireRowAligned(v, "V");
        requireRowAligned(o, "O");
        float scale = binding.scale;
        if(binding.runtimeScale.has_value())
        {
            scale = static_cast<float>(
                hipdnn_plugin_sdk::toDouble(hipdnn_plugin_sdk::resolveScalarOperand(
                    *binding.runtimeScale, deviceBuffers, numDeviceBuffers)));
            if(!std::isfinite(scale) || !(scale > 0.0F))
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                    "gfx1151 WMMA attention scale must be finite and positive");
            }
        }
        const float scaleLog2 = scale * 1.4426950408889634F;
        void* bias = p.attnBias ? find(binding.bias.uid) : nullptr;
        hipStream_t stream = handle.getStream();
        const auto& kernel = state.kernelForStream(stream);

        if(!state.decode())
        {
            const auto launchPrefill = [&](const auto&... tail) {
                kernel.launch(stream,
                              q,
                              k,
                              v,
                              o,
                              lse,
                              scaleLog2,
                              p.batch,
                              p.seqLenQ,
                              p.seqLenK,
                              p.queryHeads,
                              p.kvHeads,
                              p.vHeads,
                              p.headDimQ,
                              p.headDimV,
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
                              p.lseStrides[2],
                              tail...);
            };
            if(p.attnBias)
            {
                launchPrefill(bias, p.biasStrides[0], p.biasStrides[1], p.biasStrides[2]);
            }
            else
            {
                launchPrefill();
            }
            return;
        }

        // Split-KV decode. One split writes O / LSE directly with null workspace
        // pointers. More splits write partials to the workspace and the merge companion
        // combines them. WS_CNT stays null, so the kernel never takes its fused merge,
        // whose arrival counters would have to be cleared per launch (hipDNN's workspace
        // carries no state between launches).
        const auto& plan = state.plan();
        const auto numSplits = static_cast<int32_t>(plan.numSplits);
        void* wsO = nullptr;
        void* wsLse = nullptr;
        void* const wsCnt = nullptr;
        if(plan.numSplits > 1)
        {
            if(workspace == nullptr)
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                    "gfx1151 WMMA attention decode needs its split workspace");
            }
            auto* base = static_cast<char*>(workspace);
            requireRowAligned(base, "workspace");
            wsO = base;
            wsLse = base + plan.wsLse;
        }
        const auto launchDecode = [&](const auto&... tail) {
            kernel.launch(stream,
                          q,
                          k,
                          v,
                          o,
                          lse,
                          wsO,
                          wsLse,
                          wsCnt,
                          scaleLog2,
                          p.seqLenQ,
                          p.seqLenK,
                          p.queryHeads,
                          p.kvHeads,
                          p.vHeads,
                          p.headDimQ,
                          p.headDimV,
                          p.bottomRight,
                          p.windowLeft,
                          p.windowRight,
                          p.writeLse,
                          numSplits,
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
                          p.lseStrides[2],
                          tail...);
        };
        if(p.attnBias)
        {
            launchDecode(bias, p.biasStrides[0], p.biasStrides[1], p.biasStrides[2]);
        }
        else
        {
            launchDecode();
        }
        if(plan.numSplits > 1)
        {
            state.mergeForStream(stream).launch(stream,
                                                o,
                                                lse,
                                                wsO,
                                                wsLse,
                                                p.seqLenQ,
                                                p.queryHeads,
                                                p.headDimV,
                                                numSplits,
                                                p.writeLse,
                                                p.oStrides[0],
                                                p.oStrides[1],
                                                p.oStrides[2],
                                                p.lseStrides[0],
                                                p.lseStrides[1],
                                                p.lseStrides[2]);
        }
    }

private:
    /// The traits of @p kernel, which must be of this handler's engine: the state
    /// manager's launch-name check keeps the other engine's kernels out, and this is the
    /// same guard on the metadata.
    ObjectTraits requireEngineTraits(const KernelDefinition& kernel) const
    {
        auto traits = requireTraits(kernel);
        if(traits.decode != _decode)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                "gfx1151 WMMA attention "
                    + describeDescriptor("kernel", kernel.name, kernel.kernelId) + " is a "
                    + (traits.decode ? "decode" : "prefill") + " kernel in the "
                    + (_decode ? "decode" : "prefill") + " engine");
        }
        return traits;
    }

    static DecodePlan decodePlan(const MatchContext& context,
                                 const BoundTokens& bound,
                                 const ObjectTraits& traits)
    {
        const auto problem = problemFor(context);
        const auto plan
            = problem.has_value()
                  ? decodePlanFor(*problem, boundMask(bound), traits, context.deviceProperties)
                  : std::nullopt;
        if(!plan.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "gfx1151 WMMA attention decode cannot launch this problem");
        }
        return *plan;
    }

    const compilation::KpackKernelLoader& _kpackLoader;
    bool _decode;
};

compilation::KpackModuleCache& gfx1151WmmaAttentionKpackModuleCache()
{
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

compilation::KpackModuleCache& gfx1151WmmaAttentionDecodeKpackModuleCache()
{
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

} // namespace

void resetGfx1151WmmaAttentionModuleCache()
{
    gfx1151WmmaAttentionKpackModuleCache().clear();
}

void resetGfx1151WmmaAttentionDecodeModuleCache()
{
    gfx1151WmmaAttentionDecodeKpackModuleCache().clear();
}

namespace
{

const Gfx1151WmmaAttentionDispatchHandler& gfx1151WmmaAttentionDispatchHandler()
{
    static const compilation::KpackKernelLoader s_loader(gfx1151WmmaAttentionKpackModuleCache());
    static const Gfx1151WmmaAttentionDispatchHandler s_handler(s_loader, false);
    return s_handler;
}

const Gfx1151WmmaAttentionDispatchHandler& gfx1151WmmaAttentionDecodeDispatchHandler()
{
    static const compilation::KpackKernelLoader s_loader(
        gfx1151WmmaAttentionDecodeKpackModuleCache());
    static const Gfx1151WmmaAttentionDispatchHandler s_handler(s_loader, true);
    return s_handler;
}

} // namespace

void registerGfx1151WmmaAttentionSymbols(SymbolScope<Handle>& scope)
{
    scope.add(std::string(PREFILL_GRAPH_MATCHER_SYMBOL), &gfx1151WmmaAttentionGraphMatches);
    scope.add(std::string(PREFILL_KERNEL_MATCHER_SYMBOL), &prefillKernelMatches);
    scope.add(std::string(PREFILL_SCORE_SYMBOL), &scoreKernel);
    scope.add(std::string(PREFILL_DISPATCH_SYMBOL), &gfx1151WmmaAttentionDispatchHandler());
}

void registerGfx1151WmmaAttentionDecodeSymbols(SymbolScope<Handle>& scope)
{
    scope.add(std::string(DECODE_GRAPH_MATCHER_SYMBOL), &gfx1151WmmaAttentionGraphMatches);
    scope.add(std::string(DECODE_KERNEL_MATCHER_SYMBOL), &decodeKernelMatches);
    scope.add(std::string(DECODE_SCORE_SYMBOL), &scoreKernel);
    scope.add(std::string(DECODE_DISPATCH_SYMBOL), &gfx1151WmmaAttentionDecodeDispatchHandler());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
