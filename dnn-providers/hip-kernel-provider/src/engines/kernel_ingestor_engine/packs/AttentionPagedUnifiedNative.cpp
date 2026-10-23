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
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <hip/hip_runtime.h>

#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/PluginDeviceBuffers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>
#include <hipdnn_plugin_sdk/ingestor/SymbolScope.hpp>

#include "compilation/KpackKernelLoader.hpp"
#include "compilation/KpackModuleCache.hpp"
#include "core/Handle.hpp"
#include "engines/asm_sdpa_engine/plans/SdpaPlanUtils.hpp"
#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"

/**
 * @file AttentionPagedUnifiedNative.cpp
 * @brief The hipkernel:AttentionPagedUnified engine's native half: matching and dispatch.
 *
 * The kernels are rocKE's tiled 2D unified-attention builders
 * (`rocke/library/kernels/{gfx942,gfx950}/attention_tiled_2d.py`,
 * `build_unified_attention_2d_tiled`) built with `runtime_layout=True`, packaged
 * (kind: rocke -> kpack) per architecture. In that mode the kernel takes every
 * Q/O/K/V stride, the query head count, the sliding window and the batch's query
 * layout (dense or ragged) as launch arguments, and reads the graph's real page table
 * and sequence lengths from device memory. One object therefore serves every shape
 * with the same (dtype, head_size, block_size, queries_per_kv), and one plan serves
 * every step of a serving loop: actual lengths and block ids are only ever read on the
 * device, so execute performs no host synchronization and no allocation.
 *
 * Graphs served (cuDNN paged-attention semantics):
 *  - Q `[B, Hq, Sq, D]`, either dense with arbitrary strides (uniform decode, Sq rows per
 *    sequence) or ragged at the sequence axis through an INT32 `[B+1,1,1,1]`
 *    `ragged_offset` shared with O;
 *  - paged K/V containers `[num_blocks, Hkv, block_size, D]` with arbitrary (equal)
 *    strides, including the interleaved K|V view of one cache allocation;
 *  - one INT32 page table `[B, 1, M, 1]` used for both K and V;
 *  - INT32 `seq_len_kv` (required) and `seq_len_q` (optional) `[B,1,1,1]`;
 *  - bottom-right causal masking with an optional left bound (sliding window);
 *  - an optional attention sink `sink_token` `[1, Hq, 1, 1]` in the Q/K/V/O data type
 *    with a unit head stride, served by the use_sinks objects.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

namespace
{

// The contract with the installed descriptor files, which restate these same strings.
constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.attention_paged_unified.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.attention_paged_unified.kernel_match";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.attention_paged_unified.dispatch";
constexpr std::string_view GFX942_MATCHER_SYMBOL = "hipkernel.attention_paged_unified.gfx942_match";
constexpr std::string_view GFX950_MATCHER_SYMBOL = "hipkernel.attention_paged_unified.gfx950_match";

// KMD fields. The catalog builds every kernel with num_kv_heads = 1 (head counts are
// launch arguments), so its num_query_heads field IS the queries-per-KV-head ratio.
constexpr std::string_view DTYPE_FIELD = "dtype";
constexpr std::string_view HEAD_SIZE_FIELD = "head_size";
constexpr std::string_view BLOCK_SIZE_FIELD = "block_size";
constexpr std::string_view NUM_QUERY_HEADS_FIELD = "num_query_heads";
constexpr std::string_view NUM_KV_HEADS_FIELD = "num_kv_heads";
constexpr std::string_view NUM_WARPS_FIELD = "num_warps";
constexpr std::string_view BLOCK_M_PER_WARP_FIELD = "block_m_per_warp";
/// 1 for an object built with use_sinks (it reads sink_ptr), 0 (the KMD default) otherwise.
constexpr std::string_view USE_SINKS_FIELD = "use_sinks";

// Bound tokens.
constexpr std::string_view Q_TOKEN = "attention_paged_unified.q.uid";
constexpr std::string_view K_TOKEN = "attention_paged_unified.k.uid";
constexpr std::string_view V_TOKEN = "attention_paged_unified.v.uid";
constexpr std::string_view O_TOKEN = "attention_paged_unified.o.uid";
constexpr std::string_view PAGE_TABLE_TOKEN = "attention_paged_unified.page_table.uid";
constexpr std::string_view SEQ_LEN_KV_TOKEN = "attention_paged_unified.seq_len_kv.uid";
/// UID of seq_len_q, or NO_UID.
constexpr std::string_view SEQ_LEN_Q_TOKEN = "attention_paged_unified.seq_len_q.uid";
/// UID of the ragged offset table Q and O share, or NO_UID for a dense Q.
constexpr std::string_view RAGGED_OFFSET_TOKEN = "attention_paged_unified.ragged_offset.uid";
/// UID of the sink_token tensor, or NO_UID.
constexpr std::string_view SINK_TOKEN = "attention_paged_unified.sink.uid";
constexpr std::string_view SCALE_BITS_TOKEN = "attention_paged_unified.scale_bits";
/// The kernel's sliding_window argument: `d - k < window` keeps key k for diagonal d.
constexpr std::string_view WINDOW_TOKEN = "attention_paged_unified.window";
constexpr std::string_view DTYPE_TOKEN = "attention_paged_unified.dtype";
constexpr std::string_view HEAD_SIZE_TOKEN = "attention_paged_unified.head_size";
constexpr std::string_view BLOCK_SIZE_TOKEN = "attention_paged_unified.block_size";
constexpr std::string_view QUERIES_PER_KV_TOKEN = "attention_paged_unified.queries_per_kv";

constexpr int64_t NO_UID = -1;
/// Integer codes for the bound dtype.
constexpr int64_t DTYPE_FP16 = 1;
constexpr int64_t DTYPE_BF16 = 2;

/// The window the kernel receives when the graph has no left bound: wider than any
/// sequence the int32 positions can describe, so it never masks.
constexpr int64_t NO_WINDOW = 0x3FFFFFFF;

/// hipDNN tensor axes. The LOGICAL order is always (B, H, S, D) regardless of layout; a
/// paged container's B axis is the block index and its S axis the token within a block.
constexpr uint32_t BATCH_AXIS = 0;
constexpr uint32_t HEAD_AXIS = 1;
constexpr uint32_t SEQ_AXIS = 2;
constexpr uint32_t HEAD_SIZE_AXIS = 3;
constexpr uint32_t SDPA_RANK = 4;

/// The kernel's vector loads and stores move 8 elements (16 bytes) at a time, so every
/// stride of a non-unit axis and every base pointer must keep that alignment.
constexpr int64_t VECTOR_ELEMENTS = 8;
constexpr uintptr_t VECTOR_BYTES = 16;
constexpr int64_t ELEMENT_BYTES = 2;

/// Buffer-resource record limit the kernels use for one K/V block.
constexpr int64_t MAX_BLOCK_SPAN_BYTES = 0x7FFF0000;
/// The gfx942 kernel reads page-table entries through a buffer of MAX_BLOCK_SPAN_BYTES with
/// an int32 byte offset, so a table reaches at most this many entries.
constexpr int64_t MAX_PAGE_TABLE_ENTRIES = MAX_BLOCK_SPAN_BYTES / 4;
/// A ragged Q's sequence is found by a fixed-trip binary search over the offsets
/// (attention_tiled_2d _RTL_SEQ_SEARCH_ITERS), which covers this many sequences.
constexpr int64_t MAX_RAGGED_BATCH = 65536;
constexpr int64_t INT32_LIMIT = std::numeric_limits<int32_t>::max();
/// Grid-y cap. The kernel walks query blocks with a grid-stride loop, so a smaller grid
/// is always correct; this keeps the launch inside every device's grid limits.
constexpr int64_t MAX_GRID_Y = 65535;
constexpr int64_t WAVE_LANES = 64;

/// Catalog coverage. The descriptors ship exactly these; graph_match uses them only to
/// name the reason for a decline, kernel_match compares the candidate's own metadata.
constexpr std::array<int64_t, 3> HEAD_SIZES = {64, 128, 256};
constexpr std::array<int64_t, 3> BLOCK_SIZES = {16, 32, 64};
constexpr std::array<int64_t, 8> QUERIES_PER_KV = {1, 2, 4, 5, 6, 7, 8, 16};

template <size_t N>
bool contains(const std::array<int64_t, N>& values, int64_t value)
{
    for(const auto v : values)
    {
        if(v == value)
        {
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// Matching helpers
// ---------------------------------------------------------------------------

const data_objects::TensorAttributes* findTensor(const MatchContext& context, int64_t uid)
{
    const auto& tensors = context.graph.getTensorMap();
    const auto it = tensors.find(uid);
    return it == tensors.end() ? nullptr : it->second;
}

std::optional<int64_t> checkedProduct(std::initializer_list<int64_t> factors)
{
    int64_t product = 1;
    for(const int64_t factor : factors)
    {
        if(__builtin_mul_overflow(product, factor, &product))
        {
            return std::nullopt;
        }
    }
    return product;
}

std::optional<int64_t> checkedSum(std::initializer_list<int64_t> terms)
{
    int64_t sum = 0;
    for(const int64_t term : terms)
    {
        if(__builtin_add_overflow(sum, term, &sum))
        {
            return std::nullopt;
        }
    }
    return sum;
}

/// The head of a decline line: "<engine> declined the graph [<cause>]: ". The cause is a
/// stable key per check, for log filters and tests; the text after it is for people.
struct Declined
{
    std::string_view cause;
};

std::ostream& operator<<(std::ostream& out, const Declined& declined)
{
    return out << ATTENTION_PAGED_UNIFIED_ENGINE_NAME << " declined the graph [" << declined.cause
               << "]: ";
}

bool hasSdpaForwardNode(const MatchContext& context)
{
    for(const auto& node : context.graph.nodeWrappers())
    {
        if(node->attributesType() == data_objects::NodeAttributes::SdpaAttributes)
        {
            return true;
        }
    }
    return false;
}

std::string describeOperand(std::string_view role, const data_objects::TensorAttributes& tensor)
{
    using hipdnn_data_sdk::utilities::vecToString;
    using hipdnn_flatbuffers_sdk::utilities::convertFlatBufferVectorToStdVector;
    std::ostringstream out;
    out << role << " (uid " << tensor.uid() << "): dims "
        << vecToString(convertFlatBufferVectorToStdVector(tensor.dims())) << ", strides "
        << vecToString(convertFlatBufferVectorToStdVector(tensor.strides()));
    return out.str();
}

/// The single SDPA-forward node, or nullptr. Override-shape graphs (admitted by the
/// 1.4.0 schema floor) are not served: launch arguments come from the built shapes.
const data_objects::SdpaAttributes* sdpaNode(const MatchContext& context)
{
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

/// A rank-4 device operand with positive extents and non-negative strides, read
/// directly from the variant pack (not virtual, not pass-by-value).
bool isWellFormedRank4(const data_objects::TensorAttributes& tensor)
{
    const auto* dims = tensor.dims();
    const auto* strides = tensor.strides();
    if(dims == nullptr || strides == nullptr || dims->size() != SDPA_RANK
       || strides->size() != SDPA_RANK || tensor.virtual_()
       || hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(&tensor))
    {
        return false;
    }
    for(uint32_t i = 0; i < SDPA_RANK; ++i)
    {
        if(dims->Get(i) <= 0 || strides->Get(i) < 0)
        {
            return false;
        }
    }
    return true;
}

/// Strides with every unit-extent axis set to zero: an index that is always zero makes
/// its stride irrelevant, so it must neither decline a graph nor reach the kernel.
std::array<int64_t, SDPA_RANK> effectiveStrides(const data_objects::TensorAttributes& tensor)
{
    std::array<int64_t, SDPA_RANK> strides{};
    for(uint32_t i = 0; i < SDPA_RANK; ++i)
    {
        strides[i] = tensor.dims()->Get(i) == 1 ? 0 : tensor.strides()->Get(i);
    }
    return strides;
}

int64_t dim(const data_objects::TensorAttributes& tensor, uint32_t axis)
{
    return tensor.dims()->Get(axis);
}

/// Every stride a vector load or store walks: a multiple of 8 elements and within int32.
bool vectorStridesOk(const std::array<int64_t, SDPA_RANK>& strides, bool includeBatch)
{
    for(uint32_t axis = includeBatch ? BATCH_AXIS : HEAD_AXIS; axis < HEAD_SIZE_AXIS; ++axis)
    {
        if(strides[axis] % VECTOR_ELEMENTS != 0 || strides[axis] > INT32_LIMIT)
        {
            return false;
        }
    }
    return true;
}

/// An INT32 per-sequence vector `[n, 1, 1, 1]` the kernel indexes contiguously.
bool isInt32Vector(const data_objects::TensorAttributes& tensor, int64_t length)
{
    if(!isWellFormedRank4(tensor) || tensor.data_type() != data_objects::DataType::INT32
       || tensor.ragged_offset_tensor_uid().has_value() || dim(tensor, BATCH_AXIS) != length)
    {
        return false;
    }
    for(uint32_t axis = HEAD_AXIS; axis < SDPA_RANK; ++axis)
    {
        if(dim(tensor, axis) != 1)
        {
            return false;
        }
    }
    return length == 1 || tensor.strides()->Get(BATCH_AXIS) == 1;
}

/// The extent of the elements an operand addresses within one sequence (or one block),
/// in elements, without its batch axis: the largest within-sequence offset plus one.
std::optional<int64_t> innerExtent(const data_objects::TensorAttributes& tensor)
{
    const auto strides = effectiveStrides(tensor);
    const auto head = checkedProduct({dim(tensor, HEAD_AXIS) - 1, strides[HEAD_AXIS]});
    const auto seq = checkedProduct({dim(tensor, SEQ_AXIS) - 1, strides[SEQ_AXIS]});
    if(!head.has_value() || !seq.has_value())
    {
        return std::nullopt;
    }
    return checkedSum({*head, *seq, dim(tensor, HEAD_SIZE_AXIS)});
}

std::optional<int64_t> dataTypeCode(data_objects::DataType dataType)
{
    if(dataType == data_objects::DataType::HALF)
    {
        return DTYPE_FP16;
    }
    if(dataType == data_objects::DataType::BFLOAT16)
    {
        return DTYPE_BF16;
    }
    return std::nullopt;
}

std::string_view dataTypeName(int64_t code)
{
    return code == DTYPE_FP16 ? "FP16" : "BF16";
}

/// Q's or O's layout, the part of the problem prepare() turns into kernel arguments.
struct QueryLayout
{
    int64_t seqStride = 0; ///< dense: batch stride; ragged: the offset multiplier
    int64_t tokenStride = 0;
    int64_t headStride = 0;
};

QueryLayout queryLayout(const data_objects::TensorAttributes& tensor, bool ragged)
{
    const auto strides = effectiveStrides(tensor);
    QueryLayout layout;
    layout.seqStride = ragged ? tensor.ragged_offset_multiplier() : strides[BATCH_AXIS];
    layout.tokenStride = strides[SEQ_AXIS];
    layout.headStride = strides[HEAD_AXIS];
    return layout;
}

/// Checks Q or O (identical dims already established). Logs its own declines.
bool queryOperandOk(std::string_view role,
                    const data_objects::TensorAttributes& tensor,
                    bool ragged)
{
    const auto strides = effectiveStrides(tensor);
    if(tensor.strides()->Get(HEAD_SIZE_AXIS) != 1)
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"layout"} << describeOperand(role, tensor)
                                                  << " has a non-unit head-size stride");
        return false;
    }
    const int64_t seqStride = ragged ? tensor.ragged_offset_multiplier() : strides[BATCH_AXIS];
    // A ragged sequence starts at offset * multiplier: a zero multiplier would put every
    // sequence at the buffer base.
    if(!vectorStridesOk(strides, false) || seqStride % VECTOR_ELEMENTS != 0 || seqStride < 0
       || (ragged && seqStride == 0) || seqStride > INT32_LIMIT)
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"layout"}
                               << describeOperand(role, tensor)
                               << (ragged ? " with ragged_offset_multiplier "
                                          : " with batch stride ")
                               << seqStride
                               << ": strides must be multiples of 8 elements and fit int32, "
                                  "and a ragged multiplier must be positive");
        return false;
    }
    const auto extent = innerExtent(tensor);
    if(!extent.has_value() || *extent > INT32_LIMIT)
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"size_limit"}
                               << describeOperand(role, tensor)
                               << ": one sequence spans more than 2^31-1 elements");
        return false;
    }
    return true;
}

/// Logs and declines.
std::nullopt_t decline(std::string_view cause, const std::string& reason)
{
    HIPDNN_PLUGIN_LOG_INFO(Declined{cause} << reason);
    return std::nullopt;
}

/**
 * @brief Graph-scoped applicability for the whole engine.
 *
 * Each decline of a graph with an SDPA-forward node logs its cause at INFO; any other
 * graph declines silently. Messages are built only when INFO logging is enabled.
 *
 * @warning Returning std::nullopt empties this engine's WHOLE catalog.
 */
std::optional<BoundTokens> graphMatches(const MatchContext& context)
{
    // --- 1. Node shape.
    const auto* attributesPtr = sdpaNode(context);
    if(attributesPtr == nullptr)
    {
        if(HIPDNN_PLUGIN_LOG_IS_INFO_ENABLED() && hasSdpaForwardNode(context))
        {
            HIPDNN_PLUGIN_LOG_INFO(
                Declined{"node"}
                << "the graph is not a single SDPA-forward node without override shapes ("
                << context.graph.nodeCount() << " node(s))");
        }
        return std::nullopt;
    }
    const auto& attributes = *attributesPtr;
    const bool info = HIPDNN_PLUGIN_LOG_IS_INFO_ENABLED();
    const auto reason = [info](const auto&... parts) {
        if(!info)
        {
            return std::string();
        }
        std::ostringstream out;
        (out << ... << parts);
        return out.str();
    };

    if(context.graph.getNodeWrapper(0).computeDataType() != data_objects::DataType::FLOAT)
    {
        return decline("compute_data_type", reason("compute data type must be FLOAT"));
    }

    // --- 2. Paging. This engine serves paged K/V only; the kernel reads one block table
    // for both K and V.
    const auto pageTableK = attributes.page_table_k_tensor_uid();
    const auto pageTableV = attributes.page_table_v_tensor_uid();
    if(!pageTableK.has_value() || !pageTableV.has_value())
    {
        return decline("not_paged",
                       reason("K/V without page tables (page_table_k/page_table_v) are not "
                              "served; this engine reads paged K/V containers"));
    }
    if(*pageTableK != *pageTableV)
    {
        return decline("page_table",
                       reason("page_table_k (uid ",
                              *pageTableK,
                              ") and page_table_v (uid ",
                              *pageTableV,
                              ") must be the same tensor; the kernel takes one block table"));
    }

    // --- 3. Operands.
    const auto* q = findTensor(context, attributes.q_tensor_uid());
    const auto* k = findTensor(context, attributes.k_tensor_uid());
    const auto* v = findTensor(context, attributes.v_tensor_uid());
    const auto* o = findTensor(context, attributes.o_tensor_uid());
    const auto* pageTable = findTensor(context, *pageTableK);
    const std::array<std::pair<std::string_view, const data_objects::TensorAttributes*>, 5> operands
        = {{{"Q", q}, {"K", k}, {"V", v}, {"O", o}, {"page_table", pageTable}}};
    for(const auto& [role, tensor] : operands)
    {
        if(tensor == nullptr || !isWellFormedRank4(*tensor))
        {
            return decline("operand",
                           reason(role,
                                  " is missing, virtual, pass-by-value, not rank 4, or has a "
                                  "non-positive extent or negative stride"));
        }
    }

    // --- 4. Data types.
    const auto dtype = dataTypeCode(q->data_type());
    if(!dtype.has_value() || k->data_type() != q->data_type() || v->data_type() != q->data_type()
       || o->data_type() != q->data_type())
    {
        return decline(
            "data_type",
            reason("Q/K/V/O must all be FP16 or all BFLOAT16 (Q is ", q->data_type(), ")"));
    }

    // --- 5. Shapes.
    const int64_t batch = dim(*q, BATCH_AXIS);
    const int64_t numQueryHeads = dim(*q, HEAD_AXIS);
    const int64_t seqLenQ = dim(*q, SEQ_AXIS);
    const int64_t headSize = dim(*q, HEAD_SIZE_AXIS);
    const int64_t numKvHeads = dim(*k, HEAD_AXIS);
    const int64_t blockSize = dim(*k, SEQ_AXIS);
    for(uint32_t axis = 0; axis < SDPA_RANK; ++axis)
    {
        if(dim(*v, axis) != dim(*k, axis) || dim(*o, axis) != dim(*q, axis))
        {
            return decline("shape",
                           reason(describeOperand("V", *v),
                                  " must match ",
                                  describeOperand("K", *k),
                                  ", and ",
                                  describeOperand("O", *o),
                                  " must match ",
                                  describeOperand("Q", *q)));
        }
    }
    if(dim(*k, HEAD_SIZE_AXIS) != headSize)
    {
        return decline("head_size", reason("Q and K/V head sizes differ"));
    }
    if(!contains(HEAD_SIZES, headSize))
    {
        return decline("head_size", reason("head size ", headSize, " is not one of 64, 128, 256"));
    }
    if(!contains(BLOCK_SIZES, blockSize))
    {
        return decline("block_size",
                       reason("K/V block size ", blockSize, " is not one of 16, 32, 64"));
    }
    if(numQueryHeads % numKvHeads != 0 || !contains(QUERIES_PER_KV, numQueryHeads / numKvHeads))
    {
        return decline("gqa",
                       reason("query heads ",
                              numQueryHeads,
                              " over K/V heads ",
                              numKvHeads,
                              " is not a shipped ratio (1, 2, 4, 5, 6, 7, 8, 16)"));
    }
    const int64_t queriesPerKv = numQueryHeads / numKvHeads;

    // --- 6. Q/O layout: dense, or ragged through one shared offset table.
    const bool qRagged = q->ragged_offset_tensor_uid().has_value();
    const bool oRagged = o->ragged_offset_tensor_uid().has_value();
    if(qRagged != oRagged
       || (qRagged && *q->ragged_offset_tensor_uid() != *o->ragged_offset_tensor_uid()))
    {
        return decline("ragged",
                       reason("Q and O must both be dense or both ragged through the same "
                              "ragged_offset tensor"));
    }
    if(k->ragged_offset_tensor_uid().has_value() || v->ragged_offset_tensor_uid().has_value()
       || pageTable->ragged_offset_tensor_uid().has_value())
    {
        return decline("ragged", reason("ragged K/V or page tables are not supported"));
    }
    int64_t raggedOffsetUid = NO_UID;
    if(qRagged)
    {
        raggedOffsetUid = *q->ragged_offset_tensor_uid();
        const auto* offsets = findTensor(context, raggedOffsetUid);
        if(offsets == nullptr || !isInt32Vector(*offsets, batch + 1))
        {
            return decline("ragged",
                           reason("the ragged_offset tensor must be an INT32 [B+1, 1, 1, 1] "
                                  "device vector with unit stride"));
        }
    }
    if(!queryOperandOk("Q", *q, qRagged) || !queryOperandOk("O", *o, qRagged))
    {
        return std::nullopt;
    }

    // --- 7. K/V containers: equal strides (the kernel shares one descriptor), D-contiguous.
    const auto kStrides = effectiveStrides(*k);
    if(effectiveStrides(*v) != kStrides)
    {
        return decline("kv_layout",
                       reason(describeOperand("K", *k),
                              " and ",
                              describeOperand("V", *v),
                              " must have equal strides"));
    }
    if(k->strides()->Get(HEAD_SIZE_AXIS) != 1 || v->strides()->Get(HEAD_SIZE_AXIS) != 1
       || !vectorStridesOk(kStrides, true))
    {
        return decline("kv_layout",
                       reason(describeOperand("K", *k),
                              ": K/V need a unit head-size stride and other strides that are "
                              "multiples of 8 elements within int32"));
    }
    const auto kvExtent = innerExtent(*k);
    const auto kvSpanBytes
        = kvExtent.has_value() ? checkedProduct({*kvExtent, ELEMENT_BYTES}) : std::nullopt;
    if(!kvSpanBytes.has_value() || *kvSpanBytes > MAX_BLOCK_SPAN_BYTES)
    {
        return decline("size_limit", reason("one K/V block spans more than 0x7FFF0000 bytes"));
    }

    // --- 8. Page table INT32 [B, 1, M, 1] with a unit entry stride.
    if(pageTable->data_type() != data_objects::DataType::INT32
       || dim(*pageTable, BATCH_AXIS) != batch || dim(*pageTable, HEAD_AXIS) != 1
       || dim(*pageTable, HEAD_SIZE_AXIS) != 1
       || (dim(*pageTable, SEQ_AXIS) > 1 && pageTable->strides()->Get(SEQ_AXIS) != 1))
    {
        return decline("page_table",
                       reason(describeOperand("page_table", *pageTable),
                              " must be INT32 [B, 1, max_blocks, 1] with unit block stride"));
    }
    const auto pageTableStrides = effectiveStrides(*pageTable);
    const auto pageTableExtent
        = checkedSum({checkedProduct({batch - 1, pageTableStrides[BATCH_AXIS]})
                          .value_or(std::numeric_limits<int64_t>::max()),
                      dim(*pageTable, SEQ_AXIS)});
    if(!pageTableExtent.has_value() || *pageTableExtent > MAX_PAGE_TABLE_ENTRIES)
    {
        return decline("size_limit",
                       reason("the page table spans more than ",
                              MAX_PAGE_TABLE_ENTRIES,
                              " entries (0x7FFF0000 bytes)"));
    }

    // --- 9. Sequence lengths. The kernels read the first ceil(seq_len_kv[b] / block_size)
    // entries of row b and do not clamp seq_len_kv: the contract's 0 <= seq_len_kv[b] <=
    // max_seq_len_kv <= M * block_size is a precondition of the launch.
    const auto seqLenKvUid = attributes.seq_len_kv_tensor_uid();
    if(!seqLenKvUid.has_value())
    {
        return decline("seq_len_kv", reason("paged attention needs a seq_len_kv tensor"));
    }
    const auto* seqLenKv = findTensor(context, *seqLenKvUid);
    if(seqLenKv == nullptr || !isInt32Vector(*seqLenKv, batch))
    {
        return decline("seq_len_kv",
                       reason("seq_len_kv must be an INT32 [B, 1, 1, 1] device vector with "
                              "unit stride"));
    }
    int64_t seqLenQUid = NO_UID;
    if(attributes.seq_len_q_tensor_uid().has_value())
    {
        seqLenQUid = *attributes.seq_len_q_tensor_uid();
        const auto* seqLenQTensor = findTensor(context, seqLenQUid);
        if(seqLenQTensor == nullptr || !isInt32Vector(*seqLenQTensor, batch))
        {
            return decline("seq_len_q",
                           reason("seq_len_q must be an INT32 [B, 1, 1, 1] device vector with "
                                  "unit stride"));
        }
    }

    // --- 10. Mask: bottom-right causal, optionally with a left bound (sliding window).
    asm_sdpa_engine::plan_utils::DiagonalBand band{};
    try
    {
        band = asm_sdpa_engine::plan_utils::resolveDiagonalBand(attributes);
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& e)
    {
        return decline("mask", reason(e.what()));
    }
    if(band.rightBound != 0 || band.alignment != data_objects::DiagonalAlignment::BOTTOM_RIGHT)
    {
        return decline("mask",
                       reason("only bottom-right causal masking (right_bound 0, BOTTOM_RIGHT "
                              "alignment) is served; got right_bound ",
                              band.rightBound,
                              (band.alignment == data_objects::DiagonalAlignment::BOTTOM_RIGHT
                                   ? ", BOTTOM_RIGHT"
                                   : ", TOP_LEFT")));
    }
    int64_t window = NO_WINDOW;
    if(band.leftBound >= 0)
    {
        if(band.leftBound >= NO_WINDOW - 1)
        {
            window = NO_WINDOW;
        }
        else
        {
            // hipDNN keeps key k for diagonal d iff k >= d - left_bound; the kernel keeps it
            // iff d - k < window.
            window = band.leftBound + 1;
        }
    }

    // --- 11. Every optional attribute the kernel cannot honour.
    if(attributes.attn_mask_tensor_uid().has_value())
    {
        return decline("attn_mask", reason("an additive attention mask is not supported"));
    }
    if(attributes.scale_tensor_uid().has_value())
    {
        return decline("scale_tensor",
                       reason("a scale tensor is not supported; set attn_scale_value"));
    }
    if(attributes.seed_tensor_uid().has_value() || attributes.offset_tensor_uid().has_value()
       || attributes.dropout_mask_tensor_uid().has_value()
       || attributes.dropout_scale_tensor_uid().has_value()
       || attributes.dropout_probability().has_value())
    {
        return decline("dropout", reason("dropout is not supported"));
    }
    if(attributes.block_mask_tensor_uid().has_value())
    {
        return decline("block_mask", reason("a block mask is not supported"));
    }
    // A sink is one logit per query head, which the use_sinks kernels load from sink_ptr
    // at index `query head` in the I/O dtype.
    int64_t sinkUid = NO_UID;
    if(attributes.sink_token_tensor_uid().has_value())
    {
        sinkUid = *attributes.sink_token_tensor_uid();
        const auto* sink = findTensor(context, sinkUid);
        if(sink == nullptr || !isWellFormedRank4(*sink))
        {
            return decline("sinks",
                           reason("sink_token is missing, virtual, pass-by-value, not rank 4, or "
                                  "has a non-positive extent or negative stride"));
        }
        if(sink->ragged_offset_tensor_uid().has_value() || dim(*sink, BATCH_AXIS) != 1
           || dim(*sink, HEAD_AXIS) != numQueryHeads || dim(*sink, SEQ_AXIS) != 1
           || dim(*sink, HEAD_SIZE_AXIS) != 1)
        {
            return decline("sinks",
                           reason(describeOperand("sink_token", *sink),
                                  " must be [1, ",
                                  numQueryHeads,
                                  ", 1, 1] (one logit per query head)"));
        }
        if(sink->data_type() != q->data_type())
        {
            return decline("sinks",
                           reason("sink_token data type ",
                                  sink->data_type(),
                                  " differs from Q's ",
                                  q->data_type(),
                                  ": the kernels load the sink in the I/O data type, so FLOAT "
                                  "sinks are not served"));
        }
        if(numQueryHeads > 1 && sink->strides()->Get(HEAD_AXIS) != 1)
        {
            return decline("sinks",
                           reason(describeOperand("sink_token", *sink),
                                  " needs a unit head stride; the kernels index the sink by "
                                  "query head"));
        }
    }
    if(attributes.descale_q_tensor_uid().has_value()
       || attributes.descale_k_tensor_uid().has_value()
       || attributes.descale_v_tensor_uid().has_value()
       || attributes.descale_s_tensor_uid().has_value()
       || attributes.scale_s_tensor_uid().has_value() || attributes.scale_o_tensor_uid().has_value()
       || attributes.amax_s_tensor_uid().has_value() || attributes.amax_o_tensor_uid().has_value())
    {
        return decline("fp8", reason("FP8 descale/scale/amax tensors are not supported"));
    }
    if(attributes.stats_tensor_uid().has_value() || attributes.max_tensor_uid().has_value()
       || attributes.sum_exp_tensor_uid().has_value()
       || attributes.rng_dump_tensor_uid().has_value()
       || (attributes.generate_stats().has_value() && attributes.generate_stats().value()))
    {
        return decline("stats", reason("softmax statistics outputs are not supported"));
    }
    if(attributes.alibi_mask())
    {
        return decline("alibi_mask", reason("ALiBi is not supported"));
    }
    const auto mmaCoreMode = attributes.mma_core_mode();
    if(mmaCoreMode != data_objects::DataType::UNSET && mmaCoreMode != data_objects::DataType::HALF
       && mmaCoreMode != data_objects::DataType::BFLOAT16)
    {
        return decline("mma_core_mode", reason("mma_core_mode ", mmaCoreMode, " is not supported"));
    }
    if(attributes.implementation() != data_objects::AttentionImplementation::AUTO)
    {
        return decline("implementation",
                       reason("implementation hint ",
                              attributes.implementation(),
                              " is not supported; only AUTO is"));
    }
    // Absent attn_scale_value means 1.0 (cuDNN's default).
    const float scale = attributes.attn_scale_value().value_or(1.0F);
    if(!std::isfinite(scale))
    {
        return decline("scale", reason("attn_scale ", scale, " is not finite"));
    }

    // --- 12. Launch-size limits that do not depend on the candidate.
    if(batch > INT32_LIMIT || seqLenQ > INT32_LIMIT || numQueryHeads > INT32_LIMIT)
    {
        return decline("size_limit", reason("batch, query length or head count exceeds int32"));
    }
    if(qRagged && batch > MAX_RAGGED_BATCH)
    {
        return decline("size_limit",
                       reason("a ragged Q of ",
                              batch,
                              " sequences exceeds the kernels' ",
                              MAX_RAGGED_BATCH,
                              "-sequence search"));
    }

    BoundTokens bound;
    bound[std::string(Q_TOKEN)] = attributes.q_tensor_uid();
    bound[std::string(K_TOKEN)] = attributes.k_tensor_uid();
    bound[std::string(V_TOKEN)] = attributes.v_tensor_uid();
    bound[std::string(O_TOKEN)] = attributes.o_tensor_uid();
    bound[std::string(PAGE_TABLE_TOKEN)] = *pageTableK;
    bound[std::string(SEQ_LEN_KV_TOKEN)] = *seqLenKvUid;
    bound[std::string(SEQ_LEN_Q_TOKEN)] = seqLenQUid;
    bound[std::string(RAGGED_OFFSET_TOKEN)] = raggedOffsetUid;
    bound[std::string(SINK_TOKEN)] = sinkUid;
    int32_t scaleBits = 0;
    static_assert(sizeof(scaleBits) == sizeof(scale), "float must be 32-bit to round-trip");
    std::memcpy(&scaleBits, &scale, sizeof(scale));
    bound[std::string(SCALE_BITS_TOKEN)] = static_cast<int64_t>(scaleBits);
    bound[std::string(WINDOW_TOKEN)] = window;
    bound[std::string(DTYPE_TOKEN)] = *dtype;
    bound[std::string(HEAD_SIZE_TOKEN)] = headSize;
    bound[std::string(BLOCK_SIZE_TOKEN)] = blockSize;
    bound[std::string(QUERIES_PER_KV_TOKEN)] = queriesPerKv;
    return bound;
}

/// The device's processor without target features ("gfx942:sramecc+:xnack-" -> "gfx942").
std::string_view processorOf(const MatchContext& context)
{
    const std::string_view arch = context.deviceProperties.gcnArchName;
    return arch.substr(0, arch.find(':'));
}

/// Pack criteria: each pack holds one architecture's builder output (the gfx942 and
/// gfx950 tiled kernels are different modules), so a pack serves only its own device.
bool gfx942PackMatches(const MatchContext& context, const BoundTokens& /*bound*/)
{
    return processorOf(context) == "gfx942";
}

bool gfx950PackMatches(const MatchContext& context, const BoundTokens& /*bound*/)
{
    return processorOf(context) == "gfx950";
}

int64_t readBound(const BoundTokens& bound, std::string_view token)
{
    const auto value = tryGetBoundInt(bound, token);
    if(!value.has_value())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            std::string(ATTENTION_PAGED_UNIFIED_ENGINE_NAME) + " dispatch is missing bound token '"
                + std::string(token) + "', or it does not hold an integer");
    }
    return *value;
}

/// The candidate's CTA geometry: threads per CTA and query rows (tokens) per q-block.
struct CandidateGeometry
{
    int64_t threads = 0;
    int64_t blockQ = 0;
};

/// A candidate's integer metadata field, or nullopt when it is absent or holds another
/// type. Total: a malformed record declines instead of throwing out of a match.
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

/// The geometry the candidate's baked metadata implies, mirroring the rocKE spec:
/// `block_m = block_m_per_warp * num_warps`, `block_q = block_m // queries_per_kv`,
/// `threads = num_warps * 64`. nullopt for a record that lacks a field, was not built
/// with one KV head, or yields no query rows per block.
std::optional<CandidateGeometry> candidateGeometry(const KernelDefinition& kernel)
{
    const auto numWarps = integerMetadata(kernel, NUM_WARPS_FIELD);
    const auto blockMPerWarp = integerMetadata(kernel, BLOCK_M_PER_WARP_FIELD);
    const auto queries = integerMetadata(kernel, NUM_QUERY_HEADS_FIELD);
    const auto kvHeads = integerMetadata(kernel, NUM_KV_HEADS_FIELD);
    if(!numWarps.has_value() || !blockMPerWarp.has_value() || !queries.has_value()
       || !kvHeads.has_value() || *numWarps <= 0 || *numWarps > 16 || *blockMPerWarp <= 0
       || *kvHeads != 1 || *queries <= 0)
    {
        return std::nullopt;
    }
    CandidateGeometry geometry;
    geometry.threads = *numWarps * WAVE_LANES;
    geometry.blockQ = (*blockMPerWarp * *numWarps) / *queries;
    if(geometry.blockQ <= 0)
    {
        return std::nullopt;
    }
    return geometry;
}

/**
 * @brief Kernel-scoped applicability: the candidate was built for this graph's dtype,
 *        head size, block size and GQA ratio, with one KV head (runtime head counts), and
 *        with use_sinks exactly when the graph has a sink_token.
 */
bool kernelMatches(const MatchContext& /*context*/,
                   const BoundTokens& bound,
                   const KernelDefinition& kernel)
{
    const auto dtype = tryGetBoundInt(bound, DTYPE_TOKEN);
    const auto headSize = tryGetBoundInt(bound, HEAD_SIZE_TOKEN);
    const auto blockSize = tryGetBoundInt(bound, BLOCK_SIZE_TOKEN);
    const auto queriesPerKv = tryGetBoundInt(bound, QUERIES_PER_KV_TOKEN);
    const auto sinkUid = tryGetBoundInt(bound, SINK_TOKEN);
    if(!dtype.has_value() || !headSize.has_value() || !blockSize.has_value()
       || !queriesPerKv.has_value() || !sinkUid.has_value()
       || !candidateGeometry(kernel).has_value())
    {
        return false;
    }
    // An absent use_sinks is the KMD default 0. A sink object without a sink would read
    // a null sink_ptr; a plain object with a sink would drop it.
    const int64_t wantSinks = *sinkUid != NO_UID ? 1 : 0;
    return stringMetadata(kernel, DTYPE_FIELD) == std::string(dataTypeName(*dtype))
           && integerMetadata(kernel, HEAD_SIZE_FIELD) == *headSize
           && integerMetadata(kernel, BLOCK_SIZE_FIELD) == *blockSize
           && integerMetadata(kernel, NUM_QUERY_HEADS_FIELD) == *queriesPerKv
           && integerMetadata(kernel, USE_SINKS_FIELD).value_or(0) == wantSinks;
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// The kernel signature (runtime_layout=True). NAMES ARE LOAD-BEARING: they are checked
/// against the names hkp_pack recorded from the rocKE builder's parameter list, so keep
/// them identical to kernels/{gfx942,gfx950}/attention_tiled_2d.py.
std::vector<KernelArgument> kernelSignature()
{
    constexpr auto PTR = static_cast<uint32_t>(sizeof(void*));
    constexpr auto I32 = static_cast<uint32_t>(sizeof(int32_t));
    constexpr auto F32 = static_cast<uint32_t>(sizeof(float));
    std::vector<KernelArgument> signature;
    uint32_t offset = 0;
    const auto add = [&](std::string_view kind, uint32_t size, std::string_view name) {
        offset = (offset + size - 1) / size * size;
        signature.push_back(KernelArgument{std::string(kind), size, offset, std::string(name)});
        offset += size;
    };
    for(const auto* name : {"output_ptr",
                            "query_ptr",
                            "key_cache_ptr",
                            "value_cache_ptr",
                            "sink_ptr",
                            "block_tables_ptr",
                            "seq_lens_ptr",
                            "alibi_slopes_ptr",
                            "qq_bias_ptr",
                            "query_start_len_ptr"})
    {
        add("global_buffer", PTR, name);
    }
    for(const auto* name : {"scale", "k_scale", "v_scale", "out_scale", "softcap"})
    {
        add("by_value", F32, name);
    }
    for(const auto* name : {"num_seqs", "block_table_stride", "qq_bias_stride_0"})
    {
        add("by_value", I32, name);
    }
    add("global_buffer", PTR, "seq_lens_q_ptr");
    for(const auto* name : {"has_seq_lens_q",
                            "q_uniform_len",
                            "num_query_heads",
                            "sliding_window",
                            "q_seq_stride",
                            "q_tok_stride",
                            "q_head_stride",
                            "o_seq_stride",
                            "o_tok_stride",
                            "o_head_stride",
                            "kv_block_stride",
                            "kv_tok_stride",
                            "kv_head_stride",
                            "kv_block_span_bytes",
                            "q_block_grid"})
    {
        add("by_value", I32, name);
    }
    return signature;
}

/// Every by-value launch argument, resolved once at prepare().
struct LaunchScalars
{
    float scale = 0.0F;
    int32_t numSeqs = 0;
    int32_t blockTableStride = 0;
    int32_t hasSeqLensQ = 0;
    int32_t qUniformLen = 0;
    int32_t numQueryHeads = 0;
    int32_t slidingWindow = 0;
    QueryLayout q;
    QueryLayout o;
    int32_t kvBlockStride = 0;
    int32_t kvTokStride = 0;
    int32_t kvHeadStride = 0;
    int32_t kvBlockSpanBytes = 0;
    int32_t qBlockGrid = 0;
};

struct BoundUids
{
    int64_t q = 0;
    int64_t k = 0;
    int64_t v = 0;
    int64_t o = 0;
    int64_t pageTable = 0;
    int64_t seqLenKv = 0;
    int64_t seqLenQ = NO_UID;
    int64_t raggedOffset = NO_UID;
    int64_t sink = NO_UID;
};

class PreparedAttentionPagedUnified : public PreparedDispatch
{
public:
    PreparedAttentionPagedUnified(IngestorKernelCode code, BoundUids uids, LaunchScalars scalars)
        : _code(std::move(code))
        , _uids(uids)
        , _scalars(scalars)
    {
    }

    compilation::IRunnableKernel& kernelForStream(hipStream_t stream) const
    {
        return _code.kernelForStream(stream);
    }

    const BoundUids& uids() const
    {
        return _uids;
    }

    const LaunchScalars& scalars() const
    {
        return _scalars;
    }

private:
    IngestorKernelCode _code;
    BoundUids _uids;
    LaunchScalars _scalars;
};

int32_t narrow(int64_t value, std::string_view what)
{
    if(value < std::numeric_limits<int32_t>::min() || value > INT32_LIMIT)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            std::string(ATTENTION_PAGED_UNIFIED_ENGINE_NAME) + ": " + std::string(what)
                + " does not fit int32");
    }
    return static_cast<int32_t>(value);
}

const data_objects::TensorAttributes& requireTensor(const MatchContext& context, int64_t uid)
{
    const auto* tensor = findTensor(context, uid);
    if(tensor == nullptr || !isWellFormedRank4(*tensor))
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            std::string(ATTENTION_PAGED_UNIFIED_ENGINE_NAME) + ": tensor uid " + std::to_string(uid)
                + " is missing or malformed");
    }
    return *tensor;
}

class AttentionPagedUnifiedDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    explicit AttentionPagedUnifiedDispatchHandler(const compilation::KpackKernelLoader& loader)
        : _loader(loader)
    {
    }

    /// Zero: the kernel keeps all scratch in LDS and registers.
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
        // graph_match and kernel_match are the gates; this is defence in depth for a
        // caller that reaches the handler without having matched.
        if(!graphMatches(context).has_value() || !kernelMatches(context, bound, kernel))
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                std::string(ATTENTION_PAGED_UNIFIED_ENGINE_NAME)
                    + ": the graph or the kernel candidate is not supported");
        }
        const auto geometry = candidateGeometry(kernel);

        BoundUids uids;
        uids.q = readBound(bound, Q_TOKEN);
        uids.k = readBound(bound, K_TOKEN);
        uids.v = readBound(bound, V_TOKEN);
        uids.o = readBound(bound, O_TOKEN);
        uids.pageTable = readBound(bound, PAGE_TABLE_TOKEN);
        uids.seqLenKv = readBound(bound, SEQ_LEN_KV_TOKEN);
        uids.seqLenQ = readBound(bound, SEQ_LEN_Q_TOKEN);
        uids.raggedOffset = readBound(bound, RAGGED_OFFSET_TOKEN);
        uids.sink = readBound(bound, SINK_TOKEN);
        const bool ragged = uids.raggedOffset != NO_UID;

        const auto& q = requireTensor(context, uids.q);
        const auto& o = requireTensor(context, uids.o);
        const auto& k = requireTensor(context, uids.k);
        const auto& pageTable = requireTensor(context, uids.pageTable);

        LaunchScalars s;
        const auto scaleBits = static_cast<int32_t>(readBound(bound, SCALE_BITS_TOKEN));
        std::memcpy(&s.scale, &scaleBits, sizeof(s.scale));
        const int64_t batch = dim(q, BATCH_AXIS);
        const int64_t seqLenQ = dim(q, SEQ_AXIS);
        const int64_t numQueryHeads = dim(q, HEAD_AXIS);
        const int64_t numKvHeads = dim(k, HEAD_AXIS);
        s.numSeqs = narrow(batch, "batch");
        s.blockTableStride = narrow(effectiveStrides(pageTable)[BATCH_AXIS], "page-table stride");
        s.hasSeqLensQ = uids.seqLenQ != NO_UID ? 1 : 0;
        s.qUniformLen = ragged ? 0 : narrow(seqLenQ, "query length");
        s.numQueryHeads = narrow(numQueryHeads, "query heads");
        s.slidingWindow = narrow(readBound(bound, WINDOW_TOKEN), "window");
        s.q = queryLayout(q, ragged);
        s.o = queryLayout(o, ragged);
        const auto kStrides = effectiveStrides(k);
        s.kvBlockStride = narrow(kStrides[BATCH_AXIS], "K/V block stride");
        s.kvTokStride = narrow(kStrides[SEQ_AXIS], "K/V token stride");
        s.kvHeadStride = narrow(kStrides[HEAD_AXIS], "K/V head stride");
        const auto kvExtent = innerExtent(k);
        s.kvBlockSpanBytes = narrow(kvExtent.value_or(-1) * ELEMENT_BYTES, "K/V block span");
        for(const auto value : {s.q.seqStride,
                                s.q.tokenStride,
                                s.q.headStride,
                                s.o.seqStride,
                                s.o.tokenStride,
                                s.o.headStride})
        {
            narrow(value, "Q/O stride");
        }

        // q-block grid: exact for a dense Q; for a ragged Q, enough blocks for every
        // sequence when the batch's tokens fit in Sq (the serving convention) -- the kernel's
        // grid-stride loop covers any larger batch. Capped by MAX_GRID_Y either way.
        const int64_t blocksPerSequence = (seqLenQ + geometry->blockQ - 1) / geometry->blockQ;
        const int64_t gridY = ragged ? blocksPerSequence + batch : batch * blocksPerSequence;
        s.qBlockGrid = static_cast<int32_t>(std::min(std::max<int64_t>(gridY, 1), MAX_GRID_Y));

        auto code = buildIngestorKernelCode(_loader, context, kernel, kernelSignature());
        code.setBlockSize(static_cast<unsigned int>(geometry->threads), 1, 1);
        code.setGridSize(
            static_cast<unsigned int>(numKvHeads), static_cast<unsigned int>(s.qBlockGrid), 1);
        return std::make_unique<PreparedAttentionPagedUnified>(std::move(code), uids, s);
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* buffers,
                uint32_t count,
                void* /*workspace*/) const override
    {
        const auto& p = dynamic_cast<const PreparedAttentionPagedUnified&>(prepared);
        const auto& u = p.uids();
        const auto& s = p.scalars();
        const auto ptr = [&](int64_t uid, bool vectorAligned) {
            void* result = hipdnn_plugin_sdk::findDeviceBuffer(uid, buffers, count).ptr;
            if(result == nullptr
               || (vectorAligned && reinterpret_cast<uintptr_t>(result) % VECTOR_BYTES != 0)
               || reinterpret_cast<uintptr_t>(result) % sizeof(int32_t) != 0)
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                    std::string(ATTENTION_PAGED_UNIFIED_ENGINE_NAME) + ": buffer for uid "
                        + std::to_string(uid)
                        + (vectorAligned ? " is null or not 16-byte aligned"
                                         : " is null or not 4-byte aligned"));
            }
            return result;
        };
        void* q = ptr(u.q, true);
        void* k = ptr(u.k, true);
        void* v = ptr(u.v, true);
        void* o = ptr(u.o, true);
        void* pageTable = ptr(u.pageTable, false);
        void* seqLenKv = ptr(u.seqLenKv, false);
        // Absent tables are replaced by seq_len_kv, a valid INT32 buffer the kernel reads
        // only through index-0 masked loads in that case.
        void* seqLenQ = u.seqLenQ != NO_UID ? ptr(u.seqLenQ, false) : seqLenKv;
        void* raggedOffset = u.raggedOffset != NO_UID ? ptr(u.raggedOffset, false) : seqLenKv;
        const void* unused = nullptr;
        // sink_ptr is declared 16-byte aligned (attention_tiled_2d), like the Q/K/V/O
        // pointers; only the use_sinks objects read it.
        const void* sink = u.sink != NO_UID ? ptr(u.sink, true) : unused;

        p.kernelForStream(handle.getStream())
            .launch(handle.getStream(),
                    o,
                    q,
                    k,
                    v,
                    sink,
                    pageTable,
                    seqLenKv,
                    unused, // alibi_slopes_ptr
                    unused, // qq_bias_ptr
                    raggedOffset,
                    s.scale,
                    1.0F, // k_scale
                    1.0F, // v_scale
                    1.0F, // out_scale
                    0.0F, // softcap
                    s.numSeqs,
                    s.blockTableStride,
                    int32_t{0}, // qq_bias_stride_0
                    seqLenQ,
                    s.hasSeqLensQ,
                    s.qUniformLen,
                    s.numQueryHeads,
                    s.slidingWindow,
                    static_cast<int32_t>(s.q.seqStride),
                    static_cast<int32_t>(s.q.tokenStride),
                    static_cast<int32_t>(s.q.headStride),
                    static_cast<int32_t>(s.o.seqStride),
                    static_cast<int32_t>(s.o.tokenStride),
                    static_cast<int32_t>(s.o.headStride),
                    s.kvBlockStride,
                    s.kvTokStride,
                    s.kvHeadStride,
                    s.kvBlockSpanBytes,
                    s.qBlockGrid);
    }

private:
    const compilation::KpackKernelLoader& _loader;
};

compilation::KpackModuleCache& moduleCache()
{
    static compilation::KpackModuleCache s_cache;
    return s_cache;
}

} // namespace

void resetAttentionPagedUnifiedModuleCache()
{
    moduleCache().clear();
}

void registerAttentionPagedUnifiedSymbols(SymbolScope<Handle>& scope)
{
    static const compilation::KpackKernelLoader s_loader(moduleCache());
    static const AttentionPagedUnifiedDispatchHandler s_handler(s_loader);
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &graphMatches);
    scope.add(std::string(GFX942_MATCHER_SYMBOL), &gfx942PackMatches);
    scope.add(std::string(GFX950_MATCHER_SYMBOL), &gfx950PackMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &kernelMatches);
    scope.add(std::string(DISPATCH_SYMBOL), &s_handler);
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
