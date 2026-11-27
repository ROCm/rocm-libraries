// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

// Redundant with the CMake gate on purpose; see FlydslSdpaRdnaNative.cpp.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <hip/hip_runtime_api.h>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/ArchMatch.hpp>
#include <hipdnn_plugin_sdk/PluginDeviceBuffers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "compilation/KpackKernelLoader.hpp"
#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaRngDump.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaShared.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaVarlen.hpp"

/**
 * @file FlydslSdpaHeadPad.hpp
 * @brief Head dims that are not a multiple of 8, on the FlyDSL SDPA objects that serve only
 *        multiples of 8: the operands are copied into a padded workspace, the existing object
 *        runs on the copies, and O is copied back.
 *
 * Every FlyDSL SDPA object serves a Q/K and a V head dim that are multiples of 8 (its baked one,
 * or a runtime one up to its tier, in 8-column pieces). A graph whose Q/K head dim D or V head
 * dim Dv is not -- AOTriton's prime head dims, say, allocated at their own width so their rows
 * are not 16-byte aligned either -- runs through this file's wrapper instead:
 *
 * - **Matching.** The wrapper hands the engine's own graph match, kernel match and score a
 *   *padded graph*: the same graph with Q and K `ceil8(D)` wide, V and O `ceil8(Dv)` wide (256
 *   for 225-255, headPadWidth()), each
 *   a contiguous BSHD tensor (token-major when packed, its ragged multiplier its new sequence
 *   stride) declaring 16-byte alignment. Every other tensor, attribute and uid is unchanged: the
 *   scale is the graph's own (a value, a tensor, or cuDNN's default of 1, never recomputed from
 *   a width), and the bias, sink, lengths, offset tables, dropout, RNG dump and LSE do not depend
 *   on the head dim.
 * - **Dispatch.** The engine's handler prepares its object for the padded graph. Each launch
 *   copies Q, K and V into the workspace (the columns past their width zeroed, so Q·K over the
 *   padding adds exactly 0 and nothing read past a row reaches a score), runs the engine's
 *   launch with Q/K/V/O's buffers pointing at the copies, and copies O's first Dv columns back.
 *   A packed operand's copies go through its token table: each sequence at its table entry, for
 *   its per-batch length when the graph gives one (`seq_len_q` for Q and O, `seq_len_kv` for K
 *   and V: cuDNN's per-batch origins, in any order and with gaps), else for the table's
 *   difference: the binding the attention kernels read, servablePackedVarlenWithLengths(), taken
 *   on the graph as given, so copies and kernels take the same lengths; no token outside a
 *   sequence is read or written (headPadLengths()).
 * - **Rebase.** The workspace holds a packed operand's sequences at `b * S`, not at the graph's
 *   origins, which with gaps can pass `B * S` tokens: the workspace is sized for `B * S`
 *   tokens, so a sequence at `origin + length > B * S` would be written past it. Only the sides
 *   whose lengths the graph gives are rebased -- with `seq_len_kv` alone (`seqused_k`) K and V,
 *   with `seq_len_q` as well Q and O too; a side without lengths keeps its table, copied as the
 *   table's difference says. A rebased operand's copy in gathers each sequence from its table
 *   entry to `b * S`, and its copy back reverses that. The padded graph packs each rebased
 *   operand through a compact token table `[0, S, ..., B * S]`, a tensor it adds under a new
 *   uid (one per distinct original table and sequence extent: Q and K sharing a table with
 *   different extents get two) that the first copy through it writes into the workspace, and the
 *   launch hands the engine that buffer. When Q is rebased the graph's LSE, packed through Q's
 *   table, is held in the padded graph as a scratch tensor packed through Q's compact table
 *   (token-major: batch `S*H`, head 1, sequence `H`, multiplier `H`, which
 *   isPackedThrough() accepts for any `H`) and one more copy writes it out through Q's own
 *   table.
 *   The copies are the `sdpa_head_pad` op's object (`flydsl/kernels_src/kernels/head_pad/`),
 *   found beside the attention kernel's directory as the RNG dump's writer is; each launches on
 *   the attention launch's stream.
 * - **Every other graph** -- both head dims multiples of 8 -- reaches the engine's own functions
 *   with its own context, bound tokens and kernel, and the engine's prepared dispatch, workspace
 *   and launch unchanged: the wrapper adds a test of two dims at match time and a type check at
 *   launch, and no object changes.
 *
 * The copies cost two passes over Q, K and V and one over O, and the padded object computes
 * over `ceil8` columns, as the next multiple of 8 would.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::flydsl_sdpa
{

/// The copy's op, its pack beside the attention ops', and its workgroup-size field.
inline constexpr std::string_view HEAD_PAD_OP = "sdpa_head_pad";
inline constexpr std::string_view HEAD_PAD_PACK_FILE = "flydsl_sdpa_head_pad.kdp.json";
inline constexpr std::string_view HEAD_PAD_THREADS_FIELD = "block_threads";

/// The head-dim granularity every FlyDSL SDPA object serves.
inline constexpr int64_t HEAD_PAD_ALIGN = 8;
/// Each padded operand's offset in the workspace is a multiple of this.
inline constexpr size_t HEAD_PAD_BUFFER_ALIGN = 256;
/// The alignment the padded operands declare: rows of a multiple of 8 elements from a
/// 256-byte-aligned base.
inline constexpr int64_t HEAD_PAD_DECLARED_ALIGNMENT = 16;
/// The alignment the LSE scratch declares: one float per element, from a 256-byte-aligned base.
inline constexpr int64_t HEAD_PAD_STATS_DECLARED_ALIGNMENT = 4;
/// The copy's `compact` argument: the side of a rebased operand's copy that is the workspace's
/// dense run at `b * S + s` (the source when copying out, the destination when copying in).
inline constexpr int32_t HEAD_PAD_COMPACT_SOURCE = 1;
inline constexpr int32_t HEAD_PAD_COMPACT_DESTINATION = 2;

/// Past this padded width and below 256, a copy is padded to 256 (headPadWidth()).
inline constexpr int64_t HEAD_PAD_WIDE_FLOOR = 224;
inline constexpr int64_t HEAD_PAD_WIDE = 256;

/**
 * @brief The width a copy of an operand @p width wide is padded to: the next multiple of 8, or
 *        256 for 225-255.
 *
 * 232-248 run on the 256-wide tier or bucket on every family anyway, at its cost, and gfx12's aiw
 * objects -- the only ones there that drop in 16 bits -- serve dropout only up to 224 and at
 * exactly 256 (AOTriton's head dim 241 with dropout, among others). Padding those widths to 256
 * lands them on the exact 256 objects everywhere.
 */
inline int64_t headPadWidth(int64_t width)
{
    const int64_t aligned = (width + HEAD_PAD_ALIGN - 1) / HEAD_PAD_ALIGN * HEAD_PAD_ALIGN;
    return aligned > HEAD_PAD_WIDE_FLOOR && aligned < HEAD_PAD_WIDE ? HEAD_PAD_WIDE : aligned;
}

/// The 16-bit units the copy moves per element of @p dataType: 1 for HALF and BFLOAT16, 2 for
/// FLOAT; nullopt for a dtype no FlyDSL SDPA object takes.
inline std::optional<int64_t> headPadUnitsPerElement(data_objects::DataType dataType)
{
    switch(dataType)
    {
    case data_objects::DataType::HALF:
    case data_objects::DataType::BFLOAT16:
        return 1;
    case data_objects::DataType::FLOAT:
        return 2;
    default:
        return std::nullopt;
    }
}

/**
 * @brief Whether the graph is an SDPA forward whose Q/K or V head dim is not a multiple of 8:
 *        the graphs the wrapper pads. Every other graph goes to the engine's own functions.
 */
inline bool needsHeadPad(const MatchContext& context)
{
    const auto* attributes = sdpaNode(context);
    if(attributes == nullptr)
    {
        return false;
    }
    const auto* q = findTensor(context, attributes->q_tensor_uid());
    const auto* v = findTensor(context, attributes->v_tensor_uid());
    if(q == nullptr || v == nullptr || q->dims() == nullptr || v->dims() == nullptr
       || q->dims()->size() != SDPA_RANK || v->dims()->size() != SDPA_RANK)
    {
        return false;
    }
    return q->dims()->Get(HEAD_DIM_AXIS) % HEAD_PAD_ALIGN != 0
           || v->dims()->Get(HEAD_DIM_AXIS) % HEAD_PAD_ALIGN != 0;
}

/// A padded operand's element strides (batch, head, sequence): contiguous BSHD
/// `[B, S, H, width]`, which is also the token-major layout a packed operand needs.
struct HeadPadLayout
{
    int64_t batch = 0;
    int64_t head = 0;
    int64_t sequence = 0;
};

inline HeadPadLayout headPadLayout(int64_t heads, int64_t sequence, int64_t width)
{
    return HeadPadLayout{sequence * heads * width, width, heads * width};
}

/**
 * @brief The per-batch length tensors the copies of packed operands read: the query side's
 *        (Q and O) and the key side's (K and V), each NO_SEQ_LEN when a sequence's token count
 *        is its table's difference.
 *
 * Read from the binding the attention kernels themselves read, servablePackedVarlenWithLengths()
 * (the RDNA, aiw and CDNA matchers' packed rule), over the graph as given: that
 * binding is the one home of which lengths each packed operand takes. Q and O take its `seqLenQ`
 * unless that names Q's table (queryLengthsFromOffsets(): `seq_len_kv` alone, PyTorch's
 * seqused_k), K and V its `seqLenKv`.
 */
struct HeadPadLengths
{
    int64_t query = NO_SEQ_LEN;
    int64_t key = NO_SEQ_LEN;
};

/**
 * @brief HeadPadLengths for the graph @p context as given: none for a graph with no packed
 *        operand, or nullopt for a packed graph the binding declines (as the engines then do).
 */
inline std::optional<HeadPadLengths> headPadLengths(const MatchContext& context)
{
    const auto* attributes = sdpaNode(context);
    if(attributes == nullptr)
    {
        return std::nullopt;
    }
    if(!hasRaggedOperand(context, *attributes))
    {
        return HeadPadLengths{};
    }
    const auto* q = findTensor(context, attributes->q_tensor_uid());
    if(q == nullptr || q->dims() == nullptr || q->dims()->size() != SDPA_RANK)
    {
        return std::nullopt;
    }
    const auto binding
        = servablePackedVarlenWithLengths(context, *attributes, q->dims()->Get(BATCH_AXIS));
    if(!binding.has_value())
    {
        return std::nullopt;
    }
    HeadPadLengths lengths;
    lengths.query = queryLengthsFromOffsets(binding->seqLenQ, binding->raggedOffsetQ)
                        ? NO_SEQ_LEN
                        : binding->seqLenQ;
    lengths.key = binding->seqLenKv;
    return lengths;
}

/**
 * @brief The per-batch length tensor operand @p uid's copy reads (@p lengths' query side for Q
 *        and O, key side for K and V), or nullopt when the operand plays a query-side and a
 *        key-side role whose lengths differ: one copy cannot hold both sets of sequences.
 */
inline std::optional<int64_t> headPadOperandLengths(const HeadPadLengths& lengths,
                                                    const data_objects::SdpaAttributes& attributes,
                                                    int64_t uid)
{
    const bool queryRole = uid == attributes.q_tensor_uid() || uid == attributes.o_tensor_uid();
    const bool keyRole = uid == attributes.k_tensor_uid() || uid == attributes.v_tensor_uid();
    if(queryRole && keyRole && lengths.query != lengths.key)
    {
        return std::nullopt;
    }
    return queryRole ? lengths.query : lengths.key;
}

/**
 * @brief Whether the copy can read (or, for O, write) @p tensor: a rank-4 device tensor in a
 *        dtype the objects take, head-dim stride 1, every other axis of more than one element at
 *        a positive stride, the offset of its last element within int64; packed, with offsets
 *        that count tokens (its multiplier its sequence stride, as the engines require).
 */
inline bool headPadCopyable(const data_objects::TensorAttributes& tensor)
{
    const auto* dims = tensor.dims();
    const auto* strides = tensor.strides();
    if(dims == nullptr || strides == nullptr || dims->size() != SDPA_RANK
       || strides->size() != SDPA_RANK || tensor.virtual_()
       || hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(&tensor)
       || !headPadUnitsPerElement(tensor.data_type()).has_value() || dims->Get(HEAD_DIM_AXIS) <= 0
       || strides->Get(HEAD_DIM_AXIS) != 1)
    {
        return false;
    }
    int64_t last = 0;
    const bool packed = tensor.ragged_offset_tensor_uid().has_value();
    for(uint32_t axis = 0; axis < SDPA_RANK; ++axis)
    {
        const int64_t extent = dims->Get(axis);
        if(extent <= 0)
        {
            return false;
        }
        if(extent == 1 && (!packed || axis != SEQ_AXIS))
        {
            continue;
        }
        const auto reach = checkedProduct({extent - 1, strides->Get(axis)});
        if(strides->Get(axis) <= 0 || !reach.has_value()
           || *reach > std::numeric_limits<int64_t>::max() - last)
        {
            return false;
        }
        last += *reach;
    }
    return !packed || tensor.ragged_offset_multiplier() == strides->Get(SEQ_AXIS);
}

/// A compact token table the padded graph adds for rebased packed operands, `[0, S, ..., B * S]`
/// in int32, and the one the copies write to the workspace.
struct HeadPadTable
{
    /// The new tensor's uid in the padded graph.
    int64_t uid = 0;
    /// The graph's table the operands go through, and the operands' batch and sequence extents.
    int64_t original = 0;
    int64_t batch = 0;
    int64_t sequence = 0;
    /// Its place in the workspace (headPadPlan(); 0 in HeadPadGraph::tables).
    size_t offset = 0;
    size_t bytes = 0;
};

/**
 * @brief Whether the LSE @p stats of a graph with query @p q can be held as a scratch tensor and
 *        copied out through Q's own table: a FLOAT `[B, H, Sq, 1]` device tensor packed through
 *        Q's table (isPackedThrough()), every axis of more than one element at a positive
 *        stride, its elements distinct since the copy out writes each once.
 */
inline bool headPadStatsCopyable(const data_objects::TensorAttributes& stats,
                                 const data_objects::TensorAttributes& q)
{
    const auto* dims = stats.dims();
    const auto* strides = stats.strides();
    if(dims == nullptr || strides == nullptr || dims->size() != SDPA_RANK
       || strides->size() != SDPA_RANK || stats.data_type() != data_objects::DataType::FLOAT
       || stats.virtual_() || hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(&stats)
       || !q.ragged_offset_tensor_uid().has_value() || q.dims() == nullptr
       || q.dims()->size() != SDPA_RANK)
    {
        return false;
    }
    for(const uint32_t axis : {BATCH_AXIS, HEAD_AXIS, SEQ_AXIS})
    {
        if(dims->Get(axis) != q.dims()->Get(axis)
           || (dims->Get(axis) > 1 && strides->Get(axis) <= 0))
        {
            return false;
        }
    }
    return dims->Get(HEAD_DIM_AXIS) == 1 && isPackedThrough(stats, *q.ragged_offset_tensor_uid())
           && elementsDoNotOverlap(stats);
}

/// The padded graph's serialized bytes and the graph over them. Built in place and never moved:
/// the wrapper reads the bytes the vector holds.
struct HeadPadGraph
{
    explicit HeadPadGraph(std::vector<uint8_t> serialized)
        : bytes(std::move(serialized))
        , graph(bytes.data(), bytes.size())
    {
    }

    HeadPadGraph(const HeadPadGraph&) = delete;
    HeadPadGraph& operator=(const HeadPadGraph&) = delete;
    HeadPadGraph(HeadPadGraph&&) = delete;
    HeadPadGraph& operator=(HeadPadGraph&&) = delete;
    ~HeadPadGraph() = default;

    std::vector<uint8_t> bytes;
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph;
    /// The lengths the copies of packed operands read (headPadLengths()).
    HeadPadLengths lengths;
    /// The packed operands (of Q, K, V and O) rebased onto compact origins: those the graph
    /// gives lengths for. Their tensors in this graph go through a table of `tables`.
    std::set<int64_t> rebased;
    /// The compact token tables, each a tensor this graph adds, `[0, S, ..., B * S]`; none for a
    /// graph with nothing rebased.
    std::vector<HeadPadTable> tables;
    /// The LSE tensor's uid when this graph holds it as a scratch tensor packed through Q's
    /// compact table (Q is rebased and the graph's LSE is headPadStatsCopyable()), else
    /// NO_SEQ_LEN.
    int64_t scratchStats = NO_SEQ_LEN;

    /// A context over the padded graph for the same device and ranking as @p original.
    MatchContext contextFor(const MatchContext& original) const
    {
        return MatchContext{
            graph, original.deviceId, original.deviceProperties, original.rankingMetric};
    }
};

/**
 * @brief The padded graph for @p context (the file comment), or nullptr when an operand cannot
 *        be copied: Q, K, V and O each headPadCopyable, and O's elements distinct, since the copy
 *        back writes each once. Also nullptr unless K has Q's head dim and O has V's, for a
 *        packed graph the kernels' binding declines on the graph as given (headPadLengths()),
 *        when one packed tensor is both a query-side and a key-side operand whose lengths differ
 *        (headPadOperandLengths()), and when a rebased operand's token count passes an int32.
 */
inline std::shared_ptr<const HeadPadGraph> buildHeadPadGraph(const MatchContext& context)
{
    const auto* attributes = sdpaNode(context);
    if(attributes == nullptr)
    {
        return nullptr;
    }
    const int64_t qUid = attributes->q_tensor_uid();
    const int64_t kUid = attributes->k_tensor_uid();
    const int64_t vUid = attributes->v_tensor_uid();
    const int64_t oUid = attributes->o_tensor_uid();
    for(const int64_t uid : {qUid, kUid, vUid, oUid})
    {
        const auto* tensor = findTensor(context, uid);
        if(tensor == nullptr || !headPadCopyable(*tensor))
        {
            return nullptr;
        }
    }
    if(!elementsDoNotOverlap(*findTensor(context, oUid)))
    {
        return nullptr;
    }
    // K takes Q's head dim and O takes V's, the pairs the engines check; padding writes one width
    // over each pair, so a mismatch inside one multiple of 8 would otherwise reach the engine as
    // a match, and the copies would drop K's columns or read past O's padded rows.
    const auto headDim
        = [&context](int64_t uid) { return findTensor(context, uid)->dims()->Get(HEAD_DIM_AXIS); };
    if(headDim(kUid) != headDim(qUid) || headDim(oUid) != headDim(vUid))
    {
        return nullptr;
    }
    const int64_t widthQk = headPadWidth(headDim(qUid));
    const int64_t widthV = headPadWidth(headDim(vUid));

    // The lengths the kernels read on the graph as given are the ones the copies read, and they
    // decide which operands are rebased.
    const auto lengths = headPadLengths(context);
    if(!lengths.has_value())
    {
        return nullptr;
    }
    for(const int64_t uid : {qUid, kUid, vUid, oUid})
    {
        if(!headPadOperandLengths(*lengths, *attributes, uid).has_value())
        {
            return nullptr;
        }
    }

    // A packed operand whose sequences the graph gives lengths for is rebased: its copy goes
    // through a compact token table of its own, a new tensor, one per distinct (original table,
    // sequence extent) -- Q and K sharing a table but not a sequence extent need two. Operands
    // without lengths keep their table.
    int64_t nextUid = 0;
    for(const auto& entry : context.graph.getTensorMap())
    {
        nextUid = std::max(nextUid, entry.first + 1);
    }
    std::vector<HeadPadTable> tables;
    std::set<int64_t> rebased;
    std::map<int64_t, int64_t> compactTable;
    for(const int64_t uid : {qUid, kUid, vUid, oUid})
    {
        const auto& tensor = *findTensor(context, uid);
        if(!tensor.ragged_offset_tensor_uid().has_value()
           || *headPadOperandLengths(*lengths, *attributes, uid) == NO_SEQ_LEN)
        {
            continue;
        }
        const int64_t batch = tensor.dims()->Get(BATCH_AXIS);
        const int64_t sequence = tensor.dims()->Get(SEQ_AXIS);
        // The table's entries are int32 token indices.
        const auto tokens = checkedProduct({batch, sequence});
        if(!tokens.has_value() || *tokens > INT32_LIMIT)
        {
            return nullptr;
        }
        const int64_t original = *tensor.ragged_offset_tensor_uid();
        auto found = std::find_if(tables.begin(), tables.end(), [&](const HeadPadTable& table) {
            return table.original == original && table.sequence == sequence;
        });
        if(found == tables.end())
        {
            tables.push_back(HeadPadTable{nextUid++, original, batch, sequence});
            found = std::prev(tables.end());
        }
        rebased.insert(uid);
        compactTable[uid] = found->uid;
    }
    // The LSE of a rebased Q is written to a scratch tensor packed through Q's compact table and
    // copied out through Q's own (HeadPadGraph::scratchStats).
    int64_t scratchStats = NO_SEQ_LEN;
    if(attributes->stats_tensor_uid().has_value() && compactTable.count(qUid) != 0)
    {
        const auto* stats = findTensor(context, *attributes->stats_tensor_uid());
        if(stats != nullptr && headPadStatsCopyable(*stats, *findTensor(context, qUid)))
        {
            scratchStats = *attributes->stats_tensor_uid();
        }
    }

    std::unique_ptr<data_objects::GraphT> graph(context.graph.getGraph().UnPack());
    for(auto& tensor : graph->tensors)
    {
        if(tensor == nullptr)
        {
            continue;
        }
        const int64_t uid = tensor->uid;
        if(scratchStats != NO_SEQ_LEN && uid == scratchStats)
        {
            // Token-major, one float per (token, head): batch S*H, head 1, sequence H, and
            // offsets counting tokens through Q's compact table.
            const int64_t heads = tensor->dims.at(HEAD_AXIS);
            const int64_t sequence = tensor->dims.at(SEQ_AXIS);
            tensor->strides = {sequence * heads, 1, heads, 1};
            tensor->alignment = HEAD_PAD_STATS_DECLARED_ALIGNMENT;
            tensor->ragged_offset_tensor_uid = compactTable.at(qUid);
            tensor->ragged_offset_multiplier = heads;
            continue;
        }
        if(uid != qUid && uid != kUid && uid != vUid && uid != oUid)
        {
            continue;
        }
        // Q and K share D, V and O share Dv; a tensor bound to both (K and V the same uid) is
        // copied at the wider.
        int64_t width = (uid == qUid || uid == kUid) ? widthQk : widthV;
        if((uid == qUid || uid == kUid) && (uid == vUid || uid == oUid))
        {
            width = std::max(widthQk, widthV);
        }
        const int64_t heads = tensor->dims.at(HEAD_AXIS);
        const int64_t sequence = tensor->dims.at(SEQ_AXIS);
        const auto layout = headPadLayout(heads, sequence, width);
        tensor->dims.at(HEAD_DIM_AXIS) = width;
        tensor->strides = {layout.batch, layout.head, layout.sequence, 1};
        tensor->alignment = HEAD_PAD_DECLARED_ALIGNMENT;
        if(tensor->ragged_offset_tensor_uid.has_value())
        {
            tensor->ragged_offset_multiplier = layout.sequence;
        }
        if(const auto table = compactTable.find(uid); table != compactTable.end())
        {
            tensor->ragged_offset_tensor_uid = table->second;
        }
    }
    for(const auto& table : tables)
    {
        auto tensor = std::make_unique<data_objects::TensorAttributesT>();
        tensor->uid = table.uid;
        tensor->name = "head_pad_compact_offsets_" + std::to_string(table.uid);
        tensor->data_type = data_objects::DataType::INT32;
        tensor->dims = {table.batch + 1, 1, 1, 1};
        tensor->strides = {1, 1, 1, 1};
        tensor->alignment = HEAD_PAD_DECLARED_ALIGNMENT;
        graph->tensors.push_back(std::move(tensor));
    }
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(data_objects::Graph::Pack(builder, graph.get()));
    auto padded = std::make_shared<HeadPadGraph>(std::vector<uint8_t>(
        builder.GetBufferPointer(), builder.GetBufferPointer() + builder.GetSize()));
    if(!padded->graph.isValid())
    {
        return nullptr;
    }
    padded->lengths = *lengths;
    padded->rebased = std::move(rebased);
    padded->tables = std::move(tables);
    padded->scratchStats = scratchStats;
    return padded;
}

/**
 * @brief buildHeadPadGraph(), remembered for the last graph this thread padded: a match calls
 *        the kernel matcher once per candidate object, each with the same graph. Keyed by the
 *        graph's serialized bytes, so a different graph at the same address is never served
 *        another's padding; a graph without bytes is padded every time.
 */
inline std::shared_ptr<const HeadPadGraph> headPadGraph(const MatchContext& context)
{
    struct Entry
    {
        std::vector<uint8_t> key;
        std::shared_ptr<const HeadPadGraph> padded;
    };
    thread_local Entry s_last;
    const auto view = context.graph.bytes();
    if(view.data == nullptr || view.size == 0)
    {
        return buildHeadPadGraph(context);
    }
    if(s_last.padded != nullptr && s_last.key.size() == view.size
       && std::memcmp(s_last.key.data(), view.data, view.size) == 0)
    {
        return s_last.padded;
    }
    auto padded = buildHeadPadGraph(context);
    s_last.key.assign(view.data, view.data + view.size);
    s_last.padded = padded;
    return padded;
}

/// Whether @p attention's shard ships the copy's object (helperPackPath()).
inline bool hasHeadPadKernel(const KernelDefinition& attention)
{
    return hasHelperPack(attention, HEAD_PAD_OP, HEAD_PAD_PACK_FILE);
}

/**
 * @brief An engine's graph match, kernel match and score, padding the graphs needsHeadPad()
 *        names and passing every other graph through untouched.
 */
template <std::optional<BoundTokens> (*GRAPH_MATCH)(const MatchContext&),
          bool (*KERNEL_MATCH)(const MatchContext&, const BoundTokens&, const KernelDefinition&),
          double (*SCORE)(const MatchContext&, const BoundTokens&, const KernelDefinition&)>
struct HeadPadMatchers
{
    static std::optional<BoundTokens> graphMatches(const MatchContext& context)
    {
        if(!needsHeadPad(context))
        {
            return GRAPH_MATCH(context);
        }
        const auto padded = headPadGraph(context);
        if(padded == nullptr)
        {
            return std::nullopt;
        }
        return GRAPH_MATCH(padded->contextFor(context));
    }

    /// A padded graph also needs the copy's object in the candidate's shard.
    static bool kernelMatches(const MatchContext& context,
                              const BoundTokens& bound,
                              const KernelDefinition& kernel)
    {
        if(!needsHeadPad(context))
        {
            return KERNEL_MATCH(context, bound, kernel);
        }
        const auto padded = headPadGraph(context);
        return padded != nullptr && hasHeadPadKernel(kernel)
               && KERNEL_MATCH(padded->contextFor(context), bound, kernel);
    }

    static double
        score(const MatchContext& context, const BoundTokens& bound, const KernelDefinition& kernel)
    {
        if(!needsHeadPad(context))
        {
            return SCORE(context, bound, kernel);
        }
        const auto padded = headPadGraph(context);
        if(padded == nullptr)
        {
            return SCORE(context, bound, kernel);
        }
        return SCORE(padded->contextFor(context), bound, kernel);
    }
};

/// The copy's argument list, as `gen_sdpa_head_pad.py` asserts it: the source, destination,
/// token-table, per-batch-length and compact-token-table-output pointers; batch, heads, sequence,
/// units copied, units per destination row, compact sides (i32); the source's and the
/// destination's (batch, head, sequence) strides in units (i64).
inline const std::vector<KernelArgument>& headPadSignature()
{
    static const std::vector<KernelArgument> s_signature{{"global_buffer", 8, 0, ""},
                                                         {"global_buffer", 8, 8, ""},
                                                         {"global_buffer", 8, 16, ""},
                                                         {"global_buffer", 8, 24, ""},
                                                         {"global_buffer", 8, 32, ""},
                                                         {"by_value", 4, 40, ""},
                                                         {"by_value", 4, 44, ""},
                                                         {"by_value", 4, 48, ""},
                                                         {"by_value", 4, 52, ""},
                                                         {"by_value", 4, 56, ""},
                                                         {"by_value", 4, 60, ""},
                                                         {"by_value", 8, 64, ""},
                                                         {"by_value", 8, 72, ""},
                                                         {"by_value", 8, 80, ""},
                                                         {"by_value", 8, 88, ""},
                                                         {"by_value", 8, 96, ""},
                                                         {"by_value", 8, 104, ""}};
    return s_signature;
}

/// One operand's copy: the graph tensor, its padded copy's place in the workspace, and the
/// copy's extents and strides in 16-bit units.
struct HeadPadOperand
{
    int64_t uid = 0;
    /// The token table a packed operand goes through (the graph's own, even for a rebased
    /// operand: the copy reads the graph tensor's sequences through it), or NO_SEQ_LEN.
    int64_t rowOffset = NO_SEQ_LEN;
    /// The per-batch lengths a packed operand's sequences take (headPadOperandLengths()), or
    /// NO_SEQ_LEN.
    int64_t lengths = NO_SEQ_LEN;
    int32_t batch = 0;
    int32_t heads = 0;
    int32_t sequence = 0;
    /// The graph's head dim, and the padded copy's width, in units.
    int32_t units = 0;
    int32_t paddedUnits = 0;
    /// The graph tensor's (batch, head, sequence) strides, in units (0 on an axis of one).
    int64_t strideBatch = 0;
    int64_t strideHead = 0;
    int64_t strideSequence = 0;
    /// The padded copy's.
    HeadPadLayout padded;
    size_t offset = 0;
    size_t bytes = 0;
    /// The copy's grid: one thread per destination unit of a (batch, head) plane's rows, and
    /// one grid row per plane.
    int64_t planeThreads = 0;
    int64_t planes = 0;
    /// The copy's `compact` argument: HEAD_PAD_COMPACT_SOURCE or _DESTINATION for the
    /// workspace side of a rebased operand, else 0.
    int32_t compact = 0;
    /// The index in HeadPadPlan::tables of the compact token table this copy writes, or -1: one
    /// copy per table, the first operand through it.
    int32_t tableIndex = -1;
};

/// Where everything the padded launch uses sits: each distinct operand's padded copy, O's, the
/// LSE scratch of a rebased Q, each compact token table, then the engine's own workspace, last,
/// where the engine would have it alone (its zero bias, for one, stays at the workspace's end).
struct HeadPadPlan
{
    std::vector<HeadPadOperand> inputs;
    HeadPadOperand output;
    /// The LSE's copy out of its scratch, when Q is rebased and the graph has an LSE
    /// (HeadPadGraph::scratchStats).
    std::optional<HeadPadOperand> stats;
    std::vector<HeadPadTable> tables;
    size_t innerOffset = 0;
    size_t innerBytes = 0;
    size_t totalBytes = 0;
};

inline size_t headPadAlignUp(size_t bytes)
{
    return (bytes + HEAD_PAD_BUFFER_ALIGN - 1) / HEAD_PAD_BUFFER_ALIGN * HEAD_PAD_BUFFER_ALIGN;
}

/// The index in @p padded's tables of the compact table @p tensor (a rebased operand's graph
/// tensor) goes through.
inline std::optional<size_t> headPadTableIndex(const HeadPadGraph& padded,
                                               const data_objects::TensorAttributes& tensor)
{
    if(!tensor.ragged_offset_tensor_uid().has_value())
    {
        return std::nullopt;
    }
    for(size_t index = 0; index < padded.tables.size(); ++index)
    {
        if(padded.tables[index].original == *tensor.ragged_offset_tensor_uid()
           && padded.tables[index].sequence == tensor.dims()->Get(SEQ_AXIS))
        {
            return index;
        }
    }
    return std::nullopt;
}

/**
 * @brief The plan for @p context's padded launch, @p padded its buildHeadPadGraph(), with
 *        @p innerBytes of the engine's own workspace after the copies.
 * @throws HipdnnPluginException when an extent passes the copy's 32-bit arguments or a size
 *         overflows.
 */
inline HeadPadPlan
    headPadPlan(const MatchContext& context, const HeadPadGraph& padded, size_t innerBytes)
{
    const auto* attributes = sdpaNode(context);
    if(attributes == nullptr)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                       "flydsl_sdpa head pad: not an SDPA graph");
    }
    const int64_t qUid = attributes->q_tensor_uid();
    const int64_t kUid = attributes->k_tensor_uid();
    const int64_t vUid = attributes->v_tensor_uid();
    const int64_t oUid = attributes->o_tensor_uid();
    const int64_t widthQk = headPadWidth(findTensor(context, qUid)->dims()->Get(HEAD_DIM_AXIS));
    const int64_t widthV = headPadWidth(findTensor(context, vUid)->dims()->Get(HEAD_DIM_AXIS));

    HeadPadPlan plan;
    plan.innerBytes = innerBytes;
    size_t offset = 0;
    const auto operand = [&](int64_t uid) {
        const auto& tensor = *findTensor(context, uid);
        const auto* dims = tensor.dims();
        const auto* strides = tensor.strides();
        const int64_t units = *headPadUnitsPerElement(tensor.data_type());
        int64_t width = (uid == qUid || uid == kUid) ? widthQk : widthV;
        if((uid == qUid || uid == kUid) && (uid == vUid || uid == oUid))
        {
            width = std::max(widthQk, widthV);
        }
        const bool packed = tensor.ragged_offset_tensor_uid().has_value();
        const auto stride = [&](uint32_t axis) {
            return dims->Get(axis) == 1 && (!packed || axis != SEQ_AXIS)
                       ? int64_t{0}
                       : strides->Get(axis) * units;
        };
        HeadPadOperand result;
        result.uid = uid;
        result.rowOffset = tensor.ragged_offset_tensor_uid().value_or(NO_SEQ_LEN);
        // buildHeadPadGraph() has declined an operand in two roles whose lengths differ.
        result.lengths
            = headPadOperandLengths(padded.lengths, *attributes, uid).value_or(NO_SEQ_LEN);
        const int64_t heads = dims->Get(HEAD_AXIS);
        const int64_t sequence = dims->Get(SEQ_AXIS);
        const auto elements = checkedProduct({dims->Get(BATCH_AXIS), sequence, heads, width});
        const auto planeThreads = checkedProduct({sequence, width * units});
        const auto planes = checkedProduct({dims->Get(BATCH_AXIS), heads});
        const auto layout = headPadLayout(heads, sequence, width);
        if(!elements.has_value() || !planeThreads.has_value() || !planes.has_value()
           || *planeThreads > INT32_LIMIT || *planes > INT32_LIMIT
           || *elements > std::numeric_limits<int64_t>::max() / (2 * units))
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "flydsl_sdpa head pad: an operand is too large for the padding copy");
        }
        result.batch = static_cast<int32_t>(dims->Get(BATCH_AXIS));
        result.heads = static_cast<int32_t>(heads);
        result.sequence = static_cast<int32_t>(sequence);
        result.units = static_cast<int32_t>(dims->Get(HEAD_DIM_AXIS) * units);
        result.paddedUnits = static_cast<int32_t>(width * units);
        result.strideBatch = stride(BATCH_AXIS);
        result.strideHead = stride(HEAD_AXIS);
        result.strideSequence = stride(SEQ_AXIS);
        result.padded
            = HeadPadLayout{layout.batch * units, layout.head * units, layout.sequence * units};
        result.bytes = static_cast<size_t>(*elements * 2 * units);
        result.planeThreads = *planeThreads;
        result.planes = *planes;
        return result;
    };
    std::vector<bool> claimed(padded.tables.size(), false);
    for(const int64_t uid : {qUid, kUid, vUid})
    {
        bool seen = false;
        for(const auto& input : plan.inputs)
        {
            seen = seen || input.uid == uid;
        }
        if(seen)
        {
            continue;
        }
        auto input = operand(uid);
        input.offset = offset;
        offset = headPadAlignUp(offset + input.bytes);
        if(padded.rebased.count(uid) != 0)
        {
            // The copy in gathers each sequence to b * S; the first copy through a table
            // writes it.
            input.compact = HEAD_PAD_COMPACT_DESTINATION;
            const auto index = headPadTableIndex(padded, *findTensor(context, uid));
            if(index.has_value() && !claimed[*index])
            {
                claimed[*index] = true;
                input.tableIndex = static_cast<int32_t>(*index);
            }
        }
        plan.inputs.push_back(input);
    }
    plan.output = operand(oUid);
    plan.output.offset = offset;
    if(padded.rebased.count(oUid) != 0)
    {
        plan.output.compact = HEAD_PAD_COMPACT_SOURCE;
    }
    // Only O's own columns are copied back.
    plan.output.planeThreads
        = plan.output.planeThreads / plan.output.paddedUnits * plan.output.units;
    offset = headPadAlignUp(offset + plan.output.bytes);

    // The LSE of a rebased Q: a scratch tensor of one float per (token, head), token-major at
    // b * Sq + s, copied out through Q's own table.
    if(padded.scratchStats != NO_SEQ_LEN)
    {
        const auto& tensor = *findTensor(context, padded.scratchStats);
        const auto* dims = tensor.dims();
        const auto* strides = tensor.strides();
        constexpr int64_t units = 2;
        const int64_t batch = dims->Get(BATCH_AXIS);
        const int64_t heads = dims->Get(HEAD_AXIS);
        const int64_t sequence = dims->Get(SEQ_AXIS);
        const auto elements = checkedProduct({batch, sequence, heads});
        const auto planeThreads = checkedProduct({sequence, units});
        const auto planes = checkedProduct({batch, heads});
        if(!elements.has_value() || !planeThreads.has_value() || !planes.has_value()
           || *planeThreads > INT32_LIMIT || *planes > INT32_LIMIT
           || *elements > std::numeric_limits<int64_t>::max() / (2 * units))
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "flydsl_sdpa head pad: the LSE is too large for the padding copy");
        }
        // The LSE is packed: its sequence stride is real, every other axis of one has none.
        const auto stride = [&](uint32_t axis) {
            return dims->Get(axis) == 1 && axis != SEQ_AXIS ? int64_t{0}
                                                            : strides->Get(axis) * units;
        };
        HeadPadOperand stats;
        stats.uid = padded.scratchStats;
        stats.rowOffset = tensor.ragged_offset_tensor_uid().value_or(NO_SEQ_LEN);
        stats.lengths = padded.lengths.query;
        stats.batch = static_cast<int32_t>(batch);
        stats.heads = static_cast<int32_t>(heads);
        stats.sequence = static_cast<int32_t>(sequence);
        stats.units = static_cast<int32_t>(units);
        stats.paddedUnits = static_cast<int32_t>(units);
        stats.strideBatch = stride(BATCH_AXIS);
        stats.strideHead = stride(HEAD_AXIS);
        stats.strideSequence = stride(SEQ_AXIS);
        stats.padded = HeadPadLayout{sequence * heads * units, units, heads * units};
        stats.offset = offset;
        stats.bytes = static_cast<size_t>(*elements * 2 * units);
        stats.planeThreads = *planeThreads;
        stats.planes = *planes;
        stats.compact = HEAD_PAD_COMPACT_SOURCE;
        offset = headPadAlignUp(offset + stats.bytes);
        plan.stats = stats;
    }

    // The compact token tables, B + 1 int32 entries each; every one has a copy that writes it.
    for(size_t index = 0; index < padded.tables.size(); ++index)
    {
        if(!claimed[index])
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "flydsl_sdpa head pad: a compact token table has no copy to write it");
        }
        auto table = padded.tables[index];
        table.bytes = static_cast<size_t>(table.batch + 1) * sizeof(int32_t);
        table.offset = offset;
        offset = headPadAlignUp(offset + table.bytes);
        plan.tables.push_back(table);
    }
    plan.innerOffset = offset;
    plan.totalBytes = plan.innerOffset + innerBytes;
    return plan;
}

/// The engine's prepared dispatch for a padded graph, with the copies around it.
class HeadPadPrepared : public PreparedDispatch
{
public:
    HeadPadPrepared(std::unique_ptr<PreparedDispatch> inner,
                    HeadPadPlan plan,
                    std::vector<IngestorKernelCode> copies)
        : _inner(std::move(inner))
        , _plan(std::move(plan))
        , _copies(std::move(copies))
    {
    }

    const PreparedDispatch& inner() const
    {
        return *_inner;
    }

    const HeadPadPlan& plan() const
    {
        return _plan;
    }

    /// The copy for each of the plan's inputs, in order, then O's, then the LSE's when the plan
    /// has one.
    const IngestorKernelCode& copy(size_t index) const
    {
        return _copies.at(index);
    }

private:
    std::unique_ptr<PreparedDispatch> _inner;
    HeadPadPlan _plan;
    std::vector<IngestorKernelCode> _copies;
};

/**
 * @brief An engine's dispatch handler, padding the graphs needsHeadPad() names. Every other
 *        graph's workspace, prepared dispatch and launch are the engine handler's own.
 */
class HeadPadDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    /// @param inner, @param kpackLoader Must outlive this handler; all are process-lifetime.
    HeadPadDispatchHandler(const IKernelDispatchHandler<Handle>& inner,
                           const compilation::KpackKernelLoader& kpackLoader)
        : _inner(inner)
        , _kpackLoader(kpackLoader)
    {
    }

    std::set<std::string> launchNames() const override
    {
        return _inner.launchNames();
    }

    size_t workspaceBytes(const MatchContext& context,
                          const BoundTokens& bound,
                          const KernelDefinition& kernel) const override
    {
        if(!needsHeadPad(context))
        {
            return _inner.workspaceBytes(context, bound, kernel);
        }
        const auto padded = requirePadded(context);
        return headPadPlan(context,
                           *padded,
                           _inner.workspaceBytes(padded->contextFor(context), bound, kernel))
            .totalBytes;
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& bound,
                                              const KernelDefinition& kernel) const override
    {
        if(!needsHeadPad(context))
        {
            return _inner.prepare(context, bound, kernel);
        }
        const auto padded = requirePadded(context);
        const auto paddedContext = padded->contextFor(context);
        auto inner = _inner.prepare(paddedContext, bound, kernel);
        auto plan
            = headPadPlan(context, *padded, _inner.workspaceBytes(paddedContext, bound, kernel));

        const auto helper = helperKernel(
            kernel,
            std::string(hipdnn_plugin_sdk::stripArchFeatures(context.deviceProperties.gcnArchName)),
            HEAD_PAD_OP,
            HEAD_PAD_PACK_FILE,
            HEAD_PAD_THREADS_FIELD,
            "flydsl_sdpa head pad");
        const int64_t threads = helper.getIntMetadata(std::string(HEAD_PAD_THREADS_FIELD));
        if(threads <= 0)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "flydsl_sdpa head pad: the padding copy declares no workgroup size");
        }
        std::vector<IngestorKernelCode> copies;
        const auto addCopy = [&](const HeadPadOperand& operand) {
            const int64_t gridX = (operand.planeThreads + threads - 1) / threads;
            if(!launchAxisFits(gridX, threads))
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                    "flydsl_sdpa head pad: the padding copy's launch exceeds the launch limits");
            }
            auto code = buildIngestorKernelCode(_kpackLoader, context, helper, headPadSignature());
            code.setBlockSize(static_cast<unsigned int>(threads), 1, 1);
            code.setGridSize(
                static_cast<unsigned int>(gridX), static_cast<unsigned int>(operand.planes), 1);
            copies.push_back(std::move(code));
        };
        for(const auto& input : plan.inputs)
        {
            addCopy(input);
        }
        addCopy(plan.output);
        if(plan.stats.has_value())
        {
            addCopy(*plan.stats);
        }
        return std::make_unique<HeadPadPrepared>(
            std::move(inner), std::move(plan), std::move(copies));
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* workspace) const override
    {
        const auto* padded = dynamic_cast<const HeadPadPrepared*>(&prepared);
        if(padded == nullptr)
        {
            _inner.launch(handle, prepared, deviceBuffers, numDeviceBuffers, workspace);
            return;
        }
        const auto& plan = padded->plan();
        const auto stream = handle.getStream();
        auto* base = static_cast<uint8_t*>(workspace);
        const auto rowOffset = [&](const HeadPadOperand& operand) -> void* {
            return operand.rowOffset < 0 ? nullptr
                                         : hipdnn_plugin_sdk::findDeviceBuffer(
                                               operand.rowOffset, deviceBuffers, numDeviceBuffers)
                                               .ptr;
        };
        // Read by the copy only for a packed operand.
        const auto lengths = [&](const HeadPadOperand& operand) -> void* {
            return operand.rowOffset < 0 || operand.lengths < 0
                       ? nullptr
                       : hipdnn_plugin_sdk::findDeviceBuffer(
                             operand.lengths, deviceBuffers, numDeviceBuffers)
                             .ptr;
        };

        std::vector<hipdnnPluginDeviceBuffer_t> buffers(deviceBuffers,
                                                        deviceBuffers + numDeviceBuffers);
        const auto redirect = [&buffers](int64_t uid, void* ptr) {
            for(auto& buffer : buffers)
            {
                if(buffer.uid == uid)
                {
                    buffer.ptr = ptr;
                }
            }
        };
        // The compact token table a copy writes, in the workspace; every other copy gets null.
        void* const noTable = nullptr;
        const auto tableOut = [&](const HeadPadOperand& operand) -> void* {
            return operand.tableIndex < 0
                       ? noTable
                       : base + plan.tables[static_cast<size_t>(operand.tableIndex)].offset;
        };
        // Q, K and V into the workspace, the columns past their width zeroed, the sequences of a
        // rebased operand at b * S.
        for(size_t index = 0; index < plan.inputs.size(); ++index)
        {
            const auto& input = plan.inputs[index];
            void* source
                = hipdnn_plugin_sdk::findDeviceBuffer(input.uid, deviceBuffers, numDeviceBuffers)
                      .ptr;
            void* copy = base + input.offset;
            // Changing this argument list means changing headPadSignature() with it.
            padded->copy(index).kernelForStream(stream).launch(stream,
                                                               source,
                                                               copy,
                                                               rowOffset(input),
                                                               lengths(input),
                                                               tableOut(input),
                                                               input.batch,
                                                               input.heads,
                                                               input.sequence,
                                                               input.units,
                                                               input.paddedUnits,
                                                               input.compact,
                                                               input.strideBatch,
                                                               input.strideHead,
                                                               input.strideSequence,
                                                               input.padded.batch,
                                                               input.padded.head,
                                                               input.padded.sequence);
            redirect(input.uid, copy);
        }
        void* outputCopy = base + plan.output.offset;
        redirect(plan.output.uid, outputCopy);
        void* statsCopy = nullptr;
        if(plan.stats.has_value())
        {
            statsCopy = base + plan.stats->offset;
            redirect(plan.stats->uid, statsCopy);
        }
        // The compact tables the rebased operands of the inner graph go through.
        for(const auto& table : plan.tables)
        {
            buffers.push_back(hipdnnPluginDeviceBuffer_t{table.uid, base + table.offset});
        }
        _inner.launch(handle,
                      padded->inner(),
                      buffers.data(),
                      static_cast<uint32_t>(buffers.size()),
                      base + plan.innerOffset);
        // O's own columns back, and the LSE of a rebased Q, from the workspace.
        const auto copyBack = [&](size_t index, const HeadPadOperand& operand, void* source) {
            void* destination
                = hipdnn_plugin_sdk::findDeviceBuffer(operand.uid, deviceBuffers, numDeviceBuffers)
                      .ptr;
            padded->copy(index).kernelForStream(stream).launch(stream,
                                                               source,
                                                               destination,
                                                               rowOffset(operand),
                                                               lengths(operand),
                                                               noTable,
                                                               operand.batch,
                                                               operand.heads,
                                                               operand.sequence,
                                                               operand.units,
                                                               operand.units,
                                                               operand.compact,
                                                               operand.padded.batch,
                                                               operand.padded.head,
                                                               operand.padded.sequence,
                                                               operand.strideBatch,
                                                               operand.strideHead,
                                                               operand.strideSequence);
        };
        copyBack(plan.inputs.size(), plan.output, outputCopy);
        if(plan.stats.has_value())
        {
            copyBack(plan.inputs.size() + 1, *plan.stats, statsCopy);
        }
    }

private:
    static std::shared_ptr<const HeadPadGraph> requirePadded(const MatchContext& context)
    {
        auto padded = headPadGraph(context);
        if(padded == nullptr)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "flydsl_sdpa head pad: the graph's operands cannot be padded");
        }
        return padded;
    }

    const IKernelDispatchHandler<Handle>& _inner;
    const compilation::KpackKernelLoader& _kpackLoader;
};

} // namespace hip_kernel_provider::kernel_ingestor_engine::flydsl_sdpa

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
