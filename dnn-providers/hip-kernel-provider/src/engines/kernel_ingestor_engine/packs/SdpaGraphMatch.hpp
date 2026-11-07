// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <initializer_list>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "engines/kernel_ingestor_engine/packs/PackFeatures.hpp"

/**
 * @file SdpaGraphMatch.hpp
 * @brief The graph-reading helpers every ingestor pack that serves a single SDPA-forward
 *        node shares: finding the node and its operands, the operand checks made before
 *        anything indexes an axis, the problem read from Q and K, the f32 compute-type
 *        rule for the packs that apply it, the decline-line head, and, for packs serving
 *        cuDNN's paged-K/V graphs (SEM-009), the paged-graph contract checks with one
 *        cause vocabulary.
 *
 * Pack-family rules (mask kinds, dtypes, the decode rule, kernel limits) stay with the
 * family, e.g. UnifiedAttentionGraphMatch.hpp.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::sdpa_graph
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_plugin_sdk::ingestor::MatchContext;

/// hipDNN tensor axes. The LOGICAL order is always (B, H, S, D) regardless of layout.
inline constexpr uint32_t BATCH_AXIS = 0;
inline constexpr uint32_t HEAD_AXIS = 1;
inline constexpr uint32_t SEQ_AXIS = 2;
inline constexpr uint32_t HEAD_SIZE_AXIS = 3;
inline constexpr uint32_t SDPA_RANK = 4;

/// The product of @p factors, or nullopt when it does not fit in int64_t. Every product of
/// graph-controlled extents goes through this: an unchecked one is signed overflow, and a
/// wrapped value can pass a bound it should fail. Zero and negative factors are allowed (an
/// extent of 1 contributes (1 - 1) * stride = 0), unlike pack_features::checkedProduct,
/// which refuses non-positive cost features.
inline std::optional<int64_t> checkedProduct(std::initializer_list<int64_t> factors)
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

inline const data_objects::TensorAttributes* findTensor(const MatchContext& context, int64_t uid)
{
    const auto& tensors = context.graph.getTensorMap();
    auto it = tensors.find(uid);
    return it == tensors.end() ? nullptr : it->second;
}

/// The node an SDPA pack's matchers read, or nullptr if the graph is not a single
/// SDPA-forward node.
inline const data_objects::SdpaAttributes* sdpaNode(const MatchContext& context)
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

/// Total over an UNVALIDATED graph: rank, stride/dim agreement and positive extents,
/// checked before anything indexes an axis.
inline bool isWellFormedOperand(const data_objects::TensorAttributes& tensor)
{
    const auto* dims = tensor.dims();
    const auto* strides = tensor.strides();
    if(dims == nullptr || strides == nullptr || dims->size() != SDPA_RANK
       || strides->size() != SDPA_RANK)
    {
        return false;
    }

    for(const auto dim : *dims)
    {
        if(dim <= 0)
        {
            return false;
        }
    }

    return !tensor.virtual_() && !hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(&tensor);
}

/**
 * @brief Is this tensor's memory BSHD -- token-major, head varying fastest -- and gap-free?
 *
 * A kernel that takes no stride arguments reads a differently-strided tensor as if it were
 * this one, so it must be declined. Callers must have checked isWellFormedOperand().
 *
 * Unit-extent axes are exempt: a stride multiplies an index that is always 0 when the extent
 * is 1. An expected stride too large for int64_t is one no stride can equal, so that axis
 * fails unless it is unit-extent.
 */
inline bool hasBshdStrides(const data_objects::TensorAttributes& tensor)
{
    const auto* dims = tensor.dims();
    const auto* strides = tensor.strides();

    const int64_t heads = dims->Get(HEAD_AXIS);
    const int64_t sequence = dims->Get(SEQ_AXIS);
    const int64_t headSize = dims->Get(HEAD_SIZE_AXIS);

    const auto axisOk = [&](uint32_t axis, std::optional<int64_t> expected) {
        return dims->Get(axis) == 1 || (expected.has_value() && strides->Get(axis) == *expected);
    };

    return axisOk(BATCH_AXIS, pack_features::checkedProduct({sequence, heads, headSize}))
           && axisOk(HEAD_AXIS, headSize)
           && axisOk(SEQ_AXIS, pack_features::checkedProduct({heads, headSize}))
           && axisOk(HEAD_SIZE_AXIS, 1);
}

/// The graph facts a pack's matchers and prepare() need, derived once from the tensors.
struct SdpaShape
{
    int64_t batch = 0;
    int64_t seqLenQ = 0;
    int64_t seqLenKv = 0;
    int64_t numQueryHeads = 0;
    int64_t numKvHeads = 0;
    int64_t headSize = 0;
    data_objects::DataType dataType = data_objects::DataType::UNSET;
};

/// The unified attention packs' name for it.
using UnifiedAttentionProblem = SdpaShape;

/// The graph's shape, read from Q and K. Callers must have checked both with
/// isWellFormedOperand().
inline SdpaShape problemFor(const data_objects::TensorAttributes& q,
                            const data_objects::TensorAttributes& k)
{
    SdpaShape problem;
    problem.batch = q.dims()->Get(BATCH_AXIS);
    problem.numQueryHeads = q.dims()->Get(HEAD_AXIS);
    problem.seqLenQ = q.dims()->Get(SEQ_AXIS);
    problem.headSize = q.dims()->Get(HEAD_SIZE_AXIS);
    problem.numKvHeads = k.dims()->Get(HEAD_AXIS);
    problem.seqLenKv = k.dims()->Get(SEQ_AXIS);
    problem.dataType = q.data_type();
    return problem;
}

/**
 * @brief Whether a kernel that computes the softmax, accumulates and keeps the statistics in
 *        f32 meets the node's compute type.
 *
 * f32 is at least the precision FLOAT, HALF or BFLOAT16 asks for, and UNSET leaves it to the
 * engine. DOUBLE asks for more, and an integer or FP8 compute type for a different
 * computation, so those decline. Callers must have found the node with sdpaNode().
 */
inline bool computesInFloat(const MatchContext& context)
{
    const auto computeType = context.graph.getNodeWrapper(0).computeDataType();
    return computeType == data_objects::DataType::UNSET
           || computeType == data_objects::DataType::FLOAT
           || computeType == data_objects::DataType::HALF
           || computeType == data_objects::DataType::BFLOAT16;
}

// ---------------------------------------------------------------------------
// Decline logging
//
// hipDNN reports a declined graph to its caller only as "No engine configurations available
// for the graph", so each decline logs its cause at INFO. graph_match runs on every catalog
// miss, so every piece of that work sits inside HIPDNN_PLUGIN_LOG_INFO's argument: the macro
// tests the level first, and with INFO off nothing below is called.
// ---------------------------------------------------------------------------

/// The head of a decline line: "<engine> declined the graph [<cause>]: ".
struct DeclinedBy
{
    std::string_view engine;
    std::string_view cause;
};

inline std::ostream& operator<<(std::ostream& out, const DeclinedBy& declined)
{
    return out << declined.engine << " declined the graph [" << declined.cause << "]: ";
}

/// Whether any node of the graph is an SDPA-forward node. Only such a graph hears why it
/// was declined: graph_match sees every graph the catalog misses on.
inline bool hasSdpaForwardNode(const MatchContext& context)
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

/// "Q (uid 1): dims [..], strides [..]". Null dims or strides print as [].
inline std::string describeOperand(std::string_view role,
                                   const data_objects::TensorAttributes& tensor)
{
    using hipdnn_data_sdk::utilities::vecToString;
    using hipdnn_flatbuffers_sdk::utilities::convertFlatBufferVectorToStdVector;
    std::ostringstream out;
    out << role << " (uid " << tensor.uid() << "): dims "
        << vecToString(convertFlatBufferVectorToStdVector(tensor.dims())) << ", strides "
        << vecToString(convertFlatBufferVectorToStdVector(tensor.strides()));
    return out.str();
}

// ---------------------------------------------------------------------------
// Operand checks stricter than isWellFormedOperand
// ---------------------------------------------------------------------------

/// Whether the graph asks for override shapes (admitted from engine API 1.4.0). A pack
/// that derives its launch arguments from the built shapes declines such a graph.
inline bool usesOverrideShapes(const MatchContext& context)
{
    return context.graph.getGraph().is_override_shape_enabled();
}

/// Whether every stride is non-negative. Callers must have checked isWellFormedOperand().
inline bool hasNonNegativeStrides(const data_objects::TensorAttributes& tensor)
{
    for(const auto stride : *tensor.strides())
    {
        if(stride < 0)
        {
            return false;
        }
    }
    return true;
}

/// Whether @p tensor promises a base pointer aligned to a multiple of @p bytes. The backend
/// enforces the alignment a graph declares and no more, so a kernel that assumes a wider one
/// must read the declaration: a smaller one would hand it a pointer its vector accesses
/// cannot take.
inline bool declaresAlignment(const data_objects::TensorAttributes& tensor, int64_t bytes)
{
    return tensor.alignment() >= bytes && tensor.alignment() % bytes == 0;
}

/// An INT32 device vector `[length, 1, 1, 1]` read as v[i]: well formed, non-negative
/// strides, not ragged, and a unit stride along its one non-unit axis.
inline bool isInt32Vector(const data_objects::TensorAttributes& tensor, int64_t length)
{
    if(!isWellFormedOperand(tensor) || !hasNonNegativeStrides(tensor)
       || tensor.data_type() != data_objects::DataType::INT32
       || tensor.ragged_offset_tensor_uid().has_value() || tensor.dims()->Get(BATCH_AXIS) != length
       || tensor.dims()->Get(HEAD_AXIS) != 1 || tensor.dims()->Get(SEQ_AXIS) != 1
       || tensor.dims()->Get(HEAD_SIZE_AXIS) != 1)
    {
        return false;
    }
    return length == 1 || tensor.strides()->Get(BATCH_AXIS) == 1;
}

// ---------------------------------------------------------------------------
// cuDNN paged-K/V graphs (SdpaSemanticsRegister SEM-009)
//
// The contract checks every pack serving a paged SDPA-forward graph applies, with one
// cause vocabulary. Each logs its decline through DeclinedBy with the calling engine's
// name and returns false (or nullopt); the engine keeps its own kernel limits (block
// sizes, head sizes, address ranges) and calls these in place of its own copies.
//
// Cause keys: one_sided_paging, block_size, shape, page_table, seq_len_kv, seq_len_q,
// ragged, alignment, sinks, attn_mask.
// ---------------------------------------------------------------------------

/// The paged graph pages both K and V, through page_table_k and page_table_v (which may
/// be one tensor). cuDNN also allows K or V paged alone, which no paged kernel here reads.
inline bool pagesKAndV(std::string_view engine, const data_objects::SdpaAttributes& attributes)
{
    const bool pagedK = attributes.page_table_k_tensor_uid().has_value();
    const bool pagedV = attributes.page_table_v_tensor_uid().has_value();
    if(pagedK && pagedV)
    {
        return true;
    }
    HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{engine, "one_sided_paging"})
                           << (pagedK || pagedV ? (pagedK ? "only K is paged" : "only V is paged")
                                                : "neither K nor V is paged")
                           << "; the kernels page both K and V");
    return false;
}

/// The paged K and V containers `[num_blocks, H, block_size, D]` share one block size and
/// one block count: the kernels address both with one block size and bound both page ids
/// by one block count. Both must be well formed. Their head counts and head sizes are the
/// engine's to check.
inline bool pagedContainersAgree(std::string_view engine,
                                 const data_objects::TensorAttributes& k,
                                 const data_objects::TensorAttributes& v)
{
    if(v.dims()->Get(SEQ_AXIS) != k.dims()->Get(SEQ_AXIS))
    {
        HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{engine, "block_size"})
                               << "the K and V containers differ in block size ("
                               << k.dims()->Get(SEQ_AXIS) << " and " << v.dims()->Get(SEQ_AXIS)
                               << "); the kernels address both with one");
        return false;
    }
    if(v.dims()->Get(BATCH_AXIS) != k.dims()->Get(BATCH_AXIS))
    {
        HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{engine, "shape"})
                               << "the K and V containers differ in page count ("
                               << k.dims()->Get(BATCH_AXIS) << " and " << v.dims()->Get(BATCH_AXIS)
                               << "); the kernels bound both page ids by one");
        return false;
    }
    return true;
}

/// The two page tables of a paged graph, K's and V's (the same tensor when one pages both).
struct PageTables
{
    int64_t kUid = 0;
    int64_t vUid = 0;
    const data_objects::TensorAttributes* k = nullptr;
    const data_objects::TensorAttributes* v = nullptr;
};

/// The page tables: INT32 `[B, 1, M, 1]` with a unit entry stride, not packed (no
/// ragged_offset). Separate K and V tables share one width M and one batch stride: the
/// kernels read both at one row offset. A graph that does not page both K and V is
/// declined through pagesKAndV().
inline std::optional<PageTables> pageTablesFor(std::string_view engine,
                                               const MatchContext& context,
                                               const data_objects::SdpaAttributes& attributes,
                                               int64_t batch)
{
    if(!pagesKAndV(engine, attributes))
    {
        return std::nullopt;
    }
    PageTables tables;
    tables.kUid = *attributes.page_table_k_tensor_uid();
    tables.vUid = *attributes.page_table_v_tensor_uid();
    tables.k = findTensor(context, tables.kUid);
    tables.v = findTensor(context, tables.vUid);
    for(const auto& [role, table] : {std::pair{std::string_view("page_table_k"), tables.k},
                                     std::pair{std::string_view("page_table_v"), tables.v}})
    {
        if(table == nullptr || !isWellFormedOperand(*table) || !hasNonNegativeStrides(*table))
        {
            HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{engine, "page_table"})
                                   << role
                                   << " is missing, virtual, pass-by-value, not rank 4, or has "
                                      "a non-positive extent or negative stride");
            return std::nullopt;
        }
        if(table->ragged_offset_tensor_uid().has_value())
        {
            HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{engine, "page_table"})
                                   << role
                                   << " is packed (has a ragged_offset); the kernels read B rows "
                                      "of M entries");
            return std::nullopt;
        }
        const auto* dims = table->dims();
        if(table->data_type() != data_objects::DataType::INT32 || dims->Get(BATCH_AXIS) != batch
           || dims->Get(HEAD_AXIS) != 1 || dims->Get(HEAD_SIZE_AXIS) != 1
           || (dims->Get(SEQ_AXIS) > 1 && table->strides()->Get(SEQ_AXIS) != 1))
        {
            HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{engine, "page_table"})
                                   << role
                                   << " must be a device INT32 [B, 1, M, 1] tensor with a unit "
                                      "entry stride; got "
                                   << describeOperand(role, *table));
            return std::nullopt;
        }
    }
    if(tables.vUid != tables.kUid)
    {
        if(tables.v->dims()->Get(SEQ_AXIS) != tables.k->dims()->Get(SEQ_AXIS))
        {
            HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{engine, "page_table"})
                                   << "the K and V page tables differ in width ("
                                   << tables.k->dims()->Get(SEQ_AXIS) << " and "
                                   << tables.v->dims()->Get(SEQ_AXIS)
                                   << "); the kernels read both at one row offset");
            return std::nullopt;
        }
        if(batch > 1
           && tables.v->strides()->Get(BATCH_AXIS) != tables.k->strides()->Get(BATCH_AXIS))
        {
            HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{engine, "page_table"})
                                   << "the K and V page tables differ in row stride ("
                                   << tables.k->strides()->Get(BATCH_AXIS) << " and "
                                   << tables.v->strides()->Get(BATCH_AXIS)
                                   << "); the kernels read both at one row offset");
            return std::nullopt;
        }
    }
    return tables;
}

/// seq_len_q and seq_len_kv, which a paged graph carries both of (also for a dense Q).
struct SequenceLengths
{
    int64_t kvUid = 0;
    int64_t qUid = 0;
    const data_objects::TensorAttributes* kv = nullptr;
    const data_objects::TensorAttributes* q = nullptr;
};

/// The sequence lengths: INT32 `[B, 1, 1, 1]` device vectors with a unit stride. Checks
/// seq_len_kv first, then seq_len_q.
inline std::optional<SequenceLengths>
    sequenceLengthsFor(std::string_view engine,
                       const MatchContext& context,
                       const data_objects::SdpaAttributes& attributes,
                       int64_t batch)
{
    const auto lengthFor = [&](std::string_view role,
                               const flatbuffers::Optional<int64_t>& uid,
                               const data_objects::TensorAttributes*& tensor,
                               int64_t& tensorUid) {
        if(!uid.has_value())
        {
            HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{engine, role}) << "a paged graph needs " << role);
            return false;
        }
        tensor = findTensor(context, *uid);
        if(tensor == nullptr || !isInt32Vector(*tensor, batch))
        {
            HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{engine, role})
                                   << role
                                   << " must be a device INT32 [B, 1, 1, 1] vector with a unit "
                                      "stride");
            return false;
        }
        tensorUid = *uid;
        return true;
    };
    SequenceLengths lengths;
    if(!lengthFor("seq_len_kv", attributes.seq_len_kv_tensor_uid(), lengths.kv, lengths.kvUid)
       || !lengthFor("seq_len_q", attributes.seq_len_q_tensor_uid(), lengths.q, lengths.qUid))
    {
        return std::nullopt;
    }
    return lengths;
}

/// A ragged Q's offsets (RFC 0014), or none for a dense Q.
struct RaggedQuery
{
    bool ragged = false;
    int64_t offsetsUid = 0;
    const data_objects::TensorAttributes* offsets = nullptr;
};

/// Q and O are both dense, or both ragged at the sequence axis through one INT32
/// `[B+1, 1, 1, 1]` offset tensor with a unit stride; paged K and V are never ragged.
/// Call with well-formed Q, O, K and V.
inline std::optional<RaggedQuery> raggedQueryFor(std::string_view engine,
                                                 const MatchContext& context,
                                                 const data_objects::TensorAttributes& q,
                                                 const data_objects::TensorAttributes& o,
                                                 const data_objects::TensorAttributes& k,
                                                 const data_objects::TensorAttributes& v,
                                                 int64_t batch)
{
    const auto qOffsets = q.ragged_offset_tensor_uid();
    const auto oOffsets = o.ragged_offset_tensor_uid();
    if(qOffsets.has_value() != oOffsets.has_value()
       || (qOffsets.has_value() && *qOffsets != *oOffsets))
    {
        HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{engine, "ragged"})
                               << "Q and O must both be dense or both ragged through the same "
                                  "ragged_offset tensor");
        return std::nullopt;
    }
    if(k.ragged_offset_tensor_uid().has_value() || v.ragged_offset_tensor_uid().has_value())
    {
        HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{engine, "ragged"})
                               << "ragged K/V containers are not supported; paged K and V are "
                                  "addressed through their page tables");
        return std::nullopt;
    }
    RaggedQuery query;
    if(!qOffsets.has_value())
    {
        return query;
    }
    query.ragged = true;
    query.offsetsUid = *qOffsets;
    query.offsets = findTensor(context, query.offsetsUid);
    if(query.offsets == nullptr || !isInt32Vector(*query.offsets, batch + 1))
    {
        HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{engine, "ragged"})
                               << "the ragged offsets must be a device INT32 [B+1, 1, 1, 1] "
                                  "vector with a unit stride");
        return std::nullopt;
    }
    return query;
}

/// A ragged Q's or O's offsets count tokens: its ragged_offset_multiplier is its token
/// (sequence-axis) stride. The kernels take rows per sequence and query-block starts from
/// the stored offsets themselves and scale only a sequence's base; RFC 0014 also allows
/// other units, which they would misread.
inline bool countsOffsetsInTokens(std::string_view engine,
                                  std::string_view role,
                                  const data_objects::TensorAttributes& tensor)
{
    if(tensor.ragged_offset_multiplier() == tensor.strides()->Get(SEQ_AXIS))
    {
        return true;
    }
    HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{engine, "ragged"})
                           << describeOperand(role, tensor) << " with ragged_offset_multiplier "
                           << tensor.ragged_offset_multiplier()
                           << ": the multiplier must equal the sequence-axis (token) stride; "
                              "the kernels read stored offsets as token counts");
    return false;
}

/// An operand and the name a decline gives it.
using NamedOperand = std::pair<std::string_view, const data_objects::TensorAttributes*>;

/**
 * @brief The declared device-pointer alignments the kernels need (engine API 1.3.0 lets a
 *        graph declare them; 16 bytes unless set).
 *
 * The kernels load Q/K/V/O as @p vectorBytes vectors and the INT32 page tables, sequence
 * lengths and ragged offsets as 4-byte words, so a graph declaring less for any of them is
 * declined here, where hipDNN can still pick another engine, rather than at execute.
 * Null operands (absent ones) are skipped.
 */
inline bool declaresServedAlignment(std::string_view engine,
                                    std::initializer_list<NamedOperand> vectorOperands,
                                    int64_t vectorBytes,
                                    std::initializer_list<NamedOperand> wordOperands)
{
    for(const auto& [role, tensor] : vectorOperands)
    {
        if(tensor != nullptr && !declaresAlignment(*tensor, vectorBytes))
        {
            HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{engine, "alignment"})
                                   << role << " declares a " << tensor->alignment()
                                   << "-byte alignment; the kernels load Q, K, V and O as "
                                   << vectorBytes << "-byte vectors, so each must declare a "
                                   << "multiple of " << vectorBytes << " bytes");
            return false;
        }
    }
    for(const auto& [role, tensor] : wordOperands)
    {
        if(tensor != nullptr && !declaresAlignment(*tensor, int64_t{sizeof(int32_t)}))
        {
            HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{engine, "alignment"})
                                   << role << " declares a " << tensor->alignment()
                                   << "-byte alignment; the kernels read the INT32 page tables, "
                                      "sequence lengths and ragged offsets as 4-byte words");
            return false;
        }
    }
    return true;
}

/// The graph's attention sink, if it has one.
struct SinkOperand
{
    bool present = false;
    int64_t uid = 0;
    const data_objects::TensorAttributes* tensor = nullptr;
    /// FLOAT logits (cuDNN's form); otherwise the logits are in Q's data type.
    bool f32 = false;
};

/**
 * @brief The attention sink (SEM-022): sink_token `[1, Hq, 1, 1]`, one logit per query
 *        head, which the kernels load at index `query head`.
 *
 * Served as FLOAT (cuDNN's form) or in Q's data type, with a unit head stride; any other
 * data type is declined (cause `sinks`). @p minAlignmentBytes is the calling engine's sink
 * load width: a sink declaring less is declined with cause `alignment`. Absent sink:
 * present = false.
 */
inline std::optional<SinkOperand> sinkFor(std::string_view engine,
                                          const MatchContext& context,
                                          const data_objects::SdpaAttributes& attributes,
                                          int64_t numQueryHeads,
                                          data_objects::DataType queryType,
                                          int64_t minAlignmentBytes)
{
    SinkOperand sink;
    const auto uid = attributes.sink_token_tensor_uid();
    if(!uid.has_value())
    {
        return sink;
    }
    sink.present = true;
    sink.uid = *uid;
    sink.tensor = findTensor(context, sink.uid);
    const auto* tensor = sink.tensor;
    if(tensor == nullptr || !isWellFormedOperand(*tensor) || !hasNonNegativeStrides(*tensor)
       || tensor->ragged_offset_tensor_uid().has_value() || tensor->dims()->Get(BATCH_AXIS) != 1
       || tensor->dims()->Get(HEAD_AXIS) != numQueryHeads || tensor->dims()->Get(SEQ_AXIS) != 1
       || tensor->dims()->Get(HEAD_SIZE_AXIS) != 1)
    {
        HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{engine, "sinks"})
                               << "sink_token must be a device tensor [1, Hq, 1, 1] (Hq = "
                               << numQueryHeads << ", one logit per query head)");
        return std::nullopt;
    }
    if(tensor->data_type() != data_objects::DataType::FLOAT && tensor->data_type() != queryType)
    {
        HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{engine, "sinks"})
                               << "sink_token's data type is neither FLOAT nor the Q data type");
        return std::nullopt;
    }
    sink.f32 = tensor->data_type() == data_objects::DataType::FLOAT;
    if(numQueryHeads > 1 && tensor->strides()->Get(HEAD_AXIS) != 1)
    {
        HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{engine, "sinks"})
                               << "the kernels read sink_token with a unit head stride");
        return std::nullopt;
    }
    if(!declaresAlignment(*tensor, minAlignmentBytes))
    {
        HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{engine, "alignment"})
                               << "sink_token declares a " << tensor->alignment()
                               << "-byte alignment; the kernels load the sink with "
                               << minAlignmentBytes << "-byte alignment");
        return std::nullopt;
    }
    return sink;
}

/// No additive attention mask (attn_mask): the paged kernels take no bias input.
inline bool takesNoAttentionMask(std::string_view engine,
                                 const data_objects::SdpaAttributes& attributes)
{
    if(!attributes.attn_mask_tensor_uid().has_value())
    {
        return true;
    }
    HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{engine, "attn_mask"})
                           << "the paged kernels take no additive attention mask");
    return false;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::sdpa_graph
