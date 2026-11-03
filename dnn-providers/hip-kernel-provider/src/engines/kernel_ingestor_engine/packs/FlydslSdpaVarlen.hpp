// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

// Redundant with the CMake gate on purpose; see FlydslSdpaRdnaNative.cpp.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <cstdint>
#include <optional>

#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "engines/kernel_ingestor_engine/packs/FlydslSdpaShared.hpp"

/**
 * @file FlydslSdpaVarlen.hpp
 * @brief Variable sequence lengths as every FlyDSL SDPA engine reads them from a graph: per-batch
 *        lengths under `padding_mask` (padding), and ragged Q/K/V/O (packed THD).
 *
 * One function, servableVarlen(), decides what of this a graph asks for and whether it is
 * served, and both families' graph matchers call it, so the varlen rules live in one place and
 * a varlen change does not touch the matchers' other checks.
 *
 * Served: padding, cuDNN's spelling of it -- `padding_mask` with both `seq_len_q` and `seq_len_kv`
 * -- over dense Q/K/V/O whose sequence extent is each batch's longest sequence. Which engines run
 * it is each kernel matcher's to say (FlydslSdpaVarlen's binding is in the bound tokens).
 * Declined: ragged (THD) operands, and either length or the switch alone.
 *
 * Packed (THD) sequences are servablePackedVarlen()'s, which a family whose kernels read them
 * calls instead for a graph with a ragged operand (hasRaggedOperand()).
 */
namespace hip_kernel_provider::kernel_ingestor_engine::flydsl_sdpa
{

/// The uid a varlen token holds when the graph has no such tensor.
inline constexpr int64_t NO_SEQ_LEN = -1;

/// What a graph's variable-length fields bind: the per-batch length tensors under padding, or
/// NO_SEQ_LEN for a dense graph.
struct SdpaVarlen
{
    int64_t seqLenQ = NO_SEQ_LEN;
    int64_t seqLenKv = NO_SEQ_LEN;
    /// Packed (THD) sequences: the ragged-offset tables Q/O and K/V are packed through, or
    /// NO_SEQ_LEN (servablePackedVarlen).
    int64_t raggedOffsetQ = NO_SEQ_LEN;
    int64_t raggedOffsetKv = NO_SEQ_LEN;

    bool padded() const
    {
        return seqLenQ != NO_SEQ_LEN;
    }

    bool packed() const
    {
        return raggedOffsetQ != NO_SEQ_LEN;
    }
};

/**
 * @brief Whether a packed graph's query lengths are its packed sequences' own: `seq_len_kv`
 *        without `seq_len_q` (PyTorch's `seqused_k`, SEM-028). The binding then names Q's
 *        ragged-offset table as the query lengths (@p seqLenQ == @p raggedOffsetQ), which no
 *        length tensor can be (a length tensor is `[B, 1, 1, 1]`, a table `[B + 1, 1, 1, 1]`).
 */
inline bool queryLengthsFromOffsets(int64_t seqLenQ, int64_t raggedOffsetQ)
{
    return seqLenQ != NO_SEQ_LEN && seqLenQ == raggedOffsetQ;
}

/**
 * @brief The per-batch query lengths a padded launch reads, from the buffer bound to
 *        `seqLenQ`: the `seq_len_q` tensor itself, or, when the lengths are the packed
 *        sequences' own (queryLengthsFromOffsets()), Q's offset table one entry on.
 *
 * Every kernel that reads lengths under packing clamps batch b's length to its slot,
 * `offset[b + 1] - offset[b]` (RDNA prefill modification 29, CDNA modification 34). The
 * table's entry b + 1 is at least that whenever `offset[b] >= 0`, which hipDNN's ragged
 * reference requires, so the clamped length is the slot: the sequence's own token count, as
 * the references derive `seq_len_q` from the offsets (RaggedSdpaDensify.hpp). No kernel or
 * object changes, and no workspace.
 */
inline void* queryLengthsArgument(void* buffer, int64_t seqLenQ, int64_t raggedOffsetQ)
{
    if(buffer == nullptr || !queryLengthsFromOffsets(seqLenQ, raggedOffsetQ))
    {
        return buffer;
    }
    return static_cast<int32_t*>(buffer) + 1;
}

/**
 * @brief Whether @p uid names a per-batch length tensor the kernels can read: INT32 `[B, 1, 1, 1]`
 *        (cuDNN's form), one length per batch at consecutive int32s, a plain device tensor at a
 *        4-byte-aligned base.
 *
 * The kernels load batch b's length from `base[b]` as one int32, so INT64 lengths, a batch
 * stride other than 1, and a host (pass-by-value) array all decline. The values are the
 * caller's to keep in [0, S] (S the graph's sequence extent), as cuDNN leaves them; the kernels
 * do not check them.
 */
inline bool isServableSeqLenTensor(const MatchContext& context, int64_t uid, int64_t batch)
{
    const auto* tensor = findTensor(context, uid);
    if(tensor == nullptr || tensor->dims() == nullptr || tensor->strides() == nullptr
       || tensor->dims()->size() != SDPA_RANK || tensor->strides()->size() != SDPA_RANK
       || tensor->data_type() != data_objects::DataType::INT32 || tensor->virtual_()
       || hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(tensor)
       || tensor->ragged_offset_tensor_uid().has_value()
       || !declaresAlignment(*tensor, sizeof(int32_t)))
    {
        return false;
    }
    const auto* dims = tensor->dims();
    if(dims->Get(BATCH_AXIS) != batch || dims->Get(HEAD_AXIS) != 1 || dims->Get(SEQ_AXIS) != 1
       || dims->Get(HEAD_DIM_AXIS) != 1)
    {
        return false;
    }
    return batch == 1 || tensor->strides()->Get(BATCH_AXIS) == 1;
}

/// Whether @p tensor is ragged: its sequences packed end to end (THD) through an offset tensor.
inline bool isRaggedTensor(const data_objects::TensorAttributes& tensor)
{
    return tensor.ragged_offset_tensor_uid().has_value();
}

/**
 * @brief The graph's variable-length fields, or nullopt for a graph that asks for a form the
 *        kernels do not serve.
 *
 * Padding is `padding_mask` with both length tensors, each isServableSeqLenTensor for the
 * graph's @p batch. Declined, each a wrong answer rather than an error if it were ignored:
 * - a ragged (THD) Q, K, V or O, whose packed sequences a dense kernel would read as one batch;
 * - a length tensor without `padding_mask`, or `padding_mask` without both: cuDNN applies the
 *   lengths only under the switch, and the switch needs both;
 * - a length tensor the kernels cannot read.
 *
 * A Q, K, V or O the graph does not name is left to the caller, which declines it.
 */
inline std::optional<SdpaVarlen> servableVarlen(const MatchContext& context,
                                                const data_objects::SdpaAttributes& attributes,
                                                int64_t batch)
{
    for(const int64_t uid : {attributes.q_tensor_uid(),
                             attributes.k_tensor_uid(),
                             attributes.v_tensor_uid(),
                             attributes.o_tensor_uid()})
    {
        const auto* tensor = findTensor(context, uid);
        if(tensor != nullptr && isRaggedTensor(*tensor))
        {
            return std::nullopt;
        }
    }
    const bool hasSeqLenQ = attributes.seq_len_q_tensor_uid().has_value();
    const bool hasSeqLenKv = attributes.seq_len_kv_tensor_uid().has_value();
    if(!attributes.padding_mask())
    {
        if(hasSeqLenQ || hasSeqLenKv)
        {
            return std::nullopt;
        }
        return SdpaVarlen{};
    }
    if(!hasSeqLenQ || !hasSeqLenKv
       || !isServableSeqLenTensor(context, *attributes.seq_len_q_tensor_uid(), batch)
       || !isServableSeqLenTensor(context, *attributes.seq_len_kv_tensor_uid(), batch))
    {
        return std::nullopt;
    }
    return SdpaVarlen{*attributes.seq_len_q_tensor_uid(), *attributes.seq_len_kv_tensor_uid()};
}

/**
 * @brief servableVarlen(), read as hipDNN's references read a graph: both length tensors
 *        without `padding_mask` are padding too (the references' SEM-008, which keeps
 *        hipDNN's golden bundles valid; cuDNN would refuse such a graph).
 *
 * Each length must still be one the kernels read, no operand may be ragged, and one length
 * alone declines. The RDNA and CDNA engines both read dense operands' lengths this way.
 */
inline std::optional<SdpaVarlen> servableVarlenAsReferenceReads(
    const MatchContext& context, const data_objects::SdpaAttributes& attributes, int64_t batch)
{
    // Under padding_mask, servableVarlen's answer stands.
    auto varlen = servableVarlen(context, attributes, batch);
    if(varlen.has_value() || attributes.padding_mask()
       || !attributes.seq_len_q_tensor_uid().has_value()
       || !attributes.seq_len_kv_tensor_uid().has_value())
    {
        return varlen;
    }
    for(const int64_t uid : {attributes.q_tensor_uid(),
                             attributes.k_tensor_uid(),
                             attributes.v_tensor_uid(),
                             attributes.o_tensor_uid()})
    {
        const auto* tensor = findTensor(context, uid);
        if(tensor != nullptr && isRaggedTensor(*tensor))
        {
            return std::nullopt;
        }
    }
    const int64_t seqLenQ = *attributes.seq_len_q_tensor_uid();
    const int64_t seqLenKv = *attributes.seq_len_kv_tensor_uid();
    if(!isServableSeqLenTensor(context, seqLenQ, batch)
       || !isServableSeqLenTensor(context, seqLenKv, batch))
    {
        return std::nullopt;
    }
    return SdpaVarlen{seqLenQ, seqLenKv};
}

/// Whether any of the graph's Q, K, V and O is ragged (packed through an offset table).
inline bool hasRaggedOperand(const MatchContext& context,
                             const data_objects::SdpaAttributes& attributes)
{
    for(const int64_t uid : {attributes.q_tensor_uid(),
                             attributes.k_tensor_uid(),
                             attributes.v_tensor_uid(),
                             attributes.o_tensor_uid()})
    {
        const auto* tensor = findTensor(context, uid);
        if(tensor != nullptr && isRaggedTensor(*tensor))
        {
            return true;
        }
    }
    return false;
}

/**
 * @brief Whether @p uid names a ragged-offset table the kernels can read: INT32
 *        `[B + 1, 1, 1, 1]` (RFC 0014's form), consecutive int32s, a plain device tensor at a
 *        4-byte-aligned base.
 *
 * The kernels load sequence b's bounds from `base[b]` and `base[b + 1]`, and the total from
 * `base[B]`, each one int32. The values are the caller's to keep a non-decreasing token count
 * from 0 with no sequence longer than the operand's sequence extent, as hipDNN's ragged
 * reference requires; the kernels do not check them.
 */
inline bool isServableRaggedOffsetTensor(const MatchContext& context, int64_t uid, int64_t batch)
{
    const auto* tensor = findTensor(context, uid);
    if(tensor == nullptr || tensor->dims() == nullptr || tensor->strides() == nullptr
       || tensor->dims()->size() != SDPA_RANK || tensor->strides()->size() != SDPA_RANK
       || tensor->data_type() != data_objects::DataType::INT32 || tensor->virtual_()
       || hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(tensor) || isRaggedTensor(*tensor)
       || !declaresAlignment(*tensor, sizeof(int32_t)))
    {
        return false;
    }
    const auto* dims = tensor->dims();
    return dims->Get(BATCH_AXIS) == batch + 1 && dims->Get(HEAD_AXIS) == 1
           && dims->Get(SEQ_AXIS) == 1 && dims->Get(HEAD_DIM_AXIS) == 1
           && tensor->strides()->Get(BATCH_AXIS) == 1;
}

/**
 * @brief Whether @p tensor is packed token by token through the table @p offsetUid, with
 *        offsets that count tokens.
 *
 * Its ragged offset is @p offsetUid; its `ragged_offset_multiplier` is its own sequence stride,
 * so `offset[b]` is sequence b's first token and `offset[b + 1] - offset[b]` its length, as the
 * kernels read them (AITER's and AOTriton's `cu_seqlens`); and one token's head x head-dim block
 * fits within a sequence stride (RFC 0014's token-major layout, which a BHSD tensor fails).
 * Its batch stride is not read: a packed sequence starts at its offset.
 */
inline bool isPackedThrough(const data_objects::TensorAttributes& tensor, int64_t offsetUid)
{
    const auto* dims = tensor.dims();
    const auto* strides = tensor.strides();
    if(!tensor.ragged_offset_tensor_uid().has_value()
       || *tensor.ragged_offset_tensor_uid() != offsetUid || dims == nullptr || strides == nullptr
       || dims->size() != SDPA_RANK || strides->size() != SDPA_RANK)
    {
        return false;
    }
    const int64_t sequenceStride = strides->Get(SEQ_AXIS);
    if(tensor.ragged_offset_multiplier() != sequenceStride)
    {
        return false;
    }
    // The callers have checked the strides of every non-unit axis positive, so the span is at
    // least 1 and so is a sequence stride that covers it.
    int64_t tokenSpan = 1;
    for(const uint32_t axis : {HEAD_AXIS, HEAD_DIM_AXIS})
    {
        if(dims->Get(axis) > 1)
        {
            tokenSpan += (dims->Get(axis) - 1) * strides->Get(axis);
        }
    }
    return sequenceStride >= tokenSpan;
}

/**
 * @brief Packed (THD) sequences, or nullopt for a packing the kernels do not read.
 *
 * Served: compact THD, the form hipDNN's ragged reference runs -- Q and O packed through one
 * offset table, K and V through one (the same one, or another), each table
 * isServableRaggedOffsetTensor for the graph's @p batch, each operand isPackedThrough it, and
 * no per-batch lengths or `padding_mask` with it (the reference takes none; cuDNN's ragged
 * operands with lengths is later work). Which engines run it is each kernel matcher's to say.
 * Declined, each a wrong answer rather than an error if it were ignored: an operand left dense
 * beside ragged ones, Q and O (or K and V) through different tables, whose sequences the
 * kernels could not tell apart, offsets that do not count tokens, and a head-major packing.
 */
inline std::optional<SdpaVarlen> servablePackedVarlen(
    const MatchContext& context, const data_objects::SdpaAttributes& attributes, int64_t batch)
{
    const auto* q = findTensor(context, attributes.q_tensor_uid());
    const auto* k = findTensor(context, attributes.k_tensor_uid());
    const auto* v = findTensor(context, attributes.v_tensor_uid());
    const auto* o = findTensor(context, attributes.o_tensor_uid());
    if(q == nullptr || k == nullptr || v == nullptr || o == nullptr)
    {
        return std::nullopt;
    }
    // A dense Q or K packs through no table, which isPackedThrough declines.
    const int64_t offsetQ = q->ragged_offset_tensor_uid().value_or(NO_SEQ_LEN);
    const int64_t offsetKv = k->ragged_offset_tensor_uid().value_or(NO_SEQ_LEN);
    if(!isPackedThrough(*q, offsetQ) || !isPackedThrough(*o, offsetQ)
       || !isPackedThrough(*k, offsetKv) || !isPackedThrough(*v, offsetKv)
       || !isServableRaggedOffsetTensor(context, offsetQ, batch)
       || !isServableRaggedOffsetTensor(context, offsetKv, batch))
    {
        return std::nullopt;
    }
    if(attributes.padding_mask() || attributes.seq_len_q_tensor_uid().has_value()
       || attributes.seq_len_kv_tensor_uid().has_value())
    {
        return std::nullopt;
    }
    SdpaVarlen varlen;
    varlen.raggedOffsetQ = offsetQ;
    varlen.raggedOffsetKv = offsetKv;
    return varlen;
}

/**
 * @brief servablePackedVarlen(), and cuDNN's packed form with per-batch lengths: ragged
 *        operands with both length tensors, or with `seq_len_kv` alone.
 *
 * Each table entry is then a slot, sequence b's `offset[b + 1] - offset[b]` tokens, of which
 * the first `seq_len[b]` are its rows and keys; the slot's rows past the length are padded rows
 * (O = 0, LSE = -inf, as under padding), and its keys past the length are never read. The
 * lengths are read with or without `padding_mask`, as servableVarlenAsReferenceReads() reads
 * them over dense operands. `seq_len_kv` alone (PyTorch's `seqused_k`, SEM-028) takes each
 * sequence's query rows from Q's table, as the references derive `seq_len_q`: the binding
 * names Q's table as the query lengths (queryLengthsFromOffsets()). `seq_len_q` alone, or
 * `padding_mask` without `seq_len_kv`, declines. The packing itself is servablePackedVarlen()'s,
 * with the same rules. Which engines run it is each kernel matcher's to say (FlydslSdpaVarlen's
 * binding holds both the tables and the lengths).
 */
inline std::optional<SdpaVarlen> servablePackedVarlenWithLengths(
    const MatchContext& context, const data_objects::SdpaAttributes& attributes, int64_t batch)
{
    const bool hasSeqLenQ = attributes.seq_len_q_tensor_uid().has_value();
    const bool hasSeqLenKv = attributes.seq_len_kv_tensor_uid().has_value();
    if(!hasSeqLenQ && !hasSeqLenKv)
    {
        return servablePackedVarlen(context, attributes, batch);
    }
    if(!hasSeqLenKv || !isServableSeqLenTensor(context, *attributes.seq_len_kv_tensor_uid(), batch)
       || (hasSeqLenQ
           && !isServableSeqLenTensor(context, *attributes.seq_len_q_tensor_uid(), batch)))
    {
        return std::nullopt;
    }
    // The packing: servablePackedVarlen()'s operand and table checks, which it runs only for
    // a graph without lengths.
    const auto* q = findTensor(context, attributes.q_tensor_uid());
    const auto* k = findTensor(context, attributes.k_tensor_uid());
    const auto* v = findTensor(context, attributes.v_tensor_uid());
    const auto* o = findTensor(context, attributes.o_tensor_uid());
    if(q == nullptr || k == nullptr || v == nullptr || o == nullptr)
    {
        return std::nullopt;
    }
    const int64_t offsetQ = q->ragged_offset_tensor_uid().value_or(NO_SEQ_LEN);
    const int64_t offsetKv = k->ragged_offset_tensor_uid().value_or(NO_SEQ_LEN);
    if(!isPackedThrough(*q, offsetQ) || !isPackedThrough(*o, offsetQ)
       || !isPackedThrough(*k, offsetKv) || !isPackedThrough(*v, offsetKv)
       || !isServableRaggedOffsetTensor(context, offsetQ, batch)
       || !isServableRaggedOffsetTensor(context, offsetKv, batch))
    {
        return std::nullopt;
    }
    SdpaVarlen varlen;
    varlen.seqLenQ = hasSeqLenQ ? *attributes.seq_len_q_tensor_uid() : offsetQ;
    varlen.seqLenKv = *attributes.seq_len_kv_tensor_uid();
    varlen.raggedOffsetQ = offsetQ;
    varlen.raggedOffsetKv = offsetKv;
    return varlen;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::flydsl_sdpa

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
