// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <string_view>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/SavedDispatch.hpp>

/**
 * @file AttentionPagedUnifiedLaunchValues.hpp
 * @brief The launch state of the hipkernel:AttentionPagedUnified pack, and its saved form:
 *        a versioned dispatch name and the named, typed values the launch is computed from.
 *        The pack also reads its saved form back into its launch inputs.
 *
 * The launch is one kernel with no workspace. Lengths, page tables, offsets and sinks are
 * device buffers bound by uid at execute, so the uids, the by-value kernel arguments and
 * the CTA size are the whole launch state.
 *
 * The version in the dispatch name identifies the contract between the saved values and
 * the launch. Use a new version for a change to any of these:
 *   - the names, types or meanings of the values.
 *   - the kernel arguments the launch passes: their order, count, kinds, sizes or meanings.
 *   - how the launch derives grid or block from the values.
 * Keep the handler of the old version registered while plans saved under it must load.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

/// Contract v1: the runtime_layout tiled 2D kernel signature. New plans are saved under it.
inline constexpr std::string_view ATTENTION_PAGED_UNIFIED_DISPATCH_SYMBOL_V1
    = "hipkernel.attention_paged_unified.dispatch.v1";

/// The uid an absent optional operand (ragged offset, sink) saves as.
inline constexpr int64_t ATTENTION_PAGED_UNIFIED_NO_UID = -1;

inline constexpr std::string_view ATTENTION_PAGED_UNIFIED_Q_UID_VALUE = "q.uid";
inline constexpr std::string_view ATTENTION_PAGED_UNIFIED_K_UID_VALUE = "k.uid";
inline constexpr std::string_view ATTENTION_PAGED_UNIFIED_V_UID_VALUE = "v.uid";
inline constexpr std::string_view ATTENTION_PAGED_UNIFIED_O_UID_VALUE = "o.uid";
inline constexpr std::string_view ATTENTION_PAGED_UNIFIED_PAGE_TABLE_K_UID_VALUE
    = "page_table_k.uid";
inline constexpr std::string_view ATTENTION_PAGED_UNIFIED_PAGE_TABLE_V_UID_VALUE
    = "page_table_v.uid";
inline constexpr std::string_view ATTENTION_PAGED_UNIFIED_SEQ_LEN_KV_UID_VALUE = "seq_len_kv.uid";
inline constexpr std::string_view ATTENTION_PAGED_UNIFIED_SEQ_LEN_Q_UID_VALUE = "seq_len_q.uid";
inline constexpr std::string_view ATTENTION_PAGED_UNIFIED_RAGGED_OFFSET_UID_VALUE
    = "ragged_offset.uid";
inline constexpr std::string_view ATTENTION_PAGED_UNIFIED_SINK_UID_VALUE = "sink.uid";
inline constexpr std::string_view ATTENTION_PAGED_UNIFIED_SCALE_VALUE = "scale";
inline constexpr std::string_view ATTENTION_PAGED_UNIFIED_NUM_SEQS_VALUE = "num_seqs";
inline constexpr std::string_view ATTENTION_PAGED_UNIFIED_BLOCK_TABLE_STRIDE_VALUE
    = "block_table_stride";
inline constexpr std::string_view ATTENTION_PAGED_UNIFIED_NUM_QUERY_HEADS_VALUE = "num_query_heads";
inline constexpr std::string_view ATTENTION_PAGED_UNIFIED_NUM_KV_HEADS_VALUE = "num_kv_heads";
inline constexpr std::string_view ATTENTION_PAGED_UNIFIED_SLIDING_WINDOW_VALUE = "sliding_window";
inline constexpr std::string_view ATTENTION_PAGED_UNIFIED_Q_UNIFORM_LEN_VALUE = "q_uniform_len";
inline constexpr std::string_view ATTENTION_PAGED_UNIFIED_Q_SEQ_STRIDE_VALUE = "q_seq_stride";
inline constexpr std::string_view ATTENTION_PAGED_UNIFIED_Q_TOK_STRIDE_VALUE = "q_tok_stride";
inline constexpr std::string_view ATTENTION_PAGED_UNIFIED_Q_HEAD_STRIDE_VALUE = "q_head_stride";
inline constexpr std::string_view ATTENTION_PAGED_UNIFIED_O_SEQ_STRIDE_VALUE = "o_seq_stride";
inline constexpr std::string_view ATTENTION_PAGED_UNIFIED_O_TOK_STRIDE_VALUE = "o_tok_stride";
inline constexpr std::string_view ATTENTION_PAGED_UNIFIED_O_HEAD_STRIDE_VALUE = "o_head_stride";
inline constexpr std::string_view ATTENTION_PAGED_UNIFIED_KV_BLOCK_STRIDE_VALUE = "kv_block_stride";
inline constexpr std::string_view ATTENTION_PAGED_UNIFIED_KV_TOK_STRIDE_VALUE = "kv_tok_stride";
inline constexpr std::string_view ATTENTION_PAGED_UNIFIED_KV_HEAD_STRIDE_VALUE = "kv_head_stride";
inline constexpr std::string_view ATTENTION_PAGED_UNIFIED_KV_BLOCK_SPAN_BYTES_VALUE
    = "kv_block_span_bytes";
inline constexpr std::string_view ATTENTION_PAGED_UNIFIED_Q_BLOCK_GRID_VALUE = "q_block_grid";
inline constexpr std::string_view ATTENTION_PAGED_UNIFIED_SINK_IS_F32_VALUE = "sink_is_f32";
inline constexpr std::string_view ATTENTION_PAGED_UNIFIED_BLOCK_THREADS_VALUE = "block_threads";

/// The tensor uids a launch binds to device buffers. raggedOffset is NO_UID for a dense Q;
/// sink is NO_UID without attention sinks. pageTableK == pageTableV when one table pages
/// both K and V.
struct AttentionPagedUnifiedUids
{
    int64_t q = 0;
    int64_t k = 0;
    int64_t v = 0;
    int64_t o = 0;
    int64_t pageTableK = 0;
    int64_t pageTableV = 0;
    int64_t seqLenKv = 0;
    int64_t seqLenQ = 0;
    int64_t raggedOffset = ATTENTION_PAGED_UNIFIED_NO_UID;
    int64_t sink = ATTENTION_PAGED_UNIFIED_NO_UID;
};

/// Q's or O's strides, in elements. seqStride is the batch stride of a dense operand and
/// the ragged-offset multiplier of a ragged one.
struct AttentionPagedUnifiedQueryStrides
{
    int32_t seqStride = 0;
    int32_t tokenStride = 0;
    int32_t headStride = 0;
};

/// Every by-value kernel argument the launch varies.
struct AttentionPagedUnifiedScalars
{
    float scale = 0.0F;
    int32_t numSeqs = 0;
    int32_t blockTableStride = 0;
    int32_t numQueryHeads = 0;
    int32_t numKvHeads = 0;
    int32_t slidingWindow = 0;
    int32_t qUniformLen = 0; ///< query rows per sequence of a dense Q; 0 for a ragged Q
    AttentionPagedUnifiedQueryStrides q;
    AttentionPagedUnifiedQueryStrides o;
    int32_t kvBlockStride = 0;
    int32_t kvTokStride = 0;
    int32_t kvHeadStride = 0;
    int32_t kvBlockSpanBytes = 0;
    int32_t qBlockGrid = 0;
    int32_t sinkIsF32 = 0; ///< 1: the sink is FLOAT; 0: in the Q data type, or no sink
};

/// Everything a launch is computed from: block = (blockThreads, 1, 1) and
/// grid = (numKvHeads, qBlockGrid, 1).
struct AttentionPagedUnifiedLaunchInputs
{
    AttentionPagedUnifiedUids uids;
    AttentionPagedUnifiedScalars scalars;
    int32_t blockThreads = 0;
};

/// The values of the `hipkernel.attention_paged_unified.dispatch.v1` contract. `scale` is
/// stored as a double, which holds every float exactly; every other value is an int.
hipdnn_plugin_sdk::ingestor::MetadataValues
    attentionPagedUnifiedLaunchValues(const AttentionPagedUnifiedLaunchInputs& launch);

/// Reads the values of the `hipkernel.attention_paged_unified.dispatch.v1` contract. Each
/// value must be in the range the kernel takes (int32, positive where the kernel divides
/// by it or indexes with it), `scale` must be a finite float stored exactly, the head
/// counts must divide, a dense Q needs `q_uniform_len` >= 1 and a ragged one 0,
/// `sink_is_f32` must be 0 or 1, and `block_threads` must be a whole number of waves.
///
/// @throws HipdnnPluginException with a load refusal when the dispatch name is not the
///         contract, or when a value is missing, extra, of another type or out of range.
AttentionPagedUnifiedLaunchInputs readAttentionPagedUnifiedLaunchInputs(
    const hipdnn_plugin_sdk::ingestor::SavedLaunchInputs& inputs);

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
