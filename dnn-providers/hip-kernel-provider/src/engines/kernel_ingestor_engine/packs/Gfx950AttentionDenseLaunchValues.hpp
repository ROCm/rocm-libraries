// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <string_view>

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/SavedDispatch.hpp>

/**
 * @file Gfx950AttentionDenseLaunchValues.hpp
 * @brief The launch state of the gfx950 dense-attention ingestor pack, and its saved form:
 *        a versioned dispatch name and the named, typed values the launch is computed from.
 *        The pack also reads its saved form back into its launch inputs.
 *
 * The version in a dispatch name identifies the contract of its values. A change to the
 * names, types or meanings of the values needs a new version.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

/// The tensor uids and derived scalars a matched dense-attention graph binds.
struct AttentionDenseBinding
{
    int64_t q = 0;
    int64_t k = 0;
    int64_t v = 0;
    int64_t o = 0;
    int64_t causal = 0;
    int64_t slidingWindow = 0;
    float scale = 0.0F;
};

/// The graph facts the matcher and prepare() both need, derived once from the tensors.
struct AttentionDenseProblem
{
    int64_t batch = 0;
    int64_t seqLenQ = 0;
    int64_t seqLenKv = 0;
    int64_t numQueryHeads = 0;
    int64_t numKvHeads = 0;
    int64_t headSize = 0;
    data_objects::DataType dataType = data_objects::DataType::UNSET;
};

inline constexpr std::string_view GFX950_ATTENTION_DENSE_DISPATCH_SYMBOL_V1
    = "hipkernel.gfx950_attention_dense.dispatch.v1";

inline constexpr std::string_view GFX950_ATTENTION_DENSE_Q_UID_VALUE = "q.uid";
inline constexpr std::string_view GFX950_ATTENTION_DENSE_K_UID_VALUE = "k.uid";
inline constexpr std::string_view GFX950_ATTENTION_DENSE_V_UID_VALUE = "v.uid";
inline constexpr std::string_view GFX950_ATTENTION_DENSE_O_UID_VALUE = "o.uid";
inline constexpr std::string_view GFX950_ATTENTION_DENSE_CAUSAL_VALUE = "causal";
inline constexpr std::string_view GFX950_ATTENTION_DENSE_SLIDING_WINDOW_VALUE = "sliding_window";
inline constexpr std::string_view GFX950_ATTENTION_DENSE_BATCH_VALUE = "batch";
inline constexpr std::string_view GFX950_ATTENTION_DENSE_SEQLEN_Q_VALUE = "seqlen_q";
inline constexpr std::string_view GFX950_ATTENTION_DENSE_SEQLEN_KV_VALUE = "seqlen_kv";
inline constexpr std::string_view GFX950_ATTENTION_DENSE_NUM_QUERY_HEADS_VALUE = "num_query_heads";
inline constexpr std::string_view GFX950_ATTENTION_DENSE_NUM_KV_HEADS_VALUE = "num_kv_heads";
inline constexpr std::string_view GFX950_ATTENTION_DENSE_HEAD_SIZE_VALUE = "head_size";
inline constexpr std::string_view GFX950_ATTENTION_DENSE_BLOCK_M_VALUE = "block_m";
inline constexpr std::string_view GFX950_ATTENTION_DENSE_SCALE_VALUE = "scale";
inline constexpr std::string_view GFX950_ATTENTION_DENSE_DATA_TYPE_VALUE = "data_type";
inline constexpr std::string_view GFX950_ATTENTION_DENSE_STRIDE_LAYOUT_VALUE = "stride_layout";

/// The one stride layout the kernel supports. prepare() refuses every other layout.
inline constexpr std::string_view GFX950_ATTENTION_DENSE_BSHD_LAYOUT = "bshd";

/// The values of the `hipkernel.gfx950_attention_dense.dispatch.v1` contract. `scale` is
/// stored as a double, which holds every float exactly. `data_type` is the DataType
/// enumerator name.
hipdnn_plugin_sdk::ingestor::MetadataValues gfx950AttentionDenseLaunchValues(
    const AttentionDenseBinding& binding, const AttentionDenseProblem& problem, int64_t blockM);

/// Everything a dense-attention launch is computed from.
struct Gfx950AttentionDenseLaunchInputs
{
    AttentionDenseBinding binding;
    AttentionDenseProblem problem;
    int64_t blockM = 0;
};

/// Reads the values of the `hipkernel.gfx950_attention_dense.dispatch.v1` contract.
/// `batch`, `seqlen_q` and `seqlen_kv` must be in [1, INT32_MAX], because the kernel takes
/// them as 32-bit integers. The head counts and `head_size` must be in [1, INT32_MAX].
/// `block_m` must be a tile the kernel is built with. `scale` must convert to a float
/// and back without change. `data_type` must be a type the kernel is built for, and
/// `stride_layout` must be `bshd`.
///
/// @throws HipdnnPluginException with a load refusal when the dispatch name is not that
///         contract, or when a value is missing, extra, of another type or out of range.
Gfx950AttentionDenseLaunchInputs readGfx950AttentionDenseLaunchInputs(
    const hipdnn_plugin_sdk::ingestor::SavedLaunchInputs& inputs);

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
