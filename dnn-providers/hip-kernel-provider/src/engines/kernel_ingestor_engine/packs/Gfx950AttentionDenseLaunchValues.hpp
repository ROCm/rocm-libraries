// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <cstdint>
#include <optional>
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
 * The version in a dispatch name identifies the contract between the saved values and the
 * launch. A saved plan pairs an older code object with this launch. Use a new version for
 * a change to any of these:
 *   - the names, types or meanings of the values.
 *   - the kernel arguments the launch passes: their order, count, kinds, sizes or meanings.
 *   - how the launch derives grid, block or dynamic LDS from the values.
 * Keep the handler of the old version registered while plans saved under it must load.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

/// An additive bias the graph carries: its uid and the element strides of its batch,
/// query-head and query-row axes, 0 on a broadcast (extent-1) axis.
struct AdditiveBias
{
    int64_t uid = 0;
    std::array<int32_t, 3> strides{0, 0, 0};
};

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
    /// A binding with a bias is saved under v3, one without under v2.
    std::optional<AdditiveBias> bias;
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

/// Contract v1: the launch passes 8 kernel arguments (q, k, v, o, scale, batch, seqlen_q,
/// seqlen_kv), for kernels that bake the head counts. Its handler stays registered so plans
/// saved under it still load.
inline constexpr std::string_view GFX950_ATTENTION_DENSE_DISPATCH_SYMBOL_V1
    = "hipkernel.gfx950_attention_dense.dispatch.v1";

/// Contract v2: the same values, and the launch also passes num_query_heads and
/// num_kv_heads (10 kernel arguments), for kernels that read the head counts at runtime.
/// New plans without a bias are saved under it.
inline constexpr std::string_view GFX950_ATTENTION_DENSE_DISPATCH_SYMBOL_V2
    = "hipkernel.gfx950_attention_dense.dispatch.v2";

/// Contract v3: the v2 values plus the bias uid and its batch, query-head and query-row
/// element strides; the launch passes the v2 arguments, then additive_bias_ptr and the
/// three strides (14 kernel arguments). New plans with an additive bias are saved under it.
inline constexpr std::string_view GFX950_ATTENTION_DENSE_DISPATCH_SYMBOL_V3
    = "hipkernel.gfx950_attention_dense.dispatch.v3";

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
// v3 only.
inline constexpr std::string_view GFX950_ATTENTION_DENSE_BIAS_UID_VALUE = "bias.uid";
inline constexpr std::string_view GFX950_ATTENTION_DENSE_BIAS_STRIDE_BATCH_VALUE
    = "bias.stride_batch";
inline constexpr std::string_view GFX950_ATTENTION_DENSE_BIAS_STRIDE_HEAD_VALUE
    = "bias.stride_head";
inline constexpr std::string_view GFX950_ATTENTION_DENSE_BIAS_STRIDE_QUERY_VALUE
    = "bias.stride_query";

/// The one stride layout the kernel supports. prepare() refuses every other layout.
inline constexpr std::string_view GFX950_ATTENTION_DENSE_BSHD_LAYOUT = "bshd";

/// The values of the `hipkernel.gfx950_attention_dense.dispatch.v1`, `.v2` and `.v3`
/// contracts. v1 and v2 share one value set: the uids, flags, extents, tile, scale, data
/// type and layout. v3 adds the bias uid and its batch, query-head and query-row element
/// strides, which this writes exactly when the binding has a bias. `scale` is stored as a
/// double, which holds every float exactly. `data_type` is the DataType enumerator name.
hipdnn_plugin_sdk::ingestor::MetadataValues gfx950AttentionDenseLaunchValues(
    const AttentionDenseBinding& binding, const AttentionDenseProblem& problem, int64_t blockM);

/// Everything a dense-attention launch is computed from.
struct Gfx950AttentionDenseLaunchInputs
{
    AttentionDenseBinding binding;
    AttentionDenseProblem problem;
    int64_t blockM = 0;
};

/// Reads the values of the `hipkernel.gfx950_attention_dense.dispatch.v1` or `.v2`
/// contract, or `.v3` (v1 and v2 share one value set and differ only in the kernel
/// arguments the launch passes; v3 adds the bias values). `batch`, `seqlen_q` and
/// `seqlen_kv` must be in [1, INT32_MAX], because the kernel takes them as 32-bit integers.
/// The head counts and `head_size` must be in [1, INT32_MAX]. `block_m` must be a tile
/// the kernel is built with. `scale` must convert to a float and back without change.
/// `data_type` must be a type the kernel is built for, and `stride_layout` must be `bshd`.
/// v3 also needs the bias strides in [0, INT32_MAX] and the largest bias element index a
/// launch reaches ((seqlen_kv - 1) plus (extent - 1) * stride over the batch, query-head
/// and query-row axes) within int32_t; its inputs carry a bias, v1's and v2's never do.
///
/// @throws HipdnnPluginException with a load refusal when the dispatch name is none of the
///         three contracts, when a value is missing, extra, of another type or out of
///         range, or when v3's bias strides reach beyond the int32_t element range.
Gfx950AttentionDenseLaunchInputs readGfx950AttentionDenseLaunchInputs(
    const hipdnn_plugin_sdk::ingestor::SavedLaunchInputs& inputs);

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
