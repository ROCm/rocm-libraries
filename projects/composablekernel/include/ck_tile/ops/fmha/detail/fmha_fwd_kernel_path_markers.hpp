// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>
#include <utility>

#include "ck_tile/core.hpp"

namespace ck_tile {

namespace detail {

// Opt-in markers an fmha fwd pipeline may define to pick its K/V DRAM views:
// - kUsesUntransposedVKernelPath: V is consumed as (seqlen_k, hdim_v) instead
//   of being transposed to (hdim_v, seqlen_k).
// - kUsesTdmAffineDramPath: K/V views stay unpadded; the TDM box clamp
//   zero-fills reads past their bounds.
// A pipeline that declares neither takes neither path.
template <typename T>
using has_untransposed_v_kernel_path = decltype(T::kUsesUntransposedVKernelPath);

template <typename T>
using has_tdm_affine_dram_path = decltype(T::kUsesTdmAffineDramPath);

template <typename T>
inline constexpr bool uses_untransposed_v_kernel_path_v = []() {
    if constexpr(is_detected<has_untransposed_v_kernel_path, T>::value)
        return static_cast<bool>(T::kUsesUntransposedVKernelPath);
    else
        return false;
}();

template <typename T>
inline constexpr bool uses_tdm_affine_dram_path_v = []() {
    if constexpr(is_detected<has_tdm_affine_dram_path, T>::value)
        return static_cast<bool>(T::kUsesTdmAffineDramPath);
    else
        return false;
}();

template <typename Pipeline, typename = void>
struct uses_qr_tdm_lds_arena : std::false_type
{
};

template <typename Pipeline>
struct uses_qr_tdm_lds_arena<Pipeline, std::void_t<decltype(Pipeline::kUsesLdsArena)>>
    : std::bool_constant<Pipeline::kUsesLdsArena>
{
};

template <typename Pipeline>
inline constexpr bool uses_qr_tdm_lds_arena_v = uses_qr_tdm_lds_arena<Pipeline>::value;

// Opt-in markers read by FmhaFwdKernel (absent: false); a derived pipeline overrides unsupported.
template <template <typename> class Marker, typename T>
inline constexpr bool pipeline_marker_v = []() {
    if constexpr(is_detected<Marker, T>::value)
        return Marker<T>::value;
    else
        return false;
}();

// Takes k/v descales after the LDS arguments; PERHEAD/BLOCKSCALE q descales fold into scale_s.
template <typename T>
using takes_quant_descales_marker = std::bool_constant<T::kTakesQuantDescales>;
// Takes MX E8M0 descales: Pipeline::MakeMxDescales builds them from the kernel arguments.
template <typename T>
using takes_mx_descales_marker = std::bool_constant<T::kTakesMxDescales>;
// Implements query heads of one KV head merged into the query rows (Pipeline::MergedHeads).
template <typename T>
using supports_merged_query_heads_marker = std::bool_constant<T::kSupportsMergedQueryHeads>;
// hdim_q and hdim_v must equal the tile head dims (no head-dim padding at run time).
template <typename T>
using requires_exact_head_dims_marker = std::bool_constant<T::kRequiresExactHeadDims>;
// Takes a (logits transform, dropout) callback tuple as its last argument.
template <typename T>
using takes_logits_dropout_callbacks_marker =
    std::bool_constant<T::kTakesLogitsAndDropoutCallbacks>;
// With hdim_v padding, O is stored with kAlignmentOacc vectors or per element, chosen at run time.
template <typename T>
using splits_output_store_by_alignment_marker =
    std::bool_constant<T::kSplitsOutputStoreByAlignment>;
// The kernel stores O from the K0 LDS slot with async LDS-to-global stores.
template <typename T>
using uses_async_lds_output_store_marker = std::bool_constant<T::kUsesAsyncLdsOutputStore>;
// Pipeline::IsSupportedKargs(kargs) adds host-side argument checks of its own.
template <typename T, typename Kargs>
using has_kargs_check = decltype(T::IsSupportedKargs(std::declval<const Kargs&>()));

// Order in which FmhaFwdKernel visits causal tiles (Pipeline::kCausalTileOrder).
enum class FmhaFwdCausalTileOrder
{
    kDefault,            // generic workgroup mapping
    kBatchLocal,         // batch-local causal order
    kBatchLocalXcd,      // batch-local, plus the XCD causal remap where measured
    kGlobalLongestFirst, // longest tiles across all batches and heads first
};

template <typename T>
using has_causal_tile_order = decltype(T::kCausalTileOrder);

// Pipelines staging K/V in the qr_tdm LDS arena default to the batch-local order.
template <typename T>
inline constexpr FmhaFwdCausalTileOrder causal_tile_order_v = []() {
    if constexpr(is_detected<has_causal_tile_order, T>::value)
        return static_cast<FmhaFwdCausalTileOrder>(T::kCausalTileOrder);
    else if constexpr(uses_qr_tdm_lds_arena_v<T>)
        return FmhaFwdCausalTileOrder::kBatchLocal;
    else
        return FmhaFwdCausalTileOrder::kDefault;
}();

// Whether the K/V views take the NT_HT cache hint on gfx1250 (see
// FmhaFwdKernel::kKVCoherence): TDM pipelines that stage K/V in their LDS arena,
// fp16/bf16, head dims below 256, tiles of at most 64 rows.
template <typename Pipeline>
inline constexpr bool uses_kv_nt_ht_hint_v =
    uses_qr_tdm_lds_arena_v<Pipeline> && Pipeline::kM0 <= 64 &&
    Pipeline::BlockFmhaShape::kQKHeaddim < 256 &&
    (std::is_same_v<typename Pipeline::KDataType, half_t> ||
     std::is_same_v<typename Pipeline::KDataType, bf16_t>);

} // namespace detail

} // namespace ck_tile
