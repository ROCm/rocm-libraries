// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Host-side tensor-load primitives for the quant GEMM bridges that reshuffle
 * their weight / scale tensors before the device copy (aquant, abquant, bquant).
 *
 * This header only provides the small, repeated building blocks -- loading a raw
 * host pointer into a ck_tile::HostTensor with a given layout, and the
 * unconditional pk_int4 permute. The actual *shuffle behavior* (which of
 * shuffle_b / shuffle_b_permuteN / shuffle_aq / shuffle_bq / bq_permuteN to
 * apply, and in what order) stays in the per-op source, since it is genuinely
 * op-specific and encodes each kernel family's layout contract.
 *
 * tensor_quant and rowcolquant perform no reshuffles and do not include this
 * header, so they never pull in the (heavier) shuffle utilities.
 */

#ifndef CK_TILE_DISPATCHER_QUANT_BRIDGE_SHUFFLE_HPP
#define CK_TILE_DISPATCHER_QUANT_BRIDGE_SHUFFLE_HPP

#include <algorithm>

#include "ck_tile/host/tensor_shuffle_utils.hpp"
#include "ck_tile/host/permute_pk_int4.hpp"

#include "quant_bridge_common.hpp"

namespace quant_bridge {

// Load `rows`x`cols` logical elements from a packed host pointer into a
// HostTensor with leading dim `lead`. RowMajor is a compile-time flag because
// ck_tile::host_tensor_descriptor is overloaded on bool_constant<> layout.
//
// For packed types (pk_int4_t/pk_fp4_t; PackedSize=2) the HostTensor holds only
// rows*cols/PackedSize elements and `src` already contains the packed
// representation, so we copy t.size() elements -- copying rows*cols would overrun
// the source buffer and corrupt the heap (the crash the block-scale bring-up hit
// on every i4/fp4 config before this was fixed).
template <bool RowMajor, typename T>
inline ck_tile::HostTensor<T> load_host_tensor(const T* src, int rows, int cols, int lead)
{
    ck_tile::HostTensor<T> t(
        ck_tile::host_tensor_descriptor(rows, cols, lead, ck_tile::bool_constant<RowMajor>{}));
    std::copy(src, src + t.size(), t.begin());
    return t;
}

// Apply the pk_int4 i4x4 permute in place (mirrors run_gemm_quant_example.inc:
// permute_vectors_i4x4_b, applied unconditionally to pk_int4 operands so the
// device i4->fp8/bf8 conversion sees data in the expected 0x75316420 order).
template <typename T>
inline void permute_i4_inplace(ck_tile::HostTensor<T>& t)
{
    ck_tile::permute_vectors_i4x4_b(t);
}

} // namespace quant_bridge

#endif // CK_TILE_DISPATCHER_QUANT_BRIDGE_SHUFFLE_HPP
