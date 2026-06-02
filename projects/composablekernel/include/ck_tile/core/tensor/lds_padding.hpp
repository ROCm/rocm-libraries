// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/pk_f6.hpp"

// LDS padding helpers.
//
// Precondition: this padding (12B -> 16B) is only correct for element types
// that reach LDS as a single 12-byte slot via the `buffer_load_dwordx3 ... lds`
// async path on gfx950, which writes a fixed 16-byte per-thread stride.
//
// Scope of the current rule:
// - Only single-dwordx3-per-LDS-slot types are handled (i.e. sizeof(T) == 12).
// - Wider layouts that happen to be a multiple of 12 bytes (e.g. pk_fp6x32_t
//   at 24B, which would require two consecutive dwordx3-to-LDS ops with 12->16
//   padding between them for a 32B slot) are NOT handled here. A future
//   pipeline that needs such a layout will have to extend this helper.
//
// To avoid silently changing the layout of unrelated 12-byte aggregates
// (e.g. future `float[3]` wrappers), the 12->16 rule is gated on an opt-in
// trait `needs_lds_pad<T>`. Add a `template <> struct needs_lds_pad<T>
// : std::true_type {};` specialization for any new type that travels through
// the async-load-to-LDS path.

namespace ck_tile {

template <typename T>
struct needs_lds_pad : std::false_type
{
};

// pk_fp6x16_t (= pk_f6_legacy_t<16, f6_kind::fp6>) is the 12-byte packed FP6
// type consumed by the gfx950 MX scaled-MFMA path. It reaches LDS via the
// buffer_load_dwordx3 -> LDS async path, which writes a fixed 16-byte
// per-thread stride, so it opts into the 12 -> 16 padded LDS stride.
template <>
struct needs_lds_pad<pk_fp6x16_t> : std::true_type
{
};

template <typename T>
struct lds_padding_traits
{
    static constexpr bool is_twelve_byte_type    = sizeof(T) == 12;
    static constexpr bool uses_padded_lds_stride = is_twelve_byte_type && needs_lds_pad<T>::value;
    static constexpr index_t padded_size         = uses_padded_lds_stride ? 16 : sizeof(T);
    static constexpr index_t padded_alignment    = uses_padded_lds_stride ? 16 : alignof(T);
};

// Returns the padded LDS stride for type T. Equal to sizeof(T) unless T is
// 12 bytes AND has explicitly opted in via needs_lds_pad<T>, in which case
// it returns 16 to match the buffer_load_dwordx3-to-LDS hardware stride.
template <typename T>
CK_TILE_HOST_DEVICE constexpr index_t lds_padded_sizeof()
{
    return lds_padding_traits<T>::padded_size;
}

template <typename T>
CK_TILE_HOST_DEVICE constexpr index_t lds_padded_alignof()
{
    return lds_padding_traits<T>::padded_alignment;
}

// Typed wrapper whose sizeof() == lds_padded_sizeof<T>().
// Using this for pointer arithmetic instead of raw char* keeps LLVM's
// typed GEP intact, preserving alias analysis and load coalescing.
template <typename T>
struct alignas(lds_padded_alignof<T>()) lds_padded_element
{
    static_assert(!lds_padding_traits<T>::is_twelve_byte_type || needs_lds_pad<T>::value,
                  "12-byte LDS element types must explicitly opt into LDS padding via "
                  "needs_lds_pad<T> or use a non-padded LDS path");
    static_assert(sizeof(T) <= lds_padded_sizeof<T>(), "Padded size must be at least sizeof(T)");
    static_assert(lds_padded_alignof<T>() >= alignof(T),
                  "Padded alignment must be at least alignof(T)");
    static_assert(lds_padded_sizeof<T>() % lds_padded_alignof<T>() == 0,
                  "Padded size must be a multiple of the padded alignment");
    T value;
};

} // namespace ck_tile
