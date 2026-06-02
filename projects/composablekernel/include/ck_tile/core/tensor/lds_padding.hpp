// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"

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
// trait `needs_lds_pad<T>`. The specialization for a concrete type lives next
// to that type's definition (e.g. needs_lds_pad<pk_fp6x16_t> in pk_f6.hpp) so
// that simply naming the type makes the opt-in visible. Add a
// `template <> struct needs_lds_pad<T> : std::true_type {};` next to any new
// type that travels through the async-load-to-LDS path.

namespace ck_tile {

template <typename T>
struct needs_lds_pad : std::false_type
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

// Reinterpret an LDS region (sized via lds_padded_sizeof<T>()) as an array of
// lds_padded_element<T> and return a pointer to the i-th logical element's
// payload. Indexing therefore applies the padded per-element stride (e.g. the
// gfx950 12->16B FP6 slot), and the returned &elem.value is a plain T lvalue,
// which is the well-defined access path into the wrapper. For non-padded types
// sizeof(lds_padded_element<T>) == sizeof(T), so this reduces to &base[i].
//
// This is the single choke point for the padded LDS layout: every LDS
// read/write address for a padded type goes through here, which keeps the async
// store, the scatter/gather store and the subsequent loads on one stride. It is
// also the natural place to add any future arch-gating of the padding.
template <typename T>
CK_TILE_HOST_DEVICE constexpr T* lds_padded_ptr(T* base, index_t i)
{
    return &reinterpret_cast<lds_padded_element<T>*>(base)[i].value;
}

template <typename T>
CK_TILE_HOST_DEVICE constexpr const T* lds_padded_ptr(const T* base, index_t i)
{
    return &reinterpret_cast<const lds_padded_element<T>*>(base)[i].value;
}

} // namespace ck_tile
