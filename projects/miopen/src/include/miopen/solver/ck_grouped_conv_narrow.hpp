// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Shared int32-narrowing helpers for the six grouped CK xdlops solvers
// (ConvHipImplicitGemm{,3D}Group{Fwd,Bwd,Wrw}Xdlops). Ported (verbatim, modulo
// namespace placement) from develop's src/ck_impl/ck_grouped_conv_impl_helpers.hpp
// as part of the ROCM-23997 large-tensor backport -- 7.0.2 predates the
// ck_impl/ refactor, so the inline CKArgs structs in the six solver .cpp files
// include this header directly instead.

#include <ck/ck.hpp>

#include <array>
#include <cassert>
#include <cstddef>

namespace miopen {
namespace solver {

// ---------------------------------------------------------------------------
// ToCKIndexArray — narrow a long_index_t array to a ck::index_t (int32) array.
//
// Used on the sub-INT_MAX MakeArgPtr path, where CK's int32
// MakeArgumentPointer overload accepts ck::index_t length / stride arrays
// only. For those shapes the narrowing is exact. Large-tensor (>INT_MAX)
// shapes never reach this helper: MakeArgPtr detects a large-tensor CK
// instance (IsLargeTensorCKInstance) and binds CK's int64 long_index_t
// overload with the un-narrowed members instead. The assert below therefore
// guards the contract -- if a >INT_MAX value is ever narrowed here, the
// large-tensor branch was bypassed and the result would be silently wrong.
// ---------------------------------------------------------------------------
template <typename T, std::size_t N>
constexpr std::array<ck::index_t, N> ToCKIndexArray(const std::array<T, N>& src)
{
    std::array<ck::index_t, N> dst{};
    for(std::size_t i = 0; i < N; ++i)
    {
        dst[i] = static_cast<ck::index_t>(src[i]);
        assert(static_cast<T>(dst[i]) == src[i] &&
               "ToCKIndexArray narrowed a value > INT_MAX -- "
               "RequiresLargeTensorCKInstance filter contract was bypassed");
    }
    return dst;
}

// ---------------------------------------------------------------------------
// NarrowedCKArrays3D / NarrowedCKArrays2D — bundles of int32-narrowed
// length/stride arrays handed to CK's int32 MakeArgumentPointer overload on
// the sub-INT_MAX path. Large-tensor (>INT_MAX) shapes bypass these bundles
// and bind CK's int64 long_index_t overload directly (see ToCKIndexArray).
//
// These bundles MUST be stored as members of the owning CKArgs (not as
// function-local temporaries), because CK's MakeArgumentPointer captures
// references to the array elements into the returned Argument object. If
// the bundle goes out of scope before IsSupportedArgument runs, CK reads
// freed stack memory (caught by ASAN as stack-use-after-scope).
// ---------------------------------------------------------------------------
struct NarrowedCKArrays3D
{
    std::array<ck::index_t, 6> in_l;
    std::array<ck::index_t, 6> in_s;
    std::array<ck::index_t, 6> out_l;
    std::array<ck::index_t, 6> out_s;
    std::array<ck::index_t, 6> wei_l;
    std::array<ck::index_t, 6> wei_s;
    std::array<ck::index_t, 3> filter_strides;
    std::array<ck::index_t, 3> filter_dilations;
    std::array<ck::index_t, 3> lPadding;
    std::array<ck::index_t, 3> rPadding;
};

struct NarrowedCKArrays2D
{
    std::array<ck::index_t, 5> in_l;
    std::array<ck::index_t, 5> in_s;
    std::array<ck::index_t, 5> out_l;
    std::array<ck::index_t, 5> out_s;
    std::array<ck::index_t, 5> wei_l;
    std::array<ck::index_t, 5> wei_s;
    std::array<ck::index_t, 2> filter_strides;
    std::array<ck::index_t, 2> filter_dilations;
    std::array<ck::index_t, 2> lPadding;
    std::array<ck::index_t, 2> rPadding;
};

// ---------------------------------------------------------------------------
// MakeNarrowedCKArrays — build a NarrowedCKArrays2D/3D bundle by int32-narrowing
// each int64 length/stride array via ToCKIndexArray. The 2D and 3D bundles
// share field names, so a single Bundle-parameterized helper deduplicates the
// identical field mapping across all six inline CKArgs structs. Callers pass
// their own member arrays positionally, which differ in name (2D uses
// input/output/weight/strides/dilation; 3D uses in_lengths/out_lengths/
// wei_lengths/filter_strides/filter_dilations) but not in meaning or order.
// ---------------------------------------------------------------------------
template <typename Bundle, typename LenArr, typename FilterArr>
Bundle MakeNarrowedCKArrays(const LenArr& in_lengths,
                            const LenArr& in_strides,
                            const LenArr& out_lengths,
                            const LenArr& out_strides,
                            const LenArr& wei_lengths,
                            const LenArr& wei_strides,
                            const FilterArr& filter_strides,
                            const FilterArr& filter_dilations,
                            const FilterArr& lPadding,
                            const FilterArr& rPadding)
{
    return Bundle{
        .in_l             = ToCKIndexArray(in_lengths),
        .in_s             = ToCKIndexArray(in_strides),
        .out_l            = ToCKIndexArray(out_lengths),
        .out_s            = ToCKIndexArray(out_strides),
        .wei_l            = ToCKIndexArray(wei_lengths),
        .wei_s            = ToCKIndexArray(wei_strides),
        .filter_strides   = ToCKIndexArray(filter_strides),
        .filter_dilations = ToCKIndexArray(filter_dilations),
        .lPadding         = ToCKIndexArray(lPadding),
        .rPadding         = ToCKIndexArray(rPadding),
    };
}

} // namespace solver
} // namespace miopen
