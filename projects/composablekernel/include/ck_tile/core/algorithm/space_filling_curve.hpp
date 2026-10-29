// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/container/multi_index.hpp"
#include "ck_tile/core/container/container_helper.hpp"
#include "ck_tile/core/container/statically_indexed_array.hpp"
#include "ck_tile/core/container/static_array.hpp"
#include "ck_tile/core/utility/functional.hpp"
#include "ck_tile/core/utility/type_traits.hpp"

namespace ck_tile {

namespace detail {

// Precompute the multi-dimensional index of every access of a space-filling curve with a single
// constexpr loop. Row i of the table is the index of 1D access i, in the original dimension order
// and scaled by the scalars per access.
template <index_t NumAccess,
          bool SnakeCurved,
          index_t... Lengths,
          index_t... DimAccessOrder,
          index_t... ScalarsPerAccess>
CK_TILE_HOST_DEVICE constexpr auto make_space_filling_curve_index_table(
    sequence<Lengths...>, sequence<DimAccessOrder...>, sequence<ScalarsPerAccess...>)
{
    constexpr index_t nDim = sizeof...(Lengths);

    constexpr static_array<index_t, nDim> lengths{{Lengths...}};
    constexpr static_array<index_t, nDim> order{{DimAccessOrder...}}; // new2old
    constexpr static_array<index_t, nDim> scalars{{ScalarsPerAccess...}};

    // Access lengths in access order, and the strides of the 1D access index over them
    static_array<index_t, nDim> ordered_lengths{};
    for(index_t i = 0; i < nDim; ++i)
    {
        ordered_lengths[i] = lengths[order[i]] / scalars[order[i]];
    }

    static_array<index_t, nDim> strides{};
    for(index_t i = nDim - 1, stride = 1; i >= 0; --i)
    {
        strides[i] = stride;
        stride *= ordered_lengths[i];
    }

    static_array<static_array<index_t, nDim>, NumAccess> table{};
    for(index_t idx_1d = 0; idx_1d < NumAccess; ++idx_1d)
    {
        index_t remaining = idx_1d;
        // 1D index over the dimensions visited so far. With snake curving, a dimension is swept
        // forward when this index is even, and backward when it is odd.
        index_t outer_idx_1d = 0;

        for(index_t i = 0; i < nDim; ++i)
        {
            const index_t ordered_idx = remaining / strides[i];
            remaining %= strides[i];

            const bool forward_sweep = !SnakeCurved || outer_idx_1d % 2 == 0;
            outer_idx_1d             = outer_idx_1d * ordered_lengths[i] + ordered_idx;

            table[idx_1d][order[i]] =
                (forward_sweep ? ordered_idx : ordered_lengths[i] - 1 - ordered_idx) *
                scalars[order[i]];
        }
    }

    return table;
}

} // namespace detail

template <typename TensorLengths,
          typename DimAccessOrder,
          typename ScalarsPerAccess,
          bool SnakeCurved = true> // # of scalars per access in each dimension
struct space_filling_curve
{
    static constexpr index_t TensorSize =
        reduce_on_sequence(TensorLengths{}, multiplies<>{}, number<1>{});
    static_assert(0 < TensorSize,
                  "space_filling_curve should be used to access a non-empty tensor");

    static constexpr index_t nDim = TensorLengths::size();

    // Checked before the members below divide TensorLengths by ScalarsPerAccess
    static_assert(TensorLengths::size() == ScalarsPerAccess::size());
    static_assert(TensorLengths{} % ScalarsPerAccess{} ==
                  typename uniform_sequence_gen<TensorLengths::size(), 0>::type{});

    using Index = multi_index<nDim>;

    static constexpr index_t ScalarPerVector =
        reduce_on_sequence(ScalarsPerAccess{}, multiplies<>{}, number<1>{});

    static constexpr auto access_lengths   = TensorLengths{} / ScalarsPerAccess{};
    static constexpr auto dim_access_order = DimAccessOrder{};
    static constexpr auto ordered_access_lengths =
        container_reorder_given_new2old(access_lengths, dim_access_order);

    static constexpr index_t NumAccess = TensorSize / ScalarPerVector;

    // Multi-dimensional index of every access, computed once per curve type
    static constexpr auto index_table =
        detail::make_space_filling_curve_index_table<NumAccess, SnakeCurved>(
            TensorLengths{}, DimAccessOrder{}, ScalarsPerAccess{});

    static constexpr auto I0 = number<0>{};
    static constexpr auto I1 = number<1>{};

    CK_TILE_HOST_DEVICE static constexpr index_t get_num_of_access()
    {
        return reduce_on_sequence(TensorLengths{}, multiplies<>{}, number<1>{}) / ScalarPerVector;
    }

    template <index_t AccessIdx1dHead, index_t AccessIdx1dTail>
    static CK_TILE_HOST_DEVICE constexpr auto get_step_between(number<AccessIdx1dHead>,
                                                               number<AccessIdx1dTail>)
    {
        static_assert(AccessIdx1dHead >= 0 && AccessIdx1dHead < get_num_of_access(),
                      "1D index out of range");
        static_assert(AccessIdx1dTail >= 0 && AccessIdx1dTail < get_num_of_access(),
                      "1D index out of range");

        return get_step_between_impl<AccessIdx1dHead, AccessIdx1dTail>(dims{});
    }

    template <index_t AccessIdx1d>
    static CK_TILE_HOST_DEVICE constexpr auto get_forward_step(number<AccessIdx1d>)
    {
        static_assert(AccessIdx1d < get_num_of_access(), "1D index should be larger than 0");
        return get_step_between(number<AccessIdx1d>{}, number<AccessIdx1d + 1>{});
    }

    template <index_t AccessIdx1d>
    static CK_TILE_HOST_DEVICE constexpr auto get_backward_step(number<AccessIdx1d>)
    {
        static_assert(AccessIdx1d > 0, "1D index should be larger than 0");

        return get_step_between(number<AccessIdx1d>{}, number<AccessIdx1d - 1>{});
    }

    // Value form of get_index: a multi_index instead of a tuple of number<>, for constexpr code
    // that computes with the index (for example to build an offset table)
    template <index_t AccessIdx1d>
    static CK_TILE_HOST_DEVICE constexpr Index _get_index(number<AccessIdx1d>)
    {
        Index idx;
        for(index_t i = 0; i < nDim; ++i)
        {
            idx(i) = index_table[AccessIdx1d][i];
        }
        return idx;
    }

    // Return a tuple of number<>, so the index can be used in constant expressions
    template <index_t AccessIdx1d>
    static CK_TILE_HOST_DEVICE constexpr auto get_index(number<AccessIdx1d>)
    {
        return get_index_impl<AccessIdx1d>(dims{});
    }

    private:
    using dims = typename arithmetic_sequence_gen<0, nDim, 1>::type;

    // One pack expansion over the table row, without a lambda per access
    template <index_t AccessIdx1d, index_t... IDims>
    static CK_TILE_HOST_DEVICE constexpr auto get_index_impl(sequence<IDims...>)
    {
        return make_tuple(number<index_table[AccessIdx1d][IDims]>{}...);
    }

    template <index_t AccessIdx1dHead, index_t AccessIdx1dTail, index_t... IDims>
    static CK_TILE_HOST_DEVICE constexpr auto get_step_between_impl(sequence<IDims...>)
    {
        return make_tuple(
            number<index_table[AccessIdx1dTail][IDims] - index_table[AccessIdx1dHead][IDims]>{}...);
    }
};

} // namespace ck_tile
