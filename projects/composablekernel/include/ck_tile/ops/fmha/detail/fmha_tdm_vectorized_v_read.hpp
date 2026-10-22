// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

#include <type_traits>
#include <utility>

namespace ck_tile {

namespace detail {

// V LDS tile of the vectorized KV layout: the 8 consecutive tokens of one column are adjacent
// ([rows/8][D][8]). Recognised from the window's static LDS descriptor (token 1 of column 0 is
// element 1); K tiles and linear V tiles are row-major and never match.
template <typename TileWindow>
CK_TILE_HOST_DEVICE constexpr bool fmha_tdm_is_token_contiguous_lds()
{
    using View =
        remove_cvref_t<decltype(std::declval<const TileWindow&>().get_bottom_tensor_view())>;
    using Descriptor = remove_cvref_t<typename View::TensorDesc>;
    if constexpr(!Descriptor::is_static() || Descriptor::get_num_of_top_dimension() != 2)
        return false;
    else
        return Descriptor{}.calculate_offset(make_multi_index(1, 0)) == 1;
}

} // namespace detail

// Transposed V read of a token-contiguous V LDS tile. The transposed read (ds_load_tr16_b128 on a
// row-major [token][column] tile) leaves in each lane, for every access, 8 elements that CK's
// transpose model places at output Y indices group(source + j). For the read windows of the TDM
// schedule those 8 elements are 8 consecutive tokens of one column, starting at a multiple of 8
// (Validate checks every lane at compile time), so on a token-contiguous tile they are one aligned
// 16-byte vector: a plain b128 load at the element-0 coordinate, scattered exactly as the
// transposed read scatters, fills the same registers.
struct FmhaTdmVecVRead
{
    struct RawAddress
    {
        uint32_t byte_address = 0;
        bool valid            = false;
    };

    template <typename TileWindow>
    using OutputDistribution = remove_cvref_t<decltype(make_static_tile_distribution(
        typename OutputTileDistributionTraits<
            typename TileWindow::Base::TileDstr::DstrEncode,
            typename TileWindow::Base::DataType>::TransposedDstrEncode{}))>;

    // Output Y index of element J of access IAccess (the transposed read's scatter target).
    template <index_t IAccess, index_t J, typename TileWindow>
    CK_TILE_HOST_DEVICE static constexpr auto OutputIndex()
    {
        using Base                  = typename TileWindow::Base;
        using Traits                = typename Base::Traits;
        constexpr auto input_index  = Traits::SFC_Ys::get_index(number<IAccess>{});
        constexpr auto source_index = generate_tuple(
            [&](auto dim) {
                return dim == Traits::VectorDimY ? (input_index[dim] + J) : input_index[dim];
            },
            number<Base::NDimY>{});
        return DefaultTranspose<typename Base::DataType>::group_func(source_index);
    }

    // Output Y index of element J of access IAccess as an array, evaluated once per (access,
    // element) so that the per-lane checks below stay within the constant-evaluation budget.
    template <index_t IAccess, index_t J, typename TileWindow>
    static constexpr auto kOutputIndex = to_array<index_t, OutputDistribution<TileWindow>::NDimY>(
        OutputIndex<IAccess, J, TileWindow>());

    // Window (token, column) index of element J of access IAccess for one partition. The output
    // distribution is the transposed one: its X dimensions are (column, token).
    template <index_t IAccess, index_t J, typename TileWindow, typename Partition>
    CK_TILE_HOST_DEVICE static constexpr auto BottomIndex(const Partition& partition)
    {
        constexpr auto distribution = OutputDistribution<TileWindow>{};
        const auto x                = make_tensor_adaptor_coordinate(
                           distribution.get_ps_ys_to_xs_adaptor(),
                           container_concat(partition, kOutputIndex<IAccess, J, TileWindow>))
                           .get_bottom_index();
        return make_multi_index(x[number<1>{}], x[number<0>{}]);
    }

    // Every lane of warp 0: element J is token t0 + J of the column of element 0, t0 % 8 == 0. The
    // V operand is replicated over the warps (checked on lane 0), so warp 0 covers every lane.
    template <index_t IAccess, typename TileWindow, index_t... J>
    CK_TILE_HOST_DEVICE static constexpr bool ValidateLanes(std::integer_sequence<index_t, J...>)
    {
        for(index_t warp = 1; warp < 4; ++warp)
        {
            const auto w = BottomIndex<IAccess, 0, TileWindow>(array<index_t, 2>{warp, 0});
            const auto z = BottomIndex<IAccess, 0, TileWindow>(array<index_t, 2>{0, 0});
            if(w[number<0>{}] != z[number<0>{}] || w[number<1>{}] != z[number<1>{}])
                return false;
        }
        for(index_t lane = 0; lane < 32; ++lane)
        {
            const array<index_t, 2> partition{0, lane};
            const auto x0 = BottomIndex<IAccess, 0, TileWindow>(partition);
            if(x0[number<0>{}] % 8 != 0)
                return false;
            const bool same_column_next_tokens =
                ((BottomIndex<IAccess, J, TileWindow>(partition)[number<0>{}] ==
                      x0[number<0>{}] + J &&
                  BottomIndex<IAccess, J, TileWindow>(partition)[number<1>{}] == x0[number<1>{}]) &&
                 ...);
            if(!same_column_next_tokens)
                return false;
        }
        return true;
    }

    template <index_t IAccess, typename TileWindow>
    CK_TILE_HOST_DEVICE static constexpr bool Validate()
    {
        using Traits = typename TileWindow::Base::Traits;
        static_assert(Traits::ScalarPerVector == 8 && Traits::PackedSize == 1);
        static_assert(OutputDistribution<TileWindow>::NDimP == 2 &&
                      OutputDistribution<TileWindow>::NDimX == 2);
        // The output distribution spans the transposed window: (column, token) lengths.
        constexpr auto out_lengths    = to_sequence(OutputDistribution<TileWindow>{}.get_lengths());
        constexpr auto window_lengths = typename TileWindow::WindowLengths{};
        static_assert(out_lengths[number<0>{}] == window_lengths.at(number<1>{}) &&
                      out_lengths[number<1>{}] == window_lengths.at(number<0>{}));
        return ValidateLanes<IAccess, TileWindow>(std::make_integer_sequence<index_t, 8>{});
    }

    // Byte offset of access IAccess from access 0 on the token-contiguous tile; the same for every
    // lane (checked), so an access-0 address serves the whole stage.
    template <index_t IAccess, typename TileWindow>
    CK_TILE_HOST_DEVICE static constexpr index_t ByteDelta()
    {
        using View =
            remove_cvref_t<decltype(std::declval<const TileWindow&>().get_bottom_tensor_view())>;
        using DataType          = typename TileWindow::Base::DataType;
        using Descriptor        = remove_cvref_t<typename View::TensorDesc>;
        constexpr auto delta_of = [](const array<index_t, 2>& partition) {
            constexpr auto descriptor = Descriptor{};
            return (descriptor.calculate_offset(BottomIndex<IAccess, 0, TileWindow>(partition)) -
                    descriptor.calculate_offset(BottomIndex<0, 0, TileWindow>(partition))) *
                   static_cast<index_t>(sizeof(DataType));
        };
        constexpr index_t delta = delta_of(array<index_t, 2>{0, 0});
        static_assert(
            [&] {
                for(index_t lane = 0; lane < 32; ++lane)
                    if(delta_of(array<index_t, 2>{0, lane}) != delta)
                        return false;
                return true;
            }(),
            "the vectorized V access offset depends on the lane");
        return delta;
    }

    template <index_t IAccess, typename TileWindow>
    CK_TILE_DEVICE static RawAddress Prepare(const TileWindow& window)
    {

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx125__)
        using DataType = typename TileWindow::Base::DataType;
        static_assert(Validate<IAccess, TileWindow>(),
                      "transposed V read is not 8 consecutive tokens of one column");
        constexpr auto distribution = OutputDistribution<TileWindow>{};
        const auto bottom = BottomIndex<IAccess, 0, TileWindow>(get_partition_index(distribution));
        const auto& view  = window.get_bottom_tensor_view();
        const auto coord  = make_tensor_coordinate(view.get_tensor_descriptor(),
                                                  window.get_window_origin() + bottom);
        using LdsPointer  = const __attribute__((address_space(3))) DataType*;
        const auto pointer =
            c_style_pointer_cast<LdsPointer>(view.get_buffer_view().p_data_ + coord.get_offset());
        return {static_cast<uint32_t>(reinterpret_cast<uintptr_t>(pointer)),
                coordinate_has_valid_offset_assuming_top_index_is_valid(
                    view.get_tensor_descriptor(), coord)};
#else
        ignore = window;
        return {};
#endif
    }

    template <index_t IAccess, typename DistributedTensor, typename TileWindow>
    CK_TILE_DEVICE static void
    LoadAt(DistributedTensor& dst, const TileWindow& window, const RawAddress& address)
    {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx125__)
        using Base             = typename TileWindow::Base;
        using Traits           = typename Base::Traits;
        using DataType         = typename Base::DataType;
        using VectorType       = typename Traits::vector_t;
        using NativeVector     = ext_vector_t<typename vector_traits<DataType>::scalar_type,
                                              vector_traits<VectorType>::vector_size>;
        using LdsVectorPointer = const __attribute__((address_space(3))) NativeVector*;
        static_assert(sizeof(VectorType) == 16 && sizeof(NativeVector) == 16);
        static_assert(
            std::is_same_v<remove_cvref_t<typename DistributedTensor::DataType>, DataType>);
        static_assert(Validate<IAccess, TileWindow>(),
                      "transposed V read is not 8 consecutive tokens of one column");

        const VectorType value =
            address.valid
                ? bit_cast<VectorType>(*reinterpret_cast<LdsVectorPointer>(address.byte_address))
                : window.get_bottom_tensor_view().get_buffer_view().template get<VectorType>(
                      index_t{0}, index_t{0}, false);
        constexpr auto output_distribution = typename DistributedTensor::StaticTileDistribution{};
        static_for<0, Traits::ScalarPerVector, Traits::PackedSize>{}([&](auto j) {
            constexpr auto output_index = OutputIndex<IAccess, decltype(j)::value, TileWindow>();
            constexpr index_t output_offset =
                output_distribution.get_ys_to_d_descriptor().calculate_offset(output_index) /
                Traits::PackedSize;
            dst.get_thread_buffer().template at<output_offset>() =
                value.template get_as<DataType>()[j / Traits::PackedSize];
        });
#else
        ignore = dst;
        ignore = window;
        ignore = address;
#endif
    }

    template <index_t IAccess, typename DistributedTensor, typename TileWindow>
    CK_TILE_DEVICE static void LoadAccess(DistributedTensor& dst, const TileWindow& window)
    {
        LoadAt<IAccess>(dst, window, Prepare<IAccess>(window));
    }
};

} // namespace ck_tile
