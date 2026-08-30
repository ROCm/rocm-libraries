// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

struct FmhaD192DeferredPLdsLayout
{
    static constexpr index_t kNumGroups         = 4;
    static constexpr index_t kRows              = 128;
    static constexpr index_t kColumnsPerGroup   = 32;
    static constexpr index_t kMHalves           = 2;
    static constexpr index_t kValuesPerThread   = 32;
    static constexpr index_t kValuesPerMHalf    = 16;
    static constexpr index_t kSelectedRowStride = 40;
    static constexpr index_t kArenaBytes        = 0x39000;
    static constexpr index_t kSlotBytes         = 0x3800;

    using MHalf = array<bf16_t, kValuesPerMHalf>;

    CK_TILE_HOST_DEVICE static constexpr index_t GetSlotBase(index_t group)
    {
        return group == 0 ? 0x0c800 : group == 1 ? 0x1c800 : group == 2 ? 0x29000 : 0x2c800;
    }

    template <index_t Group>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSlotBase()
    {
        static_assert(Group >= 0 && Group < kNumGroups);
        return GetSlotBase(Group);
    }

    template <index_t RowStride>
    CK_TILE_HOST_DEVICE static constexpr index_t GetGroupFootprintBytes()
    {
        static_assert(RowStride >= kColumnsPerGroup);
        return ((kRows - 1) * RowStride + kColumnsPerGroup) * sizeof(bf16_t);
    }

    template <index_t RowStride>
    CK_TILE_HOST_DEVICE static constexpr auto MakeGroupLdsDescriptor()
    {
        static_assert(RowStride >= kColumnsPerGroup);
        static_assert(GetGroupFootprintBytes<RowStride>() <= kSlotBytes);
        return make_naive_tensor_descriptor(make_tuple(number<kRows>{}, number<kColumnsPerGroup>{}),
                                            make_tuple(number<RowStride>{}, number<1>{}),
                                            number<8>{},
                                            number<1>{});
    }

    template <index_t Access, typename TileWindow>
    CK_TILE_HOST_DEVICE static constexpr index_t GetAccessThreadBufferOffset()
    {
        using Window = remove_cvref_t<TileWindow>;
        using Base   = typename Window::Base;
        using Traits = typename Base::Traits;
        using SFCYs  = typename Traits::SFC_Ys;

        static_assert(Window::NumAccessPerCoord == 4);
        static_assert(Traits::ScalarPerVector == 8);
        static_assert(Traits::PackedSize == 1);
        static_assert(Access >= 0 && Access < Window::NumAccessPerCoord);

        constexpr auto distribution = typename Base::TileDstr{};
        constexpr auto index        = SFCYs::get_index(number<Access>{});
        return distribution.get_ys_to_d_descriptor().calculate_offset(index) / Traits::PackedSize;
    }

    template <index_t Access, typename TileWindow>
    CK_TILE_DEVICE static auto LoadAccess(const TileWindow& tile_window)
    {
        using Window     = remove_cvref_t<TileWindow>;
        using Base       = typename Window::Base;
        using Traits     = typename Base::Traits;
        using SFCYs      = typename Traits::SFC_Ys;
        using VectorType = typename Traits::vector_t;

        constexpr auto distribution = typename Base::TileDstr{};
        constexpr auto input_index  = SFCYs::get_index(number<Access>{});
        const auto partition_index  = get_partition_index(distribution);
        const auto window_coord     = make_tensor_adaptor_coordinate(
            distribution.get_ps_ys_to_xs_adaptor(),
            container_concat(partition_index, to_array<index_t, input_index.size()>(input_index)));
        const auto bottom_index = tile_window.get_window_origin() + window_coord.get_bottom_index();
        const auto bottom_coord = make_tensor_coordinate(
            tile_window.get_bottom_tensor_view().get_tensor_descriptor(), bottom_index);

        return tile_window.get_bottom_tensor_view().template get_vectorized_elements<VectorType>(
            bottom_coord, 0, bool_constant<true>{});
    }

    template <index_t Access, typename TileWindow, typename VectorType>
    CK_TILE_DEVICE static void StoreAccess(const TileWindow& tile_window, const VectorType& value)
    {
        using Window = remove_cvref_t<TileWindow>;
        using Base   = typename Window::Base;
        using Traits = typename Base::Traits;
        using SFCYs  = typename Traits::SFC_Ys;

        static_assert(std::is_same_v<remove_cvref_t<VectorType>, typename Traits::vector_t>);
        constexpr auto distribution = typename Base::TileDstr{};
        constexpr auto input_index  = SFCYs::get_index(number<Access>{});
        const auto partition_index  = get_partition_index(distribution);
        const auto window_coord     = make_tensor_adaptor_coordinate(
            distribution.get_ps_ys_to_xs_adaptor(),
            container_concat(partition_index, to_array<index_t, input_index.size()>(input_index)));
        const auto bottom_index = tile_window.get_window_origin() + window_coord.get_bottom_index();
        const auto bottom_coord = make_tensor_coordinate(
            tile_window.get_bottom_tensor_view().get_tensor_descriptor(), bottom_index);

        tile_window.get_bottom_tensor_view().template set_vectorized_elements<VectorType>(
            bottom_coord, 0, value, bool_constant<true>{});
    }

    template <index_t MHalfIndex, typename TileWindow>
    CK_TILE_DEVICE static MHalf LoadMHalf(const TileWindow& tile_window)
    {
        using Window = remove_cvref_t<TileWindow>;
        using Traits = typename Window::Base::Traits;
        static_assert(MHalfIndex >= 0 && MHalfIndex < kMHalves);

        MHalf result{};
        static_for<0, 2, 1>{}([&](auto i) {
            constexpr index_t access        = MHalfIndex * 2 + decltype(i)::value;
            constexpr index_t source_offset = GetAccessThreadBufferOffset<access, Window>();
            constexpr index_t half_begin    = MHalfIndex * kValuesPerMHalf;
            static_assert(source_offset >= half_begin);
            static_assert(source_offset + Traits::ScalarPerVector <= half_begin + kValuesPerMHalf);
            const auto value = LoadAccess<access>(tile_window);
            static_for<0, Traits::ScalarPerVector, 1>{}([&](auto j) {
                result[source_offset - half_begin + decltype(j)::value] =
                    value.template get_as<bf16_t>()[j];
            });
        });
        return result;
    }

    template <index_t MHalfIndex, typename TileWindow>
    CK_TILE_DEVICE static void StoreMHalf(const TileWindow& tile_window, const MHalf& value)
    {
        using Window = remove_cvref_t<TileWindow>;
        using Traits = typename Window::Base::Traits;
        static_assert(MHalfIndex >= 0 && MHalfIndex < kMHalves);

        static_for<0, 2, 1>{}([&](auto i) {
            constexpr index_t access             = MHalfIndex * 2 + decltype(i)::value;
            constexpr index_t destination_offset = GetAccessThreadBufferOffset<access, Window>();
            constexpr index_t half_begin         = MHalfIndex * kValuesPerMHalf;
            static_assert(destination_offset >= half_begin);
            static_assert(destination_offset + Traits::ScalarPerVector <=
                          half_begin + kValuesPerMHalf);

            typename Traits::vector_t vector;
            static_for<0, Traits::ScalarPerVector, 1>{}([&](auto j) {
                vector.template get_as<bf16_t>()[j] =
                    value[destination_offset - half_begin + decltype(j)::value];
            });
            StoreAccess<access>(tile_window, vector);
        });
    }

    template <index_t MHalfIndex, typename GroupTensor>
    CK_TILE_DEVICE static MHalf ExtractMHalf(const GroupTensor& group)
    {
        static_assert(MHalfIndex >= 0 && MHalfIndex < kMHalves);
        static_assert(GroupTensor::get_thread_buffer_size() == kValuesPerThread);

        MHalf result{};
        constexpr index_t begin = MHalfIndex * kValuesPerMHalf;
        static_for<0, kValuesPerMHalf, 1>{}([&](auto i) {
            result[decltype(i)::value] = group.get_thread_buffer()[begin + decltype(i)::value];
        });
        return result;
    }

    CK_TILE_HOST_DEVICE static constexpr bool ValidateSlotLayout()
    {
        constexpr index_t kK0Begin = 0x00000;
        constexpr index_t kK0End   = 0x0c800;
        constexpr index_t kK1Begin = 0x10000;
        constexpr index_t kK1End   = 0x1c800;
        constexpr index_t kV0Begin = 0x20000;
        constexpr index_t kV0End   = 0x29000;
        constexpr index_t kV1Begin = 0x30000;
        constexpr index_t kV1End   = kArenaBytes;

        constexpr index_t kSlot0Begin = GetSlotBase<0>();
        constexpr index_t kSlot1Begin = GetSlotBase<1>();
        constexpr index_t kSlot2Begin = GetSlotBase<2>();
        constexpr index_t kSlot3Begin = GetSlotBase<3>();

        return kK0Begin == 0 && kK0End <= kSlot0Begin && kSlot0Begin + kSlotBytes <= kK1Begin &&
               kK1End <= kSlot1Begin && kSlot1Begin + kSlotBytes <= kV0Begin &&
               kV0End <= kSlot2Begin && kSlot2Begin + kSlotBytes <= kSlot3Begin &&
               kSlot3Begin + kSlotBytes <= kV1Begin && kV1End == kArenaBytes;
    }
};

static_assert(FmhaD192DeferredPLdsLayout::ValidateSlotLayout());
static_assert(FmhaD192DeferredPLdsLayout::GetGroupFootprintBytes<32>() == 0x2000);
static_assert(FmhaD192DeferredPLdsLayout::GetGroupFootprintBytes<40>() <=
              FmhaD192DeferredPLdsLayout::kSlotBytes);
static_assert(FmhaD192DeferredPLdsLayout::GetGroupFootprintBytes<48>() <=
              FmhaD192DeferredPLdsLayout::kSlotBytes);
static_assert(FmhaD192DeferredPLdsLayout::GetGroupFootprintBytes<56>() <=
              FmhaD192DeferredPLdsLayout::kSlotBytes);

} // namespace ck_tile
