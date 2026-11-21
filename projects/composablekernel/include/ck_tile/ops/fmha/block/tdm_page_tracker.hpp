// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/core/tensor/tile_window.hpp"

namespace ck_tile {

// Pages K or V for the TDM pipelines. The interface follows PageBlockNavigator for a tensor
// paged along dimension 0 (seqlen rows): windows hold page-local origins over a per-page view
// (last_view for the last page), and to_global_window_origin() recovers the sequence row.
// Two differences matter to a TDM pipeline:
// - A window past the last page stays on the last page with its origin past last_view, so a
//   look-ahead TDM load there reads zero rows; PageBlockNavigator would give it a null page.
// - The block table is read through the constant address space, i.e. with scalar loads. A
//   generic read inside a loop with TDM or barrier intrinsics becomes a VMEM load with an
//   immediate wait.
template <typename DataType_, typename TensorView_>
struct TdmPageTracker
{
    using DataType     = DataType_;
    using TensorView   = remove_cvref_t<TensorView_>;
    using WindowOrigin = multi_index<2>;
    static_assert(
        std::is_same_v<remove_cvref_t<DataType>, remove_cvref_t<typename TensorView::DataType>>);

    CK_TILE_HOST_DEVICE constexpr TdmPageTracker(copy_const_t<DataType, void>* physical_blocks,
                                                 long_index_t block_stride,
                                                 long_index_t fixed_offset,
                                                 const int32_t* block_indices,
                                                 index_t num_blocks,
                                                 index_t page_block_size,
                                                 const TensorView& complete_view,
                                                 const TensorView& last_view)
        : physical_blocks_{reinterpret_cast<DataType*>(physical_blocks)},
          block_stride_{block_stride},
          fixed_offset_{fixed_offset},
          block_indices_{block_indices},
          num_blocks_{num_blocks},
          page_block_size_{page_block_size},
          complete_view_{complete_view},
          last_view_{last_view}
    {
    }

    // Returns (page index, window) for a window at sequence row origin.at(0).
    template <typename WindowLengths, typename... TileDistribution>
    CK_TILE_HOST_DEVICE auto make_tile_window(const WindowLengths& window_lengths,
                                              const WindowOrigin& origin,
                                              const TileDistribution&... tile_distribution) const
    {
        const index_t block = get_block_index(origin);
        auto window         = ck_tile::make_tile_window(view_of(block),
                                                window_lengths,
                                                to_local_window_origin(block, origin),
                                                tile_distribution...);
        window.set_bottom_tensor_view_data_ptr(get_block_ptr(block));
        return make_tuple(block, window);
    }

    // Moves the window by step rows and returns its new page index.
    template <typename TileWindow>
    CK_TILE_HOST_DEVICE index_t
    move_tile_window(index_t block_index,
                     TileWindow& tile_window,
                     const typename remove_cvref_t<TileWindow>::BottomTensorIndex& step) const
    {
        ck_tile::move_tile_window(tile_window, step);

        const WindowOrigin origin =
            to_global_window_origin(block_index, tile_window.get_window_origin());
        const index_t new_block = get_block_index(origin);

        tile_window.bottom_tensor_view_.desc_ = view_of(new_block).get_tensor_descriptor();
        tile_window.set_window_origin(to_local_window_origin(new_block, origin));
        tile_window.set_bottom_tensor_view_data_ptr(get_block_ptr(new_block));
        return new_block;
    }

    // Physical page id of the window after a move by step (same as PageBlockNavigator), to
    // pass to the move_tile_window overload below so the table is read once per move.
    template <typename TileWindow>
    CK_TILE_HOST_DEVICE int32_t
    prefetch_table_id(index_t block_index,
                      const TileWindow& tile_window,
                      const typename remove_cvref_t<TileWindow>::BottomTensorIndex& step) const
    {
        return load_block_index(get_block_index(moved_origin(block_index, tile_window, step)));
    }

    template <typename TileWindow>
    CK_TILE_HOST_DEVICE index_t
    move_tile_window(index_t block_index,
                     TileWindow& tile_window,
                     const typename remove_cvref_t<TileWindow>::BottomTensorIndex& step,
                     int32_t physical_id) const
    {
        const WindowOrigin origin = moved_origin(block_index, tile_window, step);
        const index_t new_block   = get_block_index(origin);
        relocate(tile_window,
                 origin,
                 new_block,
                 physical_blocks_ + physical_id * block_stride_ + fixed_offset_);
        return new_block;
    }

    CK_TILE_HOST_DEVICE WindowOrigin to_global_window_origin(index_t block_index,
                                                             const WindowOrigin& local) const
    {
        return make_multi_index(block_index * page_block_size_ + local.at(number<0>{}),
                                local.at(number<1>{}));
    }

    // Page of a sequence row; saturates at the last page (see the class comment).
    CK_TILE_HOST_DEVICE index_t get_block_index(const WindowOrigin& origin) const
    {
        return min(integer_divide_floor(origin.at(number<0>{}), page_block_size_), num_blocks_ - 1);
    }

    CK_TILE_HOST_DEVICE WindowOrigin to_local_window_origin(index_t block_index,
                                                            const WindowOrigin& origin) const
    {
        return make_multi_index(origin.at(number<0>{}) - block_index * page_block_size_,
                                origin.at(number<1>{}));
    }

    CK_TILE_HOST_DEVICE const TensorView& view_of(index_t block_index) const
    {
        return block_index == num_blocks_ - 1 ? last_view_ : complete_view_;
    }

    CK_TILE_HOST_DEVICE DataType* get_block_ptr(index_t block_index) const
    {
        return physical_blocks_ + load_block_index(block_index) * block_stride_ + fixed_offset_;
    }

    private:
    template <typename TileWindow>
    CK_TILE_HOST_DEVICE WindowOrigin
    moved_origin(index_t block_index,
                 const TileWindow& tile_window,
                 const typename remove_cvref_t<TileWindow>::BottomTensorIndex& step) const
    {
        return to_global_window_origin(block_index, tile_window.get_window_origin() + step);
    }

    template <typename TileWindow>
    CK_TILE_HOST_DEVICE void relocate(TileWindow& tile_window,
                                      const WindowOrigin& origin,
                                      index_t block_index,
                                      DataType* page) const
    {
        tile_window.bottom_tensor_view_.desc_ = view_of(block_index).get_tensor_descriptor();
        tile_window.set_window_origin(to_local_window_origin(block_index, origin));
        tile_window.set_bottom_tensor_view_data_ptr(page);
    }

    CK_TILE_HOST_DEVICE int32_t load_block_index(index_t block_index) const
    {
#if defined(__HIP_DEVICE_COMPILE__)
        return cast_pointer_to_constant_address_space(block_indices_)[block_index];
#else
        return block_indices_[block_index];
#endif
    }

    DataType* physical_blocks_;
    long_index_t block_stride_;
    long_index_t fixed_offset_;
    const int32_t* block_indices_;
    index_t num_blocks_;
    index_t page_block_size_;
    TensorView complete_view_;
    TensorView last_view_;
};

template <typename DataType, typename TensorView>
CK_TILE_HOST_DEVICE auto make_tdm_page_tracker(copy_const_t<DataType, void>* physical_blocks,
                                               long_index_t block_stride,
                                               long_index_t fixed_offset,
                                               const int32_t* block_indices,
                                               index_t num_blocks,
                                               index_t page_block_size,
                                               const TensorView& complete_view,
                                               const TensorView& last_view)
{
    return TdmPageTracker<DataType, TensorView>(physical_blocks,
                                                block_stride,
                                                fixed_offset,
                                                block_indices,
                                                num_blocks,
                                                page_block_size,
                                                complete_view,
                                                last_view);
}

} // namespace ck_tile
