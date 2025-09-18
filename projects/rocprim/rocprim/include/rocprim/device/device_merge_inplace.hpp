// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef ROCPRIM_DEVICE_DEVICE_MERGE_INPLACE_HPP_
#define ROCPRIM_DEVICE_DEVICE_MERGE_INPLACE_HPP_

#include "../common.hpp"
#include "../config.hpp"
#include "../functional.hpp"

#include "../detail/merge_path.hpp"
#include "../detail/temp_storage.hpp"
#include "../detail/various.hpp"

#include "../block/block_store.hpp"
#include "../device/config_types.hpp"
#include "../device/device_merge_inplace_config.hpp"
#include "../intrinsics/bit.hpp"
#include "../intrinsics/thread.hpp"
#include "../thread/thread_search.hpp"

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <limits>

BEGIN_ROCPRIM_NAMESPACE

/// \addtogroup devicemodule
/// @{

namespace detail
{

/// \brief implementation detail of merge inplace
template<size_t GlobalMergeBlockSize,
         size_t BlockMergeBlockSize,
         size_t BlockMergeIPT,
         typename IteratorT,
         typename OffsetT,
         typename BinaryFunction>
struct merge_inplace_impl
{
    using iterator_t = IteratorT;
    using value_t    = typename std::iterator_traits<iterator_t>::value_type;
    using offset_t   = OffsetT;

    static constexpr size_t global_merge_block_size      = GlobalMergeBlockSize;
    static constexpr size_t block_merge_block_size       = BlockMergeBlockSize;
    static constexpr size_t block_merge_items_per_thread = BlockMergeIPT;
    static constexpr size_t block_merge_items_per_block
        = block_merge_block_size * block_merge_items_per_thread;

    static constexpr offset_t no_split = -1;

    struct pivot_t
    {
        // rocprim::merge_path_search uses '.x' and '.y', but that isn't very descriptive.
        // so we union it with more descriptive names
        union
        {
            offset_t left;
            offset_t x;
        };
        union
        {
            offset_t right;
            offset_t y;
        };

        ROCPRIM_DEVICE
        pivot_t& offset(offset_t left_offset, offset_t right_offset)
        {
            left += left_offset;
            right += right_offset;
            return *this;
        }
    };

    /// \brief describes two ranges [begin, split) and [split, end)
    struct work_t
    {
        offset_t       begin;
        offset_t       split;
        offset_t       end;

        ROCPRIM_DEVICE ROCPRIM_INLINE
        constexpr bool is_valid() const
        {
            return begin <= split && split <= end;
        }

        ROCPRIM_DEVICE ROCPRIM_INLINE
        constexpr bool has_work() const
        {
            return begin < split && split < end;
        }

        ROCPRIM_DEVICE ROCPRIM_INLINE
        offset_t total_size() const
        {
            return end - begin;
        }

        ROCPRIM_DEVICE ROCPRIM_INLINE
        offset_t left_size() const
        {
            return split - begin;
        }

        ROCPRIM_DEVICE ROCPRIM_INLINE
        offset_t right_size() const
        {
            return end - split;
        }

        static constexpr ROCPRIM_DEVICE ROCPRIM_INLINE
        work_t invalid_work()
        {
            return work_t{0, no_split, 0};
        }
    };

    struct scratch_t
    {
        offset_t iteration;
        bool     is_large_merge;
    };

    /// \brief finds the `work_t` and its id by descending the binary tree `work_tree`.
    ROCPRIM_DEVICE ROCPRIM_INLINE
    static work_t reconstruct_work(const offset_t  worker_global_id,
                                   const offset_t* work_tree,
                                   work_t          work,
                                   uint32_t&       work_id,
                                   const uint32_t  iteration)
    {
        // the work tree is a binary search tree were each node is the position of the split
        // if we decent from a parent node into its left child, the upperbound will be the
        // split position contained in the parent node. likewise, the lowerbound is set when
        // we decent to the right.
        work_id    = 1;
        work.split = work_tree[work_id];

        // we need an upper bound since another thread may have already written the next level
        for(uint32_t i = 0; i < iteration; ++i)
        {
            // move to next layer in binary tree
            work_id <<= 1;

            // check which side of the binary tree we descend
            if(worker_global_id >= work.split)
            {
                // worker id is right of split
                work.begin = work.split;
                work_id |= 1;
            }
            else
            {
                work.end = work.split;
            }
            work.split = work_tree[work_id];

            // early exit if we encounter a leaf!
            if(work.split == no_split)
            {
                break;
            }
        }
        return work;
    }

    /// \brief Reconstructs `work_t` from an index by traveling up the tree to the root.
    /// If `find_split` is `true`, also try to first find the first id that is valid
    /// while discarding duplicates.
    template<bool find_split = false>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    static work_t reconstruct_work_from_id(work_t          work,
                                           const offset_t* work_tree,
                                           offset_t        work_id,
                                           const uint32_t  iteration)
    {
        bool need_begin = true;
        bool need_end   = true;
        work.split      = work_tree[work_id];

        for(uint32_t i = 0; i < iteration && (need_begin || need_end); ++i)
        {
            // odd ids are right branches
            bool is_right = work_id & 1;

            // if this isn't a split, move a layer up
            if(find_split && work.split == no_split)
            {
                // if the sibling is also not a split, prune the right branch
                if(is_right && work_tree[work_id & ~1] == no_split)
                {
                    return work_t::invalid_work();
                }

                work_id >>= 1;
                work.split = work_tree[work_id];
                continue;
            }

            // move to the next layer
            work_id >>= 1;

            // if the parent is a leaf do not update our internal state
            if(work_tree[work_id] == no_split)
            {
                // cut off the right branch and let the left branch inherit the parent
                if(is_right)
                {
                    return work_t::invalid_work();
                }
                continue;
            }

            // if we're the right child, then the parent split is the left bound
            if(is_right && work.begin == 0)
            {
                work.begin = work_tree[work_id];
                need_begin = false;
            }

            // if we're the left child, then the parent split is the right bound
            if(!is_right && need_end)
            {
                work.end = work_tree[work_id];
                need_end = false;
            }
        }

        return work;
    }

    using block_merge_block_store
        = block_store<value_t, block_merge_block_size, block_merge_items_per_thread>;

    static auto get_num_global_divisions(size_t left_size, size_t right_size)
    {
        const offset_t max_size = max(left_size, right_size);
        const int32_t  set_bits = std::numeric_limits<size_t>::digits - clz(max_size);

        // compute 2 + ceil(log_2(max(left, right))) - log_2(items_per_thread)
        return max(2, 2 + set_bits - Log2<block_merge_items_per_block>::VALUE);
    }

    struct context_t
    {
        offset_t       iteration;
        BinaryFunction compare_function;

        const offset_t left_work_size;
        const offset_t right_work_size;
        const uint32_t grid_thread_id;
        const uint32_t grid_thread_count;

        // creates a default work struct
        ROCPRIM_DEVICE ROCPRIM_INLINE
        work_t get_initial_work() const
        {
            return work_t{0, left_work_size, left_work_size + right_work_size};
        }

        ROCPRIM_DEVICE ROCPRIM_INLINE
        offset_t get_total_work_size() const
        {
            return left_work_size + right_work_size;
        }
    };

    ROCPRIM_DEVICE ROCPRIM_INLINE
    static void find_pivots(const context_t& context,
                            iterator_t       data,
                            const offset_t*  work_tree,
                            pivot_t*         pivot_heap,
                            const scratch_t* /* scratch*/)
    {
        const work_t   initial_work  = context.get_initial_work();
        const uint32_t work_id_begin = 1 << context.iteration;
        const uint32_t work_id_end   = work_id_begin * 2;

        // for all work, get the optimal pivot
        for(uint32_t work_id = work_id_begin + context.grid_thread_id; work_id < work_id_end;
            work_id += context.grid_thread_count)
        {
            const work_t work
                = reconstruct_work_from_id(initial_work, work_tree, work_id, context.iteration);

            // skip invalid work
            if(!work.is_valid() || work.total_size() <= block_merge_items_per_block)
            {
                continue;
            }

            pivot_t pivot;
            rocprim::merge_path_search(work.total_size() / 2,
                                       data + work.begin,
                                       data + work.split,
                                       work.left_size(),
                                       work.right_size(),
                                       pivot,
                                       context.compare_function);

            pivot_heap[work_id] = pivot.offset(work.begin, work.split);
        }
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    static void rotate_p1(const context_t& context,
                          iterator_t       data,
                          const offset_t*  work_tree,
                          const pivot_t*   pivot_heap,
                          const scratch_t* /* scratch */)
    {
        const work_t initial_work = context.get_initial_work();

        // first reverse to start rotation
        // we can use `floor(N / 2)`  number of workers for a rotation of total size N since
        // each worker will swap 2 items
        for(offset_t worker_global_id = context.grid_thread_id * 2;
            worker_global_id < context.get_total_work_size();
            worker_global_id += context.grid_thread_count * 2)
        {
            uint32_t     work_id = 1;
            const work_t work    = reconstruct_work(worker_global_id,
                                                 work_tree,
                                                 initial_work,
                                                 work_id,
                                                 context.iteration);

            // skip invalid work
            if(!work.has_work() || work.total_size() <= block_merge_items_per_block)
            {
                continue;
            }

            const offset_t work_offset = (worker_global_id - work.begin) / 2;
            const pivot_t  pivot       = pivot_heap[work_id];

            if(work_offset >= (pivot.right - pivot.left) / 2)
            {
                continue;
            }

            const offset_t mid_l = (pivot.left + work.split) / 2 - pivot.left;

            // reverse the left and right array separately
            const bool is_left = work_offset < mid_l;

            const offset_t left_start     = is_left ? pivot.left : work.split;
            const offset_t right_end      = is_left ? work.split : pivot.right; // exclusive
            const offset_t reverse_offset = is_left ? work_offset : work_offset - mid_l;

            const offset_t left_index  = left_start + reverse_offset;
            const offset_t right_index = right_end - reverse_offset - 1;

            if(left_index != right_index)
            {
                rocprim::swap(data[left_index], data[right_index]);
            }
        }
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    static void rotate_p2(const context_t& context,
                          iterator_t       data,
                          const offset_t*  work_tree,
                          const pivot_t*   pivot_heap,
                          const scratch_t* /* scratch */)
    {
        const work_t initial_work = context.get_initial_work();

        // second reverse step to complete rotation. as before,
        // we only use half the number of workers.
        for(offset_t worker_global_id = context.grid_thread_id * 2;
            worker_global_id < context.get_total_work_size();
            worker_global_id += context.grid_thread_count * 2)
        {
            uint32_t     work_id = 1;
            const work_t work    = reconstruct_work(worker_global_id,
                                                 work_tree,
                                                 initial_work,
                                                 work_id,
                                                 context.iteration);

            // skip invalid work
            if(!work.has_work() || work.total_size() <= block_merge_items_per_block)
            {
                continue;
            }

            const offset_t work_offset = (worker_global_id - work.begin) / 2;
            const pivot_t  pivot       = pivot_heap[work_id];

            if(work_offset >= (pivot.right - pivot.left) / 2)
            {
                continue;
            }

            const auto left_index  = pivot.left + work_offset;
            const auto right_index = pivot.right - work_offset - 1;

            if(left_index != right_index)
            {
                rocprim::swap(data[left_index], data[right_index]);
            }
        }
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    static void update_work_tree(const context_t& context,
                                 const iterator_t /* data */,
                                 offset_t*      work_tree,
                                 const pivot_t* pivot_heap,
                                 scratch_t*     scratch)
    {
        const work_t   initial_work       = context.get_initial_work();
        const uint32_t work_id_begin      = 1 << context.iteration;
        const uint32_t work_id_end        = work_id_begin * 2;
        bool           is_large_work_item = false;

        // enqueue future work by adding it to the work tree
        for(uint32_t work_id = work_id_begin + context.grid_thread_id; work_id < work_id_end;
            work_id += context.grid_thread_count)
        {
            const work_t work
                = reconstruct_work_from_id(initial_work, work_tree, work_id, context.iteration);

            // default splits:
            offset_t new_split   = work.split;
            offset_t left_split  = no_split;
            offset_t right_split = no_split;

            // if this node isn't a leaf and the work should not be done in the block level merge
            if(work.is_valid() && work.total_size() > block_merge_items_per_block)
            {
                const pivot_t pivot = pivot_heap[work_id];
                if(!(pivot.left == work.split && pivot.right == work.end))
                {
                    // the pivots describe the child work, but we have to adjust
                    // the work's split since we rotated around that value
                    new_split   = pivot.left + pivot.right - work.split;
                    left_split  = pivot.left;
                    right_split = pivot.right;
                }

                const offset_t left_size  = left_split == no_split ? 0 : new_split - work.begin;
                const offset_t right_size = right_split == no_split ? 0 : work.end - new_split;

                is_large_work_item |= max(left_size, right_size) > block_merge_items_per_block;

                // mark empty nodes in the tree as a leaf via `no_split`
                if(left_size == 0)
                {
                    left_split = no_split;
                }
                if(right_size == 0)
                {
                    right_split = no_split;
                }
            }

            // write offset for the descendents
            const uint32_t child_work_id = work_id << 1;

            // write descendents to global memory
            work_tree[work_id]           = new_split;
            work_tree[child_work_id]     = left_split;
            work_tree[child_work_id | 1] = right_split;
        }

        if(is_large_work_item)
        {
            scratch->is_large_merge = true;
        }
    }

    // dispatch that sets up the required context for a merge phase.
    // this is a device function because global kernels require the
    // typename argument symbol to be available on host which is not
    // the case here.
    template<typename DispatchFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    static void dispatch_merge_phase(DispatchFunction dispatch_function,
                                     iterator_t       data,
                                     offset_t         left_size,
                                     offset_t         right_size,
                                     BinaryFunction   compare_function,
                                     offset_t*        work_tree,
                                     pivot_t*         pivot_heap,
                                     scratch_t*       scratch)
    {
        uint32_t grid_thread_id  = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t grid_thread_dim = gridDim.x * blockDim.x;

        context_t context{scratch->iteration,
                          compare_function,
                          left_size,
                          right_size,
                          grid_thread_id,
                          grid_thread_dim};

        dispatch_function(context, data, work_tree, pivot_heap, scratch);
    }

// macro to convert the device dispatch into a kernel to prevent
// copy-a-pasta'ing the boiler plate a bunch of times.
#define ROCPRIM_DETAIL_GENERATE_DISPATCH_KERNEL(NAME)                         \
    ROCPRIM_KERNEL static void NAME##_kernel(iterator_t     data,             \
                                             offset_t       left_size,        \
                                             offset_t       right_size,       \
                                             BinaryFunction compare_function, \
                                             offset_t*      work_tree,        \
                                             pivot_t*       pivot_heap,       \
                                             scratch_t*     scratch)          \
    {                                                                         \
        dispatch_merge_phase(NAME,                                            \
                             data,                                            \
                             left_size,                                       \
                             right_size,                                      \
                             compare_function,                                \
                             work_tree,                                       \
                             pivot_heap,                                      \
                             scratch);                                        \
    }

    ROCPRIM_DETAIL_GENERATE_DISPATCH_KERNEL(find_pivots);
    ROCPRIM_DETAIL_GENERATE_DISPATCH_KERNEL(rotate_p1);
    ROCPRIM_DETAIL_GENERATE_DISPATCH_KERNEL(rotate_p2);
    ROCPRIM_DETAIL_GENERATE_DISPATCH_KERNEL(update_work_tree);
#undef ROCPRIM_DETAIL_GENERATE_DISPATCH_KERNEL

    static ROCPRIM_KERNEL
    void block_merge_kernel(iterator_t     data,
                            size_t         num_items,
                            BinaryFunction compare_function,
                            offset_t*      work_tree,
                            scratch_t*     scratch_storage)
    {
        // this kernel does not use grid-wide sync, so no need for
        // cooperative groups
        const uint32_t grid_size       = rocprim::detail::grid_size<0>();
        const uint32_t block_id        = rocprim::flat_block_id();
        const uint32_t block_thread_id = rocprim::flat_block_thread_id();
        const uint32_t iteration       = scratch_storage->iteration;

        work_t initial_work{0, no_split, num_items};

        // domain of the work ids
        const uint32_t work_id_begin = 1 << iteration;
        const uint32_t work_id_end   = work_id_begin * 2;

        value_t thread_data[block_merge_items_per_thread];

        // grid stride over the work ids
        for(uint32_t work_id = work_id_begin + block_id; work_id < work_id_end;
            work_id += grid_size)
        {
            const work_t work
                = reconstruct_work_from_id<true>(initial_work, work_tree, work_id, iteration);

            bool has_work = work.has_work() && work.total_size() <= block_merge_items_per_block;

            if(has_work)
            {
                // divide work over threads via merge path
                const offset_t diagonal = block_merge_items_per_thread * block_thread_id;

                pivot_t pivot;
                rocprim::merge_path_search(diagonal,
                                           data + work.begin,
                                           data + work.split,
                                           work.left_size(),
                                           work.right_size(),
                                           pivot,
                                           compare_function);
                pivot.offset(work.begin, work.split);

                // serial merge
                range_t<offset_t> range{
                    pivot.left,
                    work.split,
                    pivot.right,
                    work.end,
                };
                serial_merge(data, thread_data, range, compare_function);

                // there are no partial blocks working on this, so a
                // block sync in this conditional can be done safely
                rocprim::syncthreads();

                block_merge_block_store{}.store(data + work.begin, thread_data, work.total_size());
            }
        }
    }
};
} // namespace detail

/// \brief Parallel merge inplace primitive for device level.
///
/// The `merge_inplace` function performs a device-wide merge in place. It merges two ordered sets
/// of input values based on a comparison function using significantly less temporary storage
/// compared to `merge`.
///
/// \warning This function prioritizes temporary storage over speed. In most cases using `merge`
/// and a device copy is significantly faster.
///
/// \par Overview
/// * The function can write intermediate values to the data array while the algorithm is running.
/// * Returns the required size of `temporary_storage` in `storage_size` if `temporary_storage` is a
/// null pointer.
/// * Accepts a custom `compare_function`.
///
/// \tparam Config Configuration of the primitive, must be `default_config`.
/// \tparam Iterator Random access iterator type for the input and output range. Must meet the
/// requirements of `std::random_access_iterator`.
/// \tparam BinaryFunction Binary function type that is used for the comparison.
///
/// \param [in] temporary_storage Pointer to a device-accessible temporary storage. When a null
/// pointer is passed the required allocation size in bytes is written to `storage_size` and the
/// function returns `hipSuccess` without performing the merge operation.
/// \param [in,out] storage_size Reference to size in bytes of `temporary_storage`.
/// \param [in,out] data Iterator to the first value to merge.
/// \param [in] left_size Number of elements in the first input range.
/// \param [in] right_size Number of elements in the second input range.
/// \param [in] compare_function Binary operation function that will be used for comparison. The
/// signature of the function should be equivalent to the following: `bool f(const T &a, const T &b);`.
/// The signature does not need to have `const &`, but the function object must not modify
/// the objects passed to it. The default value is `BinaryFunction()`.
/// \param [in] stream The HIP stream object. Default is `0` (`hipStreamDefault`).
/// \param [in] debug_synchronous If `true`, forces a device synchronization after every kernel
/// launch in order to check for errors. Default value is `false`.
///
/// \returns `hipSuccess` (`0`) after succesful merge; otherwise a HIP runtime error of type
/// `hipError_t`.
///
/// \par Example
/// \parblock
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// size_t left_size;  // e.g. 4
/// size_t right_size; // e.g. 4
/// int*   data;       // e.g. [1, 3, 5, 7, 0, 2, 4, 6]
/// // output: [0, 1, 2, 3, 4, 5, 6, 7]
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
///
/// rocprim::merge_inplace(
///     temporary_storage_ptr,
///     temporary_storage_size_bytes,
///     data,
///     left_size,
///     right_size);
///
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// rocprim::merge_inplace(
///     temporary_storage_ptr,
///     temporary_storage_size_bytes,
///     data,
///     left_size,
///     right_size);
/// \endcode
/// \endparblock
template<class Config = default_config,
         class Iterator,
         class BinaryFunction
         = ::rocprim::less<typename std::iterator_traits<Iterator>::value_type>>
inline hipError_t merge_inplace(void*             temporary_storage,
                                size_t&           storage_size,
                                Iterator          data,
                                size_t            left_size,
                                size_t            right_size,
                                BinaryFunction    compare_function  = BinaryFunction(),
                                const hipStream_t stream            = 0,
                                bool              debug_synchronous = false)
{
    using config = detail::default_or_custom_config<Config, merge_inplace_config<>>;

    constexpr size_t global_block_size      = config::global_merge_block_size;
    constexpr size_t block_block_size       = config::block_merge_block_size;
    constexpr size_t block_items_per_thread = config::block_merge_items_per_thread;

    using impl = detail::merge_inplace_impl<global_block_size,
                                            block_block_size,
                                            block_items_per_thread,
                                            Iterator,
                                            size_t,
                                            BinaryFunction>;

    typename impl::offset_t*  work_storage    = nullptr;
    typename impl::pivot_t*   pivot_storage   = nullptr;
    typename impl::scratch_t* scratch_storage = nullptr;

    size_t num_divisions = impl::get_num_global_divisions(left_size, right_size);

    ROCPRIM_RETURN_ON_ERROR(detail::temp_storage::partition(
        temporary_storage,
        storage_size,
        detail::temp_storage::make_linear_partition(
            detail::temp_storage::ptr_aligned_array(&work_storage, 2ULL << num_divisions),
            detail::temp_storage::ptr_aligned_array(&pivot_storage, 1ULL << num_divisions),
            detail::temp_storage::ptr_aligned_array(&scratch_storage, 1))));

    if(temporary_storage == nullptr)
    {
        if(debug_synchronous)
        {
            std::cout << "device_merge_inplace\n"
                      << "  left  size     : " << left_size << "\n"
                      << "  right size     : " << right_size << "\n"
                      << "  num iterations : " << num_divisions << "\n"
                      << "  requires " << storage_size << " bytes of temporary storage"
                      << std::endl;
        }
        return hipSuccess;
    }

    if(left_size == 0 || right_size == 0)
    {
        return hipSuccess;
    }

    std::chrono::high_resolution_clock::time_point t_time;

    if(debug_synchronous)
    {
        t_time = std::chrono::high_resolution_clock::now();
    }

    bool is_large_merge               = left_size + right_size > impl::block_merge_items_per_block;
    typename impl::offset_t iteration = 0;

    ROCPRIM_RETURN_ON_ERROR(hipMemcpyAsync(&work_storage[1],
                                           &left_size,
                                           sizeof(decltype(*work_storage)),
                                           hipMemcpyHostToDevice,
                                           stream));
    ROCPRIM_RETURN_ON_ERROR(hipMemcpyAsync(&scratch_storage->iteration,
                                           &iteration,
                                           sizeof(decltype(scratch_storage->iteration)),
                                           hipMemcpyHostToDevice,
                                           stream));

    if(debug_synchronous)
    {
        ROCPRIM_RETURN_ON_ERROR(hipStreamSynchronize(stream));

        float delta = rocprim::detail::update_time_point(t_time);
        std::cout << "  init memcpy: " << delta << "ms\n";
    }

    int grid_dim = 0;
    while(is_large_merge)
    {
        // clear this flag
        ROCPRIM_RETURN_ON_ERROR(hipMemset(&scratch_storage->is_large_merge, false, sizeof(bool)));

        ROCPRIM_RETURN_ON_ERROR(detail::grid_dim_for_max_active_blocks(grid_dim,
                                                                       global_block_size,
                                                                       impl::find_pivots_kernel,
                                                                       stream));
        impl::find_pivots_kernel<<<grid_dim, global_block_size>>>(data,
                                                                  left_size,
                                                                  right_size,
                                                                  compare_function,
                                                                  work_storage,
                                                                  pivot_storage,
                                                                  scratch_storage);
        ROCPRIM_RETURN_ON_ERROR(hipGetLastError());
        if(debug_synchronous)
        {
            ROCPRIM_RETURN_ON_ERROR(hipStreamSynchronize(stream));

            float delta = rocprim::detail::update_time_point(t_time);
            std::cout << "  find_pivots_kernel: " << delta << "ms\n";
        }

        ROCPRIM_RETURN_ON_ERROR(detail::grid_dim_for_max_active_blocks(grid_dim,
                                                                       global_block_size,
                                                                       impl::rotate_p1_kernel,
                                                                       stream));
        impl::rotate_p1_kernel<<<grid_dim, global_block_size>>>(data,
                                                                left_size,
                                                                right_size,
                                                                compare_function,
                                                                work_storage,
                                                                pivot_storage,
                                                                scratch_storage);
        ROCPRIM_RETURN_ON_ERROR(hipGetLastError());
        if(debug_synchronous)
        {
            ROCPRIM_RETURN_ON_ERROR(hipStreamSynchronize(stream));

            float delta = rocprim::detail::update_time_point(t_time);
            std::cout << "  rotate_p1_kernel: " << delta << "ms\n";
        }

        ROCPRIM_RETURN_ON_ERROR(detail::grid_dim_for_max_active_blocks(grid_dim,
                                                                       global_block_size,
                                                                       impl::rotate_p2_kernel,
                                                                       stream));
        impl::rotate_p2_kernel<<<grid_dim, global_block_size>>>(data,
                                                                left_size,
                                                                right_size,
                                                                compare_function,
                                                                work_storage,
                                                                pivot_storage,
                                                                scratch_storage);
        ROCPRIM_RETURN_ON_ERROR(hipGetLastError());
        if(debug_synchronous)
        {
            ROCPRIM_RETURN_ON_ERROR(hipStreamSynchronize(stream));

            float delta = rocprim::detail::update_time_point(t_time);
            std::cout << "  rotate_p2_kernel: " << delta << "ms\n";
        }

        ROCPRIM_RETURN_ON_ERROR(
            detail::grid_dim_for_max_active_blocks(grid_dim,
                                                   global_block_size,
                                                   impl::update_work_tree_kernel,
                                                   stream));
        impl::update_work_tree_kernel<<<grid_dim, global_block_size>>>(data,
                                                                       left_size,
                                                                       right_size,
                                                                       compare_function,
                                                                       work_storage,
                                                                       pivot_storage,
                                                                       scratch_storage);
        ROCPRIM_RETURN_ON_ERROR(hipGetLastError());
        if(debug_synchronous)
        {
            float delta = rocprim::detail::update_time_point(t_time);
            std::cout << "  update_work_tree_kernel: " << delta << "ms\n";
        }

        ROCPRIM_RETURN_ON_ERROR(hipMemcpy(&is_large_merge,
                                          &scratch_storage->is_large_merge,
                                          sizeof(bool),
                                          hipMemcpyDeviceToHost));

        ++iteration;
        ROCPRIM_RETURN_ON_ERROR(hipMemcpy(&scratch_storage->iteration,
                                          &iteration,
                                          sizeof(typename impl::offset_t),
                                          hipMemcpyHostToDevice));
        if(debug_synchronous)
        {
            float delta = rocprim::detail::update_time_point(t_time);
            std::cout << "  iteration memcpy: " << delta << "ms\n";
        }
    }

    // since we potentially may have more blocks than is legally
    // allowed on, we simplify launching by doing grid stride per block
    int block_merge_grid_size = 0;
    ROCPRIM_RETURN_ON_ERROR(detail::grid_dim_for_max_active_blocks(block_merge_grid_size,
                                                                   block_block_size,
                                                                   impl::block_merge_kernel,
                                                                   stream));

    const int min_grid_size = rocprim::min(
        block_merge_grid_size,
        static_cast<int>(rocprim::detail::ceiling_div(left_size + right_size, block_block_size)));

    if(debug_synchronous)
    {
        std::cout << "block_merge_kernel\n"
                  << "  grid_size     : " << min_grid_size << "\n"
                  << "  block_size    : " << block_block_size << std::endl;
    }

    // each of the sub merging problem can be solved within a block
    impl::block_merge_kernel<<<min_grid_size, block_block_size, 0, stream>>>(data,
                                                                             left_size + right_size,
                                                                             compare_function,
                                                                             work_storage,
                                                                             scratch_storage);
    ROCPRIM_RETURN_ON_ERROR(hipGetLastError());
    if(debug_synchronous)
    {
        ROCPRIM_RETURN_ON_ERROR(hipStreamSynchronize(stream));

        float delta = rocprim::detail::update_time_point(t_time);
        std::cout << "  block_merge_kernel: " << delta << "ms\n";
    }

    return hipSuccess;
}

/// @}
// end of group devicemodule

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_MERGE_INPLACE_HPP_
