// Copyright (c) 2017-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_BLOCK_DETAIL_BLOCK_REDUCE_RAKING_REDUCE_HPP_
#define ROCPRIM_BLOCK_DETAIL_BLOCK_REDUCE_RAKING_REDUCE_HPP_

#include <type_traits>

#include "../../config.hpp"
#include "../../detail/various.hpp"

#include "../../functional.hpp"
#include "../../intrinsics.hpp"

#include "../../warp/warp_reduce.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

// Class for fast storage/load of large object's arrays in local memory
// for sequential access from consecutive threads.
// For small types reproduces array
template<class T, int n, typename = void>
class fast_array
{
public:
    ROCPRIM_HOST_DEVICE T get(int index) const
    {
        return data[index];
    }

    ROCPRIM_HOST_DEVICE void set(int index, T value)
    {
        data[index] = value;
    }

private:
    T data[n];
};

// For large types reduces bank conflicts to minimum
// by values sliced into int32_t and each slice stored continuously.
// Treatment of []= operator by proxy objects
#ifndef DOXYGEN_SHOULD_SKIP_THIS
template<class T, int n>
class fast_array<T, n, std::enable_if_t<(sizeof(T) > sizeof(int32_t))>>
{
public:
    ROCPRIM_HOST_DEVICE T get(int index) const
    {
        T result;
        ROCPRIM_UNROLL
        for(int i = 0; i < words_no; i++)
        {
            const size_t s = std::min(sizeof(int32_t), sizeof(T) - i * sizeof(int32_t));
#ifdef __HIP_CPU_RT__
            std::memcpy(reinterpret_cast<char*>(&result) + i * sizeof(int32_t),
                        data + index + i * n,
                        s);
#else
            __builtin_memcpy(reinterpret_cast<char*>(&result) + i * sizeof(int32_t),
                             data + index + i * n,
                             s);
#endif
        }
        return result;
    }

    ROCPRIM_HOST_DEVICE void set(int index, T value)
    {
        ROCPRIM_UNROLL
        for(int i = 0; i < words_no; i++)
        {
            const size_t s = std::min(sizeof(int32_t), sizeof(T) - i * sizeof(int32_t));
#ifdef __HIP_CPU_RT__
            std::memcpy(data + index + i * n,
                        reinterpret_cast<const char*>(&value) + i * sizeof(int32_t),
                        s);
#else
            __builtin_memcpy(data + index + i * n,
                             reinterpret_cast<const char*>(&value) + i * sizeof(int32_t),
                             s);
#endif
        }
    }

private:
    static constexpr int words_no = rocprim::detail::ceiling_div(sizeof(T), sizeof(int32_t));

    int32_t data[words_no * n];
};
#endif // DOXYGEN_SHOULD_SKIP_THIS

template<class T,
         unsigned int BlockSizeX,
         unsigned int BlockSizeY,
         unsigned int BlockSizeZ,
         bool         CommutativeOnly = false>
class block_reduce_raking_reduce
{
    static constexpr unsigned int BlockSize = BlockSizeX * BlockSizeY * BlockSizeZ;

    // Warp reduce, warp_reduce_crosslane does not require shared memory (storage), but
    // logical warp size must be a power of two.
    static constexpr unsigned int warp_size_
        = detail::get_min_warp_size(BlockSize, ::rocprim::device_warp_size());

    static constexpr unsigned int segment_len = ceiling_div(BlockSize, warp_size_);

    static constexpr bool block_multiple_warp_     = !(BlockSize % warp_size_);
    static constexpr bool block_smaller_than_warp_ = (BlockSize < warp_size_);
    using warp_reduce_prefix_type = ::rocprim::detail::warp_reduce_crosslane<T, warp_size_, false>;

    struct storage_type_
    {
        fast_array<T, BlockSize> threads;
    };

public:
    using storage_type = detail::raw_storage<storage_type_>;

    /// \brief Computes a thread block-wide reduction using specified reduction operator. The return value is only valid for thread<sub>0</sub>.
    /// \param input     [in]  Calling thread's input to be reduced
    /// \param output    [out] Variable containing reduction output
    /// \param storage   [in]  Temporary Storage used for the Reduction
    /// \param reduce_op [in]  Binary reduction operator
    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void
        reduce(T input, T& output, storage_type& storage, BinaryFunction reduce_op)
    {
        this->reduce_impl(::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>(),
                          input,
                          output,
                          storage,
                          reduce_op);
    }

    /// \brief Computes a thread block-wide reduction using specified reduction operator. The return value is only valid for thread<sub>0</sub>.
    /// \param input     [in]  Calling thread's input to be reduced
    /// \param output    [out] Variable containing reduction output
    /// \param reduce_op [in]  Binary reduction operator
    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE void reduce(T input, T& output, BinaryFunction reduce_op)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        this->reduce(input, output, storage, reduce_op);
    }

    /// \brief Computes a thread block-wide reduction using specified reduction operator. The return value is only valid for thread<sub>0</sub>.
    /// \param input     [in]  Calling thread's input array to be reduced
    /// \param output    [out] Variable containing reduction output
    /// \param storage   [in]  Temporary Storage used for the Reduction
    /// \param reduce_op [in]  Binary reduction operator
    template<unsigned int ItemsPerThread, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void reduce(T (&input)[ItemsPerThread],
                                              T&             output,
                                              storage_type&  storage,
                                              BinaryFunction reduce_op)
    {
        // Reduce thread items
        T thread_input = input[0];
        ROCPRIM_UNROLL
        for(unsigned int i = 1; i < ItemsPerThread; i++)
        {
            thread_input = reduce_op(thread_input, input[i]);
        }

        // Reduction of reduced values to get partials
        const auto flat_tid = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        this->reduce_impl(flat_tid, thread_input, output, storage, reduce_op);
    }

    /// \brief Computes a thread block-wide reduction using specified reduction operator. The return value is only valid for thread<sub>0</sub>.
    /// \param input     [in]  Calling thread's input array to be reduced
    /// \param output    [out] Variable containing reduction output
    /// \param reduce_op [in]  Binary reduction operator
    template<unsigned int ItemsPerThread, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE void
        reduce(T (&input)[ItemsPerThread], T& output, BinaryFunction reduce_op)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        this->reduce(input, output, storage, reduce_op);
    }

    /// \brief Computes a thread block-wide reduction using specified reduction operator. The return value is only valid for thread<sub>0</sub>.
    /// \param input       [in]  Calling thread's input partial reductions
    /// \param output      [out] Variable containing reduction output
    /// \param valid_items [in]  Number of valid elements (should be equal to or less than BlockSize)
    /// \param storage     [in]  Temporary Storage used for reduction
    /// \param reduce_op   [in]  Binary reduction operator
    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void reduce(T              input,
                                              T&             output,
                                              unsigned int   valid_items,
                                              storage_type&  storage,
                                              BinaryFunction reduce_op)
    {
        this->reduce_impl(::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>(),
                          input,
                          output,
                          valid_items,
                          storage,
                          reduce_op);
    }

    /// \brief Computes a thread block-wide reduction using specified reduction operator. The return value is only valid for thread<sub>0</sub>.
    /// \param input       [in]  Calling thread's input partial reductions
    /// \param output      [out] Variable containing reduction output
    /// \param valid_items [in]  Number of valid elements (should be equal to or less than BlockSize)
    /// \param reduce_op   [in]  Binary reduction operator
    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE void
        reduce(T input, T& output, unsigned int valid_items, BinaryFunction reduce_op)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        this->reduce(input, output, valid_items, storage, reduce_op);
    }

private:
    template<class BinaryFunction, bool FunctionCommutativeOnly = CommutativeOnly>
    ROCPRIM_DEVICE ROCPRIM_INLINE auto reduce_impl(const unsigned int flat_tid,
                                                   T                  input,
                                                   T&                 output,
                                                   storage_type&      storage,
                                                   BinaryFunction     reduce_op) ->
        typename std::enable_if<(FunctionCommutativeOnly), void>::type
    {
        storage_type_& storage_ = storage.get();
        if(flat_tid >= warp_size_)
        {
            storage_.threads.set(flat_tid, input);
        }
        ::rocprim::syncthreads();

        if(flat_tid < warp_size_)
        {
            unsigned int thread_index     = flat_tid;
            T            thread_reduction = input;
            ROCPRIM_UNROLL
            for(unsigned int i = 1; i < segment_len; i++)
            {
                thread_index += warp_size_;
                if(block_multiple_warp_ || (thread_index < BlockSize))
                {
                    thread_reduction
                        = reduce_op(thread_reduction, storage_.threads.get(thread_index));
                }
            }
            warp_reduce<block_smaller_than_warp_, warp_reduce_prefix_type>(thread_reduction,
                                                                           output,
                                                                           BlockSize,
                                                                           reduce_op);
        }
    }

    template<class BinaryFunction, bool FunctionCommutativeOnly = CommutativeOnly>
    ROCPRIM_DEVICE ROCPRIM_INLINE auto reduce_impl(const unsigned int flat_tid,
                                                   T                  input,
                                                   T&                 output,
                                                   storage_type&      storage,
                                                   BinaryFunction     reduce_op) ->
        typename std::enable_if<(!FunctionCommutativeOnly), void>::type
    {
        storage_type_& storage_ = storage.get();
        storage_.threads.set(flat_tid, input);
        ::rocprim::syncthreads();

        constexpr unsigned int active_lanes = ceiling_div(BlockSize, segment_len);

        if(flat_tid < active_lanes)
        {
            unsigned int thread_index     = segment_len * flat_tid;
            T            thread_reduction = storage_.threads.get(thread_index);
            ROCPRIM_UNROLL
            for(unsigned int i = 1; i < segment_len; i++)
            {
                ++thread_index;
                if(block_multiple_warp_ || (thread_index < BlockSize))
                {
                    thread_reduction
                        = reduce_op(thread_reduction, storage_.threads.get(thread_index));
                }
            }
            warp_reduce<!block_multiple_warp_, warp_reduce_prefix_type>(thread_reduction,
                                                                        output,
                                                                        active_lanes,
                                                                        reduce_op);
        }
    }

    template<bool UseValid, class WarpReduce, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE auto
        warp_reduce(T input, T& output, const unsigned int valid_items, BinaryFunction reduce_op) ->
        typename std::enable_if<UseValid>::type
    {
        WarpReduce().reduce(input, output, valid_items, reduce_op);
    }

    template<bool UseValid, class WarpReduce, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE auto
        warp_reduce(T input, T& output, const unsigned int valid_items, BinaryFunction reduce_op) ->
        typename std::enable_if<!UseValid>::type
    {
        (void)valid_items;
        WarpReduce().reduce(input, output, reduce_op);
    }

    template<class BinaryFunction, bool FunctionCommutativeOnly = CommutativeOnly>
    ROCPRIM_DEVICE ROCPRIM_INLINE auto reduce_impl(const unsigned int flat_tid,
                                                   T                  input,
                                                   T&                 output,
                                                   const unsigned int valid_items,
                                                   storage_type&      storage,
                                                   BinaryFunction     reduce_op) ->
        typename std::enable_if<(FunctionCommutativeOnly), void>::type
    {
        storage_type_& storage_ = storage.get();
        if((flat_tid >= warp_size_) && (flat_tid < valid_items))
        {
            storage_.threads.set(flat_tid, input);
        }
        ::rocprim::syncthreads();

        if(flat_tid < warp_size_)
        {
            T thread_reduction = input;
            for(unsigned int i = warp_size_ + flat_tid; i < valid_items; i += warp_size_)
            {
                thread_reduction = reduce_op(thread_reduction, storage_.threads.get(i));
            }
            warp_reduce_prefix_type().reduce(thread_reduction, output, valid_items, reduce_op);
        }
    }

    template<class BinaryFunction, bool FunctionCommutativeOnly = CommutativeOnly>
    ROCPRIM_DEVICE ROCPRIM_INLINE auto reduce_impl(const unsigned int flat_tid,
                                                   T                  input,
                                                   T&                 output,
                                                   const unsigned int valid_items,
                                                   storage_type&      storage,
                                                   BinaryFunction     reduce_op) ->
        typename std::enable_if<(!FunctionCommutativeOnly), void>::type
    {
        storage_type_& storage_ = storage.get();
        if(flat_tid < valid_items)
        {
            storage_.threads.set(flat_tid, input);
        }
        ::rocprim::syncthreads();

        unsigned int thread_index = segment_len * flat_tid;
        if(thread_index < valid_items)
        {
            T thread_reduction = storage_.threads.get(thread_index);
            ROCPRIM_UNROLL
            for(unsigned int i = 1; i < segment_len; i++)
            {
                ++thread_index;
                if(thread_index < valid_items)
                {
                    thread_reduction
                        = reduce_op(thread_reduction, storage_.threads.get(thread_index));
                }
            }
            // not ceiling_div here as not constexpr and this is faster
            warp_reduce_prefix_type().reduce(thread_reduction,
                                             output,
                                             (valid_items + segment_len - 1) / segment_len,
                                             reduce_op);
        }
    }
};
} // end namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_BLOCK_DETAIL_BLOCK_REDUCE_RAKING_REDUCE_HPP_
