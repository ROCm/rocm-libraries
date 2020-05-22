// Copyright (c) 2017-2020 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_BLOCK_DETAIL_BLOCK_SCAN_WARP_SCAN_HPP_
#define ROCPRIM_BLOCK_DETAIL_BLOCK_SCAN_WARP_SCAN_HPP_

#include <type_traits>

#include "../../config.hpp"
#include "../../detail/various.hpp"

#include "../../intrinsics.hpp"
#include "../../functional.hpp"

#include "../../warp/warp_scan.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<
    class T,
    unsigned int BlockSizeX,
    unsigned int BlockSizeY,
    unsigned int BlockSizeZ
>
class block_scan_warp_scan
{
    static constexpr unsigned int BlockSize = BlockSizeX * BlockSizeY * BlockSizeZ;
    // Select warp size
    static constexpr unsigned int warp_size_ =
        detail::get_min_warp_size(BlockSize, ::rocprim::warp_size());
    // Number of warps in block
    static constexpr unsigned int warps_no_ = (BlockSize + warp_size_ - 1) / warp_size_;

    // typedef of warp_scan primitive that will be used to perform warp-level
    // inclusive/exclusive scan operations on input values.
    // warp_scan_crosslane is an implementation of warp_scan that does not need storage,
    // but requires logical warp size to be a power of two.
    using warp_scan_input_type = ::rocprim::detail::warp_scan_crosslane<T, warp_size_>;
    // typedef of warp_scan primitive that will be used to get prefix values for
    // each warp (scanned carry-outs from warps before it).
    using warp_scan_prefix_type = ::rocprim::detail::warp_scan_crosslane<T, detail::next_power_of_two(warps_no_)>;

    struct storage_type_
    {
        T warp_prefixes[warps_no_];
        // ---------- Shared memory optimisation ----------
        // Since warp_scan_input and warp_scan_prefix are typedef of warp_scan_crosslane,
        // we don't need to allocate any temporary memory for them.
        // If we just use warp_scan, we would need to add following union to this struct:
        // union
        // {
        //     typename warp_scan_input::storage_type wscan[warps_no_];
        //     typename warp_scan_prefix::storage_type wprefix_scan;
        // };
        // and use storage_.wscan[warp_id] and storage.wprefix_scan when calling
        // warp_scan_input().inclusive_scan(..) and warp_scan_prefix().inclusive_scan(..).
    };

public:
    using storage_type = detail::raw_storage<storage_type_>;

    template<class BinaryFunction>
    ROCPRIM_DEVICE inline
    void inclusive_scan(T input,
                        T& output,
                        storage_type& storage,
                        BinaryFunction scan_op)
    {
        this->inclusive_scan_impl(
            ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>(),
            input, output, storage, scan_op
        );
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE inline
    void inclusive_scan(T input,
                        T& output,
                        BinaryFunction scan_op)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        this->inclusive_scan(input, output, storage, scan_op);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE inline
    void inclusive_scan(T input,
                        T& output,
                        T& reduction,
                        storage_type& storage,
                        BinaryFunction scan_op)
    {
        storage_type_& storage_ = storage.get();
        this->inclusive_scan(input, output, storage, scan_op);
        // Save reduction result
        reduction = storage_.warp_prefixes[warps_no_ - 1];
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE inline
    void inclusive_scan(T input,
                        T& output,
                        T& reduction,
                        BinaryFunction scan_op)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        this->inclusive_scan(input, output, reduction, storage, scan_op);
    }

    template<class PrefixCallback, class BinaryFunction>
    ROCPRIM_DEVICE inline
    void inclusive_scan(T input,
                        T& output,
                        storage_type& storage,
                        PrefixCallback& prefix_callback_op,
                        BinaryFunction scan_op)
    {
        const auto flat_tid = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        const auto warp_id = ::rocprim::warp_id(flat_tid);
        storage_type_& storage_ = storage.get();
        this->inclusive_scan_impl(flat_tid, input, output, storage, scan_op);
        // Include block prefix (this operation overwrites storage_.warp_prefixes[warps_no_ - 1])
        T block_prefix = this->get_block_prefix(
            flat_tid, warp_id,
            storage_.warp_prefixes[warps_no_ - 1], // block reduction
            prefix_callback_op, storage
        );
        output = scan_op(block_prefix, output);
    }

    template<unsigned int ItemsPerThread, class BinaryFunction>
    ROCPRIM_DEVICE inline
    void inclusive_scan(T (&input)[ItemsPerThread],
                        T (&output)[ItemsPerThread],
                        storage_type& storage,
                        BinaryFunction scan_op)
    {
        // Reduce thread items
        T thread_input = input[0];
        #pragma unroll
        for(unsigned int i = 1; i < ItemsPerThread; i++)
        {
            thread_input = scan_op(thread_input, input[i]);
        }

        // Scan of reduced values to get prefixes
        const auto flat_tid = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        this->exclusive_scan_impl(
            flat_tid,
            thread_input, thread_input, // input, output
            storage,
            scan_op
        );

        // Include prefix (first thread does not have prefix)
        output[0] = input[0];
        if(flat_tid != 0)
        {
            output[0] = scan_op(thread_input, input[0]);
        }
        // Final thread-local scan
        #pragma unroll
        for(unsigned int i = 1; i < ItemsPerThread; i++)
        {
            output[i] = scan_op(output[i-1], input[i]);
        }
    }

    template<unsigned int ItemsPerThread, class BinaryFunction>
    ROCPRIM_DEVICE inline
    void inclusive_scan(T (&input)[ItemsPerThread],
                        T (&output)[ItemsPerThread],
                        BinaryFunction scan_op)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        this->inclusive_scan(input, output, storage, scan_op);
    }

    template<unsigned int ItemsPerThread, class BinaryFunction>
    ROCPRIM_DEVICE inline
    void inclusive_scan(T (&input)[ItemsPerThread],
                        T (&output)[ItemsPerThread],
                        T& reduction,
                        storage_type& storage,
                        BinaryFunction scan_op)
    {
        storage_type_& storage_ = storage.get();
        this->inclusive_scan(input, output, storage, scan_op);
        // Save reduction result
        reduction = storage_.warp_prefixes[warps_no_ - 1];
    }

    template<unsigned int ItemsPerThread, class BinaryFunction>
    ROCPRIM_DEVICE inline
    void inclusive_scan(T (&input)[ItemsPerThread],
                        T (&output)[ItemsPerThread],
                        T& reduction,
                        BinaryFunction scan_op)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        this->inclusive_scan(input, output, reduction, storage, scan_op);
    }

    template<
        class PrefixCallback,
        unsigned int ItemsPerThread,
        class BinaryFunction
    >
    ROCPRIM_DEVICE inline
    void inclusive_scan(T (&input)[ItemsPerThread],
                        T (&output)[ItemsPerThread],
                        storage_type& storage,
                        PrefixCallback& prefix_callback_op,
                        BinaryFunction scan_op)
    {
        storage_type_& storage_ = storage.get();
        // Reduce thread items
        T thread_input = input[0];
        #pragma unroll
        for(unsigned int i = 1; i < ItemsPerThread; i++)
        {
            thread_input = scan_op(thread_input, input[i]);
        }

        // Scan of reduced values to get prefixes
        const auto flat_tid = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        this->exclusive_scan_impl(
            flat_tid,
            thread_input, thread_input, // input, output
            storage,
            scan_op
        );

        // this operation overwrites storage_.warp_prefixes[warps_no_ - 1]
        T block_prefix = this->get_block_prefix(
            flat_tid, ::rocprim::warp_id(flat_tid),
            storage_.warp_prefixes[warps_no_ - 1], // block reduction
            prefix_callback_op, storage
        );

        // Include prefix (first thread does not have prefix)
        output[0] = input[0];
        if(flat_tid != 0)
        {
            output[0] = scan_op(thread_input, input[0]);
        }
        // Include block prefix
        output[0] = scan_op(block_prefix, output[0]);
        // Final thread-local scan
        #pragma unroll
        for(unsigned int i = 1; i < ItemsPerThread; i++)
        {
            output[i] = scan_op(output[i-1], input[i]);
        }
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE inline
    void exclusive_scan(T input,
                        T& output,
                        T init,
                        storage_type& storage,
                        BinaryFunction scan_op)
    {
        this->exclusive_scan_impl(
            ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>(),
            input, output, init, storage, scan_op
        );
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE inline
    void exclusive_scan(T input,
                        T& output,
                        T init,
                        BinaryFunction scan_op)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        this->exclusive_scan(
            input, output, init, storage, scan_op
        );
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE inline
    void exclusive_scan(T input,
                        T& output,
                        T init,
                        T& reduction,
                        storage_type& storage,
                        BinaryFunction scan_op)
    {
        storage_type_& storage_ = storage.get();
        this->exclusive_scan(
            input, output, init, storage, scan_op
        );
        // Save reduction result
        reduction = storage_.warp_prefixes[warps_no_ - 1];
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE inline
    void exclusive_scan(T input,
                        T& output,
                        T init,
                        T& reduction,
                        BinaryFunction scan_op)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        this->exclusive_scan(
            input, output, init, reduction, storage, scan_op
        );
    }

    template<class PrefixCallback, class BinaryFunction>
    ROCPRIM_DEVICE inline
    void exclusive_scan(T input,
                        T& output,
                        storage_type& storage,
                        PrefixCallback& prefix_callback_op,
                        BinaryFunction scan_op)
    {
        const auto flat_tid = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        const auto warp_id = ::rocprim::warp_id(flat_tid);
        storage_type_& storage_ = storage.get();
        this->exclusive_scan_impl(
            flat_tid, input, output, storage, scan_op
        );
        // Include block prefix (this operation overwrites storage_.warp_prefixes[warps_no_ - 1])
        T block_prefix = this->get_block_prefix(
            flat_tid, warp_id,
            storage_.warp_prefixes[warps_no_ - 1], // block reduction
            prefix_callback_op, storage
        );
        output = scan_op(block_prefix, output);
        if(flat_tid == 0) output = block_prefix;
    }

    template<unsigned int ItemsPerThread, class BinaryFunction>
    ROCPRIM_DEVICE inline
    void exclusive_scan(T (&input)[ItemsPerThread],
                        T (&output)[ItemsPerThread],
                        T init,
                        storage_type& storage,
                        BinaryFunction scan_op)
    {
        // Reduce thread items
        T thread_input = input[0];
        #pragma unroll
        for(unsigned int i = 1; i < ItemsPerThread; i++)
        {
            thread_input = scan_op(thread_input, input[i]);
        }

        // Scan of reduced values to get prefixes
        const auto flat_tid = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        this->exclusive_scan_impl(
            flat_tid,
            thread_input, thread_input, // input, output
            init,
            storage,
            scan_op
        );

        // Include init value
        T prev = input[0];
        T exclusive = init;
        if(flat_tid != 0)
        {
            exclusive = thread_input;
        }
        output[0] = exclusive;

        #pragma unroll
        for(unsigned int i = 1; i < ItemsPerThread; i++)
        {
            exclusive = scan_op(exclusive, prev);
            prev = input[i];
            output[i] = exclusive;
        }
    }

    template<unsigned int ItemsPerThread, class BinaryFunction>
    ROCPRIM_DEVICE inline
    void exclusive_scan(T (&input)[ItemsPerThread],
                        T (&output)[ItemsPerThread],
                        T init,
                        BinaryFunction scan_op)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        this->exclusive_scan(input, output, init, storage, scan_op);
    }

    template<unsigned int ItemsPerThread, class BinaryFunction>
    ROCPRIM_DEVICE inline
    void exclusive_scan(T (&input)[ItemsPerThread],
                        T (&output)[ItemsPerThread],
                        T init,
                        T& reduction,
                        storage_type& storage,
                        BinaryFunction scan_op)
    {
        storage_type_& storage_ = storage.get();
        this->exclusive_scan(input, output, init, storage, scan_op);
        // Save reduction result
        reduction = storage_.warp_prefixes[warps_no_ - 1];
    }

    template<unsigned int ItemsPerThread, class BinaryFunction>
    ROCPRIM_DEVICE inline
    void exclusive_scan(T (&input)[ItemsPerThread],
                        T (&output)[ItemsPerThread],
                        T init,
                        T& reduction,
                        BinaryFunction scan_op)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        this->exclusive_scan(input, output, init, reduction, storage, scan_op);
    }

    template<
        class PrefixCallback,
        unsigned int ItemsPerThread,
        class BinaryFunction
    >
    ROCPRIM_DEVICE inline
    void exclusive_scan(T (&input)[ItemsPerThread],
                        T (&output)[ItemsPerThread],
                        storage_type& storage,
                        PrefixCallback& prefix_callback_op,
                        BinaryFunction scan_op)
    {
        storage_type_& storage_ = storage.get();
        // Reduce thread items
        T thread_input = input[0];
        #pragma unroll
        for(unsigned int i = 1; i < ItemsPerThread; i++)
        {
            thread_input = scan_op(thread_input, input[i]);
        }

        // Scan of reduced values to get prefixes
        const auto flat_tid = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        this->exclusive_scan_impl(
            flat_tid,
            thread_input, thread_input, // input, output
            storage,
            scan_op
        );

        // this operation overwrites storage_.warp_prefixes[warps_no_ - 1]
        T block_prefix = this->get_block_prefix(
            flat_tid, ::rocprim::warp_id(flat_tid),
            storage_.warp_prefixes[warps_no_ - 1], // block reduction
            prefix_callback_op, storage
        );

        // Include init value and block prefix
        T prev = input[0];
        T exclusive = block_prefix;
        if(flat_tid != 0)
        {
            exclusive = scan_op(block_prefix, thread_input);
        }
        output[0] = exclusive;

        #pragma unroll
        for(unsigned int i = 1; i < ItemsPerThread; i++)
        {
            exclusive = scan_op(exclusive, prev);
            prev = input[i];
            output[i] = exclusive;
        }
    }

private:
    template<class BinaryFunction, unsigned int BlockSize_ = BlockSize>
    ROCPRIM_DEVICE inline
    auto inclusive_scan_impl(const unsigned int flat_tid,
                             T input,
                             T& output,
                             storage_type& storage,
                             BinaryFunction scan_op)
        -> typename std::enable_if<(BlockSize_ > ::rocprim::warp_size())>::type
    {
        storage_type_& storage_ = storage.get();
        // Perform warp scan
        warp_scan_input_type().inclusive_scan(
            // not using shared mem, see note in storage_type
            input, output, scan_op
        );

        // i-th warp will have its prefix stored in storage_.warp_prefixes[i-1]
        const auto warp_id = ::rocprim::warp_id(flat_tid);
        this->calculate_warp_prefixes(flat_tid, warp_id, output, storage, scan_op);

        // Use warp prefix to calculate the final scan results for every thread
        if(warp_id != 0)
        {
            auto warp_prefix = storage_.warp_prefixes[warp_id - 1];
            output = scan_op(warp_prefix, output);
        }
    }

    // When BlockSize is less than warp_size we dont need the extra prefix calculations.
    template<class BinaryFunction, unsigned int BlockSize_ = BlockSize>
    ROCPRIM_DEVICE inline
    auto inclusive_scan_impl(unsigned int flat_tid,
                             T input,
                             T& output,
                             storage_type& storage,
                             BinaryFunction scan_op)
        -> typename std::enable_if<!(BlockSize_ > ::rocprim::warp_size())>::type
    {
        (void) storage;
        (void) flat_tid;
        storage_type_& storage_ = storage.get();
        // Perform warp scan
        warp_scan_input_type().inclusive_scan(
            // not using shared mem, see note in storage_type
            input, output, scan_op
        );

        if(flat_tid == BlockSize_ - 1)
        {
            storage_.warp_prefixes[0] = output;
        }
        ::rocprim::syncthreads();
    }

    // Exclusive scan with initial value when BlockSize is bigger than warp_size
    template<class BinaryFunction, unsigned int BlockSize_ = BlockSize>
    ROCPRIM_DEVICE inline
    auto exclusive_scan_impl(const unsigned int flat_tid,
                             T input,
                             T& output,
                             T init,
                             storage_type& storage,
                             BinaryFunction scan_op)
        -> typename std::enable_if<(BlockSize_ > ::rocprim::warp_size())>::type
    {
        storage_type_& storage_ = storage.get();
        // Perform warp scan on input values
        warp_scan_input_type().inclusive_scan(
            // not using shared mem, see note in storage_type
            input, output, scan_op
        );

        // i-th warp will have its prefix stored in storage_.warp_prefixes[i-1]
        const auto warp_id = ::rocprim::warp_id(flat_tid);
        this->calculate_warp_prefixes(flat_tid, warp_id, output, storage, scan_op);

        // Include initial value in warp prefixes, and fix warp prefixes
        // for exclusive scan (first warp prefix is init)
        auto warp_prefix = init;
        if(warp_id != 0)
        {
            warp_prefix = scan_op(init, storage_.warp_prefixes[warp_id-1]);
        }

        // Use warp prefix to calculate the final scan results for every thread
        output = scan_op(warp_prefix, output); // include warp prefix in scan results
        output = warp_shuffle_up(output, 1, warp_size_); // shift to get exclusive results
        if(::rocprim::lane_id() == 0)
        {
            output = warp_prefix;
        }
    }

    // Exclusive scan with initial value when BlockSize is less than warp_size.
    // When BlockSize is less than warp_size we dont need the extra prefix calculations.
    template<class BinaryFunction, unsigned int BlockSize_ = BlockSize>
    ROCPRIM_DEVICE inline
    auto exclusive_scan_impl(const unsigned int flat_tid,
                             T input,
                             T& output,
                             T init,
                             storage_type& storage,
                             BinaryFunction scan_op)
        -> typename std::enable_if<!(BlockSize_ > ::rocprim::warp_size())>::type
    {
        (void) flat_tid;
        (void) storage;
        (void) init;
        storage_type_& storage_ = storage.get();
        // Perform warp scan on input values
        warp_scan_input_type().inclusive_scan(
            // not using shared mem, see note in storage_type
            input, output, scan_op
        );

        if(flat_tid == BlockSize_ - 1)
        {
            storage_.warp_prefixes[0] = output;
        }
        ::rocprim::syncthreads();

        // Use warp prefix to calculate the final scan results for every thread
        output = scan_op(init, output); // include warp prefix in scan results
        output = warp_shuffle_up(output, 1, warp_size_); // shift to get exclusive results
        if(::rocprim::lane_id() == 0)
        {
            output = init;
        }
    }

    // Exclusive scan with unknown initial value
    template<class BinaryFunction, unsigned int BlockSize_ = BlockSize>
    ROCPRIM_DEVICE inline
    auto exclusive_scan_impl(const unsigned int flat_tid,
                             T input,
                             T& output,
                             storage_type& storage,
                             BinaryFunction scan_op)
        -> typename std::enable_if<(BlockSize_ > ::rocprim::warp_size())>::type
    {
        storage_type_& storage_ = storage.get();
        // Perform warp scan on input values
        warp_scan_input_type().inclusive_scan(
            // not using shared mem, see note in storage_type
            input, output, scan_op
        );

        // i-th warp will have its prefix stored in storage_.warp_prefixes[i-1]
        const auto warp_id = ::rocprim::warp_id(flat_tid);
        this->calculate_warp_prefixes(flat_tid, warp_id, output, storage, scan_op);

        // Use warp prefix to calculate the final scan results for every thread
        T warp_prefix;
        if(warp_id != 0)
        {
            warp_prefix = storage_.warp_prefixes[warp_id - 1];
            output = scan_op(warp_prefix, output);
        }
        output = warp_shuffle_up(output, 1, warp_size_); // shift to get exclusive results
        if(::rocprim::lane_id() == 0)
        {
            output = warp_prefix;
        }
    }

    // Exclusive scan with unknown initial value, when BlockSize less than warp_size.
    // When BlockSize is less than warp_size we dont need the extra prefix calculations.
    template<class BinaryFunction, unsigned int BlockSize_ = BlockSize>
    ROCPRIM_DEVICE inline
    auto exclusive_scan_impl(const unsigned int flat_tid,
                             T input,
                             T& output,
                             storage_type& storage,
                             BinaryFunction scan_op)
        -> typename std::enable_if<!(BlockSize_ > ::rocprim::warp_size())>::type
    {
        (void) flat_tid;
        (void) storage;
        storage_type_& storage_ = storage.get();
        // Perform warp scan on input values
        warp_scan_input_type().inclusive_scan(
            // not using shared mem, see note in storage_type
            input, output, scan_op
        );

        if(flat_tid == BlockSize_ - 1)
        {
            storage_.warp_prefixes[0] = output;
        }
        ::rocprim::syncthreads();
        output = warp_shuffle_up(output, 1, warp_size_); // shift to get exclusive results
    }

    // i-th warp will have its prefix stored in storage_.warp_prefixes[i-1]
    template<class BinaryFunction, unsigned int BlockSize_ = BlockSize>
    ROCPRIM_DEVICE inline
    void calculate_warp_prefixes(const unsigned int flat_tid,
                                 const unsigned int warp_id,
                                 T inclusive_input,
                                 storage_type& storage,
                                 BinaryFunction scan_op)
    {
        storage_type_& storage_ = storage.get();
        // Save the warp reduction result, that is the scan result
        // for last element in each warp
        if(flat_tid == ::rocprim::min((warp_id+1) * warp_size_, BlockSize_) - 1)
        {
            storage_.warp_prefixes[warp_id] = inclusive_input;
        }
        ::rocprim::syncthreads();

        // Scan the warp reduction results and store in storage_.warp_prefixes
        if(flat_tid < warps_no_)
        {
            auto warp_prefix = storage_.warp_prefixes[flat_tid];
            warp_scan_prefix_type().inclusive_scan(
                // not using shared mem, see note in storage_type
                warp_prefix, warp_prefix, scan_op
            );
            storage_.warp_prefixes[flat_tid] = warp_prefix;
        }
        ::rocprim::syncthreads();
    }

    // THIS OVERWRITES storage_.warp_prefixes[warps_no_ - 1]
    template<class PrefixCallback>
    ROCPRIM_DEVICE inline
    T get_block_prefix(const unsigned int flat_tid,
                       const unsigned int warp_id,
                       const T reduction,
                       PrefixCallback& prefix_callback_op,
                       storage_type& storage)
    {
        storage_type_& storage_ = storage.get();
        if(warp_id == 0)
        {
            T block_prefix = prefix_callback_op(reduction);
            if(flat_tid == 0)
            {
                // Reuse storage_.warp_prefixes[warps_no_ - 1] to store block prefix
                storage_.warp_prefixes[warps_no_ - 1] = block_prefix;
            }
        }
        ::rocprim::syncthreads();
        return storage_.warp_prefixes[warps_no_ - 1];
    }
};

} // end namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_BLOCK_DETAIL_BLOCK_SCAN_WARP_SCAN_HPP_
