/******************************************************************************
 * Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2021-2024, Advanced Micro Devices, Inc.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#ifndef ROCPRIM_BLOCK_BLOCK_RUN_LENGTH_DECODE_HPP_
#define ROCPRIM_BLOCK_BLOCK_RUN_LENGTH_DECODE_HPP_

#include "../block/block_scan.hpp"
#include "../config.hpp"
#include "../detail/temp_storage.hpp"
#include "../detail/various.hpp"
#include "../functional.hpp"
#include "../intrinsics/thread.hpp"
#include "../thread/thread_search.hpp"

BEGIN_ROCPRIM_NAMESPACE

/**
 * \brief The block_run_length_decode class supports decoding a run-length encoded array of items. That is, given
 * the two arrays `run_value[N]` and `run_lengths[N]`, `run_value[i]` is repeated `run_lengths[i]` many times in the output
 * array.
 * \note Trailing runs of length 0 are supported (i.e., they may only appear at the end of the `run_lengths` array).
 * A run of length zero may not be followed by a run length that is not zero.
 *
 * \par Examples
 * \parblock
 * Due to the nature of the run-length decoding algorithm ("decompression"), the output size of the run-length decoded
 * array is runtime-dependent and potentially without any upper bound. To address this, `block_run_length_decode` allows
 * retrieving a "window" from the run-length decoded array. The window's offset can be specified and `BLOCK_THREADS *
 * DECODED_ITEMS_PER_THREAD` (i.e., referred to as `window_size`) decoded items from the specified window will be returned.
 *
 * \par
 * \code
 * __global__ void ExampleKernel(...)
 * {
 *   // Specialising block_run_length_decode to run-length decode items of type uint64_t
 *   using RunItemT = uint64_t;
 *   // Type large enough to index into the run-length decoded array
 *   using RunLengthT = uint32_t;
 *
 *   // Specialising block_run_length_decode for a 1D block of 128 threads
 *   constexpr int BLOCK_DIM_X = 128;
 *   // Specialising block_run_length_decode to have each thread contribute 2 run-length encoded runs
 *   constexpr int RUNS_PER_THREAD = 2;
 *   // Specialising block_run_length_decode to have each thread hold 4 run-length decoded items
 *   constexpr int DECODED_ITEMS_PER_THREAD = 4;
 *
 *   // Specialize BlockRadixSort for a 1D block of 128 threads owning 4 integer items each
 *   using block_run_length_decodeT =
 *     hipcub::block_run_length_decode<RunItemT, BLOCK_DIM_X, RUNS_PER_THREAD, DECODED_ITEMS_PER_THREAD>;
 *
 *   // Allocate shared memory for block_run_length_decode
 *   __shared__ typename block_run_length_decodeT::TempStorage temp_storage;
 *
 *   // The run-length encoded items and how often they shall be repeated in the run-length decoded output
 *   RunItemT run_values[RUNS_PER_THREAD];
 *   RunLengthT run_lengths[RUNS_PER_THREAD];
 *   ...
 *
 *   // Initialize the block_run_length_decode with the runs that we want to run-length decode
 *   uint32_t total_decoded_size = 0;
 *   block_run_length_decodeT block_rld(temp_storage, run_values, run_lengths, total_decoded_size);
 *
 *   // Run-length decode ("decompress") the runs into a window buffer of limited size. This is repeated until all runs
 *   // have been decoded.
 *   uint32_t decoded_window_offset = 0U;
 *   while (decoded_window_offset < total_decoded_size)
 *   {
 *     RunLengthT relative_offsets[DECODED_ITEMS_PER_THREAD];
 *     RunItemT decoded_items[DECODED_ITEMS_PER_THREAD];
 *
 *     // The number of decoded items that are valid within this window (aka pass) of run-length decoding
 *     uint32_t num_valid_items = total_decoded_size - decoded_window_offset;
 *     block_rld.run_length_decode(decoded_items, relative_offsets, decoded_window_offset);
 *
 *     decoded_window_offset += BLOCK_DIM_X * DECODED_ITEMS_PER_THREAD;
 *
 *     ...
 *   }
 * }
 * \endcode
 * \par
 * Suppose the set of input \p run_values across the block of threads is
 * <tt>{ [0, 1], [2, 3], [4, 5], [6, 7], ..., [254, 255] }</tt> and
 * \p run_lengths is <tt>{ [1, 2], [3, 4], [5, 1], [2, 3], ..., [5, 1] }</tt>.
 * \par
 * The corresponding output \p decoded_items in those threads will be <tt>{ [0, 1, 1, 2], [2, 2, 3, 3], [3, 3, 4, 4],
 * [4, 4, 4, 5], ..., [169, 169, 170, 171] }</tt> and \p relative_offsets will be <tt>{ [0, 0, 1, 0], [1, 2, 0, 1], [2,
 * 3, 0, 1], [2, 3, 4, 0], ..., [3, 4, 0, 0] }</tt> during the first iteration of the while loop.
 * \endparblock
 *
 * \tparam ItemT The data type of the items being run-length decoded
 * \tparam BLOCK_DIM_X The thread block length in threads along the X dimension
 * \tparam RUNS_PER_THREAD The number of consecutive runs that each thread contributes
 * \tparam DECODED_ITEMS_PER_THREAD The maximum number of decoded items that each thread holds
 * \tparam DecodedOffsetT Type used to index into the block's decoded items (large enough to hold the sum over all the
 * runs' lengths)
 * \tparam BLOCK_DIM_Y The thread block length in threads along the Y dimension
 * \tparam BLOCK_DIM_Z The thread block length in threads along the Z dimension
 */
template<typename ItemT,
         unsigned int BlockSizeX,
         int          RUNS_PER_THREAD,
         int          DECODED_ITEMS_PER_THREAD,
         typename DecodedOffsetT = uint32_t,
         unsigned int BlockSizeY = 1,
         unsigned int BlockSizeZ = 1>
class block_run_length_decode
{
private:
    /// The thread block size in threads
    static constexpr int BLOCK_THREADS = BlockSizeX * BlockSizeY * BlockSizeZ;

    /// The number of runs that the block decodes (out-of-bounds items may be padded with run lengths of '0')
    static constexpr int BLOCK_RUNS = BLOCK_THREADS * RUNS_PER_THREAD;

    /// block_scan used to determine the beginning of each run (i.e., prefix sum over the runs' length)
    using block_scan_type = rocprim::block_scan<DecodedOffsetT,
                                                BlockSizeX,
                                                rocprim::block_scan_algorithm::using_warp_scan,
                                                BlockSizeY,
                                                BlockSizeZ>;

    /// Type used to index into the block's runs
    using RunOffsetT = uint32_t;

    /// Shared memory type required by this thread block
    union storage_type_
    {
        typename block_scan_type::storage_type offset_scan;
        struct
        {
            ItemT          run_values[BLOCK_RUNS];
            DecodedOffsetT run_offsets[BLOCK_RUNS];
        } runs;
    };

    ROCPRIM_DEVICE ROCPRIM_INLINE storage_type_& private_storage()
    {
        ROCPRIM_SHARED_MEMORY storage_type private_storage;
        return private_storage.get();
    }

    storage_type_& temp_storage;

    uint32_t linear_tid;

public:
    /// \brief Struct used to allocate a temporary memory that is required for thread
    /// communication during operations provided by related parallel primitive.
    ///
    /// Depending on the implemention the operations exposed by parallel primitive may
    /// require a temporary storage for thread communication. The storage should be allocated
    /// using keywords <tt>__shared__</tt>. It can be aliased to
    /// an externally allocated memory, or be a part of a union type with other storage types
    /// to increase shared memory reusability.
    ROCPRIM_DETAIL_SUPPRESS_DEPRECATION_WITH_PUSH
    using storage_type = detail::raw_storage<storage_type_>;
    ROCPRIM_DETAIL_SUPPRESS_DEPRECATION_POP

    /**
   * \brief Constructor specialised for user-provided temporary storage, initializing using the runs' lengths. The
   * algorithm's temporary storage may not be repurposed between the constructor call and subsequent
   * <b>run_length_decode</b> calls.
   */
    template<typename RunLengthT, typename TotalDecodedSizeT>
    ROCPRIM_DEVICE ROCPRIM_INLINE
        block_run_length_decode(storage_type& temp_storage,
                                ItemT (&run_values)[RUNS_PER_THREAD],
                                RunLengthT (&run_lengths)[RUNS_PER_THREAD],
                                TotalDecodedSizeT& total_decoded_size)
        : temp_storage(temp_storage.get())
        , linear_tid(::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>())
    {
        init_with_run_lengths(run_values, run_lengths, total_decoded_size);
    }

    /**
     * \brief Constructor specialised for user-provided temporary storage, initializing using the runs' offsets. The
     * algorithm's temporary storage may not be repurposed between the constructor call and subsequent
     * <b>run_length_decode</b> calls.
     */
    template<typename UserRunOffsetT>
    ROCPRIM_DEVICE ROCPRIM_INLINE
        block_run_length_decode(storage_type& temp_storage,
                                ItemT (&run_values)[RUNS_PER_THREAD],
                                UserRunOffsetT (&run_offsets)[RUNS_PER_THREAD])
        : temp_storage(temp_storage.get())
        , linear_tid(::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>())
    {
        init_with_run_offsets(run_values, run_offsets);
    }

    /**
     * \brief Constructor specialised for static temporary storage, initializing using the runs' lengths.
     */
    template<typename RunLengthT, typename TotalDecodedSizeT>
    ROCPRIM_DEVICE ROCPRIM_INLINE
        block_run_length_decode(ItemT (&run_values)[RUNS_PER_THREAD],
                                RunLengthT (&run_lengths)[RUNS_PER_THREAD],
                                TotalDecodedSizeT& total_decoded_size)
        : temp_storage(private_storage())
        , linear_tid(::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>())
    {
        init_with_run_lengths(run_values, run_lengths, total_decoded_size);
    }

    /**
     * \brief Constructor specialised for static temporary storage, initializing using the runs' offsets.
     */
    template<typename UserRunOffsetT>
    ROCPRIM_DEVICE ROCPRIM_INLINE
        block_run_length_decode(ItemT (&run_values)[RUNS_PER_THREAD],
                                UserRunOffsetT (&run_offsets)[RUNS_PER_THREAD])
        : temp_storage(private_storage())
        , linear_tid(::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>())
    {
        init_with_run_offsets(run_values, run_offsets);
    }

private:
    template<typename RunOffsetT>
    ROCPRIM_DEVICE ROCPRIM_INLINE void
        init_with_run_offsets(ItemT (&run_values)[RUNS_PER_THREAD],
                              RunOffsetT (&run_offsets)[RUNS_PER_THREAD])
    {
        // Keep the runs' items and the offsets of each run's beginning in the temporary storage
        RunOffsetT thread_dst_offset
            = static_cast<RunOffsetT>(linear_tid) * static_cast<RunOffsetT>(RUNS_PER_THREAD);

#pragma unroll
        for(int i = 0; i < RUNS_PER_THREAD; ++i, ++thread_dst_offset)
        {
            temp_storage.runs.run_values[thread_dst_offset]  = run_values[i];
            temp_storage.runs.run_offsets[thread_dst_offset] = run_offsets[i];
        }

        // Ensure run offsets and run values have been writen to shared memory
        syncthreads();
    }

    template<typename RunLengthT, typename TotalDecodedSizeT>
    ROCPRIM_DEVICE ROCPRIM_INLINE void
        init_with_run_lengths(ItemT (&run_values)[RUNS_PER_THREAD],
                              RunLengthT (&run_lengths)[RUNS_PER_THREAD],
                              TotalDecodedSizeT& total_decoded_size)
    {
        // Compute the offset for the beginning of each run
        DecodedOffsetT run_offsets[RUNS_PER_THREAD];
#pragma unroll
        for(int i = 0; i < RUNS_PER_THREAD; ++i)
        {
            run_offsets[i] = static_cast<DecodedOffsetT>(run_lengths[i]);
        }

        DecodedOffsetT decoded_size_aggregate{};
        block_scan_type().exclusive_scan(run_offsets,
                                         run_offsets,
                                         0,
                                         decoded_size_aggregate,
                                         temp_storage.offset_scan,
                                         rocprim::plus<DecodedOffsetT>{});
        total_decoded_size = static_cast<TotalDecodedSizeT>(decoded_size_aggregate);

        // Ensure the prefix scan's temporary storage can be reused (may be superfluous, but depends on scan implementation)
        syncthreads();

        init_with_run_offsets(run_values, run_offsets);
    }

public:
    /**
     * \brief Run-length decodes the runs previously passed via a call to Init(...) and returns the run-length decoded
     * items in a blocked arrangement to \p decoded_items. If the number of run-length decoded items exceeds the
     * run-length decode buffer (i.e., <b>DECODED_ITEMS_PER_THREAD * BLOCK_THREADS</b>), only the items that fit within
     * the buffer are returned. Subsequent calls to <b>run_length_decode</b> adjusting \p from_decoded_offset can be
     * used to retrieve the remaining run-length decoded items. Calling __syncthreads() between any two calls to
     * <b>run_length_decode</b> is not required.
     * \p item_offsets can be used to retrieve each run-length decoded item's relative index within its run. E.g., the
     * run-length encoded array of `3, 1, 4` with the respective run lengths of `2, 1, 3` would yield the run-length
     * decoded array of `3, 3, 1, 4, 4, 4` with the relative offsets of `0, 1, 0, 0, 1, 2`.
     *
     * \param[out] decoded_items The run-length decoded items to be returned in a blocked arrangement
     * \param[out] item_offsets The run-length decoded items' relative offset within the run they belong to
     * \param[in] from_decoded_offset If invoked with from_decoded_offset that is larger than total_decoded_size results
     * in undefined behavior.
     */
    template<typename RelativeOffsetT>
    ROCPRIM_DEVICE ROCPRIM_INLINE void
        run_length_decode(ItemT (&decoded_items)[DECODED_ITEMS_PER_THREAD],
                          RelativeOffsetT (&item_offsets)[DECODED_ITEMS_PER_THREAD],
                          DecodedOffsetT from_decoded_offset = 0)
    {
        // The (global) offset of the first item decoded by this thread
        DecodedOffsetT thread_decoded_offset
            = from_decoded_offset + linear_tid * DECODED_ITEMS_PER_THREAD;

        // The run that the first decoded item of this thread belongs to
        // If this thread's <thread_decoded_offset> is already beyond the total decoded size, it will be assigned to the
        // last run
        RunOffsetT current_run
            = rocprim::static_upper_bound<BLOCK_RUNS>(temp_storage.runs.run_offsets,
                                                      BLOCK_RUNS,
                                                      thread_decoded_offset)
              - static_cast<RunOffsetT>(1U);

        // Set the current_run_end to thread_decoded_offset to trigger new run branch in the first iteration
        DecodedOffsetT current_run_begin, current_run_end = thread_decoded_offset;

        ItemT val{};

#pragma unroll
        for(DecodedOffsetT i = 0; i < DECODED_ITEMS_PER_THREAD; ++i, ++thread_decoded_offset)
        {
            // If we are in a new run...
            if(thread_decoded_offset == current_run_end)
            {
                // The value of the new run
                val = temp_storage.runs.run_values[current_run];

                // The run bounds
                current_run_begin = temp_storage.runs.run_offsets[current_run];
                current_run_end   = temp_storage.runs.run_offsets[++current_run];
            }

            // Decode the current run by storing the run's value
            decoded_items[i] = val;
            item_offsets[i]  = thread_decoded_offset - current_run_begin;
        }
    }

    /**
     * \brief Run-length decodes the runs previously passed via a call to Init(...) and returns the run-length decoded
     * items in a blocked arrangement to \p decoded_items. If the number of run-length decoded items exceeds the
     * run-length decode buffer (i.e., <b>DECODED_ITEMS_PER_THREAD * BLOCK_THREADS</b>), only the items that fit within
     * the buffer are returned. Subsequent calls to <b>run_length_decode</b> adjusting \p from_decoded_offset can be
     * used to retrieve the remaining run-length decoded items. Calling __syncthreads() between any two calls to
     * <b>run_length_decode</b> is not required.
     *
     * \param[out] decoded_items The run-length decoded items to be returned in a blocked arrangement
     * \param[in] from_decoded_offset If invoked with from_decoded_offset that is larger than total_decoded_size results
     * in undefined behavior.
     */
    ROCPRIM_DEVICE ROCPRIM_INLINE void
        run_length_decode(ItemT (&decoded_items)[DECODED_ITEMS_PER_THREAD],
                          DecodedOffsetT from_decoded_offset = 0)
    {
        DecodedOffsetT item_offsets[DECODED_ITEMS_PER_THREAD];
        run_length_decode(decoded_items, item_offsets, from_decoded_offset);
    }
};

END_ROCPRIM_NAMESPACE

#endif
