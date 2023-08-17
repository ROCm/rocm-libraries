/******************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2019-2023, Advanced Micro Devices, Inc.  All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/
#pragma once

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HIP

#include <thrust/detail/cstdint.h>
#include <thrust/detail/mpl/math.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/distance.h>
#include <thrust/extrema.h>
#include <thrust/pair.h>
#include <thrust/set_operations.h>
#include <thrust/system/hip/detail/execution_policy.h>
#include <thrust/system/hip/detail/get_value.h>
#include <thrust/system/hip/detail/par_to_seq.h>
#include <thrust/system/hip/detail/util.h>


// rocprim include
#include <rocprim/rocprim.hpp>

THRUST_NAMESPACE_BEGIN

namespace hip_rocprim
{

namespace __set_operations
{
    template <bool UpperBound, class IntT, class Size, class It, class T, class Comp>
    THRUST_HIP_DEVICE_FUNCTION void
    binary_search_iteration(It data, Size &begin, Size &end, T key, int shift, Comp comp)
    {
        IntT scale = (1 << shift) - 1;
        Size  mid   = ((begin + scale * end) >> shift);

        T    key2 = data[mid];
        bool pred = UpperBound ? !comp(key, key2) : comp(key2, key);
        if(pred)
            begin = mid + static_cast<Size>(1);
        else
            end = mid;
    }

    template <bool UpperBound, class Size, class T, class It, class Comp>
    THRUST_HIP_DEVICE_FUNCTION int
    binary_search(It data, Size count, T key, Comp comp)
    {
        Size begin = 0;
        Size end   = count;
        while(begin < end)
            binary_search_iteration<UpperBound, int>(data, begin, end, key, 1, comp);
        return begin;
    }

    template <bool UpperBound, class Size, class IntT, class T, class It, class Comp>
    THRUST_HIP_DEVICE_FUNCTION Size
    biased_binary_search(It data, Size count, T key, IntT levels, Comp comp)
    {
        Size begin = 0;
        Size end   = count;

        if(levels >= 4 && begin < end)
            binary_search_iteration<UpperBound, IntT>(data, begin, end, key, 9, comp);
        if(levels >= 3 && begin < end)
            binary_search_iteration<UpperBound, IntT>(data, begin, end, key, 7, comp);
        if(levels >= 2 && begin < end)
            binary_search_iteration<UpperBound, IntT>(data, begin, end, key, 5, comp);
        if(levels >= 1 && begin < end)
            binary_search_iteration<UpperBound, IntT>(data, begin, end, key, 4, comp);

        while(begin < end)
            binary_search_iteration<UpperBound, IntT>(data, begin, end, key, 1, comp);
        return begin;
    }

    template <bool UpperBound, class Size, class It1, class It2, class Comp>
    THRUST_HIP_DEVICE_FUNCTION Size
    merge_path(It1 a, Size aCount, It2 b, Size bCount, Size diag, Comp comp)
    {
        typedef typename thrust::iterator_traits<It1>::value_type T;

        Size begin = thrust::max((Size)0, diag - bCount);
        Size end   = thrust::min(diag, aCount);

        while(begin < end)
        {
            Size  mid  = (begin + end) >> 1;
            T    aKey = a[mid];
            T    bKey = b[diag - 1 - mid];
            bool pred = UpperBound ? comp(aKey, bKey) : !comp(bKey, aKey);
            if(pred)
                begin = mid + 1;
            else
                end = mid;
        }
        return begin;
    }

    template <class It1, class It2, class Size, class Size2, class CompareOp>
    pair<Size, Size> THRUST_HIP_DEVICE_FUNCTION
    balanced_path(It1       keys1,
                  It2       keys2,
                  Size      num_keys1,
                  Size      num_keys2,
                  Size      diag,
                  Size2     levels,
                  CompareOp compare_op)
    {
        typedef typename iterator_traits<It1>::value_type T;

        Size index1 = merge_path<false>(keys1, num_keys1, keys2, num_keys2, diag, compare_op);
        Size index2 = diag - index1;

        bool star = false;
        if(index2 < num_keys2)
        {
            T x = keys2[index2];

            // Search for the beginning of the duplicate run in both A and B.
            Size start1 = biased_binary_search<false>(keys1, index1, x, levels, compare_op);
            Size start2 = biased_binary_search<false>(keys2, index2, x, levels, compare_op);

            // The distance between x's merge path and its lower_bound is its rank.
            // We add up the a and b ranks and evenly distribute them to
            // get a stairstep path.
            Size run1      = index1 - start1;
            Size run2      = index2 - start2;
            Size total_run = run1 + run2;

            // Attempt to advance b and regress a.
            Size advance2 = max<Size>(total_run >> 1, total_run - run1);
            Size end2     = min<Size>(num_keys2, start2 + advance2 + 1);

            Size run_end2
                = index2 + binary_search<true>(keys2 + index2, end2 - index2, x, compare_op);
            run2 = run_end2 - start2;

            advance2      = min<Size>(advance2, run2);
            Size advance1 = total_run - advance2;

            bool round_up = (advance1 == advance2 + 1) && (advance2 < run2);
            if(round_up)
                star = true;

            index1 = start1 + advance1;
        }
        return thrust::make_pair(index1, (diag - index1) + star);
    } // func balanced_path

    //---------------------------------------------------------------------
    // Utility functions
    //---------------------------------------------------------------------

    template <unsigned int BlockSize, unsigned int ItemsPerThread>
    using set_operations_config = rocprim::kernel_config<BlockSize, ItemsPerThread>;

    template <class Key, class Value>
    struct set_operations_config_803
    {
        static constexpr unsigned int item_scale = ::rocprim::detail::ceiling_div<unsigned int>(
            ::rocprim::max(sizeof(Key), sizeof(Value)), sizeof(int));

        using type = set_operations_config<256, ::rocprim::max(1u, 16u / item_scale)>;
    };

    template <class Key, class Value>
    struct set_operations_config_900
    {
        static constexpr unsigned int item_scale = ::rocprim::detail::ceiling_div<unsigned int>(
            ::rocprim::max(sizeof(Key), sizeof(Value)), sizeof(int));

        using type = set_operations_config<256, ::rocprim::max(1u, 16u / item_scale)>;
    };

    template <unsigned int TargetArch, class Key, class Value>
    struct default_set_operations_config
        : rocprim::detail::select_arch<
              TargetArch,
              rocprim::detail::select_arch_case<803, set_operations_config_803<Key, Value>>,
              rocprim::detail::select_arch_case<900, set_operations_config_900<Key, Value>>,
              set_operations_config_900<Key, Value>>
    {
    };

    template <class Config,
              class KeysIt1,
              class KeysIt2,
              class ValuesIt1,
              class ValuesIt2,
              class KeysOutputIt,
              class ValuesOutputIt,
              class Size,
              class CompareOp,
              class SetOp,
              bool HAS_VALUES>
    class SetOpAgent
    {
        using key_type   = typename std::iterator_traits<KeysIt1>::value_type;
        using value_type = typename std::iterator_traits<ValuesIt1>::value_type;

        static constexpr int BLOCK_THREADS    = Config::block_size;
        static constexpr int ITEMS_PER_THREAD = Config::items_per_thread;

    public:
        template <bool IS_FULL_TILE, class T, class It1, class It2>
        THRUST_HIP_DEVICE_FUNCTION void
        gmem_to_reg(T (&output)[ITEMS_PER_THREAD], It1 input1, It2 input2, int count1, int count2)
        {
            const unsigned int thread_id = ::rocprim::detail::block_thread_id<0>();
            if(IS_FULL_TILE)
            {
#pragma unroll
                for(int ITEM = 0; ITEM < ITEMS_PER_THREAD - 1; ++ITEM)
                {
                    int idx      = BLOCK_THREADS * ITEM + thread_id;
                    output[ITEM] = (idx < count1) ? static_cast<T>(input1[idx])
                                                  : static_cast<T>(input2[idx - count1]);
                }

                // last ITEM might be a conditional load even for full tiles
                // please check first before attempting to load.
                int ITEM = ITEMS_PER_THREAD - 1;
                int idx  = BLOCK_THREADS * ITEM + thread_id;
                if(idx < count1 + count2)
                    output[ITEM] = (idx < count1) ? static_cast<T>(input1[idx])
                                                  : static_cast<T>(input2[idx - count1]);
            }
            else
            {
#pragma unroll
                for(int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
                {
                    int idx = BLOCK_THREADS * ITEM + thread_id;
                    if(idx < count1 + count2)
                    {
                        output[ITEM] = (idx < count1) ? static_cast<T>(input1[idx])
                                                      : static_cast<T>(input2[idx - count1]);
                    }
                }
            }
        }

        template <class T, class It>
        THRUST_HIP_DEVICE_FUNCTION void
        reg_to_shared(It output, T (&input)[ITEMS_PER_THREAD])
        {
            const unsigned int thread_id = ::rocprim::detail::block_thread_id<0>();
            #pragma unroll
            for(int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
            {
                int idx     = BLOCK_THREADS * ITEM + thread_id;
                output[idx] = input[ITEM];
            }
        }

        template <class OutputIt, class T, class SharedIt>
        THRUST_HIP_DEVICE_FUNCTION
        void scatter(OutputIt output,
                     T (&input)[ITEMS_PER_THREAD],
                     SharedIt shared,
                     int      active_mask,
                     Size     thread_output_prefix,
                     Size     tile_output_prefix,
                     int      tile_output_count)
        {
            int local_scatter_idx = thread_output_prefix - tile_output_prefix;
            #pragma unroll
            for(int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
            {
                if(active_mask & (1 << ITEM))
                {
                    shared[local_scatter_idx++] = input[ITEM];
                }
            }
            ::rocprim::syncthreads();

            const unsigned int thread_id = ::rocprim::detail::block_thread_id<0>();
            for(int item = thread_id; item < tile_output_count; item += BLOCK_THREADS)
            {
                output[tile_output_prefix + item] = shared[item];
            }
        }

        THRUST_HIP_DEVICE_FUNCTION
        int serial_set_op(key_type* keys,
                          int       keys1_beg,
                          int       keys2_beg,
                          int       keys1_count,
                          int       keys2_count,
                          key_type (&output)[ITEMS_PER_THREAD],
                          int (&indices)[ITEMS_PER_THREAD],
                          CompareOp compare_op,
                          SetOp     set_op)
        {
            int active_mask = set_op(
                keys, keys1_beg, keys2_beg, keys1_count, keys2_count, output, indices, compare_op);

            return active_mask;
        }

        template <bool IS_LAST_TILE, class LookBackScanState>
        THRUST_HIP_DEVICE_FUNCTION
        void consume_tile(Size               tile_idx,
                          LookBackScanState& lookback_scan_state,
                          KeysIt1            keys1_in,
                          KeysIt2            keys2_in,
                          ValuesIt1          values1_in,
                          ValuesIt2          values2_in,
                          KeysOutputIt       keys_out,
                          ValuesOutputIt     values_out,
                          CompareOp          compare_op,
                          SetOp              set_op,
                          pair<Size, Size>*  partitions,
                          Size*              output_count)
        {
            using block_scan_type = ::rocprim::block_scan<Size, BLOCK_THREADS>;

            using offset_scan_prefix_op_type
                = ::rocprim::detail::offset_lookback_scan_prefix_op<Size, LookBackScanState>;

            ROCPRIM_SHARED_MEMORY union
            {
                struct
                {
                    typename block_scan_type::storage_type            scan;
                    typename offset_scan_prefix_op_type::storage_type prefix_op;
                };

                struct
                {
                    int offset[BLOCK_THREADS];

                    union
                    {
                        // Allocate extra shmem than truely neccessary
                        // This will permit to avoid range checks in
                        // serial set operations, e.g. serial_set_difference
                        typename ::rocprim::detail::raw_storage<
                            key_type[BLOCK_THREADS + ITEMS_PER_THREAD * BLOCK_THREADS]>
                            keys_shared;
                        typename ::rocprim::detail::raw_storage<
                            value_type[BLOCK_THREADS + ITEMS_PER_THREAD * BLOCK_THREADS]>
                            values_shared;
                    };
                };
            } storage;

            pair<Size, Size> partition_beg = partitions[tile_idx + 0];
            pair<Size, Size> partition_end = partitions[tile_idx + 1];

            Size keys1_beg = partition_beg.first;
            Size keys1_end = partition_end.first;
            Size keys2_beg = partition_beg.second;
            Size keys2_end = partition_end.second;

            // number of keys per tile
            //
            int num_keys1 = static_cast<int>(keys1_end - keys1_beg);
            int num_keys2 = static_cast<int>(keys2_end - keys2_beg);

            // load keys into shared memory for further processing
            key_type keys_loc[ITEMS_PER_THREAD];

            gmem_to_reg<!IS_LAST_TILE>(
                keys_loc, keys1_in + keys1_beg, keys2_in + keys2_beg, num_keys1, num_keys2);

            reg_to_shared(&storage.keys_shared.get()[0], keys_loc);

            ::rocprim::syncthreads();

            int diag_loc = min<int>(ITEMS_PER_THREAD * threadIdx.x, num_keys1 + num_keys2);

            pair<int, int> partition_loc = balanced_path(&storage.keys_shared.get()[0],
                                                         &storage.keys_shared.get()[num_keys1],
                                                         num_keys1,
                                                         num_keys2,
                                                         diag_loc,
                                                         4,
                                                         compare_op);

            int keys1_beg_loc = partition_loc.first;
            int keys2_beg_loc = partition_loc.second;

            // compute difference between next and current thread
            // to obtain number of elements per thread
            int value = threadIdx.x == 0 ? (num_keys1 << 16) | num_keys2
                                         : (partition_loc.first << 16) | partition_loc.second;

            int dst             = threadIdx.x == 0 ? BLOCK_THREADS - 1 : threadIdx.x - 1;
            storage.offset[dst] = value;

            ::rocprim::syncthreads();

            pair<int, int> partition1_loc = thrust::make_pair(storage.offset[threadIdx.x] >> 16,
                                                              storage.offset[threadIdx.x] & 0xFFFF);

            int keys1_end_loc = partition1_loc.first;
            int keys2_end_loc = partition1_loc.second;

            int num_keys1_loc = keys1_end_loc - keys1_beg_loc;
            int num_keys2_loc = keys2_end_loc - keys2_beg_loc;

            // perform serial set operation
            //
            int indices[ITEMS_PER_THREAD];

            int active_mask = serial_set_op(&storage.keys_shared.get()[0],
                                            keys1_beg_loc,
                                            keys2_beg_loc + num_keys1,
                                            num_keys1_loc,
                                            num_keys2_loc,
                                            keys_loc,
                                            indices,
                                            compare_op,
                                            set_op);
            ::rocprim::syncthreads();

            // look-back scan over thread_output_count
            // to compute global thread_output_base and tile_otput_count;
            Size tile_output_count    = 0;
            Size thread_output_prefix = 0;
            Size tile_output_prefix   = 0;
            Size thread_output_count  = static_cast<Size>(__popc(active_mask));

            if(tile_idx == 0) // first tile
            {
                block_scan_type().exclusive_scan(thread_output_count,
                                                 thread_output_prefix,
                                                 Size(0),
                                                 tile_output_count,
                                                 storage.scan,
                                                 ::rocprim::plus<Size>());
                if(threadIdx.x == 0)
                {
                    // Update tile status if this is not the last tile
                    if(!IS_LAST_TILE)
                    {
                        lookback_scan_state.set_complete(0, tile_output_count);
                    }
                }
            }
            else
            {
                auto prefix_op
                    = offset_scan_prefix_op_type(tile_idx, lookback_scan_state, storage.prefix_op);
                block_scan_type().exclusive_scan(thread_output_count,
                                                 thread_output_prefix,
                                                 storage.scan,
                                                 prefix_op,
                                                 ::rocprim::plus<Size>());

                ::rocprim::syncthreads();

                tile_output_count  = prefix_op.get_reduction();
                tile_output_prefix = prefix_op.get_exclusive_prefix();
            }

            ::rocprim::syncthreads();

            // scatter results
            //
            scatter(keys_out,
                    keys_loc,
                    &storage.keys_shared.get()[0],
                    active_mask,
                    thread_output_prefix,
                    tile_output_prefix,
                    tile_output_count);

            if(HAS_VALUES)
            {
                value_type values_loc[ITEMS_PER_THREAD];
                gmem_to_reg<!IS_LAST_TILE>(values_loc,
                                           values1_in + keys1_beg,
                                           values2_in + keys2_beg,
                                           num_keys1,
                                           num_keys2);

                ::rocprim::syncthreads();

                reg_to_shared(&storage.values_shared.get()[0], values_loc);

                ::rocprim::syncthreads();

                // gather items from shared mem
                //
                #pragma unroll
                for(int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
                {
                    if(active_mask & (1 << ITEM))
                    {
                        values_loc[ITEM] = storage.values_shared.get()[indices[ITEM]];
                    }
                }

                ::rocprim::syncthreads();

                scatter(values_out,
                        values_loc,
                        &storage.values_shared.get()[0],
                        active_mask,
                        thread_output_prefix,
                        tile_output_prefix,
                        tile_output_count);
            }

            if(IS_LAST_TILE && threadIdx.x == 0)
            {
                *output_count = tile_output_prefix + tile_output_count;
            }
        }
    };

    //---------------------------------------------------------------------
    // Serial set operations
    //---------------------------------------------------------------------

    // serial_set_intersection
    // -----------------------
    // emit A if A and B are in range and equal.
    struct serial_set_intersection
    {
        // max_input_size <= 32
        template <class T, class CompareOp, int ITEMS_PER_THREAD>
        int THRUST_HIP_DEVICE_FUNCTION
        operator()(T*  keys,
                   int keys1_beg,
                   int keys2_beg,
                   int keys1_count,
                   int keys2_count,
                   T (&output)[ITEMS_PER_THREAD],
                   int (&indices)[ITEMS_PER_THREAD],
                   CompareOp compare_op)
        {
            int active_mask = 0;

            int aBegin = keys1_beg;
            int bBegin = keys2_beg;
            int aEnd   = keys1_beg + keys1_count;
            int bEnd   = keys2_beg + keys2_count;

            T aKey = keys[aBegin];
            T bKey = keys[bBegin];

            #pragma unroll
            for(int i = 0; i < ITEMS_PER_THREAD; ++i)
            {
                bool pA = compare_op(aKey, bKey);
                bool pB = compare_op(bKey, aKey);

                // The outputs must come from A by definition of set interection.
                output[i]  = aKey;
                indices[i] = aBegin;

                if((aBegin < aEnd) && (bBegin < bEnd) && pA == pB)
                    active_mask |= 1 << i;

                if(!pB)
                {
                    aKey = keys[++aBegin];
                }
                if(!pA)
                {
                    bKey = keys[++bBegin];
                }
            }
            return active_mask;
        }
    }; // struct serial_set_intersection

    // serial_set_symmetric_difference
    // ---------------------
    // emit A if A < B and emit B if B < A.
    struct serial_set_symmetric_difference
    {
        // max_input_size <= 32
        template <class T, class CompareOp, int ITEMS_PER_THREAD>
        int THRUST_HIP_DEVICE_FUNCTION
        operator()(T*  keys,
                   int keys1_beg,
                   int keys2_beg,
                   int keys1_count,
                   int keys2_count,
                   T (&output)[ITEMS_PER_THREAD],
                   int (&indices)[ITEMS_PER_THREAD],
                   CompareOp compare_op)
        {
            int active_mask = 0;

            int aBegin = keys1_beg;
            int bBegin = keys2_beg;
            int aEnd   = keys1_beg + keys1_count;
            int bEnd   = keys2_beg + keys2_count;
            int end    = aEnd + bEnd;

            T aKey = keys[aBegin];
            T bKey = keys[bBegin];

#pragma unroll
            for(int i = 0; i < ITEMS_PER_THREAD; ++i)
            {
                bool pB = aBegin >= aEnd;
                bool pA = !pB && bBegin >= bEnd;

                if(!pA && !pB)
                {
                    pA = compare_op(aKey, bKey);
                    pB = !pA && compare_op(bKey, aKey);
                }

                // The outputs must come from A by definition of set difference.
                output[i]  = pA ? aKey : bKey;
                indices[i] = pA ? aBegin : bBegin;

                if(aBegin + bBegin < end && pA != pB)
                    active_mask |= 1 << i;

                if(!pB)
                {
                    aKey = keys[++aBegin];
                }
                if(!pA)
                {
                    bKey = keys[++bBegin];
                }
            }
            return active_mask;
        }
    }; // struct set_symmetric_difference

    // serial_set_difference
    // ---------------------
    // emit A if A < B
    struct serial_set_difference
    {
        // max_input_size <= 32
        template <class T, class CompareOp, int ITEMS_PER_THREAD>
        int THRUST_HIP_DEVICE_FUNCTION
        operator()(T*  keys,
                   int keys1_beg,
                   int keys2_beg,
                   int keys1_count,
                   int keys2_count,
                   T (&output)[ITEMS_PER_THREAD],
                   int (&indices)[ITEMS_PER_THREAD],
                   CompareOp compare_op)
        {
            int active_mask = 0;

            int aBegin = keys1_beg;
            int bBegin = keys2_beg;
            int aEnd   = keys1_beg + keys1_count;
            int bEnd   = keys2_beg + keys2_count;
            int end    = aEnd + bEnd;

            T aKey = keys[aBegin];
            T bKey = keys[bBegin];

            #pragma unroll
            for(int i = 0; i < ITEMS_PER_THREAD; ++i)
            {
                bool pB = aBegin >= aEnd;
                bool pA = !pB && bBegin >= bEnd;

                if(!pA && !pB)
                {
                    pA = compare_op(aKey, bKey);
                    pB = !pA && compare_op(bKey, aKey);
                }

                // The outputs must come from A by definition of set difference.
                output[i]  = aKey;
                indices[i] = aBegin;

                if(aBegin + bBegin < end && pA)
                    active_mask |= 1 << i;

                if(!pB)
                {
                    aKey = keys[++aBegin];
                }
                if(!pA)
                {
                    bKey = keys[++bBegin];
                }
            }
            return active_mask;
        }
    }; // struct set_difference

    // serial_set_union
    // ----------------
    // emit A if A <= B else emit B
    struct serial_set_union
    {
        // max_input_size <= 32
        template <class T, class CompareOp, int ITEMS_PER_THREAD>
        int THRUST_HIP_DEVICE_FUNCTION
        operator()(T*  keys,
                   int keys1_beg,
                   int keys2_beg,
                   int keys1_count,
                   int keys2_count,
                   T (&output)[ITEMS_PER_THREAD],
                   int (&indices)[ITEMS_PER_THREAD],
                   CompareOp compare_op)
        {
            int active_mask = 0;

            int aBegin = keys1_beg;
            int bBegin = keys2_beg;
            int aEnd   = keys1_beg + keys1_count;
            int bEnd   = keys2_beg + keys2_count;
            int end    = aEnd + bEnd;

            T aKey = keys[aBegin];
            T bKey = keys[bBegin];

            #pragma unroll
            for(int i = 0; i < ITEMS_PER_THREAD; ++i)
            {
                bool pB = aBegin >= aEnd;
                bool pA = !pB && bBegin >= bEnd;

                if(!pA && !pB)
                {
                    pA = compare_op(aKey, bKey);
                    pB = !pA && compare_op(bKey, aKey);
                }

                // Output A in case of a tie, so check if b < a.
                output[i]  = pB ? bKey : aKey;
                indices[i] = pB ? bBegin : aBegin;

                if(aBegin + bBegin < end)
                    active_mask |= 1 << i;

                if(!pB)
                {
                    aKey = keys[++aBegin];
                }
                if(!pA)
                {
                    bKey = keys[++bBegin];
                }
            }
            return active_mask;
        }
    }; // struct set_union

    template <class Config,
              bool HAS_VALUES,
              class KeysIt1,
              class KeysIt2,
              class ValuesIt1,
              class ValuesIt2,
              class Size,
              class KeysOutputIt,
              class ValuesOutputIt,
              class CompareOp,
              class SetOp,
              class LookBackScanState>
    __global__
    THRUST_HIP_LAUNCH_BOUNDS_DEFAULT
    void lookback_set_op_kernel(KeysIt1                                         keys1,
                                KeysIt2                                         keys2,
                                ValuesIt1                                       values1,
                                ValuesIt2                                       values2,
                                KeysOutputIt                                    keys_output,
                                ValuesOutputIt                                  values_output,
                                CompareOp                                       compare_op,
                                SetOp                                           set_op,
                                pair<Size, Size>*                               partitions,
                                Size*                                           output_count,
                                LookBackScanState                               lookback_scan_state,
                                rocprim::detail::ordered_block_id<unsigned int> ordered_bid)
    {
        ROCPRIM_SHARED_MEMORY
        typename rocprim::detail::ordered_block_id<unsigned int>::storage_type storage_ordered_bid;

        const int num_tiles = gridDim.x;
        const int tile_idx
            = ordered_bid.get(::rocprim::flat_block_thread_id(), storage_ordered_bid);

        SetOpAgent<Config,
                   KeysIt1,
                   KeysIt2,
                   ValuesIt1,
                   ValuesIt2,
                   KeysOutputIt,
                   ValuesOutputIt,
                   Size,
                   CompareOp,
                   SetOp,
                   HAS_VALUES>
            agent;

        if(tile_idx < num_tiles - 1)
        {
            agent.template consume_tile<false>(tile_idx,
                                               lookback_scan_state,
                                               keys1,
                                               keys2,
                                               values1,
                                               values2,
                                               keys_output,
                                               values_output,
                                               compare_op,
                                               set_op,
                                               partitions,
                                               output_count);
        }
        else
        {
            agent.template consume_tile<true>(tile_idx,
                                              lookback_scan_state,
                                              keys1,
                                              keys2,
                                              values1,
                                              values2,
                                              keys_output,
                                              values_output,
                                              compare_op,
                                              set_op,
                                              partitions,
                                              output_count);
        }
    }

#define ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR(name, size, start)                         \
    {                                                                                          \
        auto error = hipPeekAtLastError();                                                     \
        if(error != hipSuccess)                                                                \
            return error;                                                                      \
        if(debug_synchronous)                                                                  \
        {                                                                                      \
            std::cout << name << "(" << size << ")";                                           \
            auto error = hipStreamSynchronize(stream);                                         \
            if(error != hipSuccess)                                                            \
                return error;                                                                  \
            auto end = std::chrono::high_resolution_clock::now();                              \
            auto d   = std::chrono::duration_cast<std::chrono::duration<double>>(end - start); \
            std::cout << " " << d.count() * 1000 << " ms" << '\n';                             \
        }                                                                                      \
    }

    template <bool HAS_VALUES,
              class KeysIt1,
              class KeysIt2,
              class ValuesIt1,
              class ValuesIt2,
              class Size,
              class KeysOutputIt,
              class ValuesOutputIt,
              class CompareOp,
              class SetOp>
    hipError_t THRUST_HIP_RUNTIME_FUNCTION
    doit_step(void*          temporary_storage,
              size_t&        storage_size,
              KeysIt1        keys1,
              KeysIt2        keys2,
              ValuesIt1      values1,
              ValuesIt2      values2,
              Size           num_keys1,
              Size           num_keys2,
              KeysOutputIt   keys_output,
              ValuesOutputIt values_output,
              Size*          output_count,
              CompareOp      compare_op,
              SetOp          set_op,
              hipStream_t    stream,
              bool           debug_synchronous)
    {
        using key_type   = typename std::iterator_traits<KeysIt1>::value_type;
        using value_type = typename std::iterator_traits<ValuesIt1>::value_type;

        using config = default_set_operations_config<ROCPRIM_TARGET_ARCH, key_type, value_type>;

        using block_state_type      = ::rocprim::detail::lookback_scan_state<Size>;
        using ordered_block_id_type = ::rocprim::detail::ordered_block_id<unsigned int>;

        constexpr unsigned int block_size       = config::block_size;
        constexpr unsigned int items_per_thread = config::items_per_thread;
        constexpr unsigned int items_per_block  = block_size * items_per_thread - 1;

        Size keys_total = num_keys1 + num_keys2;
        if(keys_total == 0)
            return hipErrorInvalidValue;

        hipError_t status = hipSuccess;

        const unsigned int number_of_blocks = (keys_total + items_per_block - 1) / items_per_block;

        // Calculate required temporary storage
        size_t scan_state_bytes
            = ::rocprim::detail::align_size(block_state_type::get_storage_size(number_of_blocks));
        size_t ordered_block_id_bytes
            = ::rocprim::detail::align_size(ordered_block_id_type::get_storage_size());
        size_t partition_storage_bytes = (number_of_blocks + 1) * sizeof(pair<Size, Size>);
        if(temporary_storage == nullptr)
        {
            // storage_size is never zero
            storage_size = scan_state_bytes + ordered_block_id_bytes + partition_storage_bytes;
            return hipSuccess;
        }

        // Start point for time measurements
        std::chrono::high_resolution_clock::time_point start;
        if(debug_synchronous)
        {
            std::cout << "keys_total " << keys_total << '\n';
            std::cout << "number_of_blocks " << number_of_blocks << '\n';
            std::cout << "block_size " << block_size << '\n';
            std::cout << "items_per_thread " << items_per_thread << '\n';
            std::cout << "items_per_block " << items_per_block << '\n';
        }

        auto ptr = reinterpret_cast<char*>(temporary_storage);
        // Create and initialize lookback_scan_state obj
        auto blocks_state = block_state_type::create(ptr, number_of_blocks);
        ptr += scan_state_bytes;
        // Create and initialize ordered_block_id obj
        auto ordered_bid
            = ordered_block_id_type::create(reinterpret_cast<ordered_block_id_type::id_type*>(ptr));
        ptr += ordered_block_id_bytes;
        pair<Size, Size>* partitions = reinterpret_cast<pair<Size, Size>*>(ptr);

        if(debug_synchronous)
            start = std::chrono::high_resolution_clock::now();
        auto grid_size = (number_of_blocks + block_size - 1) / block_size;
        hipLaunchKernelGGL(HIP_KERNEL_NAME(rocprim::detail::init_lookback_scan_state_kernel),
                           dim3(grid_size),
                           dim3(block_size),
                           0,
                           stream,
                           blocks_state,
                           number_of_blocks,
                           ordered_bid);
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR(
            "init_lookback_scan_state_kernel", number_of_blocks, start)

        status = __parallel_for::parallel_for(
            number_of_blocks + 1,
            [=] __device__(Size idx) mutable {
                Size partition_at = min<Size>(idx * items_per_block, num_keys1 + num_keys2);
                partitions[idx]   = balanced_path(
                    keys1, keys2, num_keys1, num_keys2, partition_at, 4ll, compare_op);
            },
            stream);
        if(status != hipSuccess)
            return status;

        if(debug_synchronous)
            start = std::chrono::high_resolution_clock::now();
        hipLaunchKernelGGL(HIP_KERNEL_NAME(lookback_set_op_kernel<config, HAS_VALUES>),
                           dim3(number_of_blocks),
                           dim3(block_size),
                           0,
                           stream,
                           keys1,
                           keys2,
                           values1,
                           values2,
                           keys_output,
                           values_output,
                           compare_op,
                           set_op,
                           partitions,
                           output_count,
                           blocks_state,
                           ordered_bid);
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("lookback_set_op_kernel", keys_total, start)

        return status;
    }

#undef ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR

    template <bool HAS_VALUES,
              typename Derived,
              typename KeysIt1,
              typename KeysIt2,
              typename ValuesIt1,
              typename ValuesIt2,
              typename KeysOutputIt,
              typename ValuesOutputIt,
              typename CompareOp,
              typename SetOp>
    THRUST_HIP_RUNTIME_FUNCTION
    pair<KeysOutputIt, ValuesOutputIt>
    set_operations(execution_policy<Derived>& policy,
                   KeysIt1                    keys1_first,
                   KeysIt1                    keys1_last,
                   KeysIt2                    keys2_first,
                   KeysIt2                    keys2_last,
                   ValuesIt1                  values1_first,
                   ValuesIt2                  values2_first,
                   KeysOutputIt               keys_output,
                   ValuesOutputIt             values_output,
                   CompareOp                  compare_op,
                   SetOp                      set_op)
    {
        typedef typename iterator_traits<KeysIt1>::difference_type size_type;
        size_type num_keys1 = static_cast<size_type>(thrust::distance(keys1_first, keys1_last));
        size_type num_keys2 = static_cast<size_type>(thrust::distance(keys2_first, keys2_last));

        if(num_keys1 + num_keys2 == 0)
            return thrust::make_pair(keys_output, values_output);

        size_t      temp_storage_bytes = 0;
        hipStream_t stream             = hip_rocprim::stream(policy);
        bool        debug_sync         = THRUST_HIP_DEBUG_SYNC_FLAG;

        hip_rocprim::throw_on_error(doit_step<HAS_VALUES>(NULL,
                                                          temp_storage_bytes,
                                                          keys1_first,
                                                          keys2_first,
                                                          values1_first,
                                                          values2_first,
                                                          num_keys1,
                                                          num_keys2,
                                                          keys_output,
                                                          values_output,
                                                          reinterpret_cast<size_type*>(NULL),
                                                          compare_op,
                                                          set_op,
                                                          stream,
                                                          debug_sync),
                                    "set_operations failed on 1st step");

        temp_storage_bytes = rocprim::detail::align_size(temp_storage_bytes);

        // Allocate temporary storage.
        thrust::detail::temporary_array<thrust::detail::uint8_t, Derived>
            tmp(policy, temp_storage_bytes + sizeof(size_type));
        void *ptr = static_cast<void*>(tmp.data().get());

        size_type* d_output_count = reinterpret_cast<size_type*>(
            reinterpret_cast<char*>(ptr) + temp_storage_bytes);

        hip_rocprim::throw_on_error(doit_step<HAS_VALUES>(ptr,
                                                          temp_storage_bytes,
                                                          keys1_first,
                                                          keys2_first,
                                                          values1_first,
                                                          values2_first,
                                                          num_keys1,
                                                          num_keys2,
                                                          keys_output,
                                                          values_output,
                                                          d_output_count,
                                                          compare_op,
                                                          set_op,
                                                          stream,
                                                          debug_sync),
                                    "set_operations failed on 2nd step");

        size_type output_count = hip_rocprim::get_value(policy, d_output_count);

        return thrust::make_pair(keys_output + output_count, values_output + output_count);
    }
}

//-------------------------
// Thrust API entry points
//-------------------------

template <class Derived, class ItemsIt1, class ItemsIt2, class OutputIt, class CompareOp>
OutputIt THRUST_HIP_FUNCTION
set_difference(execution_policy<Derived>& policy,
               ItemsIt1                   items1_first,
               ItemsIt1                   items1_last,
               ItemsIt2                   items2_first,
               ItemsIt2                   items2_last,
               OutputIt                   result,
               CompareOp                  compare)
{
    using dummy_type  = typename thrust::iterator_value<ItemsIt1>::type;
    using set_op_type = typename __set_operations::serial_set_difference;

    THRUST_HIP_PRESERVE_KERNELS_WORKAROUND((__set_operations::set_operations<false,
                                                                             Derived,
                                                                             ItemsIt1,
                                                                             ItemsIt2,
                                                                             dummy_type*,
                                                                             dummy_type*,
                                                                             OutputIt,
                                                                             dummy_type*,
                                                                             CompareOp,
                                                                             set_op_type>));
#if __THRUST_HAS_HIPRT__
    dummy_type* null_ = NULL;
    return __set_operations::set_operations<false>(policy,
                                                   items1_first,
                                                   items1_last,
                                                   items2_first,
                                                   items2_last,
                                                   null_,
                                                   null_,
                                                   result,
                                                   null_,
                                                   compare,
                                                   set_op_type())
        .first;
#else
    return thrust::set_difference(cvt_to_seq(derived_cast(policy)),
                                  items1_first,
                                  items1_last,
                                  items2_first,
                                  items2_last,
                                  result,
                                  compare);
#endif
}

template <class Derived, class ItemsIt1, class ItemsIt2, class OutputIt>
OutputIt THRUST_HIP_FUNCTION
set_difference(execution_policy<Derived>& policy,
               ItemsIt1                   items1_first,
               ItemsIt1                   items1_last,
               ItemsIt2                   items2_first,
               ItemsIt2                   items2_last,
               OutputIt                   result)
{
    typedef typename thrust::iterator_value<ItemsIt1>::type value_type;
    return hip_rocprim::set_difference(
        policy, items1_first, items1_last, items2_first, items2_last, result, less<value_type>());
}

/*****************************/

template <class Derived, class ItemsIt1, class ItemsIt2, class OutputIt, class CompareOp>
OutputIt THRUST_HIP_FUNCTION
set_intersection(execution_policy<Derived>& policy,
                 ItemsIt1                   items1_first,
                 ItemsIt1                   items1_last,
                 ItemsIt2                   items2_first,
                 ItemsIt2                   items2_last,
                 OutputIt                   result,
                 CompareOp                  compare)
{
    using dummy_type  = typename thrust::iterator_value<ItemsIt1>::type;
    using set_op_type = typename __set_operations::serial_set_intersection;

    THRUST_HIP_PRESERVE_KERNELS_WORKAROUND((__set_operations::set_operations<false,
                                                                             Derived,
                                                                             ItemsIt1,
                                                                             ItemsIt2,
                                                                             dummy_type*,
                                                                             dummy_type*,
                                                                             OutputIt,
                                                                             dummy_type*,
                                                                             CompareOp,
                                                                             set_op_type>));
#if __THRUST_HAS_HIPRT__
    dummy_type* null_ = NULL;
    return __set_operations::set_operations<false>(policy,
                                                   items1_first,
                                                   items1_last,
                                                   items2_first,
                                                   items2_last,
                                                   null_,
                                                   null_,
                                                   result,
                                                   null_,
                                                   compare,
                                                   set_op_type())
        .first;
#else
    return thrust::set_intersection(cvt_to_seq(derived_cast(policy)),
                                    items1_first,
                                    items1_last,
                                    items2_first,
                                    items2_last,
                                    result,
                                    compare);
#endif
}

template <class Derived, class ItemsIt1, class ItemsIt2, class OutputIt>
OutputIt THRUST_HIP_FUNCTION
set_intersection(execution_policy<Derived>& policy,
                 ItemsIt1                   items1_first,
                 ItemsIt1                   items1_last,
                 ItemsIt2                   items2_first,
                 ItemsIt2                   items2_last,
                 OutputIt                   result)
{
    typedef typename thrust::iterator_value<ItemsIt1>::type value_type;
    return hip_rocprim::set_intersection(
        policy, items1_first, items1_last, items2_first, items2_last, result, less<value_type>());
}

/*****************************/

template <class Derived, class ItemsIt1, class ItemsIt2, class OutputIt, class CompareOp>
OutputIt THRUST_HIP_FUNCTION
set_symmetric_difference(execution_policy<Derived>& policy,
                         ItemsIt1                   items1_first,
                         ItemsIt1                   items1_last,
                         ItemsIt2                   items2_first,
                         ItemsIt2                   items2_last,
                         OutputIt                   result,
                         CompareOp                  compare)
{
    using dummy_type  = typename thrust::iterator_value<ItemsIt1>::type;
    using set_op_type = typename __set_operations::serial_set_symmetric_difference;

    THRUST_HIP_PRESERVE_KERNELS_WORKAROUND((__set_operations::set_operations<false,
                                                                             Derived,
                                                                             ItemsIt1,
                                                                             ItemsIt2,
                                                                             dummy_type*,
                                                                             dummy_type*,
                                                                             OutputIt,
                                                                             dummy_type*,
                                                                             CompareOp,
                                                                             set_op_type>));
#if __THRUST_HAS_HIPRT__
    dummy_type* null_ = NULL;
    return __set_operations::set_operations<false>(policy,
                                                   items1_first,
                                                   items1_last,
                                                   items2_first,
                                                   items2_last,
                                                   null_,
                                                   null_,
                                                   result,
                                                   null_,
                                                   compare,
                                                   set_op_type())
        .first;
#else
    return thrust::set_symmetric_difference(cvt_to_seq(derived_cast(policy)),
                                            items1_first,
                                            items1_last,
                                            items2_first,
                                            items2_last,
                                            result,
                                            compare);
#endif
}

template <class Derived, class ItemsIt1, class ItemsIt2, class OutputIt>
OutputIt THRUST_HIP_FUNCTION
set_symmetric_difference(execution_policy<Derived>& policy,
                         ItemsIt1                   items1_first,
                         ItemsIt1                   items1_last,
                         ItemsIt2                   items2_first,
                         ItemsIt2                   items2_last,
                         OutputIt                   result)
{
    typedef typename thrust::iterator_value<ItemsIt1>::type value_type;
    return hip_rocprim::set_symmetric_difference(
        policy, items1_first, items1_last, items2_first, items2_last, result, less<value_type>());
}

/*****************************/

template <class Derived, class ItemsIt1, class ItemsIt2, class OutputIt, class CompareOp>
OutputIt THRUST_HIP_FUNCTION
set_union(execution_policy<Derived>& policy,
          ItemsIt1                   items1_first,
          ItemsIt1                   items1_last,
          ItemsIt2                   items2_first,
          ItemsIt2                   items2_last,
          OutputIt                   result,
          CompareOp                  compare)
{
    using dummy_type  = typename thrust::iterator_value<ItemsIt1>::type;
    using set_op_type = typename __set_operations::serial_set_union;

    THRUST_HIP_PRESERVE_KERNELS_WORKAROUND((__set_operations::set_operations<false,
                                                                             Derived,
                                                                             ItemsIt1,
                                                                             ItemsIt2,
                                                                             dummy_type*,
                                                                             dummy_type*,
                                                                             OutputIt,
                                                                             dummy_type*,
                                                                             CompareOp,
                                                                             set_op_type>));
#if __THRUST_HAS_HIPRT__
    dummy_type* null_ = NULL;
    return __set_operations::set_operations<false>(policy,
                                                   items1_first,
                                                   items1_last,
                                                   items2_first,
                                                   items2_last,
                                                   null_,
                                                   null_,
                                                   result,
                                                   null_,
                                                   compare,
                                                   set_op_type())
        .first;
#else
    return thrust::set_union(cvt_to_seq(derived_cast(policy)),
                             items1_first,
                             items1_last,
                             items2_first,
                             items2_last,
                             result,
                             compare);
#endif
}

template <class Derived, class ItemsIt1, class ItemsIt2, class OutputIt>
OutputIt THRUST_HIP_FUNCTION
set_union(execution_policy<Derived>& policy,
          ItemsIt1                   items1_first,
          ItemsIt1                   items1_last,
          ItemsIt2                   items2_first,
          ItemsIt2                   items2_last,
          OutputIt                   result)
{
    typedef typename thrust::iterator_value<ItemsIt1>::type value_type;
    return hip_rocprim::set_union(
        policy, items1_first, items1_last, items2_first, items2_last, result, less<value_type>());
}

/*****************************/
/*****************************/
/*****     *_by_key      *****/
/*****************************/
/*****************************/

/*****************************/

template <class Derived,
          class KeysIt1,
          class KeysIt2,
          class ItemsIt1,
          class ItemsIt2,
          class KeysOutputIt,
          class ItemsOutputIt,
          class CompareOp>
pair<KeysOutputIt, ItemsOutputIt> THRUST_HIP_FUNCTION
set_difference_by_key(execution_policy<Derived>& policy,
                      KeysIt1                    keys1_first,
                      KeysIt1                    keys1_last,
                      KeysIt2                    keys2_first,
                      KeysIt2                    keys2_last,
                      ItemsIt1                   items1_first,
                      ItemsIt2                   items2_first,
                      KeysOutputIt               keys_result,
                      ItemsOutputIt              items_result,
                      CompareOp                  compare_op)
{
    using set_op_type = typename __set_operations::serial_set_difference;

    THRUST_HIP_PRESERVE_KERNELS_WORKAROUND((__set_operations::set_operations<true,
                                                                             Derived,
                                                                             KeysIt1,
                                                                             KeysIt2,
                                                                             ItemsIt1,
                                                                             ItemsIt2,
                                                                             KeysOutputIt,
                                                                             ItemsOutputIt,
                                                                             CompareOp,
                                                                             set_op_type>));
#if __THRUST_HAS_HIPRT__
    return __set_operations::set_operations<true>(policy,
                                                  keys1_first,
                                                  keys1_last,
                                                  keys2_first,
                                                  keys2_last,
                                                  items1_first,
                                                  items2_first,
                                                  keys_result,
                                                  items_result,
                                                  compare_op,
                                                  set_op_type());
#else
    return thrust::set_difference_by_key(cvt_to_seq(derived_cast(policy)),
                                         keys1_first,
                                         keys1_last,
                                         keys2_first,
                                         keys2_last,
                                         items1_first,
                                         items2_first,
                                         keys_result,
                                         items_result,
                                         compare_op);
#endif
}

template <class Derived,
          class KeysIt1,
          class KeysIt2,
          class ItemsIt1,
          class ItemsIt2,
          class KeysOutputIt,
          class ItemsOutputIt>
pair<KeysOutputIt, ItemsOutputIt> THRUST_HIP_FUNCTION
set_difference_by_key(execution_policy<Derived>& policy,
                      KeysIt1                    keys1_first,
                      KeysIt1                    keys1_last,
                      KeysIt2                    keys2_first,
                      KeysIt2                    keys2_last,
                      ItemsIt1                   items1_first,
                      ItemsIt2                   items2_first,
                      KeysOutputIt               keys_result,
                      ItemsOutputIt              items_result)
{
    typedef typename thrust::iterator_value<KeysIt1>::type value_type;
    return hip_rocprim::set_difference_by_key(policy,
                                              keys1_first,
                                              keys1_last,
                                              keys2_first,
                                              keys2_last,
                                              items1_first,
                                              items2_first,
                                              keys_result,
                                              items_result,
                                              less<value_type>());
}

/*****************************/

template <class Derived,
          class KeysIt1,
          class KeysIt2,
          class ItemsIt1,
          class ItemsIt2,
          class KeysOutputIt,
          class ItemsOutputIt,
          class CompareOp>
pair<KeysOutputIt, ItemsOutputIt> THRUST_HIP_FUNCTION
set_intersection_by_key(execution_policy<Derived>& policy,
                        KeysIt1                    keys1_first,
                        KeysIt1                    keys1_last,
                        KeysIt2                    keys2_first,
                        KeysIt2                    keys2_last,
                        ItemsIt1                   items1_first,
                        KeysOutputIt               keys_result,
                        ItemsOutputIt              items_result,
                        CompareOp                  compare_op)
{
    using set_op_type = typename __set_operations::serial_set_intersection;

    THRUST_HIP_PRESERVE_KERNELS_WORKAROUND((__set_operations::set_operations<true,
                                                                             Derived,
                                                                             KeysIt1,
                                                                             KeysIt2,
                                                                             ItemsIt1,
                                                                             ItemsIt1,
                                                                             KeysOutputIt,
                                                                             ItemsOutputIt,
                                                                             CompareOp,
                                                                             set_op_type>));
#if __THRUST_HAS_HIPRT__
    return __set_operations::set_operations<true>(policy,
                                                  keys1_first,
                                                  keys1_last,
                                                  keys2_first,
                                                  keys2_last,
                                                  items1_first,
                                                  items1_first,
                                                  keys_result,
                                                  items_result,
                                                  compare_op,
                                                  set_op_type());
#else
    return thrust::set_intersection_by_key(cvt_to_seq(derived_cast(policy)),
                                           keys1_first,
                                           keys1_last,
                                           keys2_first,
                                           keys2_last,
                                           items1_first,
                                           keys_result,
                                           items_result,
                                           compare_op);
#endif
}

template <class Derived,
          class KeysIt1,
          class KeysIt2,
          class ItemsIt1,
          class ItemsIt2,
          class KeysOutputIt,
          class ItemsOutputIt>
pair<KeysOutputIt, ItemsOutputIt> THRUST_HIP_FUNCTION
set_intersection_by_key(execution_policy<Derived>& policy,
                        KeysIt1                    keys1_first,
                        KeysIt1                    keys1_last,
                        KeysIt2                    keys2_first,
                        KeysIt2                    keys2_last,
                        ItemsIt1                   items1_first,
                        KeysOutputIt               keys_result,
                        ItemsOutputIt              items_result)
{
    typedef typename thrust::iterator_value<KeysIt1>::type value_type;
    return hip_rocprim::set_intersection_by_key(policy,
                                                keys1_first,
                                                keys1_last,
                                                keys2_first,
                                                keys2_last,
                                                items1_first,
                                                keys_result,
                                                items_result,
                                                less<value_type>());
}

/*****************************/

template <class Derived,
          class KeysIt1,
          class KeysIt2,
          class ItemsIt1,
          class ItemsIt2,
          class KeysOutputIt,
          class ItemsOutputIt,
          class CompareOp>
pair<KeysOutputIt, ItemsOutputIt> THRUST_HIP_FUNCTION
set_symmetric_difference_by_key(execution_policy<Derived>& policy,
                                KeysIt1                    keys1_first,
                                KeysIt1                    keys1_last,
                                KeysIt2                    keys2_first,
                                KeysIt2                    keys2_last,
                                ItemsIt1                   items1_first,
                                ItemsIt2                   items2_first,
                                KeysOutputIt               keys_result,
                                ItemsOutputIt              items_result,
                                CompareOp                  compare_op)
{
    using set_op_type = typename __set_operations::serial_set_symmetric_difference;

    THRUST_HIP_PRESERVE_KERNELS_WORKAROUND((__set_operations::set_operations<true,
                                                                             Derived,
                                                                             KeysIt1,
                                                                             KeysIt2,
                                                                             ItemsIt1,
                                                                             ItemsIt2,
                                                                             KeysOutputIt,
                                                                             ItemsOutputIt,
                                                                             CompareOp,
                                                                             set_op_type>));
#if __THRUST_HAS_HIPRT__
    return __set_operations::set_operations<true>(policy,
                                                  keys1_first,
                                                  keys1_last,
                                                  keys2_first,
                                                  keys2_last,
                                                  items1_first,
                                                  items2_first,
                                                  keys_result,
                                                  items_result,
                                                  compare_op,
                                                  set_op_type());
#else
    return thrust::set_symmetric_difference_by_key(cvt_to_seq(derived_cast(policy)),
                                                   keys1_first,
                                                   keys1_last,
                                                   keys2_first,
                                                   keys2_last,
                                                   items1_first,
                                                   items2_first,
                                                   keys_result,
                                                   items_result,
                                                   compare_op);
#endif
}

template <class Derived,
          class KeysIt1,
          class KeysIt2,
          class ItemsIt1,
          class ItemsIt2,
          class KeysOutputIt,
          class ItemsOutputIt>
pair<KeysOutputIt, ItemsOutputIt> THRUST_HIP_FUNCTION
set_symmetric_difference_by_key(execution_policy<Derived>& policy,
                                KeysIt1                    keys1_first,
                                KeysIt1                    keys1_last,
                                KeysIt2                    keys2_first,
                                KeysIt2                    keys2_last,
                                ItemsIt1                   items1_first,
                                ItemsIt2                   items2_first,
                                KeysOutputIt               keys_result,
                                ItemsOutputIt              items_result)
{
    typedef typename thrust::iterator_value<KeysIt1>::type value_type;
    return hip_rocprim::set_symmetric_difference_by_key(policy,
                                                        keys1_first,
                                                        keys1_last,
                                                        keys2_first,
                                                        keys2_last,
                                                        items1_first,
                                                        items2_first,
                                                        keys_result,
                                                        items_result,
                                                        less<value_type>());
}

/*****************************/

template <class Derived,
          class KeysIt1,
          class KeysIt2,
          class ItemsIt1,
          class ItemsIt2,
          class KeysOutputIt,
          class ItemsOutputIt,
          class CompareOp>
pair<KeysOutputIt, ItemsOutputIt> THRUST_HIP_FUNCTION
set_union_by_key(execution_policy<Derived>& policy,
                 KeysIt1                    keys1_first,
                 KeysIt1                    keys1_last,
                 KeysIt2                    keys2_first,
                 KeysIt2                    keys2_last,
                 ItemsIt1                   items1_first,
                 ItemsIt2                   items2_first,
                 KeysOutputIt               keys_result,
                 ItemsOutputIt              items_result,
                 CompareOp                  compare_op)
{
    using set_op_type = typename __set_operations::serial_set_union;

    THRUST_HIP_PRESERVE_KERNELS_WORKAROUND((__set_operations::set_operations<true,
                                                                             Derived,
                                                                             KeysIt1,
                                                                             KeysIt2,
                                                                             ItemsIt1,
                                                                             ItemsIt2,
                                                                             KeysOutputIt,
                                                                             ItemsOutputIt,
                                                                             CompareOp,
                                                                             set_op_type>));
#if __THRUST_HAS_HIPRT__
    return __set_operations::set_operations<true>(policy,
                                                  keys1_first,
                                                  keys1_last,
                                                  keys2_first,
                                                  keys2_last,
                                                  items1_first,
                                                  items2_first,
                                                  keys_result,
                                                  items_result,
                                                  compare_op,
                                                  set_op_type());
#else
    return thrust::set_union_by_key(cvt_to_seq(derived_cast(policy)),
                                    keys1_first,
                                    keys1_last,
                                    keys2_first,
                                    keys2_last,
                                    items1_first,
                                    items2_first,
                                    keys_result,
                                    items_result,
                                    compare_op);
#endif
}

template <class Derived,
          class KeysIt1,
          class KeysIt2,
          class ItemsIt1,
          class ItemsIt2,
          class KeysOutputIt,
          class ItemsOutputIt>
pair<KeysOutputIt, ItemsOutputIt> THRUST_HIP_FUNCTION
set_union_by_key(execution_policy<Derived>& policy,
                 KeysIt1                    keys1_first,
                 KeysIt1                    keys1_last,
                 KeysIt2                    keys2_first,
                 KeysIt2                    keys2_last,
                 ItemsIt1                   items1_first,
                 ItemsIt2                   items2_first,
                 KeysOutputIt               keys_result,
                 ItemsOutputIt              items_result)
{
    typedef typename thrust::iterator_value<KeysIt1>::type value_type;
    return hip_rocprim::set_union_by_key(policy,
                                         keys1_first,
                                         keys1_last,
                                         keys2_first,
                                         keys2_last,
                                         items1_first,
                                         items2_first,
                                         keys_result,
                                         items_result,
                                         less<value_type>());
}

} // namespace hip_rocprim
THRUST_NAMESPACE_END
#endif
