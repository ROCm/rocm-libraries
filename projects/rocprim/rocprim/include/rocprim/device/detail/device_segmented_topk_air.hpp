// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DETAIL_DEVICE_SEGMENTED_TOPK_AIR_HPP_
#define ROCPRIM_DEVICE_DETAIL_DEVICE_SEGMENTED_TOPK_AIR_HPP_

#include "../../detail/temp_storage.hpp"
#include "../../iterator/counting_iterator.hpp"
#include "../../iterator/discard_iterator.hpp"
#include "../config_types.hpp"
#include "../device_segmented_topk_air_config.hpp"
#include "./device_segmented_reduce.hpp"
#include "./device_topk_air.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{
// TODO: This algorithm can be optimized by using the same logic of device_segmented_radix_sort
// Use Partitioner to separage small and large segments and run different kernel for different
// segments
template<unsigned int BlockSize,
         unsigned int ItemsPerThread,
         unsigned int RadixBits,
         unsigned int CandidateBufferCoefficient,
         unsigned int ThreadCounterLimit,
         bool         SelectMin,
         bool         Adaptive,
         typename KeysInputIterator,
         typename KeysOutputIterator,
         typename ValuesInputIterator,
         typename ValuesOutputIterator,
         typename SizeIn,
         typename SizeOut,
         class OffsetIterator,
         typename Decomposer,
         bool UseThreadCounter  = true,
         bool UseNativeOperator = true,
         bool KillNegativeZeros = false,
         class BaseType = device_topk_air_impl<BlockSize,
                                                   ItemsPerThread,
                                                   RadixBits,
                                                   CandidateBufferCoefficient,
                                                   ThreadCounterLimit,
                                                   SelectMin,
                                                   Adaptive,
                                                   KeysInputIterator,
                                                   KeysOutputIterator,
                                                   ValuesInputIterator,
                                                   ValuesOutputIterator,
                                                   SizeIn,
                                                   SizeOut,
                                                   Decomposer,
                                                   UseThreadCounter,
                                                   UseNativeOperator,
                                                   KillNegativeZeros>>
struct device_segmented_topk_air_impl : BaseType
{
    using key_in_t =
        typename device_topk_air_helper::iterator_traits<KeysInputIterator>::value_type;
    using key_out_t =
        typename device_topk_air_helper::iterator_traits<KeysOutputIterator>::value_type;
    using value_in_t =
        typename device_topk_air_helper::iterator_traits<ValuesInputIterator>::value_type;
    using value_out_t =
        typename device_topk_air_helper::iterator_traits<ValuesOutputIterator>::value_type;

    static_assert(!std::is_same_v<key_in_t, empty_type>, "Invalid KeysInputIterator");
    static_assert(!std::is_same_v<key_out_t, empty_type>, "Invalid KeysOutputIterator");
    static_assert(std::is_same_v<key_in_t, key_out_t>,
                  "KeysInputIterator and KeysOutputIterator must have the same value_type");
    static_assert(std::is_same_v<value_in_t, value_out_t>,
                  "ValuesInputIterator and ValuesOutputIterator must have the same value_type");
    static_assert(rocprim::is_integral<SizeIn>::value, "SizeIn must be integral");
    static_assert(rocprim::is_integral<SizeOut>::value, "SizeOut must be integral");
    static_assert(
        sizeof(SizeIn) >= sizeof(int) && sizeof(SizeIn) <= sizeof(std::int64_t),
        "The SizeIn must be a integral type with size between 32 bits and 64 bits. This is because "
        "atomic operation does not support any smaller or larger integral types");

    static constexpr auto block_size          = BlockSize;
    static constexpr auto items_per_thread    = ItemsPerThread;
    static constexpr auto items_per_block     = block_size * items_per_thread;
    static constexpr auto bits_per_iteration  = RadixBits;
    static constexpr auto bits_last_iteration = (sizeof(key_in_t) * 8) % bits_per_iteration == 0
                                                    ? bits_per_iteration
                                                    : (sizeof(key_in_t) * 8) % bits_per_iteration;

    // Also know as `radix_size` in other algorithms
    static constexpr auto num_buckets                = 1u << RadixBits;
    static constexpr auto num_buckets_last_iteration = 1u << bits_last_iteration;
    static constexpr auto num_iterations = ceiling_div((sizeof(key_in_t) * 8), bits_per_iteration);

    static constexpr unsigned int bins_per_thread = ceiling_div(num_buckets, block_size);

    static constexpr bool output_value
        = !std::is_same_v<value_in_t, empty_type> || !std::is_same_v<value_out_t, empty_type>;

    static constexpr auto thread_counter_limit         = ThreadCounterLimit;
    static constexpr auto candidate_buffer_coefficient = CandidateBufferCoefficient;

    using key_codec = decltype(::rocprim::traits::get<key_in_t>().template radix_key_codec<true>());
    using digit_t
        = decltype(key_codec::template extract_digit<Decomposer>(key_in_t{}, 0, 0, Decomposer{}));
    using segments_size_t = unsigned int;

    // Used by thread counter
    using count_t = unsigned char;

    using common_size_t = std::common_type_t<SizeIn, SizeOut>;

    // Using functions from the regular topk
    using BaseType::equal_last_n_bits;
    using BaseType::less_last_n_bits;
    using BaseType::extract_digit_flip_xaxis;
    template<size_t HistogramSize>
    using histogram_t = typename BaseType::template histogram_t<HistogramSize>;

    // Scan over histogram, so use SizeOut
    using block_scan_t = block_scan<SizeOut, block_size>;

    struct storage_type
    {};

    // Initialize histogram bin counts to zeros
    template<unsigned int HistogramSize, unsigned int ActualSize>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE static void
    init_histogram(histogram_t<ActualSize> &histogram, const unsigned int thread_id)
    {
        static_assert(HistogramSize <= ActualSize,
                      "HistogramSize is larger than the size of input histogram");
        std::remove_cv_t<decltype(HistogramSize)> histo_offset = 0;

        // Strip threads for initializing
        ROCPRIM_UNROLL
        for(; histo_offset + block_size <= HistogramSize; histo_offset += block_size)
        {
            histogram[histo_offset + thread_id] = 0;
        }
        // Finish up with guarded initialization if necessary
        if((HistogramSize % block_size != 0) && (histo_offset + thread_id < HistogramSize))
        {
            histogram[histo_offset + thread_id] = 0;
        }
    }

    template<unsigned int Iteration, class SharedStorageType, class F>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE static void
    thread_histogram_and_filter_prev(
        SharedStorageType& storage,
        KeysInputIterator keys_input,
        KeysOutputIterator keys_output,
        ValuesInputIterator values_input,
        ValuesOutputIterator values_output,
        SizeOut K,
        Decomposer decomposer,
        const SizeIn index,
        F record_to_histogram_fn
    )
    {
        // TODO: the adaptive optimization needs to be added later
        const bool load_adaptive = false;
        [[maybe_unused]]
        const bool store_adaptive
            = false;

        [[maybe_unused]]
        SizeIn* in_idx_buf;
        [[maybe_unused]]
        SizeIn* out_idx_buf;

        std::conditional_t<Adaptive, SizeIn[items_per_thread], rocprim::empty_type> thread_out_buf;
        std::remove_cv_t<decltype(items_per_thread)> thread_out_buf_size = 0;

        const auto key = keys_input[index];

        auto write = [&]()
        {
            const auto segment_output_pos
                = ::rocprim::detail::atomic_add(&storage.output_pos, 1) + (K * block_id<0>());
            keys_output[segment_output_pos] = key;
            if constexpr(output_value)
            {
                values_output[segment_output_pos] = values_input[index];
            }
        };

        if constexpr(Iteration == 0) // First Iteration
        { // For first iteration, every thing from the input is input
            record_to_histogram_fn(BaseType::template  extract_digit_of_cur_iteration<Iteration>(key, decomposer));
        }
        else
        {
            const auto [category, candidate_digit]
                = BaseType::template identify_candidate<Iteration>(key,
                                                storage.chosen_bins,
                                                load_adaptive,
                                                decomposer);
            // Items which are in the previous be is the input of this iteration
            switch(category)
            {
                case BaseType::candidate_category::input:
                    record_to_histogram_fn(candidate_digit);
                    if constexpr(Adaptive)
                    {
                        if(store_adaptive)
                        {
                            thread_out_buf[thread_out_buf_size] = index;
                            ++thread_out_buf_size;
                        }
                    }
                    break;

                case BaseType::candidate_category::candidate:
                    write(); // Write this into output buffer
                    break;

                default: break;
            }
        }
    }

    template<unsigned int Iteration, class SharedStorageType>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE static void
    launch_thread_histogram_and_filter_prev(
        SharedStorageType& storage,
        KeysInputIterator keys_input,
        KeysOutputIterator keys_output,
        ValuesInputIterator values_input,
        ValuesOutputIterator values_output,
        SizeOut K,
        Decomposer decomposer,
        const SizeIn index
    )
    {
        if constexpr(UseThreadCounter
                     && items_per_thread
                            != 1 // When items_per_thread is 1 UseThreadCounter is useless
                     && (items_per_thread < ~(count_t{0})) // Ensure count_t is capable
                     && items_per_thread < thread_counter_limit // Ensure thread_counter is fast
        )
        {
            digit_t                                      thread_digit[items_per_thread];
            count_t                                      thread_counter[items_per_thread];
            std::remove_cv_t<decltype(items_per_thread)> thread_counter_size = 0;

            auto record_to_counter_fn = [&](digit_t digit)
            {
                if(thread_counter_size == 0)
                { // When thread_counter_size add digit directly to the first
                    thread_counter[0]   = 1;
                    thread_digit[0]     = digit;
                    thread_counter_size = 1;
                    return;
                }

                bool added = false;
                ROCPRIM_UNROLL
                for(decltype(thread_counter_size) i = 0; i < items_per_thread; ++i)
                {
                    if(i < thread_counter_size && thread_digit[i] == digit)
                    {
                        ++thread_counter[i];
                        added = true;
                        break;
                    }
                }

                if(!added)
                {
                    thread_counter[thread_counter_size] = 1;
                    thread_digit[thread_counter_size]   = digit;
                    ++thread_counter_size;
                }
            };
            thread_histogram_and_filter_prev<Iteration>(storage,
                                                        keys_input,
                                                        keys_output,
                                                        values_input,
                                                        values_output,
                                                        K,
                                                        decomposer,
                                                        index,
                                                        record_to_counter_fn);
            // Store counter into shared memory
            ROCPRIM_UNROLL
            for(decltype(thread_counter_size) i = 0; i < items_per_thread; ++i)
            {
                if(i < thread_counter_size)
                {
                    ::rocprim::detail::atomic_add(&storage.block_local_histogram[thread_digit[i]], thread_counter[i]);
                }
            }
        }
        else
        {
            auto record_to_histogram_fn
                = [&](auto digit) { ::rocprim::detail::atomic_add(&storage.block_local_histogram[digit], 1); };
            thread_histogram_and_filter_prev<Iteration>(storage,
                                                        keys_input,
                                                        keys_output,
                                                        values_input,
                                                        values_output,
                                                        K,
                                                        decomposer,
                                                        index,
                                                        record_to_histogram_fn);
        }
    }


    template<unsigned int Iteration, unsigned int HistogramSize, class SharedStorageType>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE static void
    choose_pivot_bin(
        SharedStorageType& storage,
        histogram_t<bins_per_thread> const& thread_bins,
        histogram_t<HistogramSize> const& block_local_histogram,
        SizeIn N,
        SizeOut K,
        unsigned int thread_id)
    {
        ROCPRIM_UNROLL
        for(std::remove_cv_t<decltype(bins_per_thread)> i = 0; i < bins_per_thread; ++i)
        {
            const auto global_i = i + (thread_id * bins_per_thread);
            if(global_i >= HistogramSize)
            {
                break;
            }

            // A pivot be should satisfy (cur >= K && prev < K)
            // The code is writing like this because I don't want to load data from shared memory
            // for each item, I want to load prev only when needed.
            // cur == block_local_histogram[global_i], using thread_bins because it's faster
            const auto cur = thread_bins[i];
            if(cur < static_cast<decltype(cur)>(K))
            {
                continue;
            }

            const auto prev = global_i == 0 ? 0 : block_local_histogram[global_i - 1];
            if(prev < static_cast<decltype(prev)>(K))
            {
                // Bin that contains pivot is found
                K         = K - prev;
                N         = cur - prev;
                storage.K = K;
                storage.N = N;
                storage.chosen_bins.template set<Iteration>(global_i);
                storage.stopped_at = static_cast<common_size_t>(K) == static_cast<common_size_t>(N)
                                         ? Iteration
                                         : num_iterations;
                break;
            }
        }
    }

    // TODO: This function is possible to be replaced by block_reduce or segmented_reduce
    // But I tried to use segmented_reduce with counting_iteration, it didn't work, so this
    // needs to be investigated further.
    template<class RangeType, class UnaryFunc>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE static void
    block_for_in_range(RangeType range, UnaryFunc&& fn)
    {
        static_assert(rocprim::is_integral<RangeType>::value, "RangeType must be integral");
        using common_t           = std::common_type_t<decltype(items_per_block), RangeType>;
        const auto thread_offset = block_thread_id<0>() * items_per_thread;

        if(static_cast<common_t>(items_per_block) >= static_cast<common_t>(range))
        { // Block is larger than the range

            ROCPRIM_UNROLL
            for(std::remove_const_t<decltype(items_per_thread)> i = 0; i < items_per_thread; ++i)
            {
                const auto index = i + thread_offset;
                if(static_cast<common_t>(index) < static_cast<common_t>(range))
                {
                    fn(index);
                }
            }
        }
        else
        { // Block is smaller than the range
            for(std::remove_const_t<decltype(items_per_thread)> i = 0; i < items_per_thread; ++i)
            {
                auto index = i + thread_offset;
                while(static_cast<common_t>(index) < static_cast<common_t>(range))
                {
                    fn(index);
                    index += items_per_block;
                }
            }
        }
    }

    template<class SharedStorageType>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    static void
    last_filter(
        SharedStorageType&   storage,
        KeysInputIterator    keys_input,
        KeysOutputIterator   keys_output,
        ValuesInputIterator  values_input,
        ValuesOutputIterator values_output,
        OffsetIterator       begin_offsets,
        OffsetIterator       end_offsets,
        const SizeOut        K,
        const Decomposer     decomposer
    )
    {
        if(storage.output_pos >= K)
        {
            return; // Early stop
        }

        const auto stopped_iteration = storage.stopped_at;
        const auto chosen_bins       = storage.chosen_bins;
        const auto cur_bits
            = stopped_iteration == (num_iterations - 1) ? bits_last_iteration : bits_per_iteration;
        const auto stopped        = num_iterations != stopped_iteration;
        const auto last_iteration = stopped ? stopped_iteration : num_iterations - 1;
        [[maybe_unused]]
        const auto cur_iteration
            = last_iteration + 1;

        // TODO: the adaptive optimization needs to be added later
        const bool load_adaptive = false;
        [[maybe_unused]]
        const bool store_adaptive
            = false;

        [[maybe_unused]]
        SizeIn* in_idx_buf;
        [[maybe_unused]]
        SizeIn* out_idx_buf;

        const auto last_chosed_bin = chosen_bins.get(last_iteration);

        const unsigned int segment_id   = block_id<0>();
        const auto         begin_offset = begin_offsets[segment_id];
        const auto         end_offset   = end_offsets[segment_id];

        auto reduce_op = [&](auto block_index) -> auto
        {
            const auto index = block_index + begin_offset;
            const auto key   = keys_input[index];

            auto write = [&]()
            {
                const auto segment_output_pos
                    = ::rocprim::detail::atomic_add(&storage.output_pos, 1) + (K * segment_id);
                keys_output[segment_output_pos] = key;
                if constexpr(output_value)
                {
                    values_output[segment_output_pos] = values_input[index];
                }
            };

            // Extract all digits
            digit_t digits[num_iterations];
            bool    is_candidate_in_prev_iteration = true;
            // It's actually faster to just directly extract all digits, instead of using runtime variable
            // last_iteration to determine how many iterations needs to be loaded
            rocprim::detail::constexpr_for_lt<0, num_iterations, 1>(
                [&](const auto i)
                { digits[i] = BaseType::template  extract_digit_of_cur_iteration<i>(key, decomposer); });

            // Only check the iteration before last iteration
            if(last_iteration == 0)
            {
                is_candidate_in_prev_iteration = true;
            }
            else if(load_adaptive
               && !equal_last_n_bits(storage.chosen_bins.get(last_iteration - 1),
                                     digits[last_iteration - 1],
                                     bits_per_iteration))
            {
                is_candidate_in_prev_iteration = false;
            }
            else
            {
                // Check match previous iterations
                ROCPRIM_UNROLL
                for(std::remove_cv_t<decltype(num_iterations)> j = 0; j < num_iterations; ++j)
                {
                    if(j < last_iteration
                       && !equal_last_n_bits(chosen_bins.get(j), digits[j], bits_per_iteration))
                    {
                        is_candidate_in_prev_iteration = false;
                        break;
                    }
                }
            }
            if(is_candidate_in_prev_iteration
               && less_last_n_bits(digits[last_iteration], last_chosed_bin, cur_bits))
            { // Is candidate of last iteration
                // This can be also done with thread counter, but in practice, this is super slow
                // because there are a lot of threads even do not have a candidate to store, but if
                // we use thread counter for it, we need to create a buffer to store the counter, which
                // increases the use of register, so here we use atomicAdd once we have a candidate to
                // output
                write();
            }
            else if(is_candidate_in_prev_iteration && stopped
                    && equal_last_n_bits(digits[last_iteration], last_chosed_bin, cur_bits))
            { // If stopped, then we don't need to count last_output_pos
                // Stopped means that, K = N, so all items in previous pivot
                // bin should be stored into output.
                write();
            }
            else if(is_candidate_in_prev_iteration && !stopped
                    && equal_last_n_bits(digits[last_iteration], last_chosed_bin, cur_bits)
                    && ::rocprim::detail::atomic_add(&storage.last_output_pos, 1) < storage.K)
            { // If not stopped, we need to check how many items in the pivot bin should we
                // Write to the output
                write();
            }

            return index;
        };
        block_for_in_range(end_offset - begin_offset, reduce_op);
    }

    ROCPRIM_KERNEL ROCPRIM_FORCE_INLINE
    ROCPRIM_LAUNCH_BOUNDS(ROCPRIM_DEFAULT_MAX_BLOCK_SIZE)
    static void
    large_segments_kernel(
        [[maybe_unused]] storage_type* p_global_storage,
        KeysInputIterator    keys_input,
        KeysOutputIterator   keys_output,
        ValuesInputIterator  values_input,
        ValuesOutputIterator values_output,
        [[maybe_unused]] segments_size_t segments,
        OffsetIterator       begin_offsets,
        OffsetIterator       end_offsets,
        [[maybe_unused]] const SizeIn size,
        const SizeOut        K,
        const Decomposer     decomposer)
    {
        const unsigned int segment_id = block_id<0>();
        const unsigned int thread_id  = block_thread_id<0>();

        const auto begin_offset = begin_offsets[segment_id];
        const auto end_offset   = end_offsets[segment_id];

        // Empty segment
        if(end_offset <= begin_offset)
        {
            if(K > 0)
            {
                // TODO: Rise an error here
            }
            return;
        }
        const auto num_segment_items = end_offset - begin_offset;

        ROCPRIM_SHARED_MEMORY struct
        {
            typename block_scan_t::storage_type scan;

            SizeOut      output_pos; // Initialize at Iteration 0 -> init value 0
            SizeOut      last_output_pos; // Initialize at Iteration 0 -> init value 0
            typename BaseType::digits_array chosen_bins; // Auto initialized
            unsigned int stopped_at; // Initialize at Iteration 0 -> init value 0

            histogram_t<num_buckets>
                    block_local_histogram; // Initialize in each Iteration -> init value 0
            SizeIn  N;
            SizeOut K;
        } storage;

        ::rocprim::detail::constexpr_for_lt<0, num_iterations, 1>(
            [&]([[maybe_unused]]
                auto Iteration)
            {
                // Load problem size and init local_histogram
                SizeIn  N_this_iteration;
                SizeOut K_this_iteration;

                if constexpr(Iteration == 0) // First iteration
                {
                    N_this_iteration = num_segment_items;
                    K_this_iteration = K;

                    // Initialize variables in shared_memory
                    storage.output_pos      = 0;
                    storage.last_output_pos = 0;
                    storage.stopped_at      = 0;
                    storage.chosen_bins.init();
                }
                else
                {
                    N_this_iteration = storage.N;
                    K_this_iteration = storage.K;

                    // Return earlier
                    if(static_cast<common_size_t>(K_this_iteration)
                       == static_cast<common_size_t>(N_this_iteration))
                    {
                        return; // All threads return no divergence
                    }
                }

                // The size of valid bins in the histogram or current iteration
                constexpr auto histogram_size
                    = Iteration == (num_iterations - 1) ? num_buckets_last_iteration : num_buckets;

                init_histogram<histogram_size>(storage.block_local_histogram, thread_id);
                ::rocprim::syncthreads();

                auto reduce_op = [&](auto block_index)
                {
                    launch_thread_histogram_and_filter_prev<Iteration>(storage,
                                                                       keys_input,
                                                                       keys_output,
                                                                       values_input,
                                                                       values_output,
                                                                       K,
                                                                       decomposer,
                                                                       block_index + begin_offset);
                };
                block_for_in_range(end_offset - begin_offset, reduce_op);

                // Make sure block_local_histogram write is finished
                ::rocprim::syncthreads();
                histogram_t<bins_per_thread> thread_bins;

                // Load data into register
                block_load_direct_blocked(thread_id,
                                          storage.block_local_histogram,
                                          thread_bins,
                                          histogram_size,
                                          BaseType::template  extract_digit_of_cur_iteration<Iteration>(
                                              key_codec::get_out_of_bounds_key(decomposer),
                                              decomposer));

                // Block scan
                block_scan_t{}.inclusive_scan(thread_bins,
                                              thread_bins,
                                              storage.scan,
                                              ::rocprim::plus<SizeOut>{});
                // Store data into shared memory
                block_store_direct_blocked(thread_id,
                                           storage.block_local_histogram,
                                           thread_bins,
                                           histogram_size);

                // Need to sync threads, because we will read storage.block_local_histogram[global_i - 1]
                // which is set by thread at index of (thread_id -1)
                ::rocprim::syncthreads();

                // Chose the bin which contains the pivot
                choose_pivot_bin<Iteration>(storage,
                                           thread_bins,
                                           storage.block_local_histogram,
                                           N_this_iteration,
                                           K_this_iteration,
                                           thread_id);
                ::rocprim::syncthreads();
            });

        last_filter(storage,
                    keys_input,
                    keys_output,
                    values_input,
                    values_output,
                    begin_offsets,
                    end_offsets,
                    K,
                    decomposer);
    }

    constexpr hipError_t operator()(void*                temporary_storage,
                                    size_t&              storage_size,
                                    KeysInputIterator    keys_input,
                                    KeysOutputIterator   keys_output,
                                    ValuesInputIterator  values_input,
                                    ValuesOutputIterator values_output,
                                    const SizeIn         size,
                                    const SizeOut        K,
                                    segments_size_t      segments,
                                    OffsetIterator       begin_offsets,
                                    OffsetIterator       end_offsets,
                                    const Decomposer     decomposer,
                                    const hipStream_t    stream,
                                    const bool           debug_synchronous) const
    {
        storage_type* p_global_storage;
        ROCPRIM_RETURN_ON_ERROR(detail::temp_storage::partition(
            temporary_storage,
            storage_size,
            temp_storage::make_linear_partition(
                temp_storage::ptr_aligned_array(&p_global_storage, sizeof(storage_type)))));

        if(temporary_storage == nullptr)
        {
            return hipSuccess;
        }

        if(size == 0 || K == 0)
        { // Reject, return directly
            return hipSuccess;
        }

        std::chrono::steady_clock::time_point start;
        if(debug_synchronous)
        {
            start = std::chrono::steady_clock::now();
        }
        large_segments_kernel<<<dim3(segments), dim3(block_size), 0, stream>>>(p_global_storage,
                                                                               keys_input,
                                                                               keys_output,
                                                                               values_input,
                                                                               values_output,
                                                                               segments,
                                                                               begin_offsets,
                                                                               end_offsets,
                                                                               size,
                                                                               K,
                                                                               decomposer);
        return hipSuccess;
    }
};

template<class Config,
         bool SelectMin,
         bool Adaptive,
         typename KeysInputIterator,
         typename KeysOutputIterator,
         typename ValuesInputIterator,
         typename ValuesOutputIterator,
         typename SizeIn,
         typename SizeOut,
         class OffsetIterator,
         typename Decomposer>
struct device_segmented_topk_air_impl_invoker
{
private:
    template<unsigned int BlockSize,
             unsigned int ItemsPerThread,
             unsigned int RadixBits,
             unsigned int CandidateBufferCoefficient,
             unsigned int ThreadCounterLimit,
             class ActualSizeIn>
    using simplified_type = device_segmented_topk_air_impl<BlockSize,
                                                           ItemsPerThread,
                                                           RadixBits,
                                                           CandidateBufferCoefficient,
                                                           ThreadCounterLimit,
                                                           SelectMin,
                                                           Adaptive,
                                                           KeysInputIterator,
                                                           KeysOutputIterator,
                                                           ValuesInputIterator,
                                                           ValuesOutputIterator,
                                                           ActualSizeIn,
                                                           SizeOut,
                                                           OffsetIterator,
                                                           Decomposer>;

    // If `DecaySizeIn` is true, launch topk with a decayed SizeIn according
    // to the actual runtime input size. Otherwise, launch topk with the original
    // SizeIn type.
    template<unsigned int BlockSize,
             unsigned int ItemsPerThread,
             unsigned int RadixBits,
             unsigned int CandidateBufferCoefficient,
             unsigned int ThreadCounterLimit,
             bool         DecaySizeIn = true,
             class Args>
    static inline constexpr hipError_t invoke_impl(const SizeIn& size, Args&& args)
    {
        if constexpr(DecaySizeIn)
        {
            if(device_topk_air_helper::in_range<std::uint32_t>(size))
            {
                return std::apply(simplified_type<BlockSize,
                                                  ItemsPerThread,
                                                  RadixBits,
                                                  CandidateBufferCoefficient,
                                                  ThreadCounterLimit,
                                                  std::uint32_t>{},
                                  args);
            }
            else
            {
                return std::apply(simplified_type<BlockSize,
                                                  ItemsPerThread,
                                                  RadixBits,
                                                  CandidateBufferCoefficient,
                                                  ThreadCounterLimit,
                                                  std::uint64_t>{},
                                  args);
            }
        }
        else
        {
            return std::apply(simplified_type<BlockSize,
                                              ItemsPerThread,
                                              RadixBits,
                                              CandidateBufferCoefficient,
                                              ThreadCounterLimit,
                                              SizeIn>{},
                              args);
        }
    }

public:
    template<class Args>
    static inline constexpr hipError_t invoke(const SizeIn& size, Args&& args)
    {
        using key_in_t =
            typename device_topk_air_helper::iterator_traits<KeysInputIterator>::value_type;
        using value_in_t =
            typename device_topk_air_helper::iterator_traits<ValuesInputIterator>::value_type;

        using Selector     = segmented_topk_air_config_selector<key_in_t, value_in_t, SizeIn>;
        using Targets      = typename Selector::targets;
        const auto& stream = std::get<hipStream_t const&>(args);
        target_arch target_arch{};
        ROCPRIM_RETURN_ON_ERROR(host_target_arch(stream, target_arch));
        gpu target_gpu{};
        ROCPRIM_RETURN_ON_ERROR(host_target_gpu(stream, target_gpu));

        const auto current_target = target{target_arch, target_gpu};
        const auto target_config  = most_common_config<Targets>(current_target);

        hipError_t ret = hipSuccess;
        if constexpr(std::is_same_v<Config, rocprim::default_config>)
        {
            Targets::for_each(
                [&](auto candidate)
                {
                    if(target{candidate} == target_config)
                    {
                        constexpr auto params = Selector{candidate}.params;
                        // If one day we upgraded to c++20, then we can move params into template
                        ret = invoke_impl<params.kernel_config.block_size,
                                          params.kernel_config.items_per_thread,
                                          params.radix_bits,
                                          params.candidate_buffer_coefficient,
                                          params.thread_counter_limit>(size, args);
                    }
                });
        }
        else
        {
            constexpr auto params = Config{};
            // If one day we upgraded to c++20, then we can move params into template
            ret = invoke_impl<params.kernel_config.block_size,
                              params.kernel_config.items_per_thread,
                              params.radix_bits,
                              params.candidate_buffer_coefficient,
                              params.thread_counter_limit>(size, args);
        }

        return ret;
    }

    static inline constexpr auto get_params(segmented_topk_air_config_params& params,
                                            hipStream_t                       stream)
    {
        using key_in_t =
            typename device_topk_air_helper::iterator_traits<KeysInputIterator>::value_type;
        using value_in_t =
            typename device_topk_air_helper::iterator_traits<ValuesInputIterator>::value_type;

        using Selector = segmented_topk_air_config_selector<key_in_t, value_in_t, SizeIn>;
        using Targets  = typename Selector::targets;

        target_arch target_arch{};
        ROCPRIM_RETURN_ON_ERROR(host_target_arch(stream, target_arch));
        gpu target_gpu{};
        ROCPRIM_RETURN_ON_ERROR(host_target_gpu(stream, target_gpu));

        const auto current_target = target{target_arch, target_gpu};
        const auto target_config  = most_common_config<Targets>(current_target);

        Targets::for_each(
            [&](auto candidate)
            {
                if(target{candidate} == target_config)
                {
                    using ArchConfig
                        = rocprim::detail::target_config<Config, Selector, decltype(candidate)>;
                    params = ArchConfig::params;
                }
            });
        return hipSuccess;
    }
};

template<typename Config = rocprim::default_config,
         bool         SelectMin = true,
         bool         Adaptive = false,
         typename KeysInputIterator,
         typename KeysOutputIterator,
         typename ValuesInputIterator,
         typename ValuesOutputIterator,
         typename SizeIn,
         typename SizeOut,
         class OffsetIterator,
         typename Decomposer = ::rocprim::identity_decomposer>
ROCPRIM_FORCE_INLINE hipError_t device_segmented_topk_air(void* temporary_storage,
                                                          size_t&              storage_size,
                                                          KeysInputIterator    keys_input,
                                                          KeysOutputIterator   keys_output,
                                                          ValuesInputIterator  values_input,
                                                          ValuesOutputIterator values_output,
                                                          const SizeIn         size,
                                                          const SizeOut        K,
                                                          unsigned int         segments,
                                                          OffsetIterator       begin_offsets,
                                                          OffsetIterator       end_offsets,
                                                          const Decomposer     decomposer        = {},
                                                          const hipStream_t    stream            = 0,
                                                          const bool           debug_synchronous = false)
{
    return device_segmented_topk_air_impl_invoker<Config,
                                                  SelectMin,
                                                  Adaptive,
                                                  KeysInputIterator,
                                                  KeysOutputIterator,
                                                  ValuesInputIterator,
                                                  ValuesOutputIterator,
                                                  SizeIn,
                                                  SizeOut,
                                                  OffsetIterator,
                                                  Decomposer>::invoke(size,
                                                                      std::tie(temporary_storage,
                                                                               storage_size,
                                                                               keys_input,
                                                                               keys_output,
                                                                               values_input,
                                                                               values_output,
                                                                               size,
                                                                               K,
                                                                               segments,
                                                                               begin_offsets,
                                                                               end_offsets,
                                                                               decomposer,
                                                                               stream,
                                                                               debug_synchronous));
}

} // namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DETAIL_DEVICE_SEGMENTED_TOPK_AIR_HPP_
