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

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{
// TODO: This algorithm can be optimized by using the same logic of device_segmented_radix_sort
// Use Partitioner to separage small and large segments and run different kernel for different
// segments

// TODO: can reuse the code from:
// projects/rocprim/rocprim/include/rocprim/device/detail/device_topk_air.hpp
namespace device_segmented_topk_air_helper
{

template<class T>
struct iterator_traits : public std::iterator_traits<T>
{};

template<>
struct iterator_traits<std::nullptr_t>
{
    using value_type = empty_type;
};

// TODO: can reuse the code from:
// projects/rocprim/rocprim/include/rocprim/device/detail/device_topk_air.hpp
template<class T>
struct matched_int
{
    using type = std::conditional_t<
        sizeof(T) == 1,
        uint8_t,
        std::conditional_t<
            sizeof(T) == 2,
            uint16_t,
            std::conditional_t<
                sizeof(T) == 4,
                uint32_t,
                std::conditional_t<sizeof(T) == 8,
                                   uint64_t,
                                   std::conditional_t<sizeof(T) == 16, rocprim::int128_t, void>>>>>;
};

// TODO: can be reused
template<class T, class = void>
constexpr bool has_operator_left_shift_v = false;

template<class T>
constexpr bool has_operator_left_shift_v<T, std::void_t<decltype(std::declval<T&>() << sizeof(T))>>
    = true;

} // namespace device_segmented_topk_air_helper

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
         bool KillNegativeZeros = false>
// TODO: make this a derived class of device_topk_air_impl, or add device_topk_air_impl to be a sub type
struct device_segmented_topk_air_impl
{
    using key_in_t =
        typename device_segmented_topk_air_helper::iterator_traits<KeysInputIterator>::value_type;
    using key_out_t =
        typename device_segmented_topk_air_helper::iterator_traits<KeysOutputIterator>::value_type;
    using value_in_t =
        typename device_segmented_topk_air_helper::iterator_traits<ValuesInputIterator>::value_type;
    using value_out_t = typename device_segmented_topk_air_helper::iterator_traits<
        ValuesOutputIterator>::value_type;

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

    // Max value in each is up to numeric_limits<SizeOut>::max(), so, SizeOut is used here
    template<size_t HistogramSize>
    using histogram_t = SizeOut[HistogramSize];
    // Scan over histogram, so use SizeOut
    using block_scan_t = block_scan<SizeOut, block_size>;

    // TODO: this can also be reused
    struct digits_array
    {
    private:
        using int_key_t = typename device_segmented_topk_air_helper::matched_int<key_in_t>::type;
        static constexpr auto bits_total = sizeof(key_in_t) * 8;
        int_key_t             data;

        // Runtime mask, and it will be compile time function when Iteration is constexpr
        static constexpr ROCPRIM_FORCE_INLINE auto mask(decltype(bits_per_iteration) NumBits)
        {
            return (int_key_t{1} << NumBits) - 1;
        }

    public:
        ROCPRIM_HOST_DEVICE ROCPRIM_FORCE_INLINE void init()
        {
            data = 0;
        }

        // Runtime get funtion, and it will be compile time function when Iteration is constexpr
        constexpr ROCPRIM_FORCE_INLINE digit_t get(unsigned int Iteration) const
        {
            return static_cast<digit_t>((data >> (Iteration * bits_per_iteration))
                                        & mask(Iteration == (num_iterations - 1)
                                                   ? bits_last_iteration
                                                   : bits_per_iteration));
        }

        // Compile time set function
        template<unsigned int Iteration>
        constexpr ROCPRIM_FORCE_INLINE void set(digit_t digit)
        {
            data |= (digit
                     & (Iteration == (num_iterations - 1) ? mask(bits_last_iteration)
                                                          : mask(bits_per_iteration)))
                    << (Iteration * bits_per_iteration);
        }
    };

    struct storage_type
    {};

    // TODO: can reuse from device_topk_air
    enum class candidate_category
    {
        // Item is the input of this iteration
        input,
        // Item was the cadidate identified in the last iteration
        candidate,
        // Item is neither the input nor the candidate
        discard
    };

    // TODO: can be used
    enum class flip_strategy
    {
        // Does nothing, will call extract_digit directly
        no_flip,
        // Make input type unsigned, and move all values to fit unsigned type
        input_flip,
        // Flip only two’s complement or extracted digit
        output_flip
    };

    // TODO: can reuse this function from device_topk_air
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE 
    static constexpr bool equal_last_n_bits(digit_t const& a, digit_t const& b, decltype(bits_per_iteration) n)
    {
        if constexpr(UseNativeOperator)
        {
            return a == b;
        }
        else
        {
            if(n == 0)
            {
                return true;
            }
            else if(n >= sizeof(digit_t) * 8)
            {
                return a == b;
            }
            else
            {
                return (a & ((static_cast<digit_t>(1) << n) - 1))
                       == (b & ((static_cast<digit_t>(1) << n) - 1));
            }
        }
    }

    // TODO: can reuse this function from device_topk_air
    // In the implementaion of function `extract_digit`, we are confident that unrelated bits are zeros
    // So we can directly use the native operator
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE 
    static constexpr bool less_last_n_bits(digit_t const&a, digit_t const&b, decltype(bits_per_iteration) n)
    {
        if constexpr(UseNativeOperator)
        {
            return a < b;
        }
        else
        {
            if(n == 0)
            {
                return false;
            }
            else if(n >= sizeof(digit_t) * 8)
            {
                return a < b;
            }
            else
            {
                return (a & ((static_cast<digit_t>(1) << n) - 1))
                       < (b & ((static_cast<digit_t>(1) << n) - 1));
            }
        }
    }

    // TODO: can reuse this function from device_topk_air
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

    // TODO: can be reused
    template<flip_strategy FlipStrategy, class KeyCodec>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    static auto extract_digit_flip_xaxis(key_in_t key, unsigned int start, unsigned int length, Decomposer decomposer)
    {
        static_assert(!(rocprim::is_floating_point<key_in_t>::value
                        && FlipStrategy == flip_strategy::input_flip),
                      "For floating point types, only input_flip is not supported");

        if constexpr(FlipStrategy == flip_strategy::no_flip)
        {
            return KeyCodec::template extract_digit<Decomposer>(
                key,
                start, // Start bit of the sequence of bits to extract
                length, // How many bits to extract
                decomposer);
        }
        else if constexpr(FlipStrategy == flip_strategy::input_flip)
        {
            using unsigned_t              = typename rocprim::make_unsigned<key_in_t>::type;
            constexpr auto   half_max     = ((~unsigned_t{0}) / 2) + 1;
            const unsigned_t unsigned_key = key >= 0 ? static_cast<unsigned_t>(key) + half_max
                                                     : static_cast<unsigned_t>(key + half_max);
            return KeyCodec::template extract_digit<Decomposer>(
                unsigned_key,
                start, // Start bit of the sequence of bits to extract
                length, // How many bits to extract
                decomposer);
        }
        else if constexpr(FlipStrategy == flip_strategy::output_flip)
        {
            if constexpr(rocprim::is_integral<key_in_t>::value
                         && device_segmented_topk_air_helper::has_operator_left_shift_v<key_in_t>)
            { // Builtin integral types (including rocprim::int128_t and rocprim::uint128_t)
                return KeyCodec::template extract_digit<Decomposer>(
                    key ^ (key_in_t{1} << (sizeof(key_in_t) * 8 - 1)), // Flip only two’s complement
                    start, // Start bit of the sequence of bits to extract
                    length, // How many bits to extract
                    decomposer);
            }
            else if constexpr(rocprim::is_integral<key_in_t>::value
                              && !device_segmented_topk_air_helper::has_operator_left_shift_v<
                                  key_in_t>)
            { // Custom types may not support `operator<<`, so they are `bit_cast` to integral types instead.
                using matched_int_t =
                    typename device_segmented_topk_air_helper::matched_int<key_in_t>::type;
                static_assert(!std::is_same<matched_int_t, void>::value,
                              "Input type not supported");
                static_assert(sizeof(key_in_t) == sizeof(matched_int_t),
                              "Size of mathed_int_t is not the same as key_in_t");
                auto bits = traits::radix_key_codec::bit_cast<matched_int_t>(key);
                bits ^= (matched_int_t{1} << (sizeof(key_in_t) * 8 - 1));
                // Cast back when passing bits into extract_digit, in order to let extract_digit know that this is a floating point type
                return KeyCodec::template extract_digit<Decomposer>(
                    traits::radix_key_codec::bit_cast<key_in_t>(bits),
                    start, // Start bit of the sequence of bits to extract
                    length, // How many bits to extract
                    decomposer);
            }
            else if constexpr(rocprim::is_floating_point<key_in_t>::value)
            { // Floating point types
                using matched_int_t =
                    typename device_segmented_topk_air_helper::matched_int<key_in_t>::type;
                static_assert(!std::is_same<matched_int_t, void>::value,
                              "Input type not supported");
                static_assert(sizeof(key_in_t) == sizeof(matched_int_t),
                              "Size of mathed_int_t is not the same as key_in_t");
                // Might have undefined behavior, kill negative zeros
                if constexpr(KillNegativeZeros)
                {
                    key = key == key_in_t{-0.0} ? key_in_t{+0.0} : key;
                }
                // Cast to integral type, so we can flip the two’s complement
                const auto bits = traits::radix_key_codec::bit_cast<matched_int_t>(key);
                constexpr matched_int_t mask = matched_int_t{1} << (sizeof(key_in_t) * 8 - 1);
                // For negative values, flip the whole number
                // For positive values, flip only two’s complement
                // Cast back when passing bits into extract_digit, in order to let extract_digit know that this is a floating point type
                return KeyCodec::template extract_digit<Decomposer>(
                    traits::radix_key_codec::bit_cast<key_in_t, matched_int_t>(
                        bits & mask ? ~bits : bits ^ mask),
                    start, // Start bit of the sequence of bits to extract
                    length, // How many bits to extract
                    decomposer);
            }
            else
            {
                static_assert(
                    false,
                    "key_in_t must be either rocprim::floating_point or rocprim::integral. "
                    "If you are using custom types, please specialize "
                    "rocprim::traits::define<your_type> to implement recognizable traits.");
            }
        }
        else
        {
            static_assert(false, "flip strategy is not supported");
        }
    }

    // TODO: Might be able to reuse if comfirmed that key_codec::decode does exact same thing as
    // the flip mechanism
    template<unsigned int Iteration>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE static digit_t
    extract_digit_of_cur_iteration(key_in_t const&key, Decomposer decomposer)
    {
        constexpr auto bits_total = sizeof(key_in_t) * 8;
        constexpr auto cur_bits
            = Iteration == (num_iterations - 1) ? bits_last_iteration : bits_per_iteration;
        constexpr auto start_bits
            = Iteration == (num_iterations - 1)
                  ? 0
                  : bits_total - bits_per_iteration - (Iteration * bits_per_iteration);
        constexpr auto histogram_size
            = Iteration == (num_iterations - 1) ? num_buckets_last_iteration : num_buckets;

        digit_t digit;
        if constexpr(rocprim::is_integral<key_in_t>::value && rocprim::is_signed<key_in_t>::value)
        {
            // TODO: Can also use output_flip or input_flip, need to see which is generally faster
            // need to run some benchmarks to see which is faster
            digit = extract_digit_flip_xaxis<flip_strategy::output_flip, key_codec>(
                key,
                start_bits, // Start bit of the sequence of bits to extract
                cur_bits, // How many bits to extract
                decomposer);
        }
        else if constexpr(rocprim::is_integral<key_in_t>::value
                          && rocprim::is_unsigned<key_in_t>::value)
        {
            digit = extract_digit_flip_xaxis<flip_strategy::no_flip, key_codec>(
                key,
                start_bits, // Start bit of the sequence of bits to extract
                cur_bits, // How many bits to extract
                decomposer);
        }
        else if constexpr(rocprim::is_floating_point<key_in_t>::value)
        {
            digit = extract_digit_flip_xaxis<flip_strategy::output_flip, key_codec>(
                key,
                start_bits, // Start bit of the sequence of bits to extract
                cur_bits, // How many bits to extract
                decomposer);
        }
        else
        {
            // In this else branch, key_in_t must be custom types
            static_assert(
                false,
                "please use ::rocprim::traits::define to specify what data format is key_in_t.");
        }

        if constexpr(SelectMin)
        {
            return digit;
        }
        else
        {
            return static_cast<digit_t>(histogram_size - digit - 1);
        }
    }

    // TODO: can reuse this function
    template<unsigned int Iteration>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE static std::tuple<candidate_category, digit_t>
    identify_candidate(key_in_t const&key, digits_array const&chosen_bins, bool load_adaptive, Decomposer decomposer)
    {
        static_assert(Iteration != 0, "This function can not be used for first iteration");

        // Check if this item was in the previous-previous bin
        bool was_in_prev_prev_bin = true;
        if(load_adaptive)
        {
            if constexpr(Iteration >= 2)
            {
                // Only check the iteration before last iteration
                if(!equal_last_n_bits(
                       chosen_bins.get(Iteration - 2),
                       extract_digit_of_cur_iteration<Iteration - 2>(key, decomposer),
                       bits_per_iteration))
                {
                    was_in_prev_prev_bin = false;
                }
            }
        }
        else
        {
            rocprim::detail::constexpr_for_lt<0, Iteration - 1, 1>(
                [&](const auto i)
                {
                    if(was_in_prev_prev_bin
                       && !equal_last_n_bits(chosen_bins.get(i),
                                             extract_digit_of_cur_iteration<i>(key, decomposer),
                                             bits_per_iteration))
                    {
                        was_in_prev_prev_bin = false;
                    }
                });
        }

        if(!was_in_prev_prev_bin)
        {
            return {candidate_category::discard, {}};
        }
        const auto last_digit = extract_digit_of_cur_iteration<Iteration - 1>(key, decomposer);
        // Iteration - 1 cannot be the last iteration, so we use bits_per_iteration for them
        if(equal_last_n_bits(last_digit, chosen_bins.get(Iteration - 1), bits_per_iteration))
        {
            // This key is the input
            return {candidate_category::input,
                    extract_digit_of_cur_iteration<Iteration>(key, decomposer)};
        }
        else if(less_last_n_bits(last_digit, chosen_bins.get(Iteration - 1), bits_per_iteration))
        {
            // bits are order when being extracted, so no matter selectMax or selectMin, we select the digit which is smaller
            return {candidate_category::candidate, {}};
        }
        else
        {
            return {candidate_category::discard, {}};
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
                = ::atomicAdd(&storage.output_pos, 1) + (K * block_id<0>());
            keys_output[segment_output_pos] = key;
            if constexpr(output_value)
            {
                values_output[segment_output_pos] = values_input[index];
            }
        };

        if constexpr(Iteration == 0) // First Iteration
        { // For first iteration, every thing from the input is input
            record_to_histogram_fn(extract_digit_of_cur_iteration<Iteration>(key, decomposer));
        }
        else
        {
            const auto [category, candidate_digit]
                = identify_candidate<Iteration>(key,
                                                storage.chosen_bins,
                                                load_adaptive,
                                                decomposer);
            // Items which are in the previous be is the input of this iteration
            switch(category)
            {
                case candidate_category::input:
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

                case candidate_category::candidate:
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
                    ::atomicAdd(&storage.block_local_histogram[thread_digit[i]], thread_counter[i]);
                }
            }
        }
        else
        {
            auto record_to_histogram_fn
                = [&](auto digit) { ::atomicAdd(&storage.block_local_histogram[digit], 1); };
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

    // TODO: reuse this function from device_topk_air
    template<unsigned int Iteration, unsigned int HistogramSize, class SharedStorageType>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE static void 
    chose_pivot_bin(
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
                    = ::atomicAdd(&storage.output_pos, 1) + (K * segment_id);
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
            // last_iteration to determin how many iterations needs to be loaded
            rocprim::detail::constexpr_for_lt<0, num_iterations, 1>(
                [&](const auto i)
                { digits[i] = extract_digit_of_cur_iteration<i>(key, decomposer); });

            // Only check the iteration before last iteration
            if(load_adaptive
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
                // becasue there are a lot of threads even do not have a candidate to store, but if
                // we use thread counter for it, we need to create a buffer to store the counter, which
                // increases the use of register, so here we use atomicAdd once we have a cadidate to
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
                    && ::atomicAdd(&storage.last_output_pos, 1) < storage.K)
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
            digits_array chosen_bins; // Auto initialized
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
                { // If K_this_iteration == N_this_iteration, kernel will be reject from the host
                    // so here we don't need to check and return like other iterations
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
                                          extract_digit_of_cur_iteration<Iteration>(
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
                chose_pivot_bin<Iteration>(storage,
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

    // TODO: can reuse this fucntion from the regular topk implementation
    template<class SizeType>
    static inline constexpr auto in_range(const SizeIn& size)
    {
        using common_t = std::common_type_t<SizeIn, SizeType>;
        return static_cast<common_t>(size)
               < static_cast<common_t>(std::numeric_limits<SizeType>::max());
    }

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
            if(in_range<std::uint32_t>(size))
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
        using key_in_t = typename device_segmented_topk_air_helper::iterator_traits<
            KeysInputIterator>::value_type;
        using value_in_t = typename device_segmented_topk_air_helper::iterator_traits<
            ValuesInputIterator>::value_type;

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
        // Targets::for_each(
        //     [&](auto candidate)
        //     {
        //         if(target{candidate} == target_config)
        //         {
        //             using ArchConfig
        //                 = rocprim::detail::target_config<Config, Selector, decltype(candidate)>;
        //             ret = invoke_impl<ArchConfig>(size, args);
        //         }
        //     });
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
        using key_in_t = typename device_segmented_topk_air_helper::iterator_traits<
            KeysInputIterator>::value_type;
        using value_in_t = typename device_segmented_topk_air_helper::iterator_traits<
            ValuesInputIterator>::value_type;

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
