// Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DETAIL_DEVICE_TOPK_AIR_HPP_
#define ROCPRIM_DEVICE_DETAIL_DEVICE_TOPK_AIR_HPP_

#include "../../block/block_load_func.hpp"
#include "../../block/block_scan.hpp"
#include "../../block/block_store_func.hpp"
#include "../../detail/temp_storage.hpp"
#include "../../intrinsics.hpp"
#include "../config_types.hpp"
#include "../device_topk_config.hpp"
#include "../device_transform.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{
namespace device_topk_air_helper
{

template<class T>
struct iterator_traits : public std::iterator_traits<T>
{};

template<>
struct iterator_traits<std::nullptr_t>
{
    using value_type = empty_type;
};

// Find the integral type which has the same size as input floating point type
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

template<class T, class = void>
constexpr bool has_operator_left_shift_v = false;

template<class T>
constexpr bool has_operator_left_shift_v<T, std::void_t<decltype(std::declval<T&>() << sizeof(T))>>
    = true;
} // namespace device_topk_air_helper

/// \brief This is an implementation of the TopK algorithm.
/// AIR stands for Adaptive Iteration-fused Radix TopK:
///
/// - **Adaptive** means the algorithm can automatically choose between two
///   strategies: one optimized for evenly distributed input data and another
///   optimized for radix-adversarial input distributions.
/// - **Iteration-fused** means that in each iteration, the algorithm performs
///   filtering for the current iteration and histogramming for the next
///   iteration simultaneously.
///
/// \note This algorithm only supports unstable output. Therefore, when values
/// are written to the output array, the relative order of equivalent elements
/// is not guaranteed.
///
/// \tparam Adaptive
/// Load input data from a temporary buffer instead of the original input.
/// This uses additional buffer space.
///
/// \tparam UseThreadCounter
/// Use a thread counter to store thread values in bulk.
///
/// TODO: This brings some improvements but also some performance regressions.
/// We need to tune the algorithm and determine in which cases the thread
/// counter is beneficial.
///
/// \tparam UseNativeOperator
/// Use native operators for bit-wise comparison. For regular types, we use
/// `radix_key_codec` to extract digits from input values; in these cases, it is
/// safe to use the native `operator=` and `operator<` to compare extracted
/// digits because invalid bits are initialized to 0. However, in other cases,
/// unrelated bits may not be initialized and may contain random values, so
/// native operators may not be safe to use.
///
/// \tparam KillNegativeZeros
/// When extracting digits from floating-point types, replace negative zeros with
/// positive zeros.
///
/// TODO: I think setting this to false by default is acceptable, because we
/// usually want the output to match the input exactly, even if negative zeros
/// are present. Handling negative zeros should probably be the user's
/// responsibility, so we may need to remove this parameter.
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
         typename Decomposer    = ::rocprim::identity_decomposer,
         bool UseThreadCounter  = true,
         bool UseNativeOperator = true,
         bool KillNegativeZeros = false>
struct device_topk_air_impl
{
    // Constant member variables
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

    using key_codec = decltype(::rocprim::traits::get<key_in_t>().template radix_key_codec<true>());
    using digit_t
        = decltype(key_codec::template extract_digit<Decomposer>(key_in_t{}, 0, 0, Decomposer{}));

    // Max value in each is up to numeric_limits<SizeOut>::max(), so, SizeOut is used here
    template<size_t HistogramSize>
    using histogram_t = SizeOut[HistogramSize];
    // Scan over histogram, so use SizeOut
    using block_scan_t = block_scan<SizeOut, BlockSize>;

    // Used by thread counter
    using count_t = unsigned char;

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

    // If both value_in_t and value_out_t are empty, then no value will be output.
    static constexpr bool output_value
        = !std::is_same_v<value_in_t, empty_type> || !std::is_same_v<value_out_t, empty_type>;

    /// TODO: This likely needs further tuning.
    ///
    /// \brief `candidate_buffer_coefficient` is used to determine whether
    /// or not to use an adaptive buffer to store the indices of the input data.
    ///
    /// \note If Adaptive is true, two adaptive buffers will be created.
    /// Each has a size of `sizeof(SizeIn) * (size / candidate_buffer_coefficient)`.
    static constexpr SizeIn candidate_buffer_coefficient = CandidateBufferCoefficient;

    /// \brief This is the limit of `items_per_thread` to disable `thread_counter`
    ///
    /// For smaller types such as unsigned char, thread_counter can be slow.
    /// This may be an occupancy issue and needs further investigation.
    ///
    /// It reduces the number of atomicAdd calls, but it requires more registers.
    /// If it uses too many registers, the compiler will spill them out of VGPRs,
    /// which results in significant slowdown.
    ///
    /// Also, for naturally distributed data, the benefits of thread_counter may be limited.
    ///
    /// TODO: after tuning, this needs to be adjusted, thread_counter reduces the amount.
    /// However, it's hard to tune this parameter, since it depends on items_per_thread.
    /// If during compile time, items_per_thread is 8, then it makes no sense to tryout any
    /// larger number for thread_counter_limit.
    static constexpr decltype(items_per_thread) thread_counter_limit = ThreadCounterLimit;

    /// \brief This class is used to store the bits of each iteration. Because the
    /// total number of bits across all iterations equals the bit size of
    /// key_in_t, we can use an integral type with the same size as key_in_t to
    /// store all per-iteration data.
    ///
    /// Additionally, for the selected bin of each iteration, digits can be
    /// accessed using the constexpr value `Iteration`. Therefore, constexpr
    /// version of the setter and getter functions are also provided here.
    struct digits_array
    {
    private:
        using int_key_t = typename device_topk_air_helper::matched_int<key_in_t>::type;
        static constexpr auto bits_total = sizeof(key_in_t) * 8;
        int_key_t             data;

        // Runtime mask, and it will be compile time function when Iteration is constexpr
        static constexpr ROCPRIM_FORCE_INLINE auto mask(decltype(bits_per_iteration) NumBits)
        {
            return (int_key_t{1} << NumBits) - 1;
        }

    public:
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

    /// \brief The global storage type of non-adaptive air topk
    /// \note If you need to modify `non_adaptive_storage_type`, be careful with the
    /// order of the members. A single `hipMemcpy` or `hipMemset` operation is
    /// used to initialize and reset the members in each iteration, relying on
    /// their current order.
    /// Changing the order may lead to unexpected behavior.
    struct non_adaptive_storage_type
    {
        SizeOut      output_pos; // Initialize at Iteration 0 -> init value 0
        SizeOut      last_output_pos; // Initialize at Iteration 0 -> init value 0
        digits_array chosen_bins; // Initialize at Iteration 0 -> init value 0
        unsigned int stopped_at; // Initialize at Iteration 0 -> init value 0

        histogram_t<num_buckets> histogram; // Initialize in each Iteration -> init value 0
        unsigned int num_finished_blocks; // Initialize in each Iteration -> init value 0

        SizeIn  N; // Does not need initializing
        SizeOut K; // Does not need initializing
    };

    /// \brief The global storage type of adaptive air topk
    /// \note If you need to modify `adaptive_storage_type`, be careful with the
    /// order of the members. A single `hipMemcpy` or `hipMemset` operation is
    /// used to initialize and reset the members in each iteration, relying on
    /// their current order.
    /// Changing the order may lead to unexpected behavior.
    struct adaptive_storage_type
    {
        SizeOut      output_pos; // Initialize at Iteration 0 -> init value 0
        SizeOut      last_output_pos; // Initialize at Iteration 0 -> init value 0
        digits_array chosen_bins; // Initialize at Iteration 0 -> init value 0
        unsigned int stopped_at; // Initialize at Iteration 0 -> init value 0

        histogram_t<num_buckets> histogram; // Initialize in each Iteration -> init value 0
        unsigned int num_finished_blocks; // Initialize in each Iteration -> init value 0

        SizeIn adaptive_buf_outpos; // Initialize in each Iteration -> init value 0

        SizeIn  N; // Does not need initializing
        SizeOut K; // Does not need initializing

        SizeIn adaptive_buf_size; // Does not need initializing
    };

    using storage_type
        = std::conditional_t<Adaptive, adaptive_storage_type, non_adaptive_storage_type>;

    enum class candidate_category
    {
        // Item is the input of this iteration
        input,
        // Item was the cadidate identified in the last iteration
        candidate,
        // Item is neither the input nor the candidate
        discard
    };

    enum class flip_strategy
    {
        // Does nothing, will call extract_digit directly
        no_flip,
        // Make input type unsigned, and move all values to fit unsigned type
        input_flip,
        // Flip only two’s complement or extracted digit
        output_flip
    };

    // In the implementaion of function `extract_digit`, we are confident that unrelated bits are zeros
    // So we can directly use the native operator
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

    // Initialize histogram bin counts to zeros
    template<unsigned int HistogramSize>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE static void
    init_histogram(histogram_t<HistogramSize> &histogram, const unsigned int thread_id)
    {
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

    // Merge block histogram into global histogram
    template<unsigned int Size1, unsigned int Size2>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE static void
    merge_histogram(histogram_t<Size1> & dist, histogram_t<Size2> const& src, const unsigned int thread_id)
    {
        constexpr auto                   size         = Size1 < Size2 ? Size1 : Size2;
        std::remove_cv_t<decltype(size)> histo_offset = 0;

        // Strip threads for initializing
        ROCPRIM_UNROLL
        for(; histo_offset + block_size <= size; histo_offset += block_size)
        {
            const auto idx = histo_offset + thread_id;
            ::rocprim::detail::atomic_add(dist + idx, src[idx]);
        }

        // Finish up with guarded merging if necessary
        if((size % block_size != 0) && (histo_offset + thread_id < size))
        {
            const auto idx = histo_offset + thread_id;
            ::rocprim::detail::atomic_add(dist + idx, src[idx]);
        }
    }

    /// \brief Extracts digit values for radix-based sorting or partitioning.
    /// For radix-based sorting or partitioning, digits are extracted from the most
    /// significant bit to the least significant bit. For signed or floating-point
    /// types, the position must be adjusted for negative values.
    ///
    /// This function provides three flip strategies, suitable for both integral
    /// and floating-point types.
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
                         && device_topk_air_helper::has_operator_left_shift_v<key_in_t>)
            { // Builtin integral types (including rocprim::int128_t and rocprim::uint128_t)
                return KeyCodec::template extract_digit<Decomposer>(
                    key ^ (key_in_t{1} << (sizeof(key_in_t) * 8 - 1)), // Flip only two’s complement
                    start, // Start bit of the sequence of bits to extract
                    length, // How many bits to extract
                    decomposer);
            }
            else if constexpr(rocprim::is_integral<key_in_t>::value
                              && !device_topk_air_helper::has_operator_left_shift_v<key_in_t>)
            { // Custom types may not support `operator<<`, so they are `bit_cast` to integral types instead.
                using matched_int_t = typename device_topk_air_helper::matched_int<key_in_t>::type;
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
                using matched_int_t = typename device_topk_air_helper::matched_int<key_in_t>::type;
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

    /// \brief Check previous iterations to see if key was a candidate or in the pivot bin
    /// \return [category, last_digit]
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

    /// \brief Check if need to load from or store into the adaptive buffer
    /// \return [load, store, adaptive_buffer_size]
    constexpr ROCPRIM_FORCE_INLINE static
    std::tuple<bool, bool, SizeIn>
    check_load_store_adaptive_buf(unsigned int iteration, storage_type* __restrict__ p_global_storage, SizeIn size)
    {
        if constexpr(Adaptive)
        {
            return {(iteration > 1) && p_global_storage->adaptive_buf_size,
                    (iteration >= 1)
                        && (p_global_storage->N <= (size / candidate_buffer_coefficient)),
                    p_global_storage->adaptive_buf_size};
        }
        else
        {
            return {false, false, 0};
        }
    }

    /// \brief This function calculates the histogram and calls `record_to_histogram_fn`
    /// to store the results into the histogram.
    /// It also writes data to keys_output (or values_output when using pairs).
    template<unsigned int Iteration, class OffsetT, class F>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE static void 
    thread_histogram_and_filter_prev(
        storage_type* __restrict__ p_global_storage,
        SizeIn* __restrict__ in_idx_buf,
        SizeIn* __restrict__ out_idx_buf,
        KeysInputIterator keys_input,
        KeysOutputIterator keys_output,
        ValuesInputIterator values_input,
        ValuesOutputIterator values_output,
        SizeIn size,
        Decomposer decomposer,
        OffsetT global_offset,
        F record_to_histogram_fn
    )
    {
        const auto [load_adaptive, store_adaptive, size_adaptive]
            = check_load_store_adaptive_buf(Iteration, p_global_storage, size);
        const auto chosen_bins = p_global_storage->chosen_bins;

        const auto num_valid = load_adaptive ? size_adaptive : size;
        // Swap the buffer in each iteration
        SizeIn* in_buf  = Iteration % 2 ? in_idx_buf : out_idx_buf;
        SizeIn* out_buf = Iteration % 2 ? out_idx_buf : in_idx_buf;
        std::conditional_t<Adaptive, SizeIn[items_per_thread], rocprim::empty_type> thread_out_buf;
        std::remove_cv_t<decltype(items_per_thread)> thread_out_buf_size = 0;

        ROCPRIM_UNROLL
        for(std::remove_cv_t<decltype(items_per_thread)> i = 0; i < items_per_thread; ++i)
        {
            const auto global_idx = static_cast<SizeIn>(i + global_offset);
            if(global_idx >= num_valid)
            {
                break;
            }

            const auto index = load_adaptive ? in_buf[global_idx] : global_idx;
            // Calculate the histogram and identify candidate
            const auto key = keys_input[index];
            if constexpr(Iteration == 0) // First Iteration
            { // For first iteration, every thing from the input is input
                record_to_histogram_fn(extract_digit_of_cur_iteration<Iteration>(key, decomposer));
            }
            else
            {
                const auto [category, candidate_digit]
                    = identify_candidate<Iteration>(key, chosen_bins, load_adaptive, decomposer);

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
                        {
                            // Write this into output buffer
                            // TODO: use thread counter
                            const auto output_pos
                                = ::rocprim::detail::atomic_add(&p_global_storage->output_pos, 1);
                            keys_output[output_pos] = key;
                            if constexpr(output_value)
                            {
                                values_output[output_pos] = values_input[index];
                            }
                            break;
                        }

                    default: break;
                }
            }
        }

        // Store adaptive output indices into outbuf
        if constexpr(Adaptive)
        {
            if(store_adaptive && thread_out_buf_size)
            {
                auto adaptive_buf_outpos
                    = ::rocprim::detail::atomic_add(&p_global_storage->adaptive_buf_outpos,
                                                    thread_out_buf_size);
                ROCPRIM_UNROLL
                for(decltype(thread_out_buf_size) i = 0; i < items_per_thread; ++i)
                {
                    if(i < thread_out_buf_size)
                    {
                        out_buf[adaptive_buf_outpos + i] = thread_out_buf[i];
                    }
                }
            }
        }
    }

    template<unsigned int Iteration, class OffsetT, class StorageT>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE static void 
    launch_thread_histogram_and_filter_prev(
        storage_type* __restrict__ p_global_storage,
        SizeIn* __restrict__ in_idx_buf,
        SizeIn* __restrict__ out_idx_buf,
        KeysInputIterator keys_input,
        KeysOutputIterator keys_output,
        ValuesInputIterator values_input,
        ValuesOutputIterator values_output,
        SizeIn size, 
        Decomposer decomposer,
        OffsetT global_offset,
        StorageT& storage
    )
    {
        if constexpr(UseThreadCounter
                     && items_per_thread
                            != 1 // When items_per_thread is 1 UseThreadCounter is useless
                     && (items_per_thread < ~(count_t{0})) // Ensure count_t is capable
                     && items_per_thread < thread_counter_limit // Ensure thread_counter is fast
        )
        {
            // Use thread_counter to record thread histogram values, and add to block histogram at once
            // to reduce the number of calls of atomicAdd
            // When items_per_thread is small, we can use an array directly, I don't want bother hash table
            // or any thread sort algorithms, they will be slow I think.
            // Also thread_counter should be an option.
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

            thread_histogram_and_filter_prev<Iteration>(p_global_storage,
                                                        in_idx_buf,
                                                        out_idx_buf,
                                                        keys_input,
                                                        keys_output,
                                                        values_input,
                                                        values_output,
                                                        size,
                                                        decomposer,
                                                        global_offset,
                                                        record_to_counter_fn);
            // Store counter into shared memory
            ROCPRIM_UNROLL
            for(decltype(thread_counter_size) i = 0; i < items_per_thread; ++i)
            {
                if(i < thread_counter_size)
                {
                    ::rocprim::detail::atomic_add(&storage.block_local_histogram[thread_digit[i]],
                                                  thread_counter[i]);
                }
            }
        }
        else
        {
            auto record_to_histogram_fn = [&](auto digit)
            { ::rocprim::detail::atomic_add(&storage.block_local_histogram[digit], 1); };
            thread_histogram_and_filter_prev<Iteration>(p_global_storage,
                                                        in_idx_buf,
                                                        out_idx_buf,
                                                        keys_input,
                                                        keys_output,
                                                        values_input,
                                                        values_output,
                                                        size,
                                                        decomposer,
                                                        global_offset,
                                                        record_to_histogram_fn);
        }
    }

    /// \brief Chooses the bin that contains the pivot.
    ///
    /// Stores information into:
    /// - p_global_storage->K
    /// - p_global_storage->N
    /// - p_global_storage->chosen_bins
    /// - p_global_storage->stopped_at
    template<unsigned int Iteration, unsigned int HistogramSize>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE static void 
    chose_pivot_bin(
        storage_type* __restrict__ p_global_storage,
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
                K                   = K - prev;
                N                   = cur - prev;
                p_global_storage->K = K;
                p_global_storage->N = N;
                p_global_storage->chosen_bins.template set<Iteration>(global_i);
                p_global_storage->stopped_at = K == N ? Iteration : num_iterations;
                break;
            }
        }
    }

    /// \brief Performs histogramming for the current iteration and filtering for
    /// the previous iteration.
    ///
    /// If Adaptive is enabled, this function also calls
    /// `launch_thread_histogram_and_filter_prev` to either load the adaptive
    /// buffer from two iterations earlier or to store the chosen pivot bin from
    /// the previous iteration into the adaptive buffer.
    ///
    /// Why not store the pivot bin of the current iteration?
    /// - Because the histogram result is not available until the end of this function.
    template<unsigned int Iteration>
    ROCPRIM_KERNEL ROCPRIM_FORCE_INLINE
    ROCPRIM_LAUNCH_BOUNDS(ROCPRIM_DEFAULT_MAX_BLOCK_SIZE) 
    static void 
    histogram_and_filter(
        storage_type* __restrict__                p_global_storage,
        SizeIn* __restrict__                             in_idx_buf,
        SizeIn* __restrict__                             out_idx_buf,
        KeysInputIterator                                keys_input,
        KeysOutputIterator                               keys_output,
        ValuesInputIterator                              values_input,
        ValuesOutputIterator                             values_output,
        const SizeIn             size,
        const SizeOut            K,
        const Decomposer decomposer)
    {
        constexpr auto histogram_size
            = Iteration == (num_iterations - 1) ? num_buckets_last_iteration : num_buckets;

        ROCPRIM_SHARED_MEMORY struct
        {
            union
            {
                bool                                is_last_block;
                typename block_scan_t::storage_type scan;
            } union_data;
            histogram_t<histogram_size> block_local_histogram;
        } storage;

        // Load problem size and init local_histogram
        SizeIn  N_this_iteration;
        SizeOut K_this_iteration;

        if constexpr(Iteration == 0) // First iteration
        { // If K_this_iteration == N_this_iteration, kernel will be reject from the host
            // so here we don't need to check and return like other iterations
            N_this_iteration = size;
            K_this_iteration = K;
        }
        else
        {
            N_this_iteration = p_global_storage->N;
            K_this_iteration = p_global_storage->K;

            // Return earlier
            if(K_this_iteration == N_this_iteration)
            {
                return; // All threads return no divergence
            }
        }

        // Histogram

        const auto thread_id     = detail::block_thread_id<0>();
        const auto block_id      = detail::block_id<0>();
        const auto global_offset = items_per_thread * ((block_id * block_size) + thread_id);
#ifndef __OPTIMIZE__
        // If compile with "-O0", we use the shared memory as temporary workaround
        storage.union_data.is_last_block = false;
#endif
        init_histogram(storage.block_local_histogram, thread_id);
        // Sync to make sure all write operations are done
        ::rocprim::syncthreads();

        launch_thread_histogram_and_filter_prev<Iteration>(p_global_storage,
                                                           in_idx_buf,
                                                           out_idx_buf,
                                                           keys_input,
                                                           keys_output,
                                                           values_input,
                                                           values_output,
                                                           size,
                                                           decomposer,
                                                           global_offset,
                                                           storage);
        // Make sure block_local_histogram write is finished
        ::rocprim::syncthreads();

        merge_histogram(p_global_storage->histogram, storage.block_local_histogram, thread_id);
        // Make sure this block has completed writing data into global histogram
        ::rocprim::syncthreads();

        // Above this line, there are atomic writes to global memory.
        // Below this line, there are direct loads from global memory.
        // This fence is needed to ensure the correct visibility for subsequent loads
        // from global memory, so that we read the correct values.
        // Because direct loads from global memory go through the cache, this fence
        // acts as a cache invalidation.
        // I'm not sure why this works without the fence on other architectures, but on
        // gfx950 this is strictly required. It may have a more strict caching strategy.
        ::rocprim::detail::atomic_fence_acquire_order_only();

        // Temporary fix for `__syncthreads_or`, which doesn't work when compiling with "-O0"
        // So we use shared memory for broadcasting
        auto syncdevice_and_is_last_block = [&]()
        {
#ifdef __OPTIMIZE__
            bool is_last_block = false;
            if(thread_id == 0)
            {
                const auto local_num_finished_blocks
                    = ::rocprim::detail::atomic_add(&p_global_storage->num_finished_blocks, 1);
                is_last_block = local_num_finished_blocks == (gridDim.x - 1);
            }
            // By using __syncthreads_or, all threads will communicate with each other
            // and do "OR" operation for all values.
            return __syncthreads_or(is_last_block);
#else
            if(thread_id == 0)
            {
                const auto local_num_finished_blocks
                    = ::rocprim::detail::atomic_add(&p_global_storage->num_finished_blocks, 1);
                storage.union_data.is_last_block = local_num_finished_blocks == (gridDim.x - 1);
            }
            ::rocprim::syncthreads();
            const auto x = storage.union_data.is_last_block;
            // The shared memory will be used by storage.union_data.scan
            // So we need to make sure all write operations to this memory
            // to be finished
            ::rocprim::syncthreads();
            return x;
#endif
        };

        // Make sure all write operation to the global memory is finished
        // Filter
        if(syncdevice_and_is_last_block())
        {
            // Setup the buffer size of next iteration if needed
            if constexpr(Adaptive && Iteration >= 1)
            {
                if(thread_id == 0)
                {
                    // Direct load of `p_global_storage->adaptive_buf_outpos` is ensured by the fence.
                    // Direct store of `p_global_storage->adaptive_buf_size` doesn't need to be ensured
                    // by a fence since, we don't read it later in this kernel.
                    p_global_storage->adaptive_buf_size = p_global_storage->adaptive_buf_outpos;
                }
            }

            histogram_t<bins_per_thread> thread_bins;
            // Load data into register
            // `block_load_direct_blocked` calls direct loads from the `p_global_storage->histogram`
            // A fence is added to ensure that the cache is refreshed
            block_load_direct_blocked(thread_id,
                                      p_global_storage->histogram,
                                      thread_bins,
                                      histogram_size,
                                      extract_digit_of_cur_iteration<Iteration>(
                                          key_codec::get_out_of_bounds_key(decomposer),
                                          decomposer));
            // Block scan
            block_scan_t{}.inclusive_scan(thread_bins,
                                          thread_bins,
                                          storage.union_data.scan,
                                          ::rocprim::plus<SizeOut>{});
            // Store data into shared memory
            // `storage.block_local_histogram` is not in the storage.union,
            // so there is no conflict, we don't need a `syncthreads` here.
            block_store_direct_blocked(thread_id,
                                       storage.block_local_histogram,
                                       thread_bins,
                                       histogram_size);

            // Need to sync threads, because we will read storage.block_local_histogram[global_i - 1]
            // which is set by thread at index of (thread_id -1)
            ::rocprim::syncthreads();

            // Chose the bin which contains the pivot
            chose_pivot_bin<Iteration>(p_global_storage,
                                       thread_bins,
                                       storage.block_local_histogram,
                                       N_this_iteration,
                                       K_this_iteration,
                                       thread_id);
        }
    }

    /// \brief Performs the filtering step for the final iteration.
    ///
    /// Because `histogram_and_filter` only performs filtering for earlier
    /// iterations, this function is needed to run the final round of filtering.
    ROCPRIM_KERNEL ROCPRIM_FORCE_INLINE
    ROCPRIM_LAUNCH_BOUNDS(ROCPRIM_DEFAULT_MAX_BLOCK_SIZE) 
    static void 
    last_filter(
        storage_type* __restrict__                p_global_storage,
        SizeIn* __restrict__                             in_idx_buf,
        SizeIn* __restrict__                             out_idx_buf,
        KeysInputIterator                                keys_input,
        KeysOutputIterator                               keys_output,
        ValuesInputIterator                              values_input,
        ValuesOutputIterator                             values_output,
        const SizeIn                                     size,
        const SizeOut                                    K,
        const Decomposer decomposer)
    {
        if(p_global_storage->output_pos >= K)
        {
            return; // Early stop
        }

        const auto global_offset
            = items_per_thread
              * ((detail::block_id<0>() * block_size) + detail::block_thread_id<0>());

        // Load some global memory into register
        const auto last_k            = p_global_storage->K;
        const auto stopped_iteration = p_global_storage->stopped_at;
        const auto chosen_bins       = p_global_storage->chosen_bins;

        const auto cur_bits
            = stopped_iteration == (num_iterations - 1) ? bits_last_iteration : bits_per_iteration;
        const auto stopped        = num_iterations != stopped_iteration;
        const auto last_iteration = stopped ? stopped_iteration : num_iterations - 1;
        const auto cur_iteration  = last_iteration + 1;
        const auto [load_adaptive, _, size_adaptive]
            = check_load_store_adaptive_buf(cur_iteration, p_global_storage, size);
        const auto num_valid = load_adaptive ? size_adaptive : size;
        // Swap the buffer in each iteration
        SizeIn*    in_buf          = cur_iteration % 2 ? in_idx_buf : out_idx_buf;
        const auto last_chosed_bin = chosen_bins.get(last_iteration);

        ROCPRIM_UNROLL
        for(std::remove_cv_t<decltype(items_per_thread)> i = 0; i < items_per_thread; ++i)
        {
            const auto global_idx = static_cast<SizeIn>(i + global_offset);
            if(global_idx >= num_valid)
            {
                break;
            }

            const auto index = load_adaptive ? in_buf[global_idx] : global_idx;
            // Calculate the histogram and identify candidate
            const auto key = keys_input[index];

            // Extract all digits
            digit_t digits[num_iterations];
            bool    is_candidate_in_prev_iteration = true;
            // It's actually faster to just directly extract all digits, instead of using runtime variable
            // last_iteration to determin how many iterations needs to be loaded
            rocprim::detail::constexpr_for_lt<0, num_iterations, 1>(
                [&](const auto i)
                { digits[i] = extract_digit_of_cur_iteration<i>(key, decomposer); });

            // last_iteration could be 0, which invalidates `last_iteration - 1`
            if(last_iteration == 0)
            {
                is_candidate_in_prev_iteration = true;
            }
            else if(load_adaptive
                    && !equal_last_n_bits(chosen_bins.get(last_iteration - 1),
                                          digits[last_iteration - 1],
                                          bits_per_iteration))
            { // Only check the iteration before last iteration
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
                const auto output_pos   = ::atomicAdd(&p_global_storage->output_pos, 1);
                keys_output[output_pos] = key;
                if constexpr(output_value)
                {
                    values_output[output_pos] = values_input[index];
                }
            }
            else if(is_candidate_in_prev_iteration && stopped
                    && equal_last_n_bits(digits[last_iteration], last_chosed_bin, cur_bits))
            { // If stopped, then we don't need to count last_output_pos
                // Stopped means that, K = N, so all items in previous pivot
                // bin should be stored into output.
                const auto output_pos   = ::atomicAdd(&p_global_storage->output_pos, 1);
                keys_output[output_pos] = key;
                if constexpr(output_value)
                {
                    values_output[output_pos] = values_input[index];
                }
            }
            else if(is_candidate_in_prev_iteration && !stopped
                    && equal_last_n_bits(digits[last_iteration], last_chosed_bin, cur_bits)
                    && ::rocprim::detail::atomic_add(&p_global_storage->last_output_pos, 1)
                           < last_k)
            { // If not stopped, we need to check how many items in the pivot bin should we
                // Write to the output
                const auto output_pos
                    = ::rocprim::detail::atomic_add(&p_global_storage->output_pos, 1);
                keys_output[output_pos] = key;
                if constexpr(output_value)
                {
                    values_output[output_pos] = values_input[index];
                }
            }
        }
    }

    /// \brief Launches `histogram_and_filter` for `IterationLeft` iterations.
    ///
    /// For the first iteration, `size` and `K` describe the problem size.
    /// For subsequent iterations, `p_prev_n` and `p_prev_k` are used, because
    /// only the bin that contains the pivot is carried forward to the next
    /// iteration.
    ///
    /// This is similar to a static/constexpr unroll, but since it returns
    /// `hipError_t`, it cannot be replaced by `constexpr_for_*`.
    template<unsigned int IterationLeft>
    ROCPRIM_FORCE_INLINE static constexpr hipError_t launch_iterations(
                                    storage_type*                         p_global_storage,
                                    SizeIn*                                      in_idx_buf,
                                    SizeIn*                                      out_idx_buf,
                                    KeysInputIterator                            keys_input,
                                    KeysOutputIterator                           keys_output,
                                    ValuesInputIterator                          values_input,
                                    ValuesOutputIterator                         values_output,
                                    const SizeIn                                 size,
                                    const SizeOut                                K,
                                    const Decomposer                             decomposer,
                                    unsigned int                                 num_blocks,
                                    const hipStream_t                            stream,
                                    const bool                                   debug_synchronous,
                                    const std::chrono::steady_clock::time_point& start)
    {
        if constexpr(IterationLeft)
        {
            // Initialize

            constexpr auto current_iteration = num_iterations - IterationLeft;
            if constexpr(current_iteration == 0)
            {
                const auto init_size
                    = std::distance(reinterpret_cast<char*>(p_global_storage),
                                    reinterpret_cast<char*>(&(p_global_storage->N)));
                ROCPRIM_RETURN_ON_ERROR(hipMemsetAsync(p_global_storage, 0, init_size, stream));
            }
            else
            {
                const auto init_size
                    = std::distance(reinterpret_cast<char*>(&(p_global_storage->histogram)),
                                    reinterpret_cast<char*>(&(p_global_storage->N)));
                ROCPRIM_RETURN_ON_ERROR(
                    hipMemsetAsync(&(p_global_storage->histogram), 0, init_size, stream));
            }

            // Launch histogram and filter for current iteration
            histogram_and_filter<current_iteration>
                <<<num_blocks, block_size, 0, stream>>>(p_global_storage,
                                                        in_idx_buf,
                                                        out_idx_buf,
                                                        keys_input,
                                                        keys_output,
                                                        values_input,
                                                        values_output,
                                                        size,
                                                        K,
                                                        decomposer);
            ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("rocprim::detail::histogram_and_filter",
                                                        size,
                                                        start);
            return launch_iterations<IterationLeft - 1>(p_global_storage,
                                                        in_idx_buf,
                                                        out_idx_buf,
                                                        keys_input,
                                                        keys_output,
                                                        values_input,
                                                        values_output,
                                                        size,
                                                        K,
                                                        decomposer,
                                                        num_blocks,
                                                        stream,
                                                        debug_synchronous,
                                                        start);
        }
        else
        {
            // Launch last filter
            last_filter<<<num_blocks, block_size, 0, stream>>>(p_global_storage,
                                                               in_idx_buf,
                                                               out_idx_buf,
                                                               keys_input,
                                                               keys_output,
                                                               values_input,
                                                               values_output,
                                                               size,
                                                               K,
                                                               decomposer);
            ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("rocprim::detail::last_filter",
                                                        size,
                                                        start);
            return hipSuccess;
        }
    }

    constexpr hipError_t operator()(void*                temporary_storage,
                                    size_t&              storage_size,
                                    KeysInputIterator    keys_input,
                                    KeysOutputIterator   keys_output,
                                    ValuesInputIterator  values_input,
                                    ValuesOutputIterator values_output,
                                    const SizeIn         size,
                                    const SizeOut        K,
                                    const Decomposer     decomposer        = {},
                                    const hipStream_t    stream            = 0,
                                    const bool           debug_synchronous = false) const
    {
        const auto    num_blocks = ceiling_div(size, items_per_block);
        storage_type* p_global_storage;
        SizeIn*       in_idx_buf  = nullptr;
        SizeIn*       out_idx_buf = nullptr;

        if constexpr(Adaptive)
        {
            ROCPRIM_RETURN_ON_ERROR(detail::temp_storage::partition(
                temporary_storage,
                storage_size,
                detail::temp_storage::make_linear_partition(
                    detail::temp_storage::ptr_aligned_array(&p_global_storage,
                                                            sizeof(p_global_storage)),
                    temp_storage::ptr_aligned_array(&in_idx_buf,
                                                    sizeof(SizeIn)
                                                        * (size / candidate_buffer_coefficient)),
                    temp_storage::ptr_aligned_array(&out_idx_buf,
                                                    sizeof(SizeIn)
                                                        * (size / candidate_buffer_coefficient)))));
        }
        else
        {
            ROCPRIM_RETURN_ON_ERROR(detail::temp_storage::partition(
                temporary_storage,
                storage_size,
                detail::temp_storage::make_linear_partition(
                    detail::temp_storage::ptr_aligned_array(&p_global_storage,
                                                            sizeof(p_global_storage)))));
        }

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

        if(size == K)
        { // Write output directly
            ROCPRIM_RETURN_ON_ERROR(transform(keys_input,
                                              keys_output,
                                              K,
                                              ::rocprim::identity<>(),
                                              stream,
                                              debug_synchronous));
            if constexpr(output_value)
            {
                ROCPRIM_RETURN_ON_ERROR(transform(values_input,
                                                  values_output,
                                                  K,
                                                  ::rocprim::identity<>(),
                                                  stream,
                                                  debug_synchronous));
            }
            return hipSuccess;
        }
        return launch_iterations<num_iterations>(p_global_storage,
                                                 in_idx_buf,
                                                 out_idx_buf,
                                                 keys_input,
                                                 keys_output,
                                                 values_input,
                                                 values_output,
                                                 size,
                                                 K,
                                                 decomposer,
                                                 num_blocks,
                                                 stream,
                                                 debug_synchronous,
                                                 start);
    }
};

// SizeIn is sensitive, this invoker will check which type is the actual
// SizeIn needed by the algorithm
// It's annoying to compile all of these types, but it's faster in theory
template<class Config,
         bool SelectMin,
         bool Adaptive,
         typename KeysInputIterator,
         typename KeysOutputIterator,
         typename ValuesInputIterator,
         typename ValuesOutputIterator,
         typename SizeIn,
         typename SizeOut,
         typename Decomposer>
struct device_topk_air_impl_invoker
{
private:
    template<unsigned int BlockSize,
             unsigned int ItemsPerThread,
             unsigned int RadixBits,
             unsigned int CandidateBufferCoefficient,
             unsigned int ThreadCounterLimit,
             class ActualSizeIn>
    using simplified_type = device_topk_air_impl<BlockSize,
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
                                                 Decomposer>;

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
        using key_in_t =
            typename device_topk_air_helper::iterator_traits<KeysInputIterator>::value_type;
        using value_in_t =
            typename device_topk_air_helper::iterator_traits<ValuesInputIterator>::value_type;

        using Selector     = topk_air_config_selector<key_in_t, value_in_t, SizeIn>;
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
};

template<typename Config = rocprim::default_config,
         bool         SelectMin = true,
         bool         Adaptive = true,
         typename KeysInputIterator,
         typename KeysOutputIterator,
         typename ValuesInputIterator,
         typename ValuesOutputIterator,
         typename SizeIn,
         typename SizeOut,
         typename Decomposer = ::rocprim::identity_decomposer>
ROCPRIM_FORCE_INLINE hipError_t device_topk_air(void* temporary_storage,
                                        size_t&              storage_size,
                                        KeysInputIterator    keys_input,
                                        KeysOutputIterator   keys_output,
                                        ValuesInputIterator  values_input,
                                        ValuesOutputIterator values_output,
                                        const SizeIn         size,
                                        const SizeOut        K,
                                        const Decomposer     decomposer        = {},
                                        const hipStream_t    stream            = 0,
                                        const bool           debug_synchronous = false)
{
    return device_topk_air_impl_invoker<Config,
                                        SelectMin,
                                        Adaptive,
                                        KeysInputIterator,
                                        KeysOutputIterator,
                                        ValuesInputIterator,
                                        ValuesOutputIterator,
                                        SizeIn,
                                        SizeOut,
                                        Decomposer>::invoke(size,
                                                            std::tie(temporary_storage,
                                                                     storage_size,
                                                                     keys_input,
                                                                     keys_output,
                                                                     values_input,
                                                                     values_output,
                                                                     size,
                                                                     K,
                                                                     decomposer,
                                                                     stream,
                                                                     debug_synchronous));
}

} // namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DETAIL_DEVICE_TOPK_AIR_HPP_
