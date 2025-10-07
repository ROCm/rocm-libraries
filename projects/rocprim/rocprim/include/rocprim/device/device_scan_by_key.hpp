// Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DEVICE_SCAN_BY_KEY_HPP_
#define ROCPRIM_DEVICE_DEVICE_SCAN_BY_KEY_HPP_

#include "config_types.hpp"
#include "detail/device_config_helper.hpp"
#include "detail/device_scan_by_key.hpp"
#include "detail/lookback_scan_state.hpp"
#include "detail/ordered_block_id.hpp"
#include "device_scan_by_key_config.hpp"

#include "../common.hpp"
#include "../config.hpp"
#include "../detail/temp_storage.hpp"
#include "../detail/various.hpp"
#include "../functional.hpp"
#include "../types/future_value.hpp"
#include "../types/tuple.hpp"

#include <hip/hip_runtime.h>

#include <iostream>
#include <iterator>

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<typename LookBackScanState, class BlockIdWrapper>
ROCPRIM_KERNEL ROCPRIM_LAUNCH_BOUNDS(ROCPRIM_DEFAULT_MAX_BLOCK_SIZE) void
    init_device_scan_by_key_kernel(LookBackScanState  lookback_scan_state,
                                   const unsigned int number_of_blocks,
                                   unsigned int       save_index,
                                   typename LookBackScanState::value_type* const save_dest,
                                   BlockIdWrapper                                block_id)
{
    init_lookback_scan_state_kernel_impl(lookback_scan_state,
                                         number_of_blocks,
                                         block_id,
                                         save_index,
                                         save_dest);
}

template<typename LookBackScanState,
         class KeysInputIterator,
         class ItemPerBlockType,
         class BlockIdWrapper>
ROCPRIM_KERNEL ROCPRIM_LAUNCH_BOUNDS(ROCPRIM_DEFAULT_MAX_BLOCK_SIZE) void init_device_scan_by_key_kernel(
    LookBackScanState                             lookback_scan_state,
    const unsigned int                            number_of_blocks,
    unsigned int                                  save_index,
    typename LookBackScanState::value_type* const save_dest,
    KeysInputIterator                             keys, // keys from the 0 without any offsets
    typename std::iterator_traits<
        KeysInputIterator>::value_type* __restrict__ last_keys_of_each_block,
    size_t                 num_last_keys_of_each_block,
    const ItemPerBlockType items_per_block,
    BlockIdWrapper block_id)
{
    init_lookback_scan_state_kernel_impl(lookback_scan_state,
                                         number_of_blocks,
                                         block_id,
                                         save_index,
                                         save_dest);

    // Initialize the last_keys_of_each_block
    const auto block_size = ::rocprim::detail::block_size<0>();
    const auto global_tid
        = (::rocprim::detail::block_id<0>() * block_size) + ::rocprim::detail::block_thread_id<0>();
    const auto total_threads = ::rocprim::detail::grid_size<0>() * block_size;
    using common_size_t      = typename std::common_type<decltype(num_last_keys_of_each_block),
                                                    decltype(total_threads)>::type;
    for(common_size_t i = global_tid; i < static_cast<common_size_t>(num_last_keys_of_each_block);
        i += static_cast<common_size_t>(total_threads))
    {
        last_keys_of_each_block[i] = keys[(i * items_per_block) + (items_per_block - 1)];
    }
}

template<typename Config,
         lookback_scan_determinism Determinism,
         bool                      Exclusive,
         typename KeyInputIterator,
         typename InputIterator,
         typename OutputIterator,
         typename InitialValueType,
         typename CompareFunction,
         typename BinaryFunction,
         typename LookbackScanState,
         typename AccType,
         typename WrappedBlockId>
inline hipError_t launch_device_scan_by_key(
    detail::target_arch                          arch,
    const KeyInputIterator                       keys,
    const InputIterator                          values,
    const OutputIterator                         output,
    const InitialValueType                       initial_value,
    const CompareFunction                        compare,
    const BinaryFunction                         scan_op,
    const LookbackScanState                      scan_state,
    const size_t                                 size,
    const size_t                                 starting_block,
    const size_t                                 number_of_blocks,
    const ::rocprim::tuple<AccType, bool>* const previous_last_value,
    bool                                         use_last_keys,
    const typename std::iterator_traits<KeyInputIterator>::value_type* const __restrict__ last_keys,
    WrappedBlockId ordered_bid,
    dim3                           grid,
    dim3                           block,
    size_t                         shmem,
    hipStream_t                    stream)
{

    auto kernel = [=](auto arch_config)
    {
        device_scan_by_key_kernel_impl<decltype(arch_config), Determinism, Exclusive>(
            keys,
            values,
            output,
            static_cast<AccType>(get_input_value(initial_value)),
            compare,
            scan_op,
            scan_state,
            size,
            starting_block,
            number_of_blocks,
            previous_last_value,
            use_last_keys,
            last_keys,
            ordered_bid);
    };

    return execute_launch_plan<Config>(arch, kernel, grid, block, shmem, stream);
}

/// \return 0 if in-place is not detected
template<bool Exclusive,
         typename KeysInputIterator,
         typename OutputIterator,
         typename SizeType,
         typename NumBlockType>
ROCPRIM_FORCE_INLINE ROCPRIM_HOST size_t detect_reusing_keys(
    KeysInputIterator keys,
    OutputIterator output,
    SizeType size,
    NumBlockType number_of_blocks)
{
    using key_type    = typename std::iterator_traits<KeysInputIterator>::value_type;
    using output_type = typename std::iterator_traits<OutputIterator>::value_type;
    if constexpr(std::is_same<KeysInputIterator, OutputIterator>::value)
    {
        if(static_cast<SizeType>(std::distance(keys, output)) < size)
        {
            // In this case, there is overlap on `keys` and `output`. So we allocate a temp buffer
            return sizeof(key_type) * (number_of_blocks - 1);
        }
    }
    else if constexpr(std::is_pointer<KeysInputIterator>::value
                      && std::is_pointer<OutputIterator>::value
                      && std::is_convertible<key_type, output_type>::value)
    {
        if(static_cast<SizeType>(std::distance(keys, reinterpret_cast<KeysInputIterator>(output)))
           < size)
        {
            // In this case, there is overlap on `keys` and `output`. So we allocate a temp buffer
            return sizeof(key_type) * (number_of_blocks - 1);
        }
    }
    return 0;
}

template<lookback_scan_determinism Determinism,
         bool                      Exclusive,
         typename Config,
         typename KeysInputIterator,
         typename InputIterator,
         typename OutputIterator,
         typename InitValueType,
         typename BinaryFunction,
         typename CompareFunction,
         typename AccType>
inline hipError_t scan_by_key_impl(void* const           temporary_storage,
                                   size_t&               storage_size,
                                   KeysInputIterator     keys,
                                   InputIterator         input,
                                   OutputIterator        output,
                                   const InitValueType   initial_value,
                                   const size_t          size,
                                   const BinaryFunction  scan_op,
                                   const CompareFunction compare,
                                   const hipStream_t     stream,
                                   const bool            debug_synchronous)
{
    using key_type = typename std::iterator_traits<KeysInputIterator>::value_type;

    bool use_atomic_block_id;
    ROCPRIM_RETURN_ON_ERROR(check_if_using_atomic_block_id(stream, use_atomic_block_id));
    const auto use_atomic_block_id_variant
        = ::rocprim::detail::constexpr_value_variant<bool, false, true>::create(
            use_atomic_block_id);

    bool use_sleepy_scan;
    ROCPRIM_RETURN_ON_ERROR(is_sleep_scan_state_used(stream, use_sleepy_scan));
    const auto use_sleepy_scan_variant
        = ::rocprim::detail::constexpr_value_variant<bool, false, true>::create(use_sleepy_scan);

    ROCPRIM_RETURN_ON_ERROR(std::visit(
        [&](auto use_sleepy_scan, auto use_atomic_block_id)
        {
            using config = wrapped_scan_by_key_config<Config, key_type, AccType>;

            detail::target_arch target_arch;
            ROCPRIM_RETURN_ON_ERROR(host_target_arch(stream, target_arch));
            const scan_by_key_config_params params
                = dispatch_target_arch<config, false>(target_arch);

            using wrapped_type     = ::rocprim::tuple<AccType, bool>;
            using scan_state_type  = detail::lookback_scan_state<wrapped_type, use_sleepy_scan>;
            using ordered_bid_type = block_id_wrapper<unsigned int, use_atomic_block_id>;

            const unsigned int block_size       = params.kernel_config.block_size;
            const unsigned int items_per_thread = params.kernel_config.items_per_thread;
            const unsigned int items_per_block  = block_size * items_per_thread;

            const unsigned int size_limit = params.kernel_config.size_limit;
            const unsigned int aligned_size_limit
                = std::max<size_t>(size_limit - (size_limit % static_cast<size_t>(items_per_block)),
                                   items_per_block);

            const unsigned int limited_size
                = static_cast<unsigned int>(std::min<size_t>(size, aligned_size_limit));
            const bool use_limited_size = limited_size == aligned_size_limit;

            // Number of blocks in a single launch (or the only launch if it fits)
            const unsigned int number_of_blocks = ceiling_div(limited_size, items_per_block);

            void*         scan_state_storage;
            wrapped_type* previous_last_value;

            detail::temp_storage::layout layout{};
            ROCPRIM_RETURN_ON_ERROR(
                scan_state_type::get_temp_storage_layout(number_of_blocks, stream, layout));

            typename ordered_bid_type::id_type* ordered_bid_storage;

            key_type*  last_keys_of_each_block;
            const auto last_keys_of_each_block_size_byte
                = items_per_thread
                      ? detect_reusing_keys<Exclusive>(keys,
                                                       output,
                                                       size,
                                                       ceiling_div(size, items_per_block))
                      : 0;

            const hipError_t partition_result = detail::temp_storage::partition(
                temporary_storage,
                storage_size,
                detail::temp_storage::make_linear_partition(
                    // This is valid even with offset_scan_state_with_sleep_type
                    detail::temp_storage::make_partition(&scan_state_storage, layout),
                    detail::temp_storage::ptr_aligned_array(&previous_last_value,
                                                            use_limited_size ? 1 : 0),
                    detail::temp_storage::ptr_aligned_array(&last_keys_of_each_block,
                                                            last_keys_of_each_block_size_byte),
                    detail::temp_storage::make_partition(
                        &ordered_bid_storage,
                        ordered_bid_type::get_temp_storage_layout())));
            if(partition_result != hipSuccess || temporary_storage == nullptr)
            {
                return partition_result;
            }

            if(number_of_blocks == 0u)
            {
                return hipSuccess;
            }

            scan_state_type scan_state{};
            ROCPRIM_RETURN_ON_ERROR(
                scan_state_type::create(scan_state, scan_state_storage, number_of_blocks, stream));

            auto ordered_bid = ordered_bid_type::create(ordered_bid_storage);

            // Total number of blocks in all launches
            const auto   total_number_of_blocks = ceiling_div(size, items_per_block);
            const size_t number_of_launch       = ceiling_div(size, limited_size);

            if(debug_synchronous)
            {
                std::cout << "----------------------------------\n";
                std::cout << "size:               " << size << '\n';
                std::cout << "aligned_size_limit: " << aligned_size_limit << '\n';
                std::cout << "use_limited_size:   " << std::boolalpha << use_limited_size << '\n';
                std::cout << "number_of_launch:   " << number_of_launch << '\n';
                std::cout << "block_size:         " << block_size << '\n';
                std::cout << "items_per_block:    " << items_per_block << '\n';
                std::cout << "----------------------------------\n";
            }

            for(size_t i = 0, offset = 0; i < number_of_launch; i++, offset += limited_size)
            {
                // limited_size is of type unsigned int, so current_size also fits in an unsigned int
                // size_t is necessary as type of std::min because 'size - offset' can exceed the
                // upper limit of unsigned int and converting it can lead to wrong results
                const unsigned int current_size
                    = static_cast<unsigned int>(std::min<size_t>(size - offset, limited_size));
                const unsigned int scan_blocks    = ceiling_div(current_size, items_per_block);
                const unsigned int init_grid_size = ceiling_div(scan_blocks, block_size);

                // Start point for time measurements
                std::chrono::steady_clock::time_point start;
                if(debug_synchronous)
                {
                    std::cout << "index:            " << i << '\n';
                    std::cout << "current_size:     " << current_size << '\n';
                    std::cout << "number of blocks: " << scan_blocks << '\n';

                    start = std::chrono::steady_clock::now();
                }
                if (!Exclusive && last_keys_of_each_block_size_byte != 0 && i == 0)
                {
                    init_device_scan_by_key_kernel<<<dim3(init_grid_size),
                                                     dim3(block_size),
                                                     0,
                                                     stream>>>(
                        scan_state,
                        scan_blocks,
                        number_of_blocks - 1,
                        i > 0 ? previous_last_value : nullptr,
                        keys,
                        last_keys_of_each_block,
                        last_keys_of_each_block_size_byte
                            / sizeof(decltype(*last_keys_of_each_block)),
                        items_per_block,
                        ordered_bid);
                }
                else
                {
                    init_device_scan_by_key_kernel<<<dim3(init_grid_size),
                                                     dim3(block_size),
                                                     0,
                                                     stream>>>(scan_state,
                                                               scan_blocks,
                                                               number_of_blocks - 1,
                                                               i > 0 ? previous_last_value
                                                                     : nullptr,
                                                               ordered_bid);
                }
                ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("init_lookback_scan_state_kernel",
                                                            scan_blocks,
                                                            start);

                if(debug_synchronous)
                {
                    start = std::chrono::steady_clock::now();
                }
                ROCPRIM_RETURN_ON_ERROR(launch_device_scan_by_key<config, Determinism, Exclusive>(
                    target_arch,
                    keys + offset,
                    input + offset,
                    output + offset,
                    initial_value,
                    compare,
                    scan_op,
                    scan_state,
                    size,
                    i * number_of_blocks,
                    total_number_of_blocks,
                    i > 0 ? as_const_ptr(previous_last_value) : nullptr,
                    last_keys_of_each_block_size_byte != 0,
                    last_keys_of_each_block,
                    ordered_bid,
                    dim3(scan_blocks),
                    dim3(block_size),
                    0,
                    stream));
                ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("device_scan_by_key_kernel",
                                                            current_size,
                                                            start);
            }
            return hipSuccess;
        },
        use_sleepy_scan_variant,
        use_atomic_block_id_variant));
    return hipSuccess;
}

} // namespace detail

/// \addtogroup devicemodule
/// @{

/// \brief Parallel inclusive scan-by-key primitive for device level.
///
/// inclusive_scan_by_key function performs a device-wide inclusive prefix scan-by-key
/// operation using binary \p scan_op operator.
///
/// \par Overview
/// * Supports non-commutative scan operators. However, a scan operator should be
/// associative.
/// * When used with non-associative functions (e.g. floating point arithmetic operations):
///    - the results may be non-deterministic and/or vary in precision,
///    - and bit-wise reproducibility is not guaranteed, that is, results from multiple runs
///      using the same input values on the same device may not be bit-wise identical.
///   If deterministic behavior is required, Use \link deterministic_inclusive_scan_by_key()
///   rocprim::deterministic_inclusive_scan_by_key \endlink instead.
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * Ranges specified by \p keys_input, \p values_input, and \p values_output must have
/// at least \p size elements.
///
/// \note In-place inclusive_scan_by_key
/// In these situations, in-place inclusive_scan_by_key is supported:
/// - The buffers in `keys_input` and `values_output` are overlaped to each other, and `KeysInputIterator` and `ValuesOutputIterator` are the same.
/// - The buffers in `keys_input` and `values_output` are overlaped to each other, `KeysInputIterator` and `ValuesOutputIterator` are the different,
///   but they are both pointer types, and their "std::iterator_traits<KeysInputIterator>::value_type" is convertible to
///   "std::iterator_traits<ValuesOutputIterator>::value_type"
///
/// \tparam Config [optional] Configuration of the primitive, must be `default_config` or `scan_by_key_config`.
/// \tparam KeysInputIterator random-access iterator type of the input range. It can be
/// a simple pointer type.
/// \tparam ValuesInputIterator random-access iterator type of the input range. It can be
/// a simple pointer type.
/// \tparam ValuesOutputIterator random-access iterator type of the output range. It can be
/// a simple pointer type.
/// \tparam BinaryFunction type of binary function used for scan. Default type
/// is \p rocprim::plus<T>, where \p T is a \p value_type of \p InputIterator.
/// \tparam KeyCompareFunction type of binary function used to determine keys equality. Default type
/// is \p rocprim::equal_to<T>, where \p T is a \p value_type of \p KeysInputIterator.
/// \tparam AccType accumulator type used to propagate the scanned values. Default type
/// is value type of the input iterator.
///
/// \param [in] temporary_storage pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the scan operation.
/// \param [in,out] storage_size reference to a size (in bytes) of \p temporary_storage.
/// \param [in] keys_input iterator to the first element in the range of keys.
/// \param [in] values_input iterator to the first element in the range of values to scan.
/// \param [out] values_output iterator to the first element in the output value range.
/// \param [in] size number of element in the input range.
/// \param [in] scan_op binary operation function object that will be used for scanning
/// input values.
/// The signature of the function should be equivalent to the following:
/// <tt>T f(const T &a, const T &b);</tt>. The signature does not need to have
/// <tt>const &</tt>, but function object must not modify the objects passed to it.
/// Default is BinaryFunction().
/// \param [in] key_compare_op binary operation function object that will be used to determine keys equality.
/// The signature of the function should be equivalent to the following:
/// <tt>bool f(const T &a, const T &b);</tt>. The signature does not need to have
/// <tt>const &</tt>, but function object must not modify the objects passed to it.
/// Default is KeyCompareFunction().
/// \param [in] stream [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after successful scan; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example a device-level inclusive sum-by-key operation is performed on an array of
/// integer values (<tt>short</tt>s are scanned into <tt>int</tt>s).
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t size;           // e.g., 8
/// int *   keys_input;    // e.g., [1, 1, 2, 2, 3, 3, 3, 5]
/// short * values_input;  // e.g., [1, 2, 3, 4, 5, 6, 7, 8]
/// int *   values_output; // empty array of 8 elements
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::inclusive_scan_by_key(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys_input, values_input,
///     values_output, size,
///     rocprim::plus<int>()
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform scan-by-key
/// rocprim::inclusive_scan_by_key(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys_input, values_input,
///     values_output, size,
///     rocprim::plus<int>()
/// );
/// // values_output: [1, 3, 3, 7, 5, 11, 18, 8]
/// \endcode
/// \endparblock
template<typename Config = default_config,
         typename KeysInputIterator,
         typename ValuesInputIterator,
         typename ValuesOutputIterator,
         typename BinaryFunction
         = ::rocprim::plus<typename std::iterator_traits<ValuesInputIterator>::value_type>,
         typename KeyCompareFunction
         = ::rocprim::equal_to<typename std::iterator_traits<KeysInputIterator>::value_type>,
         typename AccType = typename std::iterator_traits<ValuesInputIterator>::value_type>
inline hipError_t inclusive_scan_by_key(void* const                temporary_storage,
                                        size_t&                    storage_size,
                                        const KeysInputIterator    keys_input,
                                        const ValuesInputIterator  values_input,
                                        const ValuesOutputIterator values_output,
                                        const size_t               size,
                                        const BinaryFunction       scan_op = BinaryFunction(),
                                        const KeyCompareFunction   key_compare_op
                                        = KeyCompareFunction(),
                                        const hipStream_t stream            = 0,
                                        const bool        debug_synchronous = false)
{
    using value_type = typename std::iterator_traits<ValuesInputIterator>::value_type;
    return detail::scan_by_key_impl<detail::lookback_scan_determinism::default_determinism,
                                    false,
                                    Config,
                                    KeysInputIterator,
                                    ValuesInputIterator,
                                    ValuesOutputIterator,
                                    value_type,
                                    BinaryFunction,
                                    KeyCompareFunction,
                                    AccType>(temporary_storage,
                                             storage_size,
                                             keys_input,
                                             values_input,
                                             values_output,
                                             value_type(),
                                             size,
                                             scan_op,
                                             key_compare_op,
                                             stream,
                                             debug_synchronous);
}

/// \brief Bitwise-reproducible parallel inclusive scan-by-key primitive for device level.
///
/// This function behaves the same as <tt>inclusive_scan_by_key()</tt>, except that unlike
/// <tt>inclusive_scan_by_key()</tt>, it provides run-to-run deterministic behavior for
/// non-associative scan operators like floating point arithmetic operations.
/// Refer to the documentation for \link inclusive_scan_by_key() rocprim::inclusive_scan_by_key \endlink
/// for a detailed description of this function.
template<typename Config = default_config,
         typename KeysInputIterator,
         typename ValuesInputIterator,
         typename ValuesOutputIterator,
         typename BinaryFunction
         = ::rocprim::plus<typename std::iterator_traits<ValuesInputIterator>::value_type>,
         typename KeyCompareFunction
         = ::rocprim::equal_to<typename std::iterator_traits<KeysInputIterator>::value_type>,
         typename AccType = typename std::iterator_traits<ValuesInputIterator>::value_type>
inline hipError_t deterministic_inclusive_scan_by_key(void* const                temporary_storage,
                                                      size_t&                    storage_size,
                                                      const KeysInputIterator    keys_input,
                                                      const ValuesInputIterator  values_input,
                                                      const ValuesOutputIterator values_output,
                                                      const size_t               size,
                                                      const BinaryFunction       scan_op
                                                      = BinaryFunction(),
                                                      const KeyCompareFunction key_compare_op
                                                      = KeyCompareFunction(),
                                                      const hipStream_t stream            = 0,
                                                      const bool        debug_synchronous = false)
{
    using value_type = typename std::iterator_traits<ValuesInputIterator>::value_type;
    return detail::scan_by_key_impl<detail::lookback_scan_determinism::deterministic,
                                    false,
                                    Config,
                                    KeysInputIterator,
                                    ValuesInputIterator,
                                    ValuesOutputIterator,
                                    value_type,
                                    BinaryFunction,
                                    KeyCompareFunction,
                                    AccType>(temporary_storage,
                                             storage_size,
                                             keys_input,
                                             values_input,
                                             values_output,
                                             value_type(),
                                             size,
                                             scan_op,
                                             key_compare_op,
                                             stream,
                                             debug_synchronous);
}

/// \brief Parallel exclusive scan-by-key primitive for device level.
///
/// inclusive_scan_by_key function performs a device-wide exclusive prefix scan-by-key
/// operation using binary \p scan_op operator.
///
/// \par Overview
/// * Supports non-commutative scan operators. However, a scan operator should be
/// associative.
/// * When used with non-associative functions (e.g. floating point arithmetic operations):
///    - the results may be non-deterministic and/or vary in precision,
///    - and bit-wise reproducibility is not guaranteed, that is, results from multiple runs
///      using the same input values on the same device may not be bit-wise identical.
///   If deterministic behavior is required, Use \link deterministic_exclusive_scan_by_key()
///   rocprim::deterministic_exclusive_scan_by_key \endlink instead.
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * Ranges specified by \p keys_input, \p values_input, and \p values_output must have
/// at least \p size elements.
///
/// \note In-place exclusive_scan_by_key
/// In-place exclusive_scan_by_key is supported.
///
/// \tparam Config [optional] Configuration of the primitive, must be `default_config` or `scan_by_key_config`.
/// \tparam KeysInputIterator random-access iterator type of the input range. It can be
/// a simple pointer type.
/// \tparam ValuesInputIterator random-access iterator type of the input range. It can be
/// a simple pointer type.
/// \tparam ValuesOutputIterator random-access iterator type of the output range. It can be
/// a simple pointer type.
/// \tparam InitValueType type of the initial value.
/// \tparam BinaryFunction type of binary function used for scan. Default type
/// is \p rocprim::plus<T>, where \p T is a \p value_type of \p InputIterator.
/// \tparam KeyCompareFunction type of binary function used to determine keys equality. Default type
/// is \p rocprim::equal_to<T>, where \p T is a \p value_type of \p KeysInputIterator.
/// \tparam AccType accumulator type used to propagate the scanned values. Default type
/// is 'InitValueType', unless it's 'rocprim::future_value'. Then it will be the wrapped input type.
///
/// \param [in] temporary_storage pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the scan operation.
/// \param [in,out] storage_size reference to a size (in bytes) of \p temporary_storage.
/// \param [in] keys_input iterator to the first element in the range of keys.
/// \param [in] values_input iterator to the first element in the range of values to scan.
/// \param [out] values_output iterator to the first element in the output value range.
/// \param [in] initial_value initial value to start the scan.
/// A rocpim::future_value may be passed to use a value that will be later computed.
/// \param [in] size number of element in the input range.
/// \param [in] scan_op binary operation function object that will be used for scanning
/// input values.
/// The signature of the function should be equivalent to the following:
/// <tt>T f(const T &a, const T &b);</tt>. The signature does not need to have
/// <tt>const &</tt>, but function object must not modify the objects passed to it.
/// Default is BinaryFunction().
/// \param [in] key_compare_op binary operation function object that will be used to determine keys equality.
/// The signature of the function should be equivalent to the following:
/// <tt>bool f(const T &a, const T &b);</tt>. The signature does not need to have
/// <tt>const &</tt>, but function object must not modify the objects passed to it.
/// Default is KeyCompareFunction().
/// \param [in] stream [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after successful scan; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example a device-level inclusive sum-by-key operation is performed on an array of
/// integer values (<tt>short</tt>s are scanned into <tt>int</tt>s).
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t size;           // e.g., 8
/// int *   keys_input;    // e.g., [1, 1, 1, 2, 2, 3, 3, 4]
/// short * values_input;  // e.g., [1, 2, 3, 4, 5, 6, 7, 8]
/// int start_value;       // e.g., 9
/// int *   values_output; // empty array of 8 elements
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::exclusive_scan_by_key(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys_input, values_input,
///     values_output, start_value,
///     size,rocprim::plus<int>()
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform scan-by-key
/// rocprim::exclusive_scan_by_key(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys_input, values_input,
///     values_output, start_value,
///     size,rocprim::plus<int>()
/// );
/// // values_output: [9, 10, 12, 9, 13, 9, 15, 9]
/// \endcode
/// \endparblock
template<typename Config = default_config,
         typename KeysInputIterator,
         typename ValuesInputIterator,
         typename ValuesOutputIterator,
         typename InitialValueType,
         typename BinaryFunction
         = ::rocprim::plus<typename std::iterator_traits<ValuesInputIterator>::value_type>,
         typename KeyCompareFunction
         = ::rocprim::equal_to<typename std::iterator_traits<KeysInputIterator>::value_type>,
         typename AccType = detail::input_type_t<InitialValueType>>
inline hipError_t exclusive_scan_by_key(void* const                temporary_storage,
                                        size_t&                    storage_size,
                                        const KeysInputIterator    keys_input,
                                        const ValuesInputIterator  values_input,
                                        const ValuesOutputIterator values_output,
                                        const InitialValueType     initial_value,
                                        const size_t               size,
                                        const BinaryFunction       scan_op = BinaryFunction(),
                                        const KeyCompareFunction   key_compare_op
                                        = KeyCompareFunction(),
                                        const hipStream_t stream            = 0,
                                        const bool        debug_synchronous = false)
{
    return detail::scan_by_key_impl<detail::lookback_scan_determinism::default_determinism,
                                    true,
                                    Config,
                                    KeysInputIterator,
                                    ValuesInputIterator,
                                    ValuesOutputIterator,
                                    InitialValueType,
                                    BinaryFunction,
                                    KeyCompareFunction,
                                    AccType>(temporary_storage,
                                             storage_size,
                                             keys_input,
                                             values_input,
                                             values_output,
                                             initial_value,
                                             size,
                                             scan_op,
                                             key_compare_op,
                                             stream,
                                             debug_synchronous);
}

/// \brief Bitwise-reproducible parallel exclusive scan-by-key primitive for device level.
///
/// This function behaves the same as <tt>exclusive_scan_by_key()</tt>, except that unlike
/// <tt>exclusive_scan_by_key()</tt>, it provides run-to-run deterministic behavior for
/// non-associative scan operators like floating point arithmetic operations.
/// Refer to the documentation for \link exclusive_scan_by_key() rocprim::exclusive_scan_by_key \endlink
/// for a detailed description of this function.
template<typename Config = default_config,
         typename KeysInputIterator,
         typename ValuesInputIterator,
         typename ValuesOutputIterator,
         typename InitialValueType,
         typename BinaryFunction
         = ::rocprim::plus<typename std::iterator_traits<ValuesInputIterator>::value_type>,
         typename KeyCompareFunction
         = ::rocprim::equal_to<typename std::iterator_traits<KeysInputIterator>::value_type>,
         typename AccType = detail::input_type_t<InitialValueType>>
inline hipError_t deterministic_exclusive_scan_by_key(void* const                temporary_storage,
                                                      size_t&                    storage_size,
                                                      const KeysInputIterator    keys_input,
                                                      const ValuesInputIterator  values_input,
                                                      const ValuesOutputIterator values_output,
                                                      const InitialValueType     initial_value,
                                                      const size_t               size,
                                                      const BinaryFunction       scan_op
                                                      = BinaryFunction(),
                                                      const KeyCompareFunction key_compare_op
                                                      = KeyCompareFunction(),
                                                      const hipStream_t stream            = 0,
                                                      const bool        debug_synchronous = false)
{
    return detail::scan_by_key_impl<detail::lookback_scan_determinism::deterministic,
                                    true,
                                    Config,
                                    KeysInputIterator,
                                    ValuesInputIterator,
                                    ValuesOutputIterator,
                                    InitialValueType,
                                    BinaryFunction,
                                    KeyCompareFunction,
                                    AccType>(temporary_storage,
                                             storage_size,
                                             keys_input,
                                             values_input,
                                             values_output,
                                             initial_value,
                                             size,
                                             scan_op,
                                             key_compare_op,
                                             stream,
                                             debug_synchronous);
}

/// @}
// end of group devicemodule

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_SCAN_BY_KEY_HPP_
