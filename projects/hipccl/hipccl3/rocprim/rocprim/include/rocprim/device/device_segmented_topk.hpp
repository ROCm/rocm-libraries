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

#ifndef ROCPRIM_DEVICE_DEVICE_SEGMENTED_TOPK_HPP_
#define ROCPRIM_DEVICE_DEVICE_SEGMENTED_TOPK_HPP_

#include "../config.hpp"

#include "detail/device_segmented_topk.hpp"
#include "detail/device_segmented_topk_air.hpp"

#include <iterator>
#include <type_traits>

/// \addtogroup devicemodule
/// @{

BEGIN_ROCPRIM_NAMESPACE

#ifndef DOXYGEN_SHOULD_SKIP_THIS // Do not document

namespace detail
{

template<typename KeysInputIterator, typename BinaryFunction, typename Decomposer>
struct radix_segmented_topk_condition_checker
{
    using key_type = typename std::iterator_traits<KeysInputIterator>::value_type;

    static constexpr bool is_custom_decomposer
        = !std::is_same<Decomposer, rocprim::identity_decomposer>::value;
    static constexpr bool descending
        = std::is_same<BinaryFunction, rocprim::greater<key_type>>::value
          || std::is_same<BinaryFunction, rocprim::greater<void>>::value;
    static constexpr bool ascending = std::is_same<BinaryFunction, rocprim::less<key_type>>::value
                                      || std::is_same<BinaryFunction, rocprim::less<void>>::value;
    static constexpr bool is_radix_key_fundamental
        = rocprim::traits::radix_key_codec::radix_key_fundamental<key_type>::value;
    static constexpr bool use_radix
        = (is_radix_key_fundamental || is_custom_decomposer) && (descending || ascending);
};

template<bool UseRadix,
         class Config,
         bool Ordered,
         bool Deterministic,
         bool Stable,
         bool Descending,
         class KeysInputIterator,
         class KeysOutputIterator,
         class ValuesInputIterator,
         class ValuesOutputIterator,
         class OffsetIterator,
         class SizeIn,
         class SizeOut,
         class BinaryFunction,
         class Decomposer>
ROCPRIM_INLINE
hipError_t topk_segmented_impl(void*                                 temporary_storage,
                               size_t&                               storage_size,
                               const KeysInputIterator               keys_input,
                               const KeysOutputIterator              keys_output,
                               const ValuesInputIterator             values_input,
                               const ValuesOutputIterator            values_output,
                               const SizeIn                          size,
                               SizeOut                               K,
                               const unsigned int                    segments,
                               const OffsetIterator                  begin_offsets,
                               const OffsetIterator                  end_offsets,
                               [[maybe_unused]] const BinaryFunction compare_function
                               = BinaryFunction(),
                               const Decomposer  decomposer        = {},
                               const hipStream_t stream            = 0,
                               const bool        debug_synchronous = false)
{
    using key_type      = typename std::iterator_traits<KeysInputIterator>::value_type;
    using value_type    = typename std::iterator_traits<ValuesInputIterator>::value_type;
    using common_size_t = typename std::common_type<decltype(size), decltype(K)>::type;
    static_assert(std::is_integral<common_size_t>::value, "Size and K must be integral types.");
    static_assert(
        std::is_same<key_type,
                     typename std::iterator_traits<KeysOutputIterator>::value_type>::value,
        "KeysInputIterator and KeysOutputIterator must have the same value_type");
    static_assert(
        std::is_same<value_type,
                     typename std::iterator_traits<ValuesOutputIterator>::value_type>::value,
        "ValuesInputIterator and ValuesOutputIterator must have the same value_type");

    // Default is radix based segmented topk, check we can actually use it
    using radix_checker
        = radix_segmented_topk_condition_checker<KeysInputIterator, BinaryFunction, Decomposer>;

    if constexpr(UseRadix)
    {
        // When we request radix-based sorting, check if the input types actually support
        // the radix decomposers.
        static_assert(radix_checker::use_radix,
                      "'UseRadix = true' requires the key type to support radix-based sorting");
    }

    if (K < 0)
    {
        // K should be >= 0.
        return hipErrorInvalidValue;
    }

    // Naive segmented top-k:
    // * Radix-sortable types only
    // * Ordered
    // * NOT deterministic
    // * Stable
    // TODO: naive *could* support a custom decomposer, but this is currently not implemented.
    constexpr bool can_use_naive
        = UseRadix && !Deterministic && !radix_checker::is_custom_decomposer;

    // Segmented top-k air:
    // * Radix-sortable types only
    // * NOT ordered
    // * NOT deterministic
    // * NOT stable
    constexpr bool can_use_air = UseRadix && !Ordered && !Stable && !Deterministic;

    if constexpr(can_use_air)
    {
        return detail::device_segmented_topk_air<Config, !Descending>(temporary_storage,
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
                                                                      debug_synchronous);
    }
    else if constexpr(can_use_naive)
    {
        return detail::device_segmented_topk_impl<KeysInputIterator,
                                                  KeysOutputIterator,
                                                  ValuesInputIterator,
                                                  ValuesOutputIterator,
                                                  OffsetIterator,
                                                  SizeIn,
                                                  SizeOut,
                                                  Descending,
                                                  Decomposer>::impl(temporary_storage,
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
                                                                    debug_synchronous);
    }
    else
    {
        static_assert(false, "The requested segmented top-k implementation is not supported!");
    }

    return hipErrorNotSupported;
}

} // namespace detail

#endif // DOXYGEN_SHOULD_SKIP_THIS

/// \brief Find the largest or smallest K elements of each segment from an array of segments of keys.
///
/// Returns the K largest or smallest elements for each segment. These may be be in any order.
///
/// \tparam Config [optional] configuration of the primitive, must be `default_config` or `segmented_topk_config`(TODO).
/// \tparam Descending [optional] determines the starting direction. If \p true, select the largest K elements, and vice versa.
/// \tparam Ordered [optional] determines whether the output results are sorted by size.
/// \note Ordered output is not supported yet. This feature will be added in the future.
/// \tparam Deterministic [optional] to ensure that the results are exactly the same every time.
/// \note Deterministic output is not supported yet. This feature will be added in the future.
/// \tparam Stable [optional] determines whether elements in the output are arranged according to their relative position in the input.
/// \tparam Decomposer [optional] the type of the decomposer functor.
/// \tparam KeysInputIterator [optional] random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam KeysOutputIterator [optional] random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
/// \tparam SizeIn [optional] integral type that represents the problem size.
/// \tparam SizeOut [optional] integral type that counts the number of output elements.
///
/// \param [in] temporary_storage pointer to device-accessible temporary storage.
/// If a null pointer is provided, the required allocation size (in bytes) is written
/// to \p storage_size, and the function returns without performing the sort operation.
/// \param [in,out] storage_size reference to a size (in bytes) of \p temporary_storage.
/// \param [in] keys_input pointer to the first element in the input range. Must have at least \p size elements.
/// \param [out] keys_output pointer to the first element in the output range. Must have at least \p k*segments elements.
/// \param [in] size number of elements in the input range.
/// \param [in] K number of elements to be selected from each segment from the input.
/// \param [in] segments number of segments in the input. Each segment is a contiguous subsequence of data. Must be smaller or equal to \p size.
/// that will be processed independently.
/// \param [in] begin_offsets iterator to the first element in the range of beginning offsets.
/// Points to the starting indices of each segment within the input data array.
/// The i-th segment starts at index `begin_offsets[i]` and ends at `end_offsets[i]`.
/// For n segments, `begin_offsets` must have at least n elements.
/// \param [in] end_offsets iterator to the first element in the range of ending offsets.
/// Points to the end indices (exclusive) of each segment within the input data array.
/// For n segments, `end_offsets` must have at least n elements.
/// \param [in] decomposer decomposer functor that produces a tuple of references from the
/// input key type.
/// \param [in] stream [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous [optional] if \p true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after a successful top-segmented-k operation; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example a device-level ascending segmented top-k is performed on an array.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
/// // Prepare input and output (declare pointers, allocate device memory, etc.)
/// size_t input_size;      // e.g., 8
/// size_t k;               // e.g., 2
/// int * input;            // e.g., [2, 3, 4, -1, -2, -3, 0, 5]
/// int * output;           // empty array of 4 (segments * k) elements
/// size_t segments;        // e.g., 2
/// int * begin_offsets;    // e.g., [0, 2]
/// int * end_offsets;      // e.g., [2, 8]
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::topk_segmented(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, output, input_size, k
/// );
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
/// // perform topk
/// rocprim::topk_segmented(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, output, input_size, k
/// );
/// // output should be [2, 3, 4, 5] (order is not guaranteed)
/// \endcode
/// \endparblock
template<class Config = default_config,
         bool Descending,
         bool Ordered       = true,
         bool Deterministic = false,
         bool Stable        = false,
         class Decomposer   = ::rocprim::identity_decomposer,
         class KeysInputIterator,
         class KeysOutputIterator,
         class OffsetIterator,
         class SizeIn,
         class SizeOut>
ROCPRIM_INLINE
hipError_t segmented_topk(void*                    temporary_storage,
                          size_t&                  storage_size,
                          const KeysInputIterator  keys_input,
                          const KeysOutputIterator keys_output,
                          const SizeIn             size,
                          const SizeOut            K,
                          const unsigned int       segments,
                          const OffsetIterator     begin_offsets,
                          const OffsetIterator     end_offsets,
                          Decomposer               decomposer        = {},
                          const hipStream_t        stream            = 0,
                          const bool               debug_synchronous = false)
{
    using compare_function = std::conditional_t<
        Descending,
        rocprim::greater<typename std::iterator_traits<KeysInputIterator>::value_type>,
        rocprim::less<typename std::iterator_traits<KeysInputIterator>::value_type>>;
    return detail::topk_segmented_impl<true, Config, Ordered, Deterministic, Stable, Descending>(

        temporary_storage,
        storage_size,
        keys_input,
        keys_output,
        static_cast<rocprim::empty_type*>(nullptr),
        static_cast<rocprim::empty_type*>(nullptr),
        size,
        K,
        segments,
        begin_offsets,
        end_offsets,
        compare_function(),
        decomposer,
        stream,
        debug_synchronous);
}

/// \brief Find the largest or smallest K elements of each segment from an array of values based on their corresponding keys.
///
/// Returns the K largest or smallest (key, value) pairs for each segment. These maybe be in any order.
///
/// \tparam Config [optional] configuration of the primitive, must be `default_config` or `segmented_topk_config`(TODO).
/// \tparam Descending [optional] determines the starting direction. If \p true, select the largest K elements, and vice versa.
/// \tparam Ordered [optional] determines whether the output results are sorted by size.
/// \note Ordered output is not supported yet. This feature will be added in the future.
/// \tparam Deterministic [optional] to ensure that the results are exactly the same every time.
/// \note Deterministic output is not supported yet. This feature will be added in the future.
/// \tparam Stable [optional] determines whether elements in the output are arranged according to their relative position in the input.
/// \tparam Decomposer [optional] the type of the decomposer functor.
/// \tparam KeysInputIterator [optional] random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam KeysOutputIterator [optional] random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
/// \tparam ValuesInputIterator [optional] random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam ValuesOutputIterator [optional] random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
/// \tparam SizeIn [optional] integral type that represents the problem size.
/// \tparam SizeOut [optional] integral type that counts the number of output elements.
///
/// \param [in] temporary_storage pointer to device-accessible temporary storage.
/// If a null pointer is provided, the required allocation size (in bytes) is written
/// to \p storage_size, and the function returns without performing the sort operation.
/// \param [in,out] storage_size reference to a size (in bytes) of \p temporary_storage.
/// \param [in] keys_input pointer to the first element in the input range. Must have at least \p size elements.
/// \param [out] keys_output pointer to the first element in the output range. Must have at least \p k*segments elements.
/// \param [in] values_input pointer to the first value in the output range. Must have at least \p size elements.
/// \param [out] values_output pointer to the first value in the output range. Must have at least \p k*segments elements.
/// \param [in] size number of elements in the input range.
/// \param [in] K number of elements to be selected from the input.
/// \param [in] segments number of segments in the input. Each segment is a contiguous subsequence of data
/// that will be processed independently.
/// \param [in] begin_offsets iterator to the first element in the range of beginning offsets.
/// Points to the starting indices of each segment within the input data array.
/// The i-th segment starts at index `begin_offsets[i]` and ends at `end_offsets[i]`.
/// For n segments, `begin_offsets` must have at least n elements.
/// \param [in] end_offsets iterator to the first element in the range of ending offsets.
/// Points to the end indices (exclusive) of each segment within the input data array.
/// For n segments, `end_offsets` must have at least n elements.
/// \param [in] decomposer decomposer functor that produces a tuple of references from the
/// input key type.
/// \param [in] stream [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous [optional] if \p true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after a successful top-k operation; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example a device-level ascending top-k is performed on an array.
///
/// TODO: The full example is [on GitHub](https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocprim/example/rocprim/device/example_device_topk.cpp). TODO
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
/// // Prepare input and output (declare pointers, allocate device memory, etc.)
/// size_t input_size;      // e.g., 8
/// size_t k;               // e.g., 2
/// int * input_keys;       // e.g., [2, 3, 4, -1, -2, -3, 0, 5]
/// int * output_keys;      // empty array of 4 (segments * k) elements
/// int * input_vals;       // e.g., [0, 1, 2, 3, 4, 5, 6, 7]
/// int * output_vals;      // empty array of 4 (segments * k) elements
/// size_t segments;        // e.g., 2
/// int * begin_offsets;    // e.g., [0, 2]
/// int * end_offsets;      // e.g., [2, 8]
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::topk_segmented_pairs(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input_keys, output_keys, input_vals, output_vals, input_size, k
/// );
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
/// // perform topk
/// rocprim::topk_segmented_pairs(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input_keys, output_keys, input_vals, output_vals, input_size, k
///     segments, begin_offsets, end_offsets
/// );
/// // output_keys should be [2, 3, 4, 5] (order is not guaranteed)
/// // output_vals should be [0, 1, 2, 7] (order matches output_keys)
/// \endcode
/// \endparblock
template<class Config = default_config,
         bool Descending,
         bool Ordered       = true,
         bool Deterministic = false,
         bool Stable        = false,
         class Decomposer   = rocprim::identity_decomposer,
         class KeysInputIterator,
         class KeysOutputIterator,
         class ValuesInputIterator,
         class ValuesOutputIterator,
         class OffsetIterator,
         class SizeIn,
         class SizeOut>
ROCPRIM_INLINE
hipError_t segmented_topk_pairs(void*                      temporary_storage,
                                size_t&                    storage_size,
                                const KeysInputIterator    keys_input,
                                const KeysOutputIterator   keys_output,
                                const ValuesInputIterator  values_input,
                                const ValuesOutputIterator values_output,
                                const SizeIn               size,
                                const SizeOut              K,
                                const unsigned int         segments,
                                const OffsetIterator       begin_offsets,
                                const OffsetIterator       end_offsets,
                                const Decomposer           decomposer        = {},
                                const hipStream_t          stream            = 0,
                                const bool                 debug_synchronous = false)
{
    using compare_function = std::conditional_t<
        Descending,
        rocprim::greater<typename std::iterator_traits<KeysInputIterator>::value_type>,
        rocprim::less<typename std::iterator_traits<KeysInputIterator>::value_type>>;
    return detail::topk_segmented_impl<true, Config, Ordered, Deterministic, Stable, Descending>(

        temporary_storage,
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
        compare_function(),
        decomposer,
        stream,
        debug_synchronous);
}

END_ROCPRIM_NAMESPACE

/// @}
// end of group devicemodule

#endif // ROCPRIM_DEVICE_DEVICE_SEGMENTED_TOPK_HPP_
