// Copyright (c) 2017-2021 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_BLOCK_BLOCK_RADIX_SORT_HPP_
#define ROCPRIM_BLOCK_BLOCK_RADIX_SORT_HPP_

#include <type_traits>

#include "../config.hpp"
#include "../detail/various.hpp"
#include "../detail/radix_sort.hpp"
#include "../warp/detail/warp_scan_crosslane.hpp"

#include "../intrinsics.hpp"
#include "../functional.hpp"
#include "../types.hpp"

#include "block_exchange.hpp"
#include "block_radix_rank.hpp"

/// \addtogroup blockmodule
/// @{

BEGIN_ROCPRIM_NAMESPACE

/// \brief The block_radix_sort class is a block level parallel primitive which provides
/// methods for sorting of items (keys or key-value pairs) partitioned across threads in a block
/// using radix sort algorithm.
///
/// \tparam Key - the key type.
/// \tparam BlockSize - the number of threads in a block.
/// \tparam ItemsPerThread - the number of items contributed by each thread.
/// \tparam Value - the value type. Default type empty_type indicates
/// a keys-only sort.
///
/// \par Overview
/// * \p Key type must be an arithmetic type (that is, an integral type or a floating-point
/// type).
/// * Performance depends on \p BlockSize and \p ItemsPerThread.
///   * It is usually better for \p BlockSize to be a multiple of the size of the hardware warp.
///   * It is usually increased when \p ItemsPerThread is greater than one. However, when there
///   are too many items per thread, each thread may need so much registers and/or shared memory
///   that occupancy will fall too low, decreasing the performance.
///   * If \p Key is an integer type and the range of keys is known in advance, the performance
///   can be improved by setting \p begin_bit and \p end_bit, for example if all keys are in range
///   [100, 10000], <tt>begin_bit = 0</tt> and <tt>end_bit = 14</tt> will cover the whole range.
///
/// \par Examples
/// \parblock
/// In the examples radix sort is performed on a block of 256 threads, each thread provides
/// eight \p int value, results are returned using the same array as for input.
///
/// \code{.cpp}
/// __global__ void example_kernel(...)
/// {
///     // specialize block_radix_sort for int, block of 256 threads,
///     // and eight items per thread; key-only sort
///     using block_rsort_int = rocprim::block_radix_sort<int, 256, 8>;
///     // allocate storage in shared memory
///     __shared__ block_rsort_int::storage_type storage;
///
///     int input[8] = ...;
///     // execute block radix sort (ascending)
///     block_rsort_int().sort(
///         input,
///         storage
///     );
///     ...
/// }
/// \endcode
/// \endparblock
template<
    class Key,
    unsigned int BlockSizeX,
    unsigned int ItemsPerThread,
    class Value = empty_type,
    unsigned int BlockSizeY = 1,
    unsigned int BlockSizeZ = 1
>
class block_radix_sort
{
    static constexpr unsigned int BlockSize           = BlockSizeX * BlockSizeY * BlockSizeZ;
    static constexpr bool         with_values         = !std::is_same<Value, empty_type>::value;
    static constexpr unsigned int radix_bits_per_pass = 4;

    using bit_key_type    = typename ::rocprim::detail::radix_key_codec<Key>::bit_key_type;
    using block_rank_type = ::rocprim::block_radix_rank<BlockSizeX,
                                                        radix_bits_per_pass,
                                                        block_radix_rank_algorithm::basic_memoize,
                                                        BlockSizeY,
                                                        BlockSizeZ>;
    using bit_keys_exchange_type = ::rocprim::block_exchange<bit_key_type, BlockSizeX, ItemsPerThread, BlockSizeY, BlockSizeZ>;
    using values_exchange_type
        = ::rocprim::block_exchange<Value, BlockSizeX, ItemsPerThread, BlockSizeY, BlockSizeZ>;

    // Struct used for creating a raw_storage object for this primitive's temporary storage.
    union storage_type_
    {
        typename bit_keys_exchange_type::storage_type bit_keys_exchange;
        typename values_exchange_type::storage_type   values_exchange;
        typename block_rank_type::storage_type        rank;
    };

public:

    /// \brief Struct used to allocate a temporary memory that is required for thread
    /// communication during operations provided by related parallel primitive.
    ///
    /// Depending on the implemention the operations exposed by parallel primitive may
    /// require a temporary storage for thread communication. The storage should be allocated
    /// using keywords <tt>__shared__</tt>. It can be aliased to
    /// an externally allocated memory, or be a part of a union type with other storage types
    /// to increase shared memory reusability.
    #ifndef DOXYGEN_SHOULD_SKIP_THIS // hides storage_type implementation for Doxygen
    using storage_type = detail::raw_storage<storage_type_>;
    #else
    using storage_type = storage_type_; // only for Doxygen
    #endif

    /// \brief Performs ascending radix sort over keys partitioned across threads in a block.
    ///
    /// \param [in, out] keys - reference to an array of keys provided by a thread.
    /// \param [in] storage - reference to a temporary storage object of type storage_type.
    /// \param [in] begin_bit - [optional] index of the first (least significant) bit used in
    /// key comparison. Must be in range <tt>[0; 8 * sizeof(Key))</tt>. Default value: \p 0.
    /// \param [in] end_bit - [optional] past-the-end index (most significant) bit used in
    /// key comparison. Must be in range <tt>(begin_bit; 8 * sizeof(Key)]</tt>. Default
    /// value: \p <tt>8 * sizeof(Key)</tt>.
    ///
    /// \par Storage reusage
    /// Synchronization barrier should be placed before \p storage is reused
    /// or repurposed: \p __syncthreads() or \p rocprim::syncthreads().
    ///
    /// \par Examples
    /// \parblock
    /// In the examples radix sort is performed on a block of 128 threads, each thread provides
    /// two \p float value, results are returned using the same array as for input.
    ///
    /// \code{.cpp}
    /// __global__ void example_kernel(...)
    /// {
    ///     // specialize block_radix_sort for float, block of 128 threads,
    ///     // and two items per thread; key-only sort
    ///     using block_rsort_float = rocprim::block_radix_sort<float, 128, 2>;
    ///     // allocate storage in shared memory
    ///     __shared__ block_rsort_float::storage_type storage;
    ///
    ///     float input[2] = ...;
    ///     // execute block radix sort (ascending)
    ///     block_rsort_float().sort(
    ///         input,
    ///         storage
    ///     );
    ///     ...
    /// }
    /// \endcode
    ///
    /// If the \p input values across threads in a block are <tt>{[256, 255], ..., [4, 3], [2, 1]}}</tt>, then
    /// then after sort they will be equal <tt>{[1, 2], [3, 4]  ..., [255, 256]}</tt>.
    /// \endparblock
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort(Key (&keys)[ItemsPerThread],
              storage_type& storage,
              unsigned int begin_bit = 0,
              unsigned int end_bit = 8 * sizeof(Key))
    {
        empty_type values[ItemsPerThread];
        sort_impl<false>(keys, values, storage, begin_bit, end_bit);
    }

    /// \overload
    /// \brief Performs ascending radix sort over keys partitioned across threads in a block.
    ///
    /// * This overload does not accept storage argument. Required shared memory is
    /// allocated by the method itself.
    ///
    /// \param [in, out] keys - reference to an array of keys provided by a thread.
    /// \param [in] begin_bit - [optional] index of the first (least significant) bit used in
    /// key comparison. Must be in range <tt>[0; 8 * sizeof(Key))</tt>. Default value: \p 0.
    /// \param [in] end_bit - [optional] past-the-end index (most significant) bit used in
    /// key comparison. Must be in range <tt>(begin_bit; 8 * sizeof(Key)]</tt>. Default
    /// value: \p <tt>8 * sizeof(Key)</tt>.
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void sort(Key (&keys)[ItemsPerThread],
              unsigned int begin_bit = 0,
              unsigned int end_bit = 8 * sizeof(Key))
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        sort(keys, storage, begin_bit, end_bit);
    }

    /// \brief Performs descending radix sort over keys partitioned across threads in a block.
    ///
    /// \param [in, out] keys - reference to an array of keys provided by a thread.
    /// \param [in] storage - reference to a temporary storage object of type storage_type.
    /// \param [in] begin_bit - [optional] index of the first (least significant) bit used in
    /// key comparison. Must be in range <tt>[0; 8 * sizeof(Key))</tt>. Default value: \p 0.
    /// \param [in] end_bit - [optional] past-the-end index (most significant) bit used in
    /// key comparison. Must be in range <tt>(begin_bit; 8 * sizeof(Key)]</tt>. Default
    /// value: \p <tt>8 * sizeof(Key)</tt>.
    ///
    /// \par Storage reusage
    /// Synchronization barrier should be placed before \p storage is reused
    /// or repurposed: \p __syncthreads() or \p rocprim::syncthreads().
    ///
    /// \par Examples
    /// \parblock
    /// In the examples radix sort is performed on a block of 128 threads, each thread provides
    /// two \p float value, results are returned using the same array as for input.
    ///
    /// \code{.cpp}
    /// __global__ void example_kernel(...)
    /// {
    ///     // specialize block_radix_sort for float, block of 128 threads,
    ///     // and two items per thread; key-only sort
    ///     using block_rsort_float = rocprim::block_radix_sort<float, 128, 2>;
    ///     // allocate storage in shared memory
    ///     __shared__ block_rsort_float::storage_type storage;
    ///
    ///     float input[2] = ...;
    ///     // execute block radix sort (descending)
    ///     block_rsort_float().sort_desc(
    ///         input,
    ///         storage
    ///     );
    ///     ...
    /// }
    /// \endcode
    ///
    /// If the \p input values across threads in a block are <tt>{[1, 2], [3, 4]  ..., [255, 256]}</tt>,
    /// then after sort they will be equal <tt>{[256, 255], ..., [4, 3], [2, 1]}</tt>.
    /// \endparblock
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort_desc(Key (&keys)[ItemsPerThread],
                   storage_type& storage,
                   unsigned int begin_bit = 0,
                   unsigned int end_bit = 8 * sizeof(Key))
    {
        empty_type values[ItemsPerThread];
        sort_impl<true>(keys, values, storage, begin_bit, end_bit);
    }

    /// \overload
    /// \brief Performs descending radix sort over keys partitioned across threads in a block.
    ///
    /// * This overload does not accept storage argument. Required shared memory is
    /// allocated by the method itself.
    ///
    /// \param [in, out] keys - reference to an array of keys provided by a thread.
    /// \param [in] begin_bit - [optional] index of the first (least significant) bit used in
    /// key comparison. Must be in range <tt>[0; 8 * sizeof(Key))</tt>. Default value: \p 0.
    /// \param [in] end_bit - [optional] past-the-end index (most significant) bit used in
    /// key comparison. Must be in range <tt>(begin_bit; 8 * sizeof(Key)]</tt>. Default
    /// value: \p <tt>8 * sizeof(Key)</tt>.
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void sort_desc(Key (&keys)[ItemsPerThread],
                   unsigned int begin_bit = 0,
                   unsigned int end_bit = 8 * sizeof(Key))
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        sort_desc(keys, storage, begin_bit, end_bit);
    }

    /// \brief Performs ascending radix sort over key-value pairs partitioned across
    /// threads in a block.
    ///
    /// \pre Method is enabled only if \p Value type is different than empty_type.
    ///
    /// \param [in, out] keys - reference to an array of keys provided by a thread.
    /// \param [in, out] values - reference to an array of values provided by a thread.
    /// \param [in] storage - reference to a temporary storage object of type storage_type.
    /// \param [in] begin_bit - [optional] index of the first (least significant) bit used in
    /// key comparison. Must be in range <tt>[0; 8 * sizeof(Key))</tt>. Default value: \p 0.
    /// \param [in] end_bit - [optional] past-the-end index (most significant) bit used in
    /// key comparison. Must be in range <tt>(begin_bit; 8 * sizeof(Key)]</tt>. Default
    /// value: \p <tt>8 * sizeof(Key)</tt>.
    ///
    /// \par Storage reusage
    /// Synchronization barrier should be placed before \p storage is reused
    /// or repurposed: \p __syncthreads() or \p rocprim::syncthreads().
    ///
    /// \par Examples
    /// \parblock
    /// In the examples radix sort is performed on a block of 128 threads, each thread provides
    /// two key-value <tt>int</tt>-<tt>float</tt> pairs, results are returned using the same
    /// arrays as for input.
    ///
    /// \code{.cpp}
    /// __global__ void example_kernel(...)
    /// {
    ///     // specialize block_radix_sort for int-float pairs, block of 128
    ///     // threads, and two items per thread
    ///     using block_rsort_ii = rocprim::block_radix_sort<int, 128, 2, int>;
    ///     // allocate storage in shared memory
    ///     __shared__ block_rsort_ii::storage_type storage;
    ///
    ///     int keys[2] = ...;
    ///     float values[2] = ...;
    ///     // execute block radix sort-by-key (ascending)
    ///     block_rsort_ii().sort(
    ///         keys, values,
    ///         storage
    ///     );
    ///     ...
    /// }
    /// \endcode
    ///
    /// If the \p keys across threads in a block are <tt>{[256, 255], ..., [4, 3], [2, 1]}</tt> and
    /// the \p values are <tt>{[1, 1], [2, 2]  ..., [128, 128]}</tt>, then after sort the \p keys
    /// will be equal <tt>{[1, 2], [3, 4]  ..., [255, 256]}</tt> and the \p values will be
    /// equal <tt>{[128, 128], [127, 127]  ..., [2, 2], [1, 1]}</tt>.
    /// \endparblock
    template<bool WithValues = with_values>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort(Key (&keys)[ItemsPerThread],
              typename std::enable_if<WithValues, Value>::type (&values)[ItemsPerThread],
              storage_type& storage,
              unsigned int begin_bit = 0,
              unsigned int end_bit = 8 * sizeof(Key))
    {
        sort_impl<false>(keys, values, storage, begin_bit, end_bit);
    }

    /// \overload
    /// \brief Performs ascending radix sort over key-value pairs partitioned across
    /// threads in a block.
    ///
    /// * This overload does not accept storage argument. Required shared memory is
    /// allocated by the method itself.
    ///
    /// \pre Method is enabled only if \p Value type is different than empty_type.
    ///
    /// \param [in, out] keys - reference to an array of keys provided by a thread.
    /// \param [in, out] values - reference to an array of values provided by a thread.
    /// \param [in] begin_bit - [optional] index of the first (least significant) bit used in
    /// key comparison. Must be in range <tt>[0; 8 * sizeof(Key))</tt>. Default value: \p 0.
    /// \param [in] end_bit - [optional] past-the-end index (most significant) bit used in
    /// key comparison. Must be in range <tt>(begin_bit; 8 * sizeof(Key)]</tt>. Default
    /// value: \p <tt>8 * sizeof(Key)</tt>.
    template<bool WithValues = with_values>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void sort(Key (&keys)[ItemsPerThread],
              typename std::enable_if<WithValues, Value>::type (&values)[ItemsPerThread],
              unsigned int begin_bit = 0,
              unsigned int end_bit = 8 * sizeof(Key))
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        sort(keys, values, storage, begin_bit, end_bit);
    }

    /// \brief Performs descending radix sort over key-value pairs partitioned across
    /// threads in a block.
    ///
    /// \pre Method is enabled only if \p Value type is different than empty_type.
    ///
    /// \param [in, out] keys - reference to an array of keys provided by a thread.
    /// \param [in, out] values - reference to an array of values provided by a thread.
    /// \param [in] storage - reference to a temporary storage object of type storage_type.
    /// \param [in] begin_bit - [optional] index of the first (least significant) bit used in
    /// key comparison. Must be in range <tt>[0; 8 * sizeof(Key))</tt>. Default value: \p 0.
    /// \param [in] end_bit - [optional] past-the-end index (most significant) bit used in
    /// key comparison. Must be in range <tt>(begin_bit; 8 * sizeof(Key)]</tt>. Default
    /// value: \p <tt>8 * sizeof(Key)</tt>.
    ///
    /// \par Storage reusage
    /// Synchronization barrier should be placed before \p storage is reused
    /// or repurposed: \p __syncthreads() or \p rocprim::syncthreads().
    ///
    /// \par Examples
    /// \parblock
    /// In the examples radix sort is performed on a block of 128 threads, each thread provides
    /// two key-value <tt>int</tt>-<tt>float</tt> pairs, results are returned using the same
    /// arrays as for input.
    ///
    /// \code{.cpp}
    /// __global__ void example_kernel(...)
    /// {
    ///     // specialize block_radix_sort for int-float pairs, block of 128
    ///     // threads, and two items per thread
    ///     using block_rsort_ii = rocprim::block_radix_sort<int, 128, 2, int>;
    ///     // allocate storage in shared memory
    ///     __shared__ block_rsort_ii::storage_type storage;
    ///
    ///     int keys[2] = ...;
    ///     float values[2] = ...;
    ///     // execute block radix sort-by-key (descending)
    ///     block_rsort_ii().sort_desc(
    ///         keys, values,
    ///         storage
    ///     );
    ///     ...
    /// }
    /// \endcode
    ///
    /// If the \p keys across threads in a block are <tt>{[1, 2], [3, 4]  ..., [255, 256]}</tt> and
    /// the \p values are <tt>{[128, 128], [127, 127]  ..., [2, 2], [1, 1]}</tt>, then after sort
    /// the \p keys will be equal <tt>{[256, 255], ..., [4, 3], [2, 1]}</tt> and the \p values
    /// will be equal <tt>{[1, 1], [2, 2]  ..., [128, 128]}</tt>.
    /// \endparblock
    template<bool WithValues = with_values>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort_desc(Key (&keys)[ItemsPerThread],
                   typename std::enable_if<WithValues, Value>::type (&values)[ItemsPerThread],
                   storage_type& storage,
                   unsigned int begin_bit = 0,
                   unsigned int end_bit = 8 * sizeof(Key))
    {
        sort_impl<true>(keys, values, storage, begin_bit, end_bit);
    }

    /// \overload
    /// \brief Performs descending radix sort over key-value pairs partitioned across
    /// threads in a block.
    ///
    /// * This overload does not accept storage argument. Required shared memory is
    /// allocated by the method itself.
    ///
    /// \pre Method is enabled only if \p Value type is different than empty_type.
    ///
    /// \param [in, out] keys - reference to an array of keys provided by a thread.
    /// \param [in, out] values - reference to an array of values provided by a thread.
    /// \param [in] begin_bit - [optional] index of the first (least significant) bit used in
    /// key comparison. Must be in range <tt>[0; 8 * sizeof(Key))</tt>. Default value: \p 0.
    /// \param [in] end_bit - [optional] past-the-end index (most significant) bit used in
    /// key comparison. Must be in range <tt>(begin_bit; 8 * sizeof(Key)]</tt>. Default
    /// value: \p <tt>8 * sizeof(Key)</tt>.
    template<bool WithValues = with_values>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void sort_desc(Key (&keys)[ItemsPerThread],
                   typename std::enable_if<WithValues, Value>::type (&values)[ItemsPerThread],
                   unsigned int begin_bit = 0,
                   unsigned int end_bit = 8 * sizeof(Key))
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        sort_desc(keys, values, storage, begin_bit, end_bit);
    }

    /// \brief Performs ascending radix sort over keys partitioned across threads in a block,
    /// results are saved in a striped arrangement.
    ///
    /// \param [in, out] keys - reference to an array of keys provided by a thread.
    /// \param [in] storage - reference to a temporary storage object of type storage_type.
    /// \param [in] begin_bit - [optional] index of the first (least significant) bit used in
    /// key comparison. Must be in range <tt>[0; 8 * sizeof(Key))</tt>. Default value: \p 0.
    /// \param [in] end_bit - [optional] past-the-end index (most significant) bit used in
    /// key comparison. Must be in range <tt>(begin_bit; 8 * sizeof(Key)]</tt>. Default
    /// value: \p <tt>8 * sizeof(Key)</tt>.
    ///
    /// \par Storage reusage
    /// Synchronization barrier should be placed before \p storage is reused
    /// or repurposed: \p __syncthreads() or \p rocprim::syncthreads().
    ///
    /// \par Examples
    /// \parblock
    /// In the examples radix sort is performed on a block of 128 threads, each thread provides
    /// two \p float value, results are returned using the same array as for input.
    ///
    /// \code{.cpp}
    /// __global__ void example_kernel(...)
    /// {
    ///     // specialize block_radix_sort for float, block of 128 threads,
    ///     // and two items per thread; key-only sort
    ///     using block_rsort_float = rocprim::block_radix_sort<float, 128, 2>;
    ///     // allocate storage in shared memory
    ///     __shared__ block_rsort_float::storage_type storage;
    ///
    ///     float keys[2] = ...;
    ///     // execute block radix sort (ascending)
    ///     block_rsort_float().sort_to_striped(
    ///         keys,
    ///         storage
    ///     );
    ///     ...
    /// }
    /// \endcode
    ///
    /// If the \p input values across threads in a block are <tt>{[256, 255], ..., [4, 3], [2, 1]}}</tt>, then
    /// then after sort they will be equal <tt>{[1, 129], [2, 130]  ..., [128, 256]}</tt>.
    /// \endparblock
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort_to_striped(Key (&keys)[ItemsPerThread],
                         storage_type& storage,
                         unsigned int begin_bit = 0,
                         unsigned int end_bit = 8 * sizeof(Key))
    {
        empty_type values[ItemsPerThread];
        sort_impl<false, true>(keys, values, storage, begin_bit, end_bit);
    }

    /// \overload
    /// \brief Performs ascending radix sort over keys partitioned across threads in a block,
    /// results are saved in a striped arrangement.
    ///
    /// * This overload does not accept storage argument. Required shared memory is
    /// allocated by the method itself.
    ///
    /// \param [in, out] keys - reference to an array of keys provided by a thread.
    /// \param [in] begin_bit - [optional] index of the first (least significant) bit used in
    /// key comparison. Must be in range <tt>[0; 8 * sizeof(Key))</tt>. Default value: \p 0.
    /// \param [in] end_bit - [optional] past-the-end index (most significant) bit used in
    /// key comparison. Must be in range <tt>(begin_bit; 8 * sizeof(Key)]</tt>. Default
    /// value: \p <tt>8 * sizeof(Key)</tt>.
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void sort_to_striped(Key (&keys)[ItemsPerThread],
                         unsigned int begin_bit = 0,
                         unsigned int end_bit = 8 * sizeof(Key))
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        sort_to_striped(keys, storage, begin_bit, end_bit);
    }

    /// \brief Performs descending radix sort over keys partitioned across threads in a block,
    /// results are saved in a striped arrangement.
    ///
    /// \param [in, out] keys - reference to an array of keys provided by a thread.
    /// \param [in] storage - reference to a temporary storage object of type storage_type.
    /// \param [in] begin_bit - [optional] index of the first (least significant) bit used in
    /// key comparison. Must be in range <tt>[0; 8 * sizeof(Key))</tt>. Default value: \p 0.
    /// \param [in] end_bit - [optional] past-the-end index (most significant) bit used in
    /// key comparison. Must be in range <tt>(begin_bit; 8 * sizeof(Key)]</tt>. Default
    /// value: \p <tt>8 * sizeof(Key)</tt>.
    ///
    /// \par Storage reusage
    /// Synchronization barrier should be placed before \p storage is reused
    /// or repurposed: \p __syncthreads() or \p rocprim::syncthreads().
    ///
    /// \par Examples
    /// \parblock
    /// In the examples radix sort is performed on a block of 128 threads, each thread provides
    /// two \p float value, results are returned using the same array as for input.
    ///
    /// \code{.cpp}
    /// __global__ void example_kernel(...)
    /// {
    ///     // specialize block_radix_sort for float, block of 128 threads,
    ///     // and two items per thread; key-only sort
    ///     using block_rsort_float = rocprim::block_radix_sort<float, 128, 2>;
    ///     // allocate storage in shared memory
    ///     __shared__ block_rsort_float::storage_type storage;
    ///
    ///     float input[2] = ...;
    ///     // execute block radix sort (descending)
    ///     block_rsort_float().sort_desc_to_striped(
    ///         input,
    ///         storage
    ///     );
    ///     ...
    /// }
    /// \endcode
    ///
    /// If the \p input values across threads in a block are <tt>{[1, 2], [3, 4]  ..., [255, 256]}</tt>,
    /// then after sort they will be equal <tt>{[256, 128], ..., [130, 2], [129, 1]}</tt>.
    /// \endparblock
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort_desc_to_striped(Key (&keys)[ItemsPerThread],
                              storage_type& storage,
                              unsigned int begin_bit = 0,
                              unsigned int end_bit = 8 * sizeof(Key))
    {
        empty_type values[ItemsPerThread];
        sort_impl<true, true>(keys, values, storage, begin_bit, end_bit);
    }

    /// \overload
    /// \brief Performs descending radix sort over keys partitioned across threads in a block,
    /// results are saved in a striped arrangement.
    ///
    /// * This overload does not accept storage argument. Required shared memory is
    /// allocated by the method itself.
    ///
    /// \param [in, out] keys - reference to an array of keys provided by a thread.
    /// \param [in] begin_bit - [optional] index of the first (least significant) bit used in
    /// key comparison. Must be in range <tt>[0; 8 * sizeof(Key))</tt>. Default value: \p 0.
    /// \param [in] end_bit - [optional] past-the-end index (most significant) bit used in
    /// key comparison. Must be in range <tt>(begin_bit; 8 * sizeof(Key)]</tt>. Default
    /// value: \p <tt>8 * sizeof(Key)</tt>.
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void sort_desc_to_striped(Key (&keys)[ItemsPerThread],
                              unsigned int begin_bit = 0,
                              unsigned int end_bit = 8 * sizeof(Key))
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        sort_desc_to_striped(keys, storage, begin_bit, end_bit);
    }

    /// \brief Performs ascending radix sort over key-value pairs partitioned across
    /// threads in a block, results are saved in a striped arrangement.
    ///
    /// \pre Method is enabled only if \p Value type is different than empty_type.
    ///
    /// \param [in, out] keys - reference to an array of keys provided by a thread.
    /// \param [in, out] values - reference to an array of values provided by a thread.
    /// \param [in] storage - reference to a temporary storage object of type storage_type.
    /// \param [in] begin_bit - [optional] index of the first (least significant) bit used in
    /// key comparison. Must be in range <tt>[0; 8 * sizeof(Key))</tt>. Default value: \p 0.
    /// \param [in] end_bit - [optional] past-the-end index (most significant) bit used in
    /// key comparison. Must be in range <tt>(begin_bit; 8 * sizeof(Key)]</tt>. Default
    /// value: \p <tt>8 * sizeof(Key)</tt>.
    ///
    /// \par Storage reusage
    /// Synchronization barrier should be placed before \p storage is reused
    /// or repurposed: \p __syncthreads() or \p rocprim::syncthreads().
    ///
    /// \par Examples
    /// \parblock
    /// In the examples radix sort is performed on a block of 4 threads, each thread provides
    /// two key-value <tt>int</tt>-<tt>float</tt> pairs, results are returned using the same
    /// arrays as for input.
    ///
    /// \code{.cpp}
    /// __global__ void example_kernel(...)
    /// {
    ///     // specialize block_radix_sort for int-float pairs, block of 4
    ///     // threads, and two items per thread
    ///     using block_rsort_ii = rocprim::block_radix_sort<int, 4, 2, int>;
    ///     // allocate storage in shared memory
    ///     __shared__ block_rsort_ii::storage_type storage;
    ///
    ///     int keys[2] = ...;
    ///     float values[2] = ...;
    ///     // execute block radix sort-by-key (ascending)
    ///     block_rsort_ii().sort_to_striped(
    ///         keys, values,
    ///         storage
    ///     );
    ///     ...
    /// }
    /// \endcode
    ///
    /// If the \p keys across threads in a block are <tt>{[8, 7], [6, 5], [4, 3], [2, 1]}</tt> and
    /// the \p values are <tt>{[-1, -2], [-3, -4], [-5, -6], [-7, -8]}</tt>, then after sort the
    /// \p keys will be equal <tt>{[1, 5], [2, 6], [3, 7], [4, 8]}</tt> and the \p values will be
    /// equal <tt>{[-8, -4], [-7, -3], [-6, -2], [-5, -1]}</tt>.
    /// \endparblock
    template<bool WithValues = with_values>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort_to_striped(Key (&keys)[ItemsPerThread],
                         typename std::enable_if<WithValues, Value>::type (&values)[ItemsPerThread],
                         storage_type& storage,
                         unsigned int begin_bit = 0,
                         unsigned int end_bit = 8 * sizeof(Key))
    {
        sort_impl<false, true>(keys, values, storage, begin_bit, end_bit);
    }

    /// \overload
    /// \brief Performs ascending radix sort over key-value pairs partitioned across
    /// threads in a block, results are saved in a striped arrangement.
    ///
    /// * This overload does not accept storage argument. Required shared memory is
    /// allocated by the method itself.
    ///
    /// \param [in, out] keys - reference to an array of keys provided by a thread.
    /// \param [in, out] values - reference to an array of values provided by a thread.
    /// \param [in] begin_bit - [optional] index of the first (least significant) bit used in
    /// key comparison. Must be in range <tt>[0; 8 * sizeof(Key))</tt>. Default value: \p 0.
    /// \param [in] end_bit - [optional] past-the-end index (most significant) bit used in
    /// key comparison. Must be in range <tt>(begin_bit; 8 * sizeof(Key)]</tt>. Default
    /// value: \p <tt>8 * sizeof(Key)</tt>.
    template<bool WithValues = with_values>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void sort_to_striped(Key (&keys)[ItemsPerThread],
                         typename std::enable_if<WithValues, Value>::type (&values)[ItemsPerThread],
                         unsigned int begin_bit = 0,
                         unsigned int end_bit = 8 * sizeof(Key))
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        sort_to_striped(keys, values, storage, begin_bit, end_bit);
    }

    /// \brief Performs descending radix sort over key-value pairs partitioned across
    /// threads in a block, results are saved in a striped arrangement.
    ///
    /// \pre Method is enabled only if \p Value type is different than empty_type.
    ///
    /// \param [in, out] keys - reference to an array of keys provided by a thread.
    /// \param [in, out] values - reference to an array of values provided by a thread.
    /// \param [in] storage - reference to a temporary storage object of type storage_type.
    /// \param [in] begin_bit - [optional] index of the first (least significant) bit used in
    /// key comparison. Must be in range <tt>[0; 8 * sizeof(Key))</tt>. Default value: \p 0.
    /// \param [in] end_bit - [optional] past-the-end index (most significant) bit used in
    /// key comparison. Must be in range <tt>(begin_bit; 8 * sizeof(Key)]</tt>. Default
    /// value: \p <tt>8 * sizeof(Key)</tt>.
    ///
    /// \par Storage reusage
    /// Synchronization barrier should be placed before \p storage is reused
    /// or repurposed: \p __syncthreads() or \p rocprim::syncthreads().
    ///
    /// \par Examples
    /// \parblock
    /// In the examples radix sort is performed on a block of 4 threads, each thread provides
    /// two key-value <tt>int</tt>-<tt>float</tt> pairs, results are returned using the same
    /// arrays as for input.
    ///
    /// \code{.cpp}
    /// __global__ void example_kernel(...)
    /// {
    ///     // specialize block_radix_sort for int-float pairs, block of 4
    ///     // threads, and two items per thread
    ///     using block_rsort_ii = rocprim::block_radix_sort<int, 4, 2, int>;
    ///     // allocate storage in shared memory
    ///     __shared__ block_rsort_ii::storage_type storage;
    ///
    ///     int keys[2] = ...;
    ///     float values[2] = ...;
    ///     // execute block radix sort-by-key (descending)
    ///     block_rsort_ii().sort_desc_to_striped(
    ///         keys, values,
    ///         storage
    ///     );
    ///     ...
    /// }
    /// \endcode
    ///
    /// If the \p keys across threads in a block are <tt>{[1, 2], [3, 4], [5, 6], [7, 8]}</tt> and
    /// the \p values are <tt>{[80, 70], [60, 50], [40, 30], [20, 10]}</tt>, then after sort the
    /// \p keys will be equal <tt>{[8, 4], [7, 3], [6, 2], [5, 1]}</tt> and the \p values will be
    /// equal <tt>{[10, 50], [20, 60], [30, 70], [40, 80]}</tt>.
    /// \endparblock
    template<bool WithValues = with_values>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort_desc_to_striped(Key (&keys)[ItemsPerThread],
                              typename std::enable_if<WithValues, Value>::type (&values)[ItemsPerThread],
                              storage_type& storage,
                              unsigned int begin_bit = 0,
                              unsigned int end_bit = 8 * sizeof(Key))
    {
        sort_impl<true, true>(keys, values, storage, begin_bit, end_bit);
    }

    /// \overload
    /// \brief Performs descending radix sort over key-value pairs partitioned across
    /// threads in a block, results are saved in a striped arrangement.
    ///
    /// * This overload does not accept storage argument. Required shared memory is
    /// allocated by the method itself.
    ///
    /// \param [in, out] keys - reference to an array of keys provided by a thread.
    /// \param [in, out] values - reference to an array of values provided by a thread.
    /// \param [in] begin_bit - [optional] index of the first (least significant) bit used in
    /// key comparison. Must be in range <tt>[0; 8 * sizeof(Key))</tt>. Default value: \p 0.
    /// \param [in] end_bit - [optional] past-the-end index (most significant) bit used in
    /// key comparison. Must be in range <tt>(begin_bit; 8 * sizeof(Key)]</tt>. Default
    /// value: \p <tt>8 * sizeof(Key)</tt>.
    template<bool WithValues = with_values>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void sort_desc_to_striped(Key (&keys)[ItemsPerThread],
                              typename std::enable_if<WithValues, Value>::type (&values)[ItemsPerThread],
                              unsigned int begin_bit = 0,
                              unsigned int end_bit = 8 * sizeof(Key))
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        sort_desc_to_striped(keys, values, storage, begin_bit, end_bit);
    }

private:

    template<bool Descending, bool ToStriped = false, class SortedValue>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort_impl(Key (&keys)[ItemsPerThread],
                   SortedValue (&values)[ItemsPerThread],
                   storage_type& storage,
                   unsigned int begin_bit,
                   unsigned int end_bit)
    {
        using key_codec = ::rocprim::detail::radix_key_codec<Key, Descending>;

        bit_key_type bit_keys[ItemsPerThread];
        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            bit_keys[i] = key_codec::encode(keys[i]);
        }

        while(true)
        {
            const int pass_bits = min(radix_bits_per_pass, end_bit - begin_bit);

            unsigned int ranks[ItemsPerThread];
            block_rank_type().rank_keys(
                bit_keys,
                ranks,
                storage.get().rank,
                [begin_bit, pass_bits](const bit_key_type& key)
                { return key_codec::extract_digit(key, begin_bit, pass_bits); });
            begin_bit += radix_bits_per_pass;

            exchange_keys(storage, bit_keys, ranks);
            exchange_values(storage, values, ranks);

            if(begin_bit >= end_bit)
                break;

            // Synchronization required to make bock_rank wait on the next iteration.
            ::rocprim::syncthreads();
        }

        if ROCPRIM_IF_CONSTEXPR(ToStriped)
        {
            to_striped_keys(storage, bit_keys);
            to_striped_values(storage, values);
        }

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            keys[i] = key_codec::decode(bit_keys[i]);
        }
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    void exchange_keys(storage_type& storage,
                       bit_key_type (&bit_keys)[ItemsPerThread],
                       const unsigned int (&ranks)[ItemsPerThread])
    {
        storage_type_& storage_ = storage.get();
        ::rocprim::syncthreads(); // Storage will be reused (union), synchronization is needed
        bit_keys_exchange_type().scatter_to_blocked(bit_keys, bit_keys, ranks, storage_.bit_keys_exchange);
    }

    template<class SortedValue>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void exchange_values(storage_type& storage,
                         SortedValue (&values)[ItemsPerThread],
                         const unsigned int (&ranks)[ItemsPerThread])
    {
        storage_type_& storage_ = storage.get();
        ::rocprim::syncthreads(); // Storage will be reused (union), synchronization is needed
        values_exchange_type().scatter_to_blocked(values, values, ranks, storage_.values_exchange);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    void exchange_values(storage_type& storage,
                         empty_type (&values)[ItemsPerThread],
                         const unsigned int (&ranks)[ItemsPerThread])
    {
        (void) storage;
        (void) values;
        (void) ranks;
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    void to_striped_keys(storage_type& storage,
                         bit_key_type (&bit_keys)[ItemsPerThread])
    {
        storage_type_& storage_ = storage.get();
        ::rocprim::syncthreads(); // Storage will be reused (union), synchronization is needed
        bit_keys_exchange_type().blocked_to_striped(bit_keys, bit_keys, storage_.bit_keys_exchange);
    }

    template<class SortedValue>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void to_striped_values(storage_type& storage,
                           SortedValue (&values)[ItemsPerThread])
    {
        storage_type_& storage_ = storage.get();
        ::rocprim::syncthreads(); // Storage will be reused (union), synchronization is needed
        values_exchange_type().blocked_to_striped(values, values, storage_.values_exchange);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    void to_striped_values(storage_type& storage,
                           empty_type * values)
    {
        (void) storage;
        (void) values;
    }
};

END_ROCPRIM_NAMESPACE

/// @}
// end of group blockmodule

#endif // ROCPRIM_BLOCK_BLOCK_RADIX_SORT_HPP_
