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

#ifndef ROCPRIM_BLOCK_BLOCK_RADIX_SORT_HPP_
#define ROCPRIM_BLOCK_BLOCK_RADIX_SORT_HPP_

#include <type_traits>

#include "../config.hpp"
#include "../detail/various.hpp"
#include "../functional.hpp"
#include "../intrinsics/thread.hpp"
#include "../types.hpp"

#include "../warp/warp_exchange.hpp"
#include "block_exchange.hpp"
#include "block_radix_rank.hpp"
#include "rocprim/block/config.hpp"
#include "rocprim/intrinsics/arch.hpp"

/// \addtogroup blockmodule
/// @{

BEGIN_ROCPRIM_NAMESPACE

/// \brief The block_radix_sort class is a block level parallel primitive which provides
/// methods for sorting of items (keys or key-value pairs) partitioned across threads in a block
/// using radix sort algorithm.
///
/// \tparam Key the key type.
/// \tparam BlockSize the number of threads in a block.
/// \tparam ItemsPerThread the number of items contributed by each thread.
/// \tparam Value the value type. Default type empty_type indicates
/// a keys-only sort.
/// \tparam RadixBitsPerPass amount of bits to sort per pass. The Default is 4.
/// \tparam RadixRankAlgorithm the rank algorithm used.
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
/// \par Stability
/// \p block_radix_sort is \b stable: it preserves the relative ordering of equivalent keys.
/// That is, given two keys \p a and \p b and a binary boolean operation \p op such that:
///   * \p a precedes \p b in the input keys, and
///   * op(a, b) and op(b, a) are both false,
/// then it is \b guaranteed that \p a will precede \p b as well in the output (ordered) keys.
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
template<class Key,
         unsigned int BlockSizeX,
         unsigned int ItemsPerThread,
         class Value                                 = empty_type,
         unsigned int               BlockSizeY       = 1,
         unsigned int               BlockSizeZ       = 1,
         unsigned int               RadixBitsPerPass = 0,
         block_radix_rank_algorithm RadixRankAlgorithm
         = block_radix_rank_algorithm::default_for_radix_sort,
         block_padding_hint      PaddingHint    = block_padding_hint::lds_occupancy_bound,
         arch::wavefront::target TargetWaveSize = arch::wavefront::get_target()>
class block_radix_sort
{
    // TODO: somehow when prefer_match is true on SPIR-V, results
    // are incorrect. Block radix rank works fine though...
    static constexpr bool prefer_match = (BlockSizeX * BlockSizeY * BlockSizeZ)
                                             % arch::wavefront::size_from_target<TargetWaveSize>()
                                         == 0;

    static constexpr unsigned int radix_bits_per_pass = RadixBitsPerPass == 0
                                                            ? (prefer_match ? 8 /* match */
                                                                            : 4 /* basic_memoize */)
                                                            : RadixBitsPerPass;

    static constexpr block_radix_rank_algorithm radix_rank_algorithm
        = RadixRankAlgorithm == block_radix_rank_algorithm::default_for_radix_sort
              ? (prefer_match ? block_radix_rank_algorithm::match
                              : block_radix_rank_algorithm::basic_memoize)
              : RadixRankAlgorithm;

    static_assert(radix_bits_per_pass > 0 && radix_bits_per_pass < 32,
                  "The radix_bits_per_pass should be larger than 0 and smaller than the size "
                  "of an unsigned int");

    static constexpr unsigned int BlockSize   = BlockSizeX * BlockSizeY * BlockSizeZ;
    static constexpr bool         with_values = !std::is_same<Value, empty_type>::value;
    static constexpr bool warp_striped = radix_rank_algorithm == block_radix_rank_algorithm::match;

    ROCPRIM_DETAIL_DEVICE_STATIC_ASSERT(
        !warp_striped || (BlockSize % ::rocprim::arch::wavefront::min_size()) == 0,
        "When using 'block_radix_rank_algorithm::match', the block size should be a "
        "multiple of the warp size");

    static constexpr bool is_key_and_value_aligned
        = alignof(Key) == alignof(Value) && sizeof(Key) == sizeof(Value);

    using block_rank_type = ::rocprim::block_radix_rank<BlockSizeX,
                                                        radix_bits_per_pass,
                                                        radix_rank_algorithm,
                                                        BlockSizeY,
                                                        BlockSizeZ,
                                                        PaddingHint,
                                                        TargetWaveSize>;

    using keys_exchange_type = ::rocprim::block_exchange<Key,
                                                         BlockSizeX,
                                                         ItemsPerThread,
                                                         BlockSizeY,
                                                         BlockSizeZ,
                                                         PaddingHint,
                                                         TargetWaveSize>;

    using values_exchange_type = ::rocprim::block_exchange<Value,
                                                           BlockSizeX,
                                                           ItemsPerThread,
                                                           BlockSizeY,
                                                           BlockSizeZ,
                                                           PaddingHint,
                                                           TargetWaveSize>;

    // Struct used for creating a raw_storage object for this primitive's temporary storage.
    union storage_type_
    {
        typename keys_exchange_type::storage_type   keys_exchange;
        typename values_exchange_type::storage_type values_exchange;
        typename block_rank_type::storage_type      rank;
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
    ROCPRIM_DETAIL_SUPPRESS_DEPRECATION_WITH_PUSH
    using storage_type = detail::raw_storage<storage_type_>;
    ROCPRIM_DETAIL_SUPPRESS_DEPRECATION_POP
#else
    using storage_type = storage_type_; // only for Doxygen
#endif

        ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    block_radix_sort()
    {
        assert(!warp_striped || BlockSize % ::rocprim::arch::wavefront::size() == 0);
    }

    /// \brief Performs ascending radix sort over keys partitioned across threads in a block.
    ///
    /// \tparam Decomposer The type of the decomposer argument. Defaults to the identity decomposer.
    ///
    /// \param [in, out] keys reference to an array of keys provided by a thread.
    /// \param [in] storage reference to a temporary storage object of type storage_type.
    /// \param [in] begin_bit [optional] index of the first (least significant) bit used in
    /// key comparison. Must be in range <tt>[0; 8 * sizeof(Key))</tt>. Default value: \p 0.
    /// \param [in] end_bit [optional] past-the-end index (most significant) bit used in
    /// key comparison. Must be in range <tt>(begin_bit; 8 * sizeof(Key)]</tt>. Default
    /// value: \p <tt>8 * sizeof(Key)</tt>.
    /// \param [in] decomposer [optional] If `Key` is not an arithmetic type (integral, floating point),
    ///  a custom decomposer functor should be passed that produces a `::rocprim::tuple` of references to
    /// fundamental types from this custom type.
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
    template<class Decomposer = ::rocprim::identity_decomposer>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort(Key (&keys)[ItemsPerThread],
              storage_type& storage,
              unsigned int  begin_bit  = 0,
              unsigned int  end_bit    = 8 * sizeof(Key),
              Decomposer    decomposer = {})
    {
        empty_type values[ItemsPerThread];
        sort_impl<false>(keys, values, storage, begin_bit, end_bit, decomposer);
    }

    /// \overload
    /// \brief Performs ascending radix sort over keys partitioned across threads in a block.
    ///
    /// * This overload does not accept storage argument. Required shared memory is
    /// allocated by the method itself.
    ///
    /// \param [in, out] keys reference to an array of keys provided by a thread.
    /// \param [in] begin_bit [optional] index of the first (least significant) bit used in
    /// key comparison. Must be in range <tt>[0; 8 * sizeof(Key))</tt>. Default value: \p 0.
    /// \param [in] end_bit [optional] past-the-end index (most significant) bit used in
    /// key comparison. Must be in range <tt>(begin_bit; 8 * sizeof(Key)]</tt>. Default
    /// value: \p <tt>8 * sizeof(Key)</tt>.
    /// \param [in] decomposer [optional] If `Key` is not an arithmetic type (integral, floating point),
    ///  a custom decomposer functor should be passed that produces a `::rocprim::tuple` of references to
    /// fundamental types from this custom type.
    template<class Decomposer = ::rocprim::identity_decomposer>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE void sort(Key (&keys)[ItemsPerThread],
                                                  unsigned int begin_bit  = 0,
                                                  unsigned int end_bit    = 8 * sizeof(Key),
                                                  Decomposer   decomposer = {})
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        sort(keys, storage, begin_bit, end_bit, decomposer);
    }

    /// \brief Performs descending radix sort over keys partitioned across threads in a block.
    ///
    /// \tparam Decomposer The type of the decomposer argument. Defaults to the identity decomposer.
    ///
    /// \param [in, out] keys reference to an array of keys provided by a thread.
    /// \param [in] storage reference to a temporary storage object of type storage_type.
    /// \param [in] begin_bit [optional] index of the first (least significant) bit used in
    /// key comparison. Must be in range <tt>[0; 8 * sizeof(Key))</tt>. Default value: \p 0.
    /// \param [in] end_bit [optional] past-the-end index (most significant) bit used in
    /// key comparison. Must be in range <tt>(begin_bit; 8 * sizeof(Key)]</tt>. Default
    /// value: \p <tt>8 * sizeof(Key)</tt>.
    /// \param [in] decomposer [optional] If `Key` is not an arithmetic type (integral, floating point),
    ///  a custom decomposer functor should be passed that produces a `::rocprim::tuple` of references to
    /// fundamental types from this custom type.
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
    template<class Decomposer = ::rocprim::identity_decomposer>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort_desc(Key (&keys)[ItemsPerThread],
                   storage_type& storage,
                   unsigned int  begin_bit  = 0,
                   unsigned int  end_bit    = 8 * sizeof(Key),
                   Decomposer    decomposer = {})
    {
        empty_type values[ItemsPerThread];
        sort_impl<true>(keys, values, storage, begin_bit, end_bit, decomposer);
    }

    /// \overload
    /// \brief Performs descending radix sort over keys partitioned across threads in a block.
    ///
    /// * This overload does not accept storage argument. Required shared memory is
    /// allocated by the method itself.
    ///
    /// \tparam Decomposer The type of the decomposer argument. Defaults to the identity decomposer.
    ///
    /// \param [in, out] keys reference to an array of keys provided by a thread.
    /// \param [in] begin_bit [optional] index of the first (least significant) bit used in
    /// key comparison. Must be in range <tt>[0; 8 * sizeof(Key))</tt>. Default value: \p 0.
    /// \param [in] end_bit [optional] past-the-end index (most significant) bit used in
    /// key comparison. Must be in range <tt>(begin_bit; 8 * sizeof(Key)]</tt>. Default
    /// value: \p <tt>8 * sizeof(Key)</tt>.
    /// \param [in] decomposer [optional] If `Key` is not an arithmetic type (integral, floating point),
    ///  a custom decomposer functor should be passed that produces a `::rocprim::tuple` of references to
    /// fundamental types from this custom type.
    template<class Decomposer = ::rocprim::identity_decomposer>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE void sort_desc(Key (&keys)[ItemsPerThread],
                                                       unsigned int begin_bit  = 0,
                                                       unsigned int end_bit    = 8 * sizeof(Key),
                                                       Decomposer   decomposer = {})
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        sort_desc(keys, storage, begin_bit, end_bit, decomposer);
    }

    /// \brief Performs ascending radix sort over key-value pairs partitioned across
    /// threads in a block.
    ///
    /// \pre Method is enabled only if \p Value type is different than empty_type.
    ///
    /// \tparam Decomposer The type of the decomposer argument. Defaults to the identity decomposer.
    ///
    /// \param [in, out] keys reference to an array of keys provided by a thread.
    /// \param [in, out] values reference to an array of values provided by a thread.
    /// \param [in] storage reference to a temporary storage object of type storage_type.
    /// \param [in] begin_bit [optional] index of the first (least significant) bit used in
    /// key comparison. Must be in range <tt>[0; 8 * sizeof(Key))</tt>. Default value: \p 0.
    /// \param [in] end_bit [optional] past-the-end index (most significant) bit used in
    /// key comparison. Must be in range <tt>(begin_bit; 8 * sizeof(Key)]</tt>. Default
    /// value: \p <tt>8 * sizeof(Key)</tt>.
    /// \param [in] decomposer [optional] If `Key` is not an arithmetic type (integral, floating point),
    ///  a custom decomposer functor should be passed that produces a `::rocprim::tuple` of references to
    /// fundamental types from this custom type.
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
    template<bool WithValues = with_values, class Decomposer = ::rocprim::identity_decomposer>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort(Key (&keys)[ItemsPerThread],
              typename std::enable_if<WithValues, Value>::type (&values)[ItemsPerThread],
              storage_type& storage,
              unsigned int  begin_bit  = 0,
              unsigned int  end_bit    = 8 * sizeof(Key),
              Decomposer    decomposer = {})
    {
        sort_impl<false>(keys, values, storage, begin_bit, end_bit, decomposer);
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
    /// \tparam Decomposer The type of the decomposer argument. Defaults to the identity decomposer.
    ///
    /// \param [in, out] keys reference to an array of keys provided by a thread.
    /// \param [in, out] values reference to an array of values provided by a thread.
    /// \param [in] begin_bit [optional] index of the first (least significant) bit used in
    /// key comparison. Must be in range <tt>[0; 8 * sizeof(Key))</tt>. Default value: \p 0.
    /// \param [in] end_bit [optional] past-the-end index (most significant) bit used in
    /// key comparison. Must be in range <tt>(begin_bit; 8 * sizeof(Key)]</tt>. Default
    /// value: \p <tt>8 * sizeof(Key)</tt>.
    /// \param [in] decomposer [optional] If `Key` is not an arithmetic type (integral, floating point),
    ///  a custom decomposer functor should be passed that produces a `::rocprim::tuple` of references to
    /// fundamental types from this custom type.
    template<bool WithValues = with_values, class Decomposer = ::rocprim::identity_decomposer>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE void
        sort(Key (&keys)[ItemsPerThread],
             typename std::enable_if<WithValues, Value>::type (&values)[ItemsPerThread],
             unsigned int begin_bit  = 0,
             unsigned int end_bit    = 8 * sizeof(Key),
             Decomposer   decomposer = {})
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        sort(keys, values, storage, begin_bit, end_bit, decomposer);
    }

    /// \brief Performs descending radix sort over key-value pairs partitioned across
    /// threads in a block.
    ///
    /// \pre Method is enabled only if \p Value type is different than empty_type.
    ///
    /// \tparam Decomposer The type of the decomposer argument. Defaults to the identity decomposer.
    ///
    /// \param [in, out] keys reference to an array of keys provided by a thread.
    /// \param [in, out] values reference to an array of values provided by a thread.
    /// \param [in] storage reference to a temporary storage object of type storage_type.
    /// \param [in] begin_bit [optional] index of the first (least significant) bit used in
    /// key comparison. Must be in range <tt>[0; 8 * sizeof(Key))</tt>. Default value: \p 0.
    /// \param [in] end_bit [optional] past-the-end index (most significant) bit used in
    /// key comparison. Must be in range <tt>(begin_bit; 8 * sizeof(Key)]</tt>. Default
    /// value: \p <tt>8 * sizeof(Key)</tt>.
    /// \param [in] decomposer [optional] If `Key` is not an arithmetic type (integral, floating point),
    ///  a custom decomposer functor should be passed that produces a `::rocprim::tuple` of references to
    /// fundamental types from this custom type.
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
    template<bool WithValues = with_values, class Decomposer = ::rocprim::identity_decomposer>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort_desc(Key (&keys)[ItemsPerThread],
                   typename std::enable_if<WithValues, Value>::type (&values)[ItemsPerThread],
                   storage_type& storage,
                   unsigned int  begin_bit  = 0,
                   unsigned int  end_bit    = 8 * sizeof(Key),
                   Decomposer    decomposer = {})
    {
        sort_impl<true>(keys, values, storage, begin_bit, end_bit, decomposer);
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
    /// \tparam Decomposer The type of the decomposer argument. Defaults to the identity decomposer.
    ///
    /// \param [in, out] keys reference to an array of keys provided by a thread.
    /// \param [in, out] values reference to an array of values provided by a thread.
    /// \param [in] begin_bit [optional] index of the first (least significant) bit used in
    /// key comparison. Must be in range <tt>[0; 8 * sizeof(Key))</tt>. Default value: \p 0.
    /// \param [in] end_bit [optional] past-the-end index (most significant) bit used in
    /// key comparison. Must be in range <tt>(begin_bit; 8 * sizeof(Key)]</tt>. Default
    /// value: \p <tt>8 * sizeof(Key)</tt>.
    /// \param [in] decomposer [optional] If `Key` is not an arithmetic type (integral, floating point),
    ///  a custom decomposer functor should be passed that produces a `::rocprim::tuple` of references to
    /// fundamental types from this custom type.
    template<bool WithValues = with_values, class Decomposer = ::rocprim::identity_decomposer>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE void
        sort_desc(Key (&keys)[ItemsPerThread],
                  typename std::enable_if<WithValues, Value>::type (&values)[ItemsPerThread],
                  unsigned int begin_bit  = 0,
                  unsigned int end_bit    = 8 * sizeof(Key),
                  Decomposer   decomposer = {})
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        sort_desc(keys, values, storage, begin_bit, end_bit, decomposer);
    }

    /// \brief Performs ascending radix sort over keys partitioned across threads in a block,
    /// results are saved in a striped arrangement.
    ///
    /// \tparam Decomposer The type of the decomposer argument. Defaults to the identity decomposer.
    ///
    /// \param [in, out] keys reference to an array of keys provided by a thread.
    /// \param [in] storage reference to a temporary storage object of type storage_type.
    /// \param [in] begin_bit [optional] index of the first (least significant) bit used in
    /// key comparison. Must be in range <tt>[0; 8 * sizeof(Key))</tt>. Default value: \p 0.
    /// \param [in] end_bit [optional] past-the-end index (most significant) bit used in
    /// key comparison. Must be in range <tt>(begin_bit; 8 * sizeof(Key)]</tt>. Default
    /// value: \p <tt>8 * sizeof(Key)</tt>.
    /// \param [in] decomposer [optional] If `Key` is not an arithmetic type (integral, floating point),
    ///  a custom decomposer functor should be passed that produces a `::rocprim::tuple` of references to
    /// fundamental types from this custom type.
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
    template<class Decomposer = ::rocprim::identity_decomposer>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort_to_striped(Key (&keys)[ItemsPerThread],
                         storage_type& storage,
                         unsigned int  begin_bit  = 0,
                         unsigned int  end_bit    = 8 * sizeof(Key),
                         Decomposer    decomposer = {})
    {
        empty_type values[ItemsPerThread];
        sort_impl<false, true>(keys, values, storage, begin_bit, end_bit, decomposer);
    }

    /// \overload
    /// \brief Performs ascending radix sort over keys partitioned across threads in a block,
    /// results are saved in a striped arrangement.
    ///
    /// * This overload does not accept storage argument. Required shared memory is
    /// allocated by the method itself.
    ///
    /// \tparam Decomposer The type of the decomposer argument. Defaults to the identity decomposer.
    ///
    /// \param [in, out] keys reference to an array of keys provided by a thread.
    /// \param [in] begin_bit [optional] index of the first (least significant) bit used in
    /// key comparison. Must be in range <tt>[0; 8 * sizeof(Key))</tt>. Default value: \p 0.
    /// \param [in] end_bit [optional] past-the-end index (most significant) bit used in
    /// key comparison. Must be in range <tt>(begin_bit; 8 * sizeof(Key)]</tt>. Default
    /// value: \p <tt>8 * sizeof(Key)</tt>.
    /// \param [in] decomposer [optional] If `Key` is not an arithmetic type (integral, floating point),
    ///  a custom decomposer functor should be passed that produces a `::rocprim::tuple` of references to
    /// fundamental types from this custom type.
    template<class Decomposer = ::rocprim::identity_decomposer>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE void sort_to_striped(Key (&keys)[ItemsPerThread],
                                                             unsigned int begin_bit = 0,
                                                             unsigned int end_bit = 8 * sizeof(Key),
                                                             Decomposer   decomposer = {})
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        sort_to_striped(keys, storage, begin_bit, end_bit, decomposer);
    }

    /// \brief Performs descending radix sort over keys partitioned across threads in a block,
    /// results are saved in a striped arrangement.
    ///
    /// \tparam Decomposer The type of the decomposer argument. Defaults to the identity decomposer.
    ///
    /// \param [in, out] keys reference to an array of keys provided by a thread.
    /// \param [in] storage reference to a temporary storage object of type storage_type.
    /// \param [in] begin_bit [optional] index of the first (least significant) bit used in
    /// key comparison. Must be in range <tt>[0; 8 * sizeof(Key))</tt>. Default value: \p 0.
    /// \param [in] end_bit [optional] past-the-end index (most significant) bit used in
    /// key comparison. Must be in range <tt>(begin_bit; 8 * sizeof(Key)]</tt>. Default
    /// value: \p <tt>8 * sizeof(Key)</tt>.
    /// \param [in] decomposer [optional] If `Key` is not an arithmetic type (integral, floating point),
    ///  a custom decomposer functor should be passed that produces a `::rocprim::tuple` of references to
    /// fundamental types from this custom type.
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
    template<class Decomposer = ::rocprim::identity_decomposer>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort_desc_to_striped(Key (&keys)[ItemsPerThread],
                              storage_type& storage,
                              unsigned int  begin_bit  = 0,
                              unsigned int  end_bit    = 8 * sizeof(Key),
                              Decomposer    decomposer = {})
    {
        empty_type values[ItemsPerThread];
        sort_impl<true, true>(keys, values, storage, begin_bit, end_bit, decomposer);
    }

    /// \overload
    /// \brief Performs descending radix sort over keys partitioned across threads in a block,
    /// results are saved in a striped arrangement.
    ///
    /// * This overload does not accept storage argument. Required shared memory is
    /// allocated by the method itself.
    ///
    /// \tparam Decomposer The type of the decomposer argument. Defaults to the identity decomposer.
    ///
    /// \param [in, out] keys reference to an array of keys provided by a thread.
    /// \param [in] begin_bit [optional] index of the first (least significant) bit used in
    /// key comparison. Must be in range <tt>[0; 8 * sizeof(Key))</tt>. Default value: \p 0.
    /// \param [in] end_bit [optional] past-the-end index (most significant) bit used in
    /// key comparison. Must be in range <tt>(begin_bit; 8 * sizeof(Key)]</tt>. Default
    /// value: \p <tt>8 * sizeof(Key)</tt>.
    /// \param [in] decomposer [optional] If `Key` is not an arithmetic type (integral, floating point),
    ///  a custom decomposer functor should be passed that produces a `::rocprim::tuple` of references to
    /// fundamental types from this custom type.
    template<class Decomposer = ::rocprim::identity_decomposer>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE void sort_desc_to_striped(Key (&keys)[ItemsPerThread],
                                                                  unsigned int begin_bit = 0,
                                                                  unsigned int end_bit
                                                                  = 8 * sizeof(Key),
                                                                  Decomposer decomposer = {})
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        sort_desc_to_striped(keys, storage, begin_bit, end_bit, decomposer);
    }

    /// \brief Performs ascending radix sort over key-value pairs partitioned across
    /// threads in a block, results are saved in a striped arrangement.
    ///
    /// \pre Method is enabled only if \p Value type is different than empty_type.
    ///
    /// \tparam Decomposer The type of the decomposer argument. Defaults to the identity decomposer.
    ///
    /// \param [in, out] keys reference to an array of keys provided by a thread.
    /// \param [in, out] values reference to an array of values provided by a thread.
    /// \param [in] storage reference to a temporary storage object of type storage_type.
    /// \param [in] begin_bit [optional] index of the first (least significant) bit used in
    /// key comparison. Must be in range <tt>[0; 8 * sizeof(Key))</tt>. Default value: \p 0.
    /// \param [in] end_bit [optional] past-the-end index (most significant) bit used in
    /// key comparison. Must be in range <tt>(begin_bit; 8 * sizeof(Key)]</tt>. Default
    /// value: \p <tt>8 * sizeof(Key)</tt>.
    /// \param [in] decomposer [optional] If `Key` is not an arithmetic type (integral, floating point),
    ///  a custom decomposer functor should be passed that produces a `::rocprim::tuple` of references to
    /// fundamental types from this custom type.
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
    template<bool WithValues = with_values, class Decomposer = ::rocprim::identity_decomposer>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort_to_striped(Key (&keys)[ItemsPerThread],
                         typename std::enable_if<WithValues, Value>::type (&values)[ItemsPerThread],
                         storage_type& storage,
                         unsigned int  begin_bit  = 0,
                         unsigned int  end_bit    = 8 * sizeof(Key),
                         Decomposer    decomposer = {})
    {
        sort_impl<false, true>(keys, values, storage, begin_bit, end_bit, decomposer);
    }

    /// \overload
    /// \brief Performs ascending radix sort over key-value pairs partitioned across
    /// threads in a block, results are saved in a striped arrangement.
    ///
    /// * This overload does not accept storage argument. Required shared memory is
    /// allocated by the method itself.
    ///
    /// \tparam Decomposer The type of the decomposer argument. Defaults to the identity decomposer.
    ///
    /// \param [in, out] keys reference to an array of keys provided by a thread.
    /// \param [in, out] values reference to an array of values provided by a thread.
    /// \param [in] begin_bit [optional] index of the first (least significant) bit used in
    /// key comparison. Must be in range <tt>[0; 8 * sizeof(Key))</tt>. Default value: \p 0.
    /// \param [in] end_bit [optional] past-the-end index (most significant) bit used in
    /// key comparison. Must be in range <tt>(begin_bit; 8 * sizeof(Key)]</tt>. Default
    /// value: \p <tt>8 * sizeof(Key)</tt>.
    /// \param [in] decomposer [optional] If `Key` is not an arithmetic type (integral, floating point),
    ///  a custom decomposer functor should be passed that produces a `::rocprim::tuple` of references to
    /// fundamental types from this custom type.
    template<bool WithValues = with_values, class Decomposer = ::rocprim::identity_decomposer>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE void
        sort_to_striped(Key (&keys)[ItemsPerThread],
                        typename std::enable_if<WithValues, Value>::type (&values)[ItemsPerThread],
                        unsigned int begin_bit  = 0,
                        unsigned int end_bit    = 8 * sizeof(Key),
                        Decomposer   decomposer = {})
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        sort_to_striped(keys, values, storage, begin_bit, end_bit, decomposer);
    }

    /// \brief Performs descending radix sort over key-value pairs partitioned across
    /// threads in a block, results are saved in a striped arrangement.
    ///
    /// \pre Method is enabled only if \p Value type is different than empty_type.
    ///
    /// \tparam Decomposer The type of the decomposer argument. Defaults to the identity decomposer.
    ///
    /// \param [in, out] keys reference to an array of keys provided by a thread.
    /// \param [in, out] values reference to an array of values provided by a thread.
    /// \param [in] storage reference to a temporary storage object of type storage_type.
    /// \param [in] begin_bit [optional] index of the first (least significant) bit used in
    /// key comparison. Must be in range <tt>[0; 8 * sizeof(Key))</tt>. Default value: \p 0.
    /// \param [in] end_bit [optional] past-the-end index (most significant) bit used in
    /// key comparison. Must be in range <tt>(begin_bit; 8 * sizeof(Key)]</tt>. Default
    /// value: \p <tt>8 * sizeof(Key)</tt>.
    /// \param [in] decomposer [optional] If `Key` is not an arithmetic type (integral, floating point),
    ///  a custom decomposer functor should be passed that produces a `::rocprim::tuple` of references to
    /// fundamental types from this custom type.
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
    template<bool WithValues = with_values, class Decomposer = ::rocprim::identity_decomposer>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort_desc_to_striped(
        Key (&keys)[ItemsPerThread],
        typename std::enable_if<WithValues, Value>::type (&values)[ItemsPerThread],
        storage_type& storage,
        unsigned int  begin_bit  = 0,
        unsigned int  end_bit    = 8 * sizeof(Key),
        Decomposer    decomposer = {})
    {
        sort_impl<true, true>(keys, values, storage, begin_bit, end_bit, decomposer);
    }

    /// \overload
    /// \brief Performs descending radix sort over key-value pairs partitioned across
    /// threads in a block, results are saved in a striped arrangement.
    ///
    /// * This overload does not accept storage argument. Required shared memory is
    /// allocated by the method itself.
    ///
    /// \tparam Decomposer The type of the decomposer argument. Defaults to the identity decomposer.
    ///
    /// \param [in, out] keys reference to an array of keys provided by a thread.
    /// \param [in, out] values reference to an array of values provided by a thread.
    /// \param [in] begin_bit [optional] index of the first (least significant) bit used in
    /// key comparison. Must be in range <tt>[0; 8 * sizeof(Key))</tt>. Default value: \p 0.
    /// \param [in] end_bit [optional] past-the-end index (most significant) bit used in
    /// key comparison. Must be in range <tt>(begin_bit; 8 * sizeof(Key)]</tt>. Default
    /// value: \p <tt>8 * sizeof(Key)</tt>.
    /// \param [in] decomposer [optional] If `Key` is not an arithmetic type (integral, floating point),
    ///  a custom decomposer functor should be passed that produces a `::rocprim::tuple` of references to
    /// fundamental types from this custom type.
    template<bool WithValues = with_values, class Decomposer = ::rocprim::identity_decomposer>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE void sort_desc_to_striped(
        Key (&keys)[ItemsPerThread],
        typename std::enable_if<WithValues, Value>::type (&values)[ItemsPerThread],
        unsigned int begin_bit  = 0,
        unsigned int end_bit    = 8 * sizeof(Key),
        Decomposer   decomposer = {})
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        sort_desc_to_striped(keys, values, storage, begin_bit, end_bit, decomposer);
    }

    /// \brief Performs ascending radix sort over key-value pairs in a *warp-striped order*
    /// partitioned across threads in a block, results are saved in a striped arrangement.
    ///
    /// \see block_radix_sort::sort_to_striped
    template<bool WithValues = with_values, class Decomposer = ::rocprim::identity_decomposer>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort_warp_striped_to_striped(
        Key (&keys)[ItemsPerThread],
        typename std::enable_if<WithValues, Value>::type (&values)[ItemsPerThread],
        storage_type& storage,
        unsigned int  begin_bit  = 0,
        unsigned int  end_bit    = 8 * sizeof(Key),
        Decomposer    decomposer = {})
    {
        static_assert(warp_striped,
                      "'sort_warp_striped_to_striped' can only be used with "
                      "'block_radix_rank_algorithm::match'.");

        sort_impl<false, true, false>(keys, values, storage, begin_bit, end_bit, decomposer);
    }

    /// \brief Performs ascending radix sort over key-value pairs in a *warp-striped order*
    ///
    /// \see block_radix_sort::sort_to_striped
    /// partitioned across threads in a block, results are saved in a striped arrangement.
    template<bool WithValues = with_values, class Decomposer = ::rocprim::identity_decomposer>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort_warp_striped_to_striped(
        Key (&keys)[ItemsPerThread],
        typename std::enable_if<WithValues, Value>::type (&values)[ItemsPerThread],
        unsigned int begin_bit  = 0,
        unsigned int end_bit    = 8 * sizeof(Key),
        Decomposer   decomposer = {})
    {
        static_assert(warp_striped,
                      "'sort_warp_striped_to_striped' can only be used with "
                      "'block_radix_rank_algorithm::match'.");

        ROCPRIM_SHARED_MEMORY storage_type storage;
        sort_warp_striped_to_striped(keys, values, storage, begin_bit, end_bit, decomposer);
    }

    /// \brief Performs ascending radix sort over key-value pairs in a *warp-striped order*
    /// partitioned across threads in a block, results are saved in a striped arrangement.
    ///
    /// \see block_radix_sort::sort_to_striped
    template<bool WithValues = with_values, class Decomposer = ::rocprim::identity_decomposer>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort_warp_striped_to_striped(Key (&keys)[ItemsPerThread],
                                      storage_type& storage,
                                      unsigned int  begin_bit  = 0,
                                      unsigned int  end_bit    = 8 * sizeof(Key),
                                      Decomposer    decomposer = {})
    {
        static_assert(warp_striped,
                      "'sort_warp_striped_to_striped' can only be used with "
                      "'block_radix_rank_algorithm::match'.");

        empty_type values[ItemsPerThread];
        sort_impl<false, true, false>(keys, values, storage, begin_bit, end_bit, decomposer);
    }

    /// \brief Performs ascending radix sort over key-value pairs in a *warp-striped order*
    /// partitioned across threads in a block, results are saved in a striped arrangement.
    ///
    /// \see block_radix_sort::sort_to_striped
    template<bool WithValues = with_values, class Decomposer = ::rocprim::identity_decomposer>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort_warp_striped_to_striped(Key (&keys)[ItemsPerThread],
                                      unsigned int begin_bit  = 0,
                                      unsigned int end_bit    = 8 * sizeof(Key),
                                      Decomposer   decomposer = {})
    {
        static_assert(warp_striped,
                      "'sort_warp_striped_to_striped' can only be used with "
                      "'block_radix_rank_algorithm::match'.");

        ROCPRIM_SHARED_MEMORY storage_type storage;
        sort_warp_striped_to_striped(keys, storage, begin_bit, end_bit, decomposer);
    }

    /// \brief Performs descending radix sort over key-value pairs in a *warp-striped order*
    /// partitioned across threads in a block, results are saved in a striped arrangement.
    ///
    /// \see block_radix_sort::sort_desc_to_striped
    template<bool WithValues = with_values, class Decomposer = ::rocprim::identity_decomposer>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort_desc_warp_striped_to_striped(
        Key (&keys)[ItemsPerThread],
        typename std::enable_if<WithValues, Value>::type (&values)[ItemsPerThread],
        storage_type& storage,
        unsigned int  begin_bit  = 0,
        unsigned int  end_bit    = 8 * sizeof(Key),
        Decomposer    decomposer = {})
    {
        static_assert(warp_striped,
                      "'sort_warp_striped_to_striped' can only be used with "
                      "'block_radix_rank_algorithm::match'.");

        sort_impl<true, true, false>(keys, values, storage, begin_bit, end_bit, decomposer);
    }

    /// \brief Performs descending radix sort over key-value pairs in a *warp-striped order*
    /// partitioned across threads in a block, results are saved in a striped arrangement.
    ///
    /// \see block_radix_sort::sort_desc_to_striped
    template<bool WithValues = with_values, class Decomposer = ::rocprim::identity_decomposer>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort_desc_warp_striped_to_striped(
        Key (&keys)[ItemsPerThread],
        typename std::enable_if<WithValues, Value>::type (&values)[ItemsPerThread],
        unsigned int begin_bit  = 0,
        unsigned int end_bit    = 8 * sizeof(Key),
        Decomposer   decomposer = {})
    {
        static_assert(warp_striped,
                      "'sort_warp_striped_to_striped' can only be used with "
                      "'block_radix_rank_algorithm::match'.");

        ROCPRIM_SHARED_MEMORY storage_type storage;
        sort_desc_warp_striped_to_striped(keys, values, storage, begin_bit, end_bit, decomposer);
    }

    /// \brief Performs descending radix sort over key-value pairs in a *warp-striped order*
    /// partitioned across threads in a block, results are saved in a striped arrangement.
    ///
    /// \see block_radix_sort::sort_desc_to_striped
    template<bool WithValues = with_values, class Decomposer = ::rocprim::identity_decomposer>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort_desc_warp_striped_to_striped(Key (&keys)[ItemsPerThread],
                                           storage_type& storage,
                                           unsigned int  begin_bit  = 0,
                                           unsigned int  end_bit    = 8 * sizeof(Key),
                                           Decomposer    decomposer = {})
    {
        static_assert(warp_striped,
                      "'sort_warp_striped_to_striped' can only be used with "
                      "'block_radix_rank_algorithm::match'.");

        empty_type values[ItemsPerThread];
        sort_impl<true, true, false>(keys, values, storage, begin_bit, end_bit, decomposer);
    }

    /// \brief Performs descending radix sort over key-value pairs in a *warp-striped order*
    /// partitioned across threads in a block, results are saved in a striped arrangement.
    ///
    /// \see block_radix_sort::sort_desc_to_striped
    template<bool WithValues = with_values, class Decomposer = ::rocprim::identity_decomposer>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort_desc_warp_striped_to_striped(Key (&keys)[ItemsPerThread],
                                           unsigned int begin_bit  = 0,
                                           unsigned int end_bit    = 8 * sizeof(Key),
                                           Decomposer   decomposer = {})
    {
        static_assert(warp_striped,
                      "'sort_warp_striped_to_striped' can only be used with "
                      "'block_radix_rank_algorithm::match'.");

        ROCPRIM_SHARED_MEMORY storage_type storage;
        sort_desc_warp_striped_to_striped(keys, storage, begin_bit, end_bit, decomposer);
    }

private:
    template<class SortedValue>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void blocked_to_warp_striped(Key (&keys)[ItemsPerThread],
                                 SortedValue (&values)[ItemsPerThread],
                                 storage_type& storage,
                                 std::false_type)
    {
        keys_exchange_type().blocked_to_warp_striped(keys, keys, storage.get().keys_exchange);
        if constexpr(is_key_and_value_aligned)
        {
            // If keys and values are aligned, then the LDS for both exchanges is
            // local per wave. We can relax the data dependency!
            ::rocprim::wave_barrier();
        }
        else
        {
            ::rocprim::syncthreads();
        }
        values_exchange_type().blocked_to_warp_striped(values,
                                                       values,
                                                       storage.get().values_exchange);
    }

    template<class SortedValue>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void blocked_to_warp_striped(Key (&keys)[ItemsPerThread],
                                 SortedValue (&values)[ItemsPerThread],
                                 storage_type& /* storage */,
                                 std::true_type)
    {
        constexpr int wave_size = ::rocprim::arch::wavefront::size_from_target<TargetWaveSize>();
        using keys_warp_exchange
            = ::rocprim::warp_exchange<Key, ItemsPerThread, wave_size, TargetWaveSize>;
        using values_warp_exchange
            = ::rocprim::warp_exchange<SortedValue, ItemsPerThread, wave_size, TargetWaveSize>;

        keys_warp_exchange{}.blocked_to_striped_shuffle(keys, keys);
        values_warp_exchange{}.blocked_to_striped_shuffle(values, values);
    }

    template<bool Descending,
             bool ToStriped             = false,
             bool TryEmulateWarpStriped = true,
             class SortedValue,
             class Decomposer>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort_impl(Key (&keys)[ItemsPerThread],
                   SortedValue (&values)[ItemsPerThread],
                   storage_type& storage,
                   unsigned int  begin_bit,
                   unsigned int  end_bit,
                   Decomposer    decomposer)
    {
        using key_codec
            = decltype(::rocprim::traits::get<Key>().template radix_key_codec<Descending>());

        // 'rank_keys' may be invoked multiple times. We encode the key once and move the
        // encoded during the majority of sort to save on some compute.
        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            key_codec::encode_inplace(keys[i], decomposer);
        }

        // If we're using warp striped radix rank but our input is in a blocked layout, we
        // can emulate the correct input through an exchange to a warp striped layout.
        if constexpr(TryEmulateWarpStriped && warp_striped && ItemsPerThread > 1)
        {
            // This appears to be slower with high large items per thread.
            constexpr bool use_warp_exchange
                = arch::wavefront::size_from_target<TargetWaveSize>() % ItemsPerThread == 0
                  && ItemsPerThread <= 4;
            blocked_to_warp_striped(keys,
                                    values,
                                    storage,
                                    std::integral_constant<bool, use_warp_exchange>{});
            // Storage has been dirtied. 'rank_keys' does not always align nicely with this
            // so a full block synchronization is needed.
            ::rocprim::syncthreads();
        }

        unsigned int ranks[ItemsPerThread];
        while(true)
        {
            const int pass_bits = min(radix_bits_per_pass, end_bit - begin_bit);

            block_rank_type().rank_keys(
                keys,
                ranks,
                storage.get().rank,
                [begin_bit, pass_bits, decomposer](const Key& key) mutable
                { return key_codec::extract_digit(key, begin_bit, pass_bits, decomposer); });
            begin_bit += radix_bits_per_pass;

            if(begin_bit >= end_bit)
            {
                break;
            }

            if constexpr(warp_striped)
            {
                exchange_keys_warp_striped(storage, keys, ranks);
                exchange_values_warp_striped(storage, values, ranks);
            }
            else
            {
                exchange_keys(storage, keys, ranks);
                exchange_values(storage, values, ranks);
            }

            // Synchronization required to make block_rank wait on the next iteration.
            ::rocprim::syncthreads();
        }

        if constexpr(ToStriped)
        {
            exchange_to_striped_keys(storage, keys, ranks);
            exchange_to_striped_values(storage, values, ranks);
        }
        else
        {
            exchange_keys(storage, keys, ranks);
            exchange_values(storage, values, ranks);
        }

        // Done with 'rank_keys' so we can decode back to the original key.
        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            key_codec::decode_inplace(keys[i], decomposer);
        }
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    void exchange_keys(storage_type& storage,
                       Key (&keys)[ItemsPerThread],
                       const unsigned int (&ranks)[ItemsPerThread])
    {
        storage_type_& storage_ = storage.get();
        ::rocprim::syncthreads(); // Storage will be reused (union), synchronization is needed
        keys_exchange_type().scatter_to_blocked(keys, keys, ranks, storage_.keys_exchange);
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
        (void)storage;
        (void)values;
        (void)ranks;
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    void exchange_keys_warp_striped(storage_type& storage,
                                    Key (&keys)[ItemsPerThread],
                                    const unsigned int (&ranks)[ItemsPerThread])
    {
        storage_type_& storage_ = storage.get();
        ::rocprim::syncthreads(); // Storage will be reused (union), synchronization is needed
        keys_exchange_type().scatter_to_warp_striped(keys, keys, ranks, storage_.keys_exchange);
    }

    template<class SortedValue>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void exchange_values_warp_striped(storage_type& storage,
                                      SortedValue (&values)[ItemsPerThread],
                                      const unsigned int (&ranks)[ItemsPerThread])
    {
        storage_type_& storage_ = storage.get();
        ::rocprim::syncthreads(); // Storage will be reused (union), synchronization is needed
        values_exchange_type().scatter_to_warp_striped(values,
                                                       values,
                                                       ranks,
                                                       storage_.values_exchange);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    void exchange_values_warp_striped(storage_type& storage,
                                      empty_type (&values)[ItemsPerThread],
                                      const unsigned int (&ranks)[ItemsPerThread])
    {
        (void)storage;
        (void)values;
        (void)ranks;
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    void exchange_to_striped_keys(storage_type& storage,
                                  Key (&keys)[ItemsPerThread],
                                  const unsigned int (&ranks)[ItemsPerThread])
    {
        storage_type_& storage_ = storage.get();
        ::rocprim::syncthreads(); // Storage will be reused (union), synchronization is needed
        keys_exchange_type().scatter_to_striped(keys, keys, ranks, storage_.keys_exchange);
    }

    template<class SortedValue>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void exchange_to_striped_values(storage_type& storage,
                                    SortedValue (&values)[ItemsPerThread],
                                    const unsigned int (&ranks)[ItemsPerThread])
    {
        storage_type_& storage_ = storage.get();
        ::rocprim::syncthreads(); // Storage will be reused (union), synchronization is needed
        values_exchange_type().scatter_to_striped(values, values, ranks, storage_.values_exchange);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    void exchange_to_striped_values(storage_type& storage,
                                    empty_type*   values,
                                    const unsigned int (&ranks)[ItemsPerThread])
    {
        (void)ranks;
        (void)storage;
        (void)values;
    }
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template<class Key,
         unsigned int BlockSizeX,
         unsigned int ItemsPerThread,
         class Value,
         unsigned int               BlockSizeY,
         unsigned int               BlockSizeZ,
         unsigned int               RadixBitsPerPass,
         block_radix_rank_algorithm RadixRankAlgorithm,
         block_padding_hint         PaddingHint>
class block_radix_sort<Key,
                       BlockSizeX,
                       ItemsPerThread,
                       Value,
                       BlockSizeY,
                       BlockSizeZ,
                       RadixBitsPerPass,
                       RadixRankAlgorithm,
                       PaddingHint,
                       arch::wavefront::target::dynamic>
{
    using block_radix_sort_wave32 = block_radix_sort<Key,
                                                     BlockSizeX,
                                                     ItemsPerThread,
                                                     Value,
                                                     BlockSizeY,
                                                     BlockSizeZ,
                                                     RadixBitsPerPass,
                                                     RadixRankAlgorithm,
                                                     PaddingHint,
                                                     arch::wavefront::target::size32>;
    using block_radix_sort_wave64 = block_radix_sort<Key,
                                                     BlockSizeX,
                                                     ItemsPerThread,
                                                     Value,
                                                     BlockSizeY,
                                                     BlockSizeZ,
                                                     RadixBitsPerPass,
                                                     RadixRankAlgorithm,
                                                     PaddingHint,
                                                     arch::wavefront::target::size64>;

    using dispatch = detail::dispatch_wave_size<block_radix_sort_wave32, block_radix_sort_wave64>;

public:
    using storage_type = typename dispatch::storage_type;

    template<typename... Args>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto sort(Args&&... args)
    {
        dispatch{}([](auto impl, auto&&... args) { impl.sort(args...); }, args...);
    }

    template<typename... Args>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto sort_desc(Args&&... args)
    {
        dispatch{}([](auto impl, auto&&... args) { impl.sort_desc(args...); }, args...);
    }

    template<typename... Args>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto sort_to_striped(Args&&... args)
    {
        dispatch{}([](auto impl, auto&&... args) { impl.sort_to_striped(args...); }, args...);
    }

    template<typename... Args>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto sort_desc_to_striped(Args&&... args)
    {
        dispatch{}([](auto impl, auto&&... args) { impl.sort_desc_to_striped(args...); }, args...);
    }

    template<typename... Args>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto sort_warp_striped_to_striped(Args&&... args)
    {
        dispatch{}([](auto impl, auto&&... args) { impl.sort_warp_striped_to_striped(args...); },
                   args...);
    }

    template<typename... Args>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto sort_desc_warp_striped_to_striped(Args&&... args)
    {
        dispatch{}([](auto impl, auto&&... args)
                   { impl.sort_desc_warp_striped_to_striped(args...); },
                   args...);
    }
};
#endif // DOXYGEN_SHOULD_SKIP_THIS

END_ROCPRIM_NAMESPACE

/// @}
// end of group blockmodule

#endif // ROCPRIM_BLOCK_BLOCK_RADIX_SORT_HPP_
