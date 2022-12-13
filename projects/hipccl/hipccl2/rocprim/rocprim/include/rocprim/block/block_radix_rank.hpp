// Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_BLOCK_BLOCK_RADIX_RANK_HPP_
#define ROCPRIM_BLOCK_BLOCK_RADIX_RANK_HPP_

#include "../config.hpp"
#include "../functional.hpp"

#include "../detail/radix_sort.hpp"

#include "block_scan.hpp"

#include "detail/block_radix_rank_basic.hpp"
#include "detail/block_radix_rank_match.hpp"

/// \addtogroup blockmodule
/// @{

BEGIN_ROCPRIM_NAMESPACE

/// \brief Available algorithms for the block_radix_rank primitive.
enum class block_radix_rank_algorithm
{
    /// \brief The basic block radix rank algorithm. Keys and ranks are assumed in blocked order.
    basic,
    /// \brief The basic block radix rank algorithm, configured to memoize intermediate values. This trades
    /// register usage for less shared memory operations. Keys and ranks are assumed in blocked order.
    basic_memoize,
    /// \brief Warp-based radix ranking algorithm. Keys and ranks are assumed in warp-striped order for this algorithm.
    match,
    /// \brief The default radix ranking algorithm.
    default_algorithm = basic,
};

namespace detail
{
// Selector for block radix rank algorithm that gives the radix rank implementation.
template<block_radix_rank_algorithm Algorithm>
struct select_block_radix_rank_impl;

template<>
struct select_block_radix_rank_impl<block_radix_rank_algorithm::basic>
{
    template<unsigned int BlockSizeX,
             unsigned int RadixBits,
             unsigned int BlockSizeY,
             unsigned int BlockSizeZ>
    using type = block_radix_rank<BlockSizeX, RadixBits, false, BlockSizeY, BlockSizeZ>;
};

template<>
struct select_block_radix_rank_impl<block_radix_rank_algorithm::basic_memoize>
{
    template<unsigned int BlockSizeX,
             unsigned int RadixBits,
             unsigned int BlockSizeY,
             unsigned int BlockSizeZ>
    using type = block_radix_rank<BlockSizeX, RadixBits, true, BlockSizeY, BlockSizeZ>;
};

template<>
struct select_block_radix_rank_impl<block_radix_rank_algorithm::match>
{
    template<unsigned int BlockSizeX,
             unsigned int RadixBits,
             unsigned int BlockSizeY,
             unsigned int BlockSizeZ>
    using type = block_radix_rank_match<BlockSizeX, RadixBits, BlockSizeY, BlockSizeZ>;
};
} // namespace detail

/// \brief The block_radix_rank class is a block level parallel primitive that provides
/// methods for ranking items partitioned across threads in a block. This algorithm
/// associates each item with the index it would gain if the keys were sorted into an array,
/// according to a radix comparison. Ranking is performed in a stable manner.
///
/// \tparam BlockSizeX - the number of threads in a block's x dimension.
/// \tparam RadixBits - the maximum number of radix digit bits that comparisons are performed by.
/// \tparam MemoizeOuterScan - whether to cache digit counters in local memory. This omits loading
/// the same values from shared memory twice, at the expense of more register usage.
/// \tparam BlockSizeY - the number of threads in a block's y dimension, defaults to 1.
/// \tparam BlockSizeZ - the number of threads in a block's z dimension, defaults to 1.
///
/// \par Overview
/// * Key type must be an arithmetic type (that is, an integral type or a floating point type).
/// * Performance depends on the block size and the number of items that will be sorted per thread.
///     * It is usually better if the block size is a multiple of the size of the hardware warp.
///     * It is usually increased when there are more than one item per thread. However, when there are too
///     many items per thread, each thread may need so many registers and/or shared memory
///     that it impedes performance.
/// * Shared memory usage depends on the block size and the maximum number of radix bits that will be
/// considered when comparing keys.
///     * The storage requirement increases when more bits are considered.
/// * The maximum amount of keys that can be ranked in a block is <tt>2**16</tt>.
///
/// \par Examples
/// In the example, radix rank is performed on a block of 128 threads. Each thread provides
/// three \p float values, which are ranked according to bits 10 through 14. The results are
/// written back in a separate array of three <tt>unsigned int</tt> values.
///
/// \code{.cpp}
/// __global__ void example_kernel(...)
/// {
///     // specialize the block_radix_rank for float, block of 128 threads.
///     using block_rank_float = rocprim::block_radix_rank<float, 128>;
///     // allocate storage in shared memory
///     __shared__ block_rank_float::storage_type storage;
///
///     float        input[3] = ...;
///     unsigned int output[3] = ...;
///     // execute the block radix rank (ascending)
///     block_rank_float().rank_keys(input,
///                                  output,
///                                  storage,
///                                  10,
///                                  4);
///     ...
/// }
/// \endcode
template<unsigned int               BlockSizeX,
         unsigned int               RadixBits,
         block_radix_rank_algorithm Algorithm  = block_radix_rank_algorithm::default_algorithm,
         unsigned int               BlockSizeY = 1,
         unsigned int               BlockSizeZ = 1>
class block_radix_rank
#ifndef DOXYGEN_SHOULD_SKIP_THIS
    : private detail::select_block_radix_rank_impl<
          Algorithm>::template type<BlockSizeX, RadixBits, BlockSizeY, BlockSizeZ>
#endif
{
    using base_type = typename detail::select_block_radix_rank_impl<
        Algorithm>::template type<BlockSizeX, RadixBits, BlockSizeY, BlockSizeZ>;

public:
    static constexpr unsigned int digits_per_thread = base_type::digits_per_thread;

    /// \brief Struct used to allocate a temporary memory that is required for thread
    /// communication during operations provided by related parallel primitive.
    ///
    /// Depending on the implemention the operations exposed by parallel primitive may
    /// require a temporary storage for thread communication. The storage should be allocated
    /// using keywords \p __shared__. It can be aliased to
    /// an externally allocated memory, or be a part of a union with other storage types
    /// to increase shared memory reusability.
    using storage_type = typename base_type::storage_type;

    /// \brief Perform ascending radix rank over keys partitioned across threads in a block.
    ///
    /// \tparam Key - the key type.
    /// \tparam ItemsPerThread - the number of items contributed by each thread in the block.
    /// \param [in] keys - reference to an array of keys provided by a thread.
    /// \param [out] ranks - reference to an array where the final ranks are written to.
    /// \param [in] storage - reference to a temporary storage object of type \p storage_type.
    /// \param [in] begin_bit - index of the first (least significant) bit used in key comparison.
    /// Must be in range <tt>[0; 8 * sizeof(Key))</tt>.
    /// \param [in] pass_bits - [optional] the number of bits used in key comparison. Must be in
    /// the range <tt>(0; RadixBits]</tt>. Default value: RadixBits.
    ///
    /// \par Storage reusage
    /// A synchronization barrier should be placed before \p storage is reused
    /// or repurposed: \p __syncthreads() or \p rocprim::syncthreads().
    ///
    /// \par Examples
    /// \parblock
    /// In the example, radix rank is performed on a block of 128 threads. Each thread provides
    /// three \p float values, which are ranked according to bits 10 through 14. The results are
    /// written back in a separate array of three <tt>unsigned int</tt> values.
    ///
    /// \code{.cpp}
    /// __global__ void example_kernel(...)
    /// {
    ///     // specialize the block_radix_rank for float, block of 128 threads, and a maximum of 4 bits.
    ///     using block_rank_float = rocprim::block_radix_rank<float, 128, 4>;
    ///     // allocate storage in shared memory
    ///     __shared__ block_rank_float::storage_type storage;
    ///
    ///     float        input[3] = ...;
    ///     unsigned int output[3];
    ///     // execute the block radix rank (ascending)
    ///     block_rank_float().rank_keys(input,
    ///                                  output,
    ///                                  storage,
    ///                                  10,
    ///                                  4);
    ///     ...
    /// }
    /// \endcode
    template<typename Key, unsigned ItemsPerThread>
    ROCPRIM_DEVICE void rank_keys(const Key (&keys)[ItemsPerThread],
                                  unsigned int (&ranks)[ItemsPerThread],
                                  storage_type& storage,
                                  unsigned int  begin_bit = 0,
                                  unsigned int  pass_bits = RadixBits)
    {
        base_type::rank_keys(keys, ranks, storage, begin_bit, pass_bits);
    }

    /// \brief Perform ascending radix rank over keys partitioned across threads in a block.
    ///
    /// * This overload does not accept storage argument. Required shared memory is allocated
    /// by the method itself.
    ///
    /// \tparam Key - the key type.
    /// \tparam ItemsPerThread - the number of items contributed by each thread in the block.
    /// \param [in] keys - reference to an array of keys provided by a thread.
    /// \param [out] ranks - reference to an array where the final ranks are written to.
    /// \param [in] begin_bit - index of the first (least significant) bit used in key comparison.
    /// Must be in range <tt>[0; 8 * sizeof(Key))</tt>.
    /// \param [in] pass_bits - [optional] the number of bits used in key comparison. Must be in
    /// the range <tt>(0; RadixBits]</tt>. Default value: RadixBits.
    template<typename Key, unsigned ItemsPerThread>
    ROCPRIM_DEVICE void rank_keys(const Key (&keys)[ItemsPerThread],
                                  unsigned int (&ranks)[ItemsPerThread],
                                  unsigned int begin_bit = 0,
                                  unsigned int pass_bits = RadixBits)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        base_type::rank_keys(keys, ranks, storage, begin_bit, pass_bits);
    }

    /// \brief Perform ascending radix rank over bit keys partitioned across threads in a block.
    /// This overload accepts a callback used to extract the radix digit from a key.
    ///
    /// \tparam ItemsPerThread - the number of items contributed by each thread in the block.
    /// \param [in] keys - reference to an array of keys provided by a thread.
    /// \param [out] ranks - reference to an array where the final ranks are written to.
    /// \param [in] storage - reference to a temporary storage object of type \p storage_type.
    /// \param [in] begin_bit - index of the first (least significant) bit used in key comparison.
    /// Must be in range <tt>[0; 8 * sizeof(Key))</tt>.
    /// \param [in] pass_bits - [optional] the number of bits used in key comparison. Must be in
    /// the range <tt>(0; RadixBits]</tt>. Default value: RadixBits.
    ///
    /// \par Storage reusage
    /// A synchronization barrier should be placed before \p storage is reused
    /// or repurposed: \p __syncthreads() or \p rocprim::syncthreads().
    ///
    /// \par Examples
    /// \parblock
    /// In the example, radix rank is performed on a block of 128 threads. Each thread provides
    /// three \p float values, which are ranked according to bits 10 through 14. The results are
    /// written back in a separate array of three <tt>unsigned int</tt> values.
    ///
    /// \code{.cpp}
    /// __global__ void example_kernel(...)
    /// {
    ///     // specialize the block_radix_rank for float, block of 128 threads, and a maximum of 4 bits.
    ///     using block_rank_float = rocprim::block_radix_rank<float, 128, 4>;
    ///     // allocate storage in shared memory
    ///     __shared__ block_rank_float::storage_type storage;
    ///
    ///     float        input[3] = ...;
    ///     unsigned int output[3];
    ///     // execute the block radix rank (descending)
    ///     block_rank_float().rank_keys_desc(input,
    ///                                       output,
    ///                                       storage,
    ///                                       10,
    ///                                       4);
    ///     ...
    /// }
    /// \endcode
    template<typename Key, unsigned ItemsPerThread>
    ROCPRIM_DEVICE void rank_keys_desc(const Key (&keys)[ItemsPerThread],
                                       unsigned int (&ranks)[ItemsPerThread],
                                       storage_type& storage,
                                       unsigned int  begin_bit = 0,
                                       unsigned int  pass_bits = RadixBits)
    {
        base_type::rank_keys_desc(keys, ranks, storage, begin_bit, pass_bits);
    }

    /// \brief Perform descending radix rank over keys partitioned across threads in a block.
    ///
    /// * This overload does not accept storage argument. Required shared memory is allocated
    /// by the method itself.
    ///
    /// \tparam Key - the key type.
    /// \tparam ItemsPerThread - the number of items contributed by each thread in the block.
    /// \param [in] keys - reference to an array of keys provided by a thread.
    /// \param [out] ranks - reference to an array where the final ranks are written to.
    /// \param [in] begin_bit - index of the first (least significant) bit used in key comparison.
    /// Must be in range <tt>[0; 8 * sizeof(Key))</tt>.
    /// \param [in] pass_bits - [optional] the number of bits used in key comparison. Must be in
    /// the range <tt>(0; RadixBits]</tt>. Default value: RadixBits.
    template<typename Key, unsigned ItemsPerThread>
    ROCPRIM_DEVICE void rank_keys_desc(const Key (&keys)[ItemsPerThread],
                                       unsigned int (&ranks)[ItemsPerThread],
                                       unsigned int begin_bit = 0,
                                       unsigned int pass_bits = RadixBits)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        base_type::rank_keys_desc(keys, ranks, storage, begin_bit, pass_bits);
    }

    /// \brief Perform ascending radix rank over bit keys partitioned across threads in a block.
    /// This overload accepts a callback used to extract the radix digit from a key.
    ///
    /// \tparam Key - the key type.
    /// \tparam ItemsPerThread - the number of items contributed by each thread in the block.
    /// \tparam DigitExtractor - type of the unary function object used to extract a digit from
    /// a key.
    /// \param [in] keys - reference to an array of keys provided by a thread.
    /// \param [out] ranks - reference to an array where the final ranks are written to.
    /// \param [in] storage - reference to a temporary storage object of type \p storage_type.
    /// \param [in] digit_extractor - function object used to convert a key to a digit.
    /// The signature of the \p digit_extractor should be equivalent to the following:
    /// <tt>unsigned int f(const Key &key);</tt>. The signature does not need to have
    /// <tt>const &</tt>, but function object must not modify the objects passed to it.
    /// This function will be used during ranking to extract the digit that indicates
    /// the key's value. Values return by this function object must be in range [0; 1 << RadixBits).
    ///
    /// \par Storage reusage
    /// A synchronization barrier should be placed before \p storage is reused
    /// or repurposed: \p __syncthreads() or \p rocprim::syncthreads().
    ///
    /// \par Examples
    /// \parblock
    /// In the example, radix rank is performed on a block of 128 threads. Each thread provides
    /// three \p int values, which are ranked according to a digit callback that extracts
    /// digits 0 through 4. Results written back in a separate array of three <tt>unsigned int</tt> values.
    ///
    /// \code{.cpp}
    /// __global__ void example_kernel(...)
    /// {
    ///     // specialize the block_radix_rank for int, block of 128 threads, and a maximum of 4 bits.
    ///     using block_rank_float = rocprim::block_radix_rank<int, 128, 4>;
    ///     // allocate storage in shared memory
    ///     __shared__ block_rank_float::storage_type storage;
    ///
    ///     int          input[3] = ...;
    ///     unsigned int output[3];
    ///     // execute the block radix rank (ascending)
    ///     block_rank_float().rank_keys(input,
    ///                                  output,
    ///                                  storage,
    ///                                  [](const int& key)
    ///                                  {
    ///                                      // Rank the keys by the lower 4 bits
    ///                                      return key & 0xF;
    ///                                  });
    ///     ...
    /// }
    /// \endcode
    template<typename Key, unsigned ItemsPerThread, typename DigitExtractor>
    ROCPRIM_DEVICE void rank_keys(const Key (&keys)[ItemsPerThread],
                                  unsigned int (&ranks)[ItemsPerThread],
                                  storage_type&  storage,
                                  DigitExtractor digit_extractor)
    {
        base_type::rank_keys(keys, ranks, storage, digit_extractor);
    }

    /// \brief Perform ascending radix rank over bit keys partitioned across threads in a block.
    /// This overload accepts a callback used to extract the radix digit from a key.
    ///
    /// * This overload does not accept storage argument. Required shared memory is allocated
    /// by the method itself.
    ///
    /// \tparam Key - the key type.
    /// \tparam ItemsPerThread - the number of items contributed by each thread in the block.
    /// \tparam DigitExtractor - type of the unary function object used to extract a digit from
    /// a key.
    /// \param [in] keys - reference to an array of keys provided by a thread.
    /// \param [out] ranks - reference to an array where the final ranks are written to.
    /// \param [in] storage - reference to a temporary storage object of type \p storage_type.
    /// \param [in] digit_extractor - function object used to convert a key to a digit.
    /// The signature of the \p digit_extractor should be equivalent to the following:
    /// <tt>unsigned int f(const Key &key);</tt>. The signature does not need to have
    /// <tt>const &</tt>, but function object must not modify the objects passed to it.
    /// This function will be used during ranking to extract the digit that indicates
    /// the key's value. Values return by this function object must be in range [0; 1 << RadixBits).
    template<typename Key, unsigned ItemsPerThread, typename DigitExtractor>
    ROCPRIM_DEVICE void rank_keys(const Key (&keys)[ItemsPerThread],
                                  unsigned int (&ranks)[ItemsPerThread],
                                  DigitExtractor digit_extractor)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        base_type::rank_keys(keys, ranks, storage, digit_extractor);
    }

    /// \brief Perform descending radix rank over bit keys partitioned across threads in a block.
    /// This overload accepts a callback used to extract the radix digit from a key.
    ///
    /// \tparam Key - the key type.
    /// \tparam ItemsPerThread - the number of items contributed by each thread in the block.
    /// \tparam DigitExtractor - type of the unary function object used to extract a digit from
    /// a key.
    /// \param [in] keys - reference to an array of keys provided by a thread.
    /// \param [out] ranks - reference to an array where the final ranks are written to.
    /// \param [in] storage - reference to a temporary storage object of type \p storage_type.
    /// \param [in] digit_extractor - function object used to convert a key to a digit.
    /// The signature of the \p digit_extractor should be equivalent to the following:
    /// <tt>unsigned int f(const Key &key);</tt>. The signature does not need to have
    /// <tt>const &</tt>, but function object must not modify the objects passed to it.
    /// This function will be used during ranking to extract the digit that indicates
    /// the key's value. Values return by this function object must be in range [0; 1 << RadixBits).
    ///
    /// \par Storage reusage
    /// A synchronization barrier should be placed before \p storage is reused
    /// or repurposed: \p __syncthreads() or \p rocprim::syncthreads().
    ///
    /// \par Examples
    /// \parblock
    /// In the example, radix rank is performed on a block of 128 threads. Each thread provides
    /// three \p int values, which are ranked according to a digit callback that extracts
    /// digits 0 through 4. Results written back in a separate array of three <tt>unsigned int</tt> values.
    ///
    /// \code{.cpp}
    /// __global__ void example_kernel(...)
    /// {
    ///     // specialize the block_radix_rank for int, block of 128 threads, and a maximum of 4 bits.
    ///     using block_rank_float = rocprim::block_radix_rank<int, 128, 4>;
    ///     // allocate storage in shared memory
    ///     __shared__ block_rank_float::storage_type storage;
    ///
    ///     int          input[3] = ...;
    ///     unsigned int output[3];
    ///     // execute the block radix rank (descending))
    ///     block_rank_float().rank_keys_desc(input,
    ///                                       output,
    ///                                       storage,
    ///                                       [](const int& key)
    ///                                       {
    ///                                           // Rank the keys by the lower 4 bits
    ///                                           return key & 0xF;
    ///                                       });
    ///     ...
    /// }
    /// \endcode
    template<typename Key, unsigned ItemsPerThread, typename DigitExtractor>
    ROCPRIM_DEVICE void rank_keys_desc(const Key (&keys)[ItemsPerThread],
                                       unsigned int (&ranks)[ItemsPerThread],
                                       storage_type&  storage,
                                       DigitExtractor digit_extractor)
    {
        base_type::rank_keys_desc(keys, ranks, storage, digit_extractor);
    }

    /// \brief Perform descending radix rank over bit keys partitioned across threads in a block.
    /// This overload accepts a callback used to extract the radix digit from a key.
    ///
    /// * This overload does not accept storage argument. Required shared memory is allocated
    /// by the method itself.
    ///
    /// \tparam Key - the key type.
    /// \tparam ItemsPerThread - the number of items contributed by each thread in the block.
    /// \tparam DigitExtractor - type of the unary function object used to extract a digit from
    /// a key.
    /// \param [in] keys - reference to an array of keys provided by a thread.
    /// \param [out] ranks - reference to an array where the final ranks are written to.
    /// \param [in] storage - reference to a temporary storage object of type \p storage_type.
    /// \param [in] digit_extractor - function object used to convert a key to a digit.
    /// The signature of the \p digit_extractor should be equivalent to the following:
    /// <tt>unsinged int f(const Key &key);</tt>. The signature does not need to have
    /// <tt>const &</tt>, but function object must not modify the objects passed to it.
    /// This function will be used during ranking to extract the digit that indicates
    /// the key's value. Values return by this function object must be in range [0; 1 << RadixBits).
    template<typename Key, unsigned ItemsPerThread, typename DigitExtractor>
    ROCPRIM_DEVICE void rank_keys_desc(const Key (&keys)[ItemsPerThread],
                                       unsigned int (&ranks)[ItemsPerThread],
                                       DigitExtractor digit_extractor)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        base_type::rank_keys_desc(keys, ranks, storage, digit_extractor);
    }

    /// \brief Perform ascending radix rank over bit keys partitioned across threads in a block.
    /// This overload accepts a callback used to extract the radix digit from a key, and provides
    /// the counts of each digit and a prefix scan thereof in a blocked arrangement.
    ///
    /// \tparam Key - the key type.
    /// \tparam ItemsPerThread - the number of items contributed by each thread in the block.
    /// \tparam DigitExtractor - type of the unary function object used to extract a digit from
    /// a key.
    /// \param [in] keys - reference to an array of keys provided by a thread. Keys are expected in
    /// warp-striped arrangement.
    /// \param [out] ranks - reference to an array where the final ranks are written to. Ranks are
    /// provided in warp-striped arrangement.
    /// \param [in] storage - reference to a temporary storage object of type \p storage_type.
    /// \param [in] digit_extractor - function object used to convert a key to a digit.
    /// The signature of the \p digit_extractor should be equivalent to the following:
    /// <tt>unsigned int f(const Key &key);</tt>. The signature does not need to have
    /// <tt>const &</tt>, but function object must not modify the objects passed to it.
    /// This function will be used during ranking to extract the digit that indicates
    /// the key's value. Values return by this function object must be in range [0; 1 << RadixBits).
    /// \param [in] prefix - An exclusive prefix scan of the counts per digit.
    /// \param [in] counts - The number of keys with a particular digit in the input, per digit.
    ///
    /// \par Storage reusage
    /// A synchronization barrier should be placed before \p storage is reused
    /// or repurposed: \p __syncthreads() or \p rocprim::syncthreads().
    ///
    /// \code{.cpp}
    /// __global__ void example_kernel(...)
    /// {
    ///     // specialize the block_radix_rank for int, block of 128 threads, and a maximum of 4 bits.
    ///     using block_rank_float = rocprim::block_radix_rank<int, 128, 4>;
    ///     // allocate storage in shared memory
    ///     __shared__ block_rank_float::storage_type storage;
    ///
    ///     int          input[3] = ...;
    ///     unsigned int output[3];
    ///     unsinged int digit_prefix[block_rank_float::digits_per_thread];
    ///     unsinged int digit_counts[block_rank_float::digits_per_thread];
    ///     // execute the block radix rank (ascending)
    ///     block_rank_float().rank_keys(input,
    ///                                  output,
    ///                                  storage,
    ///                                  [](const int& key)
    ///                                  {
    ///                                      // Rank the keys by the lower 4 bits
    ///                                      return key & 0xF;
    ///                                  },
    ///                                  digit_prefix,
    ///                                  digit_counts);
    ///     ...
    /// }
    /// \endcode
    template<typename Key, unsigned ItemsPerThread, typename DigitExtractor>
    ROCPRIM_DEVICE void rank_keys(const Key (&keys)[ItemsPerThread],
                                  unsigned int (&ranks)[ItemsPerThread],
                                  storage_type&  storage,
                                  DigitExtractor digit_extractor,
                                  unsigned int (&prefix)[digits_per_thread],
                                  unsigned int (&counts)[digits_per_thread])
    {
        base_type::rank_keys(keys, ranks, storage, digit_extractor, prefix, counts);
    }
};

END_ROCPRIM_NAMESPACE

/// @}
// end of group blockmodule

#endif // ROCPRIM_BLOCK_BLOCK_RADIX_RANK_HPP_
