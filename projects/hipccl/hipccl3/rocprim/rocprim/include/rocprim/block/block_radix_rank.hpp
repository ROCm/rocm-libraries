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

/// \addtogroup blockmodule
/// @{

BEGIN_ROCPRIM_NAMESPACE

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
template<unsigned int BlockSizeX,
         unsigned int RadixBits,
         bool         MemoizeOuterScan = false,
         unsigned int BlockSizeY       = 1,
         unsigned int BlockSizeZ       = 1>
class block_radix_rank
{
    using digit_counter_type  = unsigned short;
    using packed_counter_type = unsigned int;

    using block_scan_type = ::rocprim::block_scan<packed_counter_type,
                                                  BlockSizeX,
                                                  ::rocprim::block_scan_algorithm::using_warp_scan,
                                                  BlockSizeY,
                                                  BlockSizeZ>;

    static constexpr unsigned int block_size   = BlockSizeX * BlockSizeY * BlockSizeZ;
    static constexpr unsigned int radix_digits = 1 << RadixBits;
    static constexpr unsigned int packing_ratio
        = sizeof(packed_counter_type) / sizeof(digit_counter_type);
    static constexpr unsigned int column_size = radix_digits / packing_ratio;

    // Struct used for creating a raw_storage object for this primitive's temporary storage.
    struct storage_type_
    {
        union
        {
            digit_counter_type  digit_counters[block_size * radix_digits];
            packed_counter_type packed_counters[block_size * column_size];
        };

        typename block_scan_type::storage_type block_scan;
    };

    ROCPRIM_DEVICE ROCPRIM_INLINE digit_counter_type& get_digit_counter(const unsigned int digit,
                                                                        const unsigned int thread,
                                                                        storage_type_&     storage)
    {
        const unsigned int column_counter = digit % column_size;
        const unsigned int sub_counter    = digit / column_size;
        const unsigned int counter
            = (column_counter * block_size + thread) * packing_ratio + sub_counter;
        return storage.digit_counters[counter];
    };

    ROCPRIM_DEVICE ROCPRIM_INLINE void reset_counters(const unsigned int flat_id,
                                                      storage_type_&     storage)
    {
        for(unsigned int i = flat_id; i < block_size * column_size; i += block_size)
        {
            storage.packed_counters[i] = 0;
        }
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE void
        scan_block_counters(storage_type_& storage, packed_counter_type* const packed_counters)
    {
        packed_counter_type block_reduction = 0;
        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < column_size; ++i)
        {
            block_reduction += packed_counters[i];
        }

        packed_counter_type exclusive_prefix = 0;
        packed_counter_type reduction;
        block_scan_type().exclusive_scan(block_reduction,
                                         exclusive_prefix,
                                         0,
                                         reduction,
                                         storage.block_scan);

        ROCPRIM_UNROLL
        for(unsigned int i = 1; i < packing_ratio; i <<= 1)
        {
            exclusive_prefix += reduction << (sizeof(digit_counter_type) * 8 * i);
        }

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < column_size; ++i)
        {
            packed_counter_type counter = packed_counters[i];
            packed_counters[i]          = exclusive_prefix;
            exclusive_prefix += counter;
        }
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE void scan_counters(const unsigned int flat_id,
                                                     storage_type_&     storage)
    {
        packed_counter_type* const shared_counters
            = &storage.packed_counters[flat_id * column_size];

        if ROCPRIM_IF_CONSTEXPR(MemoizeOuterScan)
        {
            packed_counter_type local_counters[column_size];
            ROCPRIM_UNROLL
            for(unsigned int i = 0; i < column_size; ++i)
            {
                local_counters[i] = shared_counters[i];
            }

            scan_block_counters(storage, local_counters);

            ROCPRIM_UNROLL
            for(unsigned int i = 0; i < column_size; ++i)
            {
                shared_counters[i] = local_counters[i];
            }
        }
        else
        {
            scan_block_counters(storage, shared_counters);
        }
    }

    template<typename Key, unsigned int ItemsPerThread, typename DigitExtractor>
    ROCPRIM_DEVICE void rank_keys_impl(const Key (&keys)[ItemsPerThread],
                                       unsigned int (&ranks)[ItemsPerThread],
                                       storage_type_& storage,
                                       DigitExtractor digit_extractor)
    {
        static_assert(block_size * ItemsPerThread < 1u << 16,
                      "The maximum amout of items that block_radix_rank can rank is 2**16.");
        const unsigned int flat_id
            = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();

        reset_counters(flat_id, storage);

        digit_counter_type  thread_prefixes[ItemsPerThread];
        digit_counter_type* digit_counters[ItemsPerThread];

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; ++i)
        {
            const unsigned int digit = digit_extractor(keys[i]);
            digit_counters[i]        = &get_digit_counter(digit, flat_id, storage);
            thread_prefixes[i]       = (*digit_counters[i])++;
        }

        ::rocprim::syncthreads();

        scan_counters(flat_id, storage);

        ::rocprim::syncthreads();

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; ++i)
        {
            ranks[i] = thread_prefixes[i] + *digit_counters[i];
        }
    }

    template<bool Descending, typename Key, unsigned int ItemsPerThread>
    ROCPRIM_DEVICE void rank_keys_impl(const Key (&keys)[ItemsPerThread],
                                       unsigned int (&ranks)[ItemsPerThread],
                                       storage_type_&     storage,
                                       const unsigned int begin_bit,
                                       const unsigned int pass_bits)
    {
        using key_codec    = ::rocprim::detail::radix_key_codec<Key, Descending>;
        using bit_key_type = typename key_codec::bit_key_type;

        bit_key_type bit_keys[ItemsPerThread];
        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; ++i)
        {
            bit_keys[i] = key_codec::encode(keys[i]);
        }

        rank_keys_impl(bit_keys,
                       ranks,
                       storage,
                       [begin_bit, pass_bits](const bit_key_type& key)
                       { return key_codec::extract_digit(key, begin_bit, pass_bits); });
    }

public:
    static constexpr unsigned int digits_per_thread
        = ::rocprim::detail::ceiling_div(radix_digits, block_size);

    /// \brief Struct used to allocate a temporary memory that is required for thread
    /// communication during operations provided by related parallel primitive.
    ///
    /// Depending on the implemention the operations exposed by parallel primitive may
    /// require a temporary storage for thread communication. The storage should be allocated
    /// using keywords \p __shared__. It can be aliased to
    /// an externally allocated memory, or be a part of a union with other storage types
    /// to increase shared memory reusability.
#ifndef DOXYGEN_SHOULD_SKIP_THIS // hides storage_type implementation for Doxygen
    using storage_type = detail::raw_storage<storage_type_>;
#else
    using storage_type = storage_type_; // only for Doxygen
#endif

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
        rank_keys_impl<false>(keys, ranks, storage.get(), begin_bit, pass_bits);
    }

    /// \brief Perform ascending radix rank over keys partitioned across threads in a block.
    ///
    /// * This overload does not accept storage argment. Required shared memory is allocated
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
        rank_keys(keys, ranks, storage.get(), begin_bit, pass_bits);
    }

    /// \brief Perform descending radix rank over keys partitioned across threads in a block.
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
        rank_keys_impl<true>(keys, ranks, storage.get(), begin_bit, pass_bits);
    }

    /// \brief Perform descending radix rank over keys partitioned across threads in a block.
    ///
    /// * This overload does not accept storage argment. Required shared memory is allocated
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
        rank_keys_desc(keys, ranks, storage.get(), begin_bit, pass_bits);
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
        rank_keys_impl(keys, ranks, storage.get(), digit_extractor);
    }

    /// \brief Perform ascending radix rank over bit keys partitioned across threads in a block.
    /// This overload accepts a callback used to extract the radix digit from a key.
    ///
    /// * This overload does not accept storage argment. Required shared memory is allocated
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
        rank_keys(keys, ranks, storage.get(), digit_extractor);
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
        rank_keys_impl(keys,
                       ranks,
                       storage.get(),
                       [&digit_extractor](const Key& key)
                       {
                           const unsigned int digit = digit_extractor(key);
                           return radix_digits - 1 - digit;
                       });
    }

    /// \brief Perform descending radix rank over bit keys partitioned across threads in a block.
    /// This overload accepts a callback used to extract the radix digit from a key.
    ///
    /// * This overload does not accept storage argment. Required shared memory is allocated
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
        rank_keys_desc(keys, ranks, storage.get(), digit_extractor);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE void
        get_exclusive_digit_prefix(unsigned int (&prefix)[digits_per_thread], storage_type& storage)
    {
        const unsigned int flat_id = ::rocprim::detail::block_thread_id<0>();
        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < digits_per_thread; ++i)
        {
            const unsigned int digit = flat_id * digits_per_thread + i;
            if(radix_digits % block_size == 0 || digit < radix_digits)
            {
                // The counter for thread 0 holds the prefix of all the digits at this point.
                prefix[i] = get_digit_counter(digit, 0, storage.get());
            }
        }
    }

    template<unsigned int ItemsPerThread>
    ROCPRIM_DEVICE ROCPRIM_INLINE void get_digit_counts(unsigned int (&counts)[digits_per_thread],
                                                        storage_type& storage)
    {
        const unsigned int flat_id = ::rocprim::detail::block_thread_id<0>();
        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < digits_per_thread; ++i)
        {
            const unsigned int digit = flat_id * digits_per_thread + i;
            if(radix_digits % block_size == 0 || digit < radix_digits)
            {
                // The counter for thread 0 holds the prefix of all the digits at this point.
                // To find the count, subtract the prefix of the next digit with that of the
                // current digit.
                const unsigned int counter = get_digit_counter(digit, 0, storage.get());
                const unsigned int next_counter
                    = digit + 1 == radix_digits ? block_size * ItemsPerThread
                                                : get_digit_counter(digit + 1, 0, storage.get());
                counts[i] = next_counter - counter;
            }
        }
    }
};

END_ROCPRIM_NAMESPACE

/// @}
// end of group blockmodule

#endif // ROCPRIM_BLOCK_BLOCK_RADIX_RANK_HPP_
