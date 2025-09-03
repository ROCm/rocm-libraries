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

#ifndef ROCPRIM_DEVICE_DETAIL_LOOKBACK_SCAN_STATE_HPP_
#define ROCPRIM_DEVICE_DETAIL_LOOKBACK_SCAN_STATE_HPP_

#include <cstring>
#include <type_traits>

#include "../../functional.hpp"
#include "../../intrinsics.hpp"
#include "../../type_traits.hpp"
#include "../../types.hpp"

#include "../../warp/detail/warp_reduce_crosslane.hpp"
#include "../../warp/detail/warp_scan_crosslane.hpp"

#include "../../detail/binary_op_wrappers.hpp"
#include "../../detail/temp_storage.hpp"
#include "../../detail/various.hpp"

#include "../config_types.hpp"
#include "rocprim/config.hpp"

// This version is specific for devices with slow __threadfence ("agent" fence which does
// L2 cache flushing and invalidation).
// Fences with "workgroup" scope are used instead to ensure ordering only but not coherence,
// they do not flush and invalidate cache.
// Global coherence of prefixes_*_values is ensured by atomic_load/atomic_store that bypass
// cache.
#ifndef ROCPRIM_DETAIL_LOOKBACK_SCAN_STATE_WITHOUT_SLOW_FENCES
    #if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx942__) || defined(__gfx950__) || defined(__gfx9_4_generic__))
        #define ROCPRIM_DETAIL_LOOKBACK_SCAN_STATE_WITHOUT_SLOW_FENCES 1
    #else
        #define ROCPRIM_DETAIL_LOOKBACK_SCAN_STATE_WITHOUT_SLOW_FENCES 0
    #endif
#endif // ROCPRIM_DETAIL_LOOKBACK_SCAN_STATE_WITHOUT_SLOW_FENCES

extern "C" {
void __builtin_amdgcn_s_sleep(int);
}
BEGIN_ROCPRIM_NAMESPACE

// Single pass prefix scan was implemented based on:
// Merrill, D. and Garland, M. Single-pass Parallel Prefix Scan with Decoupled Look-back.
// Technical Report NVR2016-002, NVIDIA Research. Mar. 2016.

namespace detail
{

enum class lookback_scan_prefix_flag : uint8_t
{
    // flag for padding, values should be discarded
    invalid = 0xFF,
    // initialized, not result in value
    empty = 0,
    // partial prefix value (from single block)
    partial = 1,
    // final prefix value
    complete = 2
};

template<typename T>
struct match_prefix_underlying_type
{
    using value_and_prefix = tuple<T, lookback_scan_prefix_flag>;
    using type
        = select_type<select_type_case<sizeof(value_and_prefix) <= sizeof(uint16_t), uint16_t>,
                      select_type_case<sizeof(value_and_prefix) <= sizeof(uint32_t), uint32_t>,
                      select_type_case<sizeof(value_and_prefix) <= sizeof(uint64_t), uint64_t>,
                      select_type_case<sizeof(value_and_prefix) <= sizeof(uint128_t), uint128_t>,
                      void>;
};

// In the original implementation, lookback scan is not deterministic
// for non-associative operations: This is because the number of lookback
// steps may vary depending on the algorithm that its used in, scanned
// operator, and the hardware of the device running on it. Usually, the
// trade-off is worth the extra speed bonus, but sometimes bitwise
// reproducibility is more important. This enum may be used to tune the
// lookback scan implementation to favor one or the other.
enum class lookback_scan_determinism
{
    // Allow the implementation to produce non-deterministic results.
    nondeterministic,
    // Do not allow the implementation to produce non-deterministic results.
    // This may come at a performance penalty, depending on algorithm and device.
    deterministic,
    // By default, prefer the speedy option.
    default_determinism = nondeterministic,
};

constexpr const int MAX_PAYLOAD_SIZE = ROCPRIM_MAX_ATOMIC_SIZE - 1;

/// \brief Optimized implementation of lookback scan, which is a parallel inclusive scan primitive for device level.
///
/// This object keeps track of prefixes status for a look-back prefix scan. Initially every prefix can be
/// either invalid (padding values) or empty. One thread in a block should later set it to partial, and later to complete.
///
/// \tparam T The accumulator type of the scan operation.
/// \tparam UseSleep [optional] If true, the execution of a wavefront is paused for a short duration, allowing other threads or processes to execute during idle periods.
/// \tparam IsSmall [optional] Dependent on the size of `T`. If it's smaller than 16 bytes (8 bytes if 16 byte atomics are possibly not supported), it's by default set to true.
template<class T, bool UseSleep = false, bool IsSmall = (sizeof(T) <= MAX_PAYLOAD_SIZE)>
struct lookback_scan_state;

/// Reduce lanes `0-valid_items` and return the result in lane 0.
template<typename F, typename T>
ROCPRIM_DEVICE ROCPRIM_INLINE
T lookback_reduce_forward_init(F scan_op, T block_prefix, unsigned int valid_items)
{
    T prefix = block_prefix;
    for(unsigned int i = 0; i < valid_items; ++i)
    {
#ifdef ROCPRIM_DETAIL_HAS_DPP_WF
        prefix = warp_move_dpp<T, 0x134 /* DPP_WF_RL1 */>(prefix);
#else
        prefix = warp_shuffle_down(prefix, 1, ::rocprim::arch::wavefront::size());
#endif
        prefix = scan_op(prefix, block_prefix);
    }
    return prefix;
}

/// Reduce all lanes with the `prefix`, which is taken from lane 0,
/// and return the result in lane 0.
template<typename F, typename T>
ROCPRIM_DEVICE ROCPRIM_INLINE
T lookback_reduce_forward(F scan_op, T prefix, T block_prefix)
{
#ifdef ROCPRIM_DETAIL_HAS_DPP_WF
    for(unsigned int i = 0; i < ::rocprim::arch::wavefront::size(); ++i)
    {
        prefix = warp_move_dpp<T, 0x134 /* DPP_WF_RL1 */>(prefix);
        prefix = scan_op(prefix, block_prefix);
    }
#elif ROCPRIM_DETAIL_USE_DPP == 1
    // If we can't rotate or shift the entire wavefront in one instruction,
    // iterate over rows of 16 lanes and use warp_readlane to communicate across rows.
    constexpr const int row_size = 16;

    for(int j = ::rocprim::arch::wavefront::size(); j > 0; j -= row_size)
    {
        prefix = warp_readlane(
            prefix,
            j /* automatically taken modulo ::rocprim::arch::wavefront::size(), first read is lane 0 */);
        prefix = scan_op(prefix, block_prefix);

        ROCPRIM_UNROLL
        for(int i = 0; i < row_size - 1; ++i)
        {
            prefix = warp_move_dpp<T, 0x101 /* DPP_ROW_SL1 */>(prefix);
            prefix = scan_op(prefix, block_prefix);
        }
    }
#else
    // If no DPP available at all, fall back to shuffles.
    for(unsigned int i = 0; i < ::rocprim::arch::wavefront::size(); ++i)
    {
        prefix = warp_shuffle(prefix, lane_id() + 1, ::rocprim::arch::wavefront::size());
        prefix = scan_op(prefix, block_prefix);
    }
#endif
    return prefix;
}

// Packed flag and prefix value are loaded/stored in one atomic operation.
template<class T, bool UseSleep>
struct lookback_scan_state<T, UseSleep, true>
{
private:
    // Type which is used in store/load operations of block prefix (flag and value).
    // It is 16-, 32- or 64-bit int and can be loaded/stored using single atomic instruction.
    using prefix_underlying_type = typename match_prefix_underlying_type<T>::type;

    // Helper struct
    struct prefix_type
    {
        T                         value;
        lookback_scan_prefix_flag flag;
    };

    static_assert(sizeof(prefix_underlying_type) >= sizeof(prefix_type), "");

public:
    // Type used for flag/flag of block prefix
    using value_type = T;

    static constexpr bool use_sleep = UseSleep;

    /// \brief Initializes the lookback_scan_state with the given temporary storage and the given grid size.
    ///
    /// \param [in,out] state the lookback_scan_state object to be initialized.
    /// \param [in] temp_storage the temporary storage necessary for the calculation. Its size can be queried with the get_storage_size function.
    /// \param [in] number_of_blocks the grid size for the kernel operation.
    /// \param [in] stream the stream which will run the kernel.
    ///
    /// \returns \p hipSuccess (\p 0) after successful scan; otherwise a HIP runtime error of
    /// type \p hipError_t.
    ROCPRIM_HOST_DEVICE
    static inline hipError_t create(lookback_scan_state& state,
                                    void*                temp_storage,
                                    const unsigned int   number_of_blocks,
                                    const hipStream_t /*stream*/)
    {
        (void)number_of_blocks;
        state.prefixes = reinterpret_cast<prefix_underlying_type*>(temp_storage);
        return hipSuccess;
    }

    /// \brief This function queries the size of the temporary storage for the lookback scan algorithm.
    ///
    /// \par Overview
    /// The lookback_scan needs a certain amount of temporary storage for the calculation. This function calculates the necessary size of the storage.
    ///
    /// \param [in] number_of_blocks the grid size for the kernel operation.
    /// \param [in] stream the stream which will run the kernel.
    /// \param [out] storage_size this parameter will contain the storage size in bytes.
    ///
    /// \returns \p hipSuccess (\p 0) after successful scan; otherwise a HIP runtime error of
    /// type \p hipError_t.
    ROCPRIM_HOST_DEVICE
    static inline hipError_t get_storage_size(const unsigned int number_of_blocks,
                                              const hipStream_t  stream,
                                              size_t&            storage_size)
    {
        unsigned int warp_size;
        hipError_t   error = ::rocprim::host_warp_size(stream, warp_size);

        storage_size = sizeof(prefix_underlying_type) * (warp_size + number_of_blocks);

        return error;
    }

    /// \brief This function queries the layout of the temporary storage for the lookback scan algorithm.
    ///
    /// \par Overview
    /// The lookback_scan needs a certain amount of temporary storage for the calculation. This function queries the layout of the storage.
    ///
    /// \param [in] number_of_blocks the grid size for the kernel operation.
    /// \param [in] stream the stream which will run the kernel.
    /// \param [out] layout this parameter will contain the storage layout.
    ///
    /// \returns \p hipSuccess (\p 0) after successful scan; otherwise a HIP runtime error of
    /// type \p hipError_t.
    ROCPRIM_HOST_DEVICE
    static inline hipError_t get_temp_storage_layout(const unsigned int            number_of_blocks,
                                                     const hipStream_t             stream,
                                                     detail::temp_storage::layout& layout)
    {
        size_t     storage_size = 0;
        hipError_t error        = get_storage_size(number_of_blocks, stream, storage_size);
        layout = detail::temp_storage::layout{storage_size, alignof(prefix_underlying_type)};
        return error;
    }

    /// \brief This device function initializes the prefixes of the lookback_scan_state instance.
    ///
    /// \param [in] block_id the prefixes are initialized per block.
    /// \param [in] number_of_blocks grid size.
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void initialize_prefix(const unsigned int block_id, const unsigned int number_of_blocks)
    {
        const unsigned int padding = ::rocprim::arch::wavefront::size();

        if(block_id < number_of_blocks)
        {
            prefix_type prefix;
            prefix.flag = lookback_scan_prefix_flag::empty;
            prefix_underlying_type p;
            memcpy(&p, &prefix, sizeof(prefix_type));
            prefixes[padding + block_id] = p;
        }
        if(block_id < padding)
        {
            prefix_type prefix;
            prefix.flag = lookback_scan_prefix_flag::invalid;
            prefix_underlying_type p;
            memcpy(&p, &prefix, sizeof(prefix_type));
            prefixes[block_id] = p;
        }
    }

    /// \brief This device function sets the given prefix to the given value and to partial flag.
    ///
    /// \param [in] block_id the index of the prefix to be updated.
    /// \param [in] value the value to update the prefix to.
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void set_partial(const unsigned int block_id, const T value)
    {
        this->set(block_id, lookback_scan_prefix_flag::partial, value);
    }

    /// \brief This device function sets the given prefix to the given value and to complete flag.
    ///
    /// \param [in] block_id the index of the prefix to be updated.
    /// \param [in] value the value to update the prefix to.
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void set_complete(const unsigned int block_id, const T value)
    {
        this->set(block_id, lookback_scan_prefix_flag::complete, value);
    }

    /// \brief This device function queries the value and the flag of the given prefix.
    ///
    /// \param [in] block_id the index of the prefix to be queried.
    /// \param [out] flag the flag of the prefix.
    /// \param [out] value the value of the prefix.
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void get(const unsigned int block_id, lookback_scan_prefix_flag& flag, T& value)
    {
        const unsigned int padding = ::rocprim::arch::wavefront::size();

        prefix_type prefix;

        const unsigned int SLEEP_MAX     = 32;
        unsigned int       times_through = 1;

        prefix_underlying_type p = ::rocprim::detail::atomic_load(&prefixes[padding + block_id]);
        memcpy(&prefix, &p, sizeof(prefix_type));
        while(prefix.flag == lookback_scan_prefix_flag::empty)
        {
            if constexpr(UseSleep)
            {
                for(unsigned int j = 0; j < times_through; j++)
                    __builtin_amdgcn_s_sleep(1);
                if(times_through < SLEEP_MAX)
                    times_through++;
            }
            prefix_underlying_type p
                = ::rocprim::detail::atomic_load(&prefixes[padding + block_id]);
            memcpy(&prefix, &p, sizeof(prefix_type));
        }

        // return
        flag  = prefix.flag;
        value = prefix.value;
    }

    /// \brief This device function queries the value of the given prefix. It should only be called after all the blocks/prefixes are complete.
    ///
    /// \param [in] block_id the index of the prefix to be queried.
    ///
    /// \returns the value of the prefix specified by the block_id.
    ROCPRIM_DEVICE ROCPRIM_INLINE
    T get_complete_value(const unsigned int block_id)
    {
        const unsigned int padding = ::rocprim::arch::wavefront::size();

        auto        p = prefixes[padding + block_id];
        prefix_type prefix{};
        memcpy(&prefix, &p, sizeof(prefix_type));
        return prefix.value;
    }

    /// \brief This device function calculates the prefix for the next block, based on this block.
    ///
    /// \tparam F [optional] The type of the scan_op parameter.
    ///
    /// \param [in] scan_op the scan operation used.
    /// \param [in] block_id the index of the prefix to be processed.
    ///
    /// \returns the value of the prefix specified by the block_id.
    template<typename F>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    T get_prefix_forward(F scan_op, unsigned int block_id_)
    {
        unsigned int lookback_block_id = block_id_ - lane_id() - 1;

        // There is one lookback scan per block, though a lookback scan is done by a single warp.
        // Because every lane of the warp checks a different lookback scan state value,
        // we need space for at least ceil(CUs / arch::wavefront::size()) items in the cache,
        // assuming that only one block is active per CU (assumes low occupancy).
        // For MI300, with 304 CUs, we have 304 / 64 = 5 items for the lookback cache.
        // Note that one item is kept in the `block_prefix` register, so we only need to
        // cache 4 values here in the worst case.
        constexpr int max_lookback_per_thread = 4;

        T   cache[max_lookback_per_thread];
        int cache_offset = 0;

        lookback_scan_prefix_flag flag;
        T                         block_prefix;
        this->get(lookback_block_id, flag, block_prefix);

        while(warp_all(flag != lookback_scan_prefix_flag::complete
                       && flag != lookback_scan_prefix_flag::invalid)
              && cache_offset < max_lookback_per_thread)
        {
            cache[cache_offset++] = block_prefix;
            lookback_block_id -= arch::wavefront::size();
            this->get(lookback_block_id, flag, block_prefix);
        }

        // If no flags are complete, we have hit either of the following edge cases:
        // - The lookback_block_id is < 0 for all lanes. In this case, we need to go
        //   forward one block and pop one invalid item off the cache.
        // - We have run out of available space in the cache. In this case, wait until
        //   any of the current lookback flags pointed to by lookback_block_id changes
        //   to complete.
        if(warp_all(flag != lookback_scan_prefix_flag::complete))
        {
            if(warp_all(flag == lookback_scan_prefix_flag::invalid))
            {
                // All invalid, so we have to move one block back to
                // get back to known civilization.
                // Don't forget to pop one item off the cache too.
                lookback_block_id += arch::wavefront::size();
                --cache_offset;
            }

            do
            {
                this->get(lookback_block_id, flag, block_prefix);
            }
            while(warp_all(flag != lookback_scan_prefix_flag::complete));
        }

        // Now just sum all these values to get the prefix
        // Note that the values are striped across the threads.
        // In the first iteration, the current prefix is at the value cache for the current
        // offset at the lowest warp number that has lookback_scan_prefix_flag::complete set.
        const auto bits   = ballot(flag == lookback_scan_prefix_flag::complete);
        const auto lowest = ctz(bits);

        // Now sum all the values from block_prefix that are lower than the current prefix.
        T prefix = lookback_reduce_forward_init(scan_op, block_prefix, lowest);

        // Now sum all from the prior cache.
        // These are all guaranteed to be PARTIAL
        while(cache_offset > 0)
        {
            block_prefix = cache[--cache_offset];
            prefix       = lookback_reduce_forward(scan_op, prefix, block_prefix);
        }

        return warp_readfirstlane(prefix);
    }

private:
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void set(const unsigned int block_id, const lookback_scan_prefix_flag flag, const T value)
    {
        const unsigned int padding = ::rocprim::arch::wavefront::size();

        prefix_type            prefix = {value, flag};
        prefix_underlying_type p;
        memcpy(&p, &prefix, sizeof(prefix_type));
        ::rocprim::detail::atomic_store(&prefixes[padding + block_id], p);
    }

    prefix_underlying_type* prefixes;
};

// Flag, partial and final prefixes are stored in separate arrays.
// Consistency ensured by memory fences between flag and prefixes load/store operations.
template<class T, bool UseSleep>
struct lookback_scan_state<T, UseSleep, false>
{

public:
    using flag_underlying_type = std::underlying_type_t<lookback_scan_prefix_flag>;
    using value_type           = T;

    static constexpr bool use_sleep = UseSleep;

    /// \brief Initializes the lookback_scan_state with the given temporary storage and the given grid size.
    ///
    /// \param [in,out] state the lookback_scan_state object to be initialized.
    /// \param [in] temp_storage the temporary storage necessary for the calculation. Its size can be queried with the get_storage_size function.
    /// \param [in] number_of_blocks the grid size for the kernel operation.
    /// \param [in] stream the stream which will run the kernel.
    ///
    /// \returns \p hipSuccess (\p 0) after successful scan; otherwise a HIP runtime error of
    /// type \p hipError_t.
    ROCPRIM_HOST_DEVICE
    static inline hipError_t create(lookback_scan_state& state,
                                    void*                temp_storage,
                                    const unsigned int   number_of_blocks,
                                    const hipStream_t    stream)
    {
        unsigned int warp_size;
        hipError_t   error = ::rocprim::host_warp_size(stream, warp_size);

        const auto n = warp_size + number_of_blocks;

        auto ptr = static_cast<char*>(temp_storage);

        state.prefixes_partial_values = ptr;
        ptr += ::rocprim::detail::align_size(n * sizeof(value_underlying_type));

        state.prefixes_complete_values = ptr;
        ptr += ::rocprim::detail::align_size(n * sizeof(value_underlying_type));

        state.prefixes_flags = reinterpret_cast<flag_underlying_type*>(ptr);

        return error;
    }

    /// \brief This function queries the size of the temporary storage for the lookback scan algorithm.
    ///
    /// \par Overview
    /// The lookback_scan needs a certain amount of temporary storage for the calculation. This function calculates the necessary size of the storage.
    ///
    /// \param [in] number_of_blocks the grid size for the kernel operation.
    /// \param [in] stream the stream which will run the kernel.
    /// \param [out] storage_size this parameter will contain the storage size in bytes.
    ///
    /// \returns \p hipSuccess (\p 0) after successful scan; otherwise a HIP runtime error of
    /// type \p hipError_t.
    ROCPRIM_HOST_DEVICE
    static inline hipError_t get_storage_size(const unsigned int number_of_blocks,
                                              const hipStream_t  stream,
                                              size_t&            storage_size)
    {
        unsigned int warp_size;
        hipError_t   error = ::rocprim::host_warp_size(stream, warp_size);
        const auto   n     = warp_size + number_of_blocks;
        // Always use sizeof(value_underlying_type) instead of sizeof(T) because storage is
        // allocated by host so it can hold both types no matter what device is used.
        storage_size = 2 * ::rocprim::detail::align_size(n * sizeof(value_underlying_type));
        storage_size += n * sizeof(lookback_scan_prefix_flag);
        return error;
    }

    /// \brief This function queries the layout of the temporary storage for the lookback scan algorithm.
    ///
    /// \par Overview
    /// The lookback_scan needs a certain amount of temporary storage for the calculation. This function queries the layout of the storage.
    ///
    /// \param [in] number_of_blocks the grid size for the kernel operation.
    /// \param [in] stream the stream which will run the kernel.
    /// \param [out] layout this parameter will contain the storage layout.
    ///
    /// \returns \p hipSuccess (\p 0) after successful scan; otherwise a HIP runtime error of
    /// type \p hipError_t.
    ROCPRIM_HOST_DEVICE
    static inline hipError_t get_temp_storage_layout(const unsigned int            number_of_blocks,
                                                     const hipStream_t             stream,
                                                     detail::temp_storage::layout& layout)
    {
        size_t storage_size = 0;
        size_t alignment    = std::max(
            {alignof(lookback_scan_prefix_flag), alignof(T), alignof(value_underlying_type)});
        hipError_t error = get_storage_size(number_of_blocks, stream, storage_size);
        layout           = detail::temp_storage::layout{storage_size, alignment};
        return error;
    }

    /// \brief This device function initializes the prefixes of the lookback_scan_state instance.
    ///
    /// \param [in] block_id the prefixes are initialized per block.
    /// \param [in] number_of_blocks grid size.
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void initialize_prefix(const unsigned int block_id, const unsigned int number_of_blocks)
    {
        const unsigned int padding = ::rocprim::arch::wavefront::size();
        if(block_id < number_of_blocks)
        {
            prefixes_flags[padding + block_id]
                = static_cast<flag_underlying_type>(lookback_scan_prefix_flag::empty);
        }
        if(block_id < padding)
        {
            prefixes_flags[block_id]
                = static_cast<flag_underlying_type>(lookback_scan_prefix_flag::invalid);
        }
    }

    /// \brief Set the given prefix to the given value and to partial flag.
    ///
    /// \param [in] block_id the index of the prefix to be updated.
    /// \param [in] value the value to update the prefix to.
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void set_partial(const unsigned int block_id, const T value)
    {
        this->set(block_id, lookback_scan_prefix_flag::partial, value);
    }

    /// \brief This device function sets the given prefix to the given value and to complete flag.
    ///
    /// \param [in] block_id the index of the prefix to be updated.
    /// \param [in] value the value to update the prefix to.
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void set_complete(const unsigned int block_id, const T value)
    {
        this->set(block_id, lookback_scan_prefix_flag::complete, value);
    }

    /// \brief This device function queries the value and the flag of the given prefix.
    ///
    /// \param [in] block_id the index of the prefix to be queried.
    /// \param [out] flag the flag of the prefix.
    /// \param [out] value the value of the prefix.
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void get(const unsigned int block_id, lookback_scan_prefix_flag& flag, T& value)
    {
        const unsigned int padding = ::rocprim::arch::wavefront::size();

        flag = this->get_flag(block_id);
#if ROCPRIM_DETAIL_LOOKBACK_SCAN_STATE_WITHOUT_SLOW_FENCES
        rocprim::detail::atomic_fence_acquire_order_only();

        const auto* values = static_cast<const value_underlying_type*>(
            flag == lookback_scan_prefix_flag::partial ? prefixes_partial_values
                                                       : prefixes_complete_values);
        value_underlying_type v;
        for(unsigned int i = 0; i < value_underlying_type::words_no; ++i)
        {
            v.words[i] = ::rocprim::detail::atomic_load(&values[padding + block_id].words[i]);
        }
        __builtin_memcpy(&value, &v, sizeof(value));
#else
        ::rocprim::detail::memory_fence_device();

        const auto* values = static_cast<const T*>(flag == lookback_scan_prefix_flag::partial
                                                       ? prefixes_partial_values
                                                       : prefixes_complete_values);
        value              = values[padding + block_id];
#endif
    }

    /// \brief This device function queries the value of the given prefix. It should only be called after all the blocks/prefixes are complete.
    ///
    /// \param [in] block_id the index of the prefix to be queried.
    ///
    /// \returns the value of the prefix specified by the block_id.
    ROCPRIM_DEVICE ROCPRIM_INLINE
    T get_complete_value(const unsigned int block_id)
    {
        const unsigned int padding = ::rocprim::arch::wavefront::size();

#if ROCPRIM_DETAIL_LOOKBACK_SCAN_STATE_WITHOUT_SLOW_FENCES
        T           value;
        const auto* values = static_cast<const value_underlying_type*>(prefixes_complete_values);
        value_underlying_type v;
        for(unsigned int i = 0; i < value_underlying_type::words_no; ++i)
        {
            v.words[i] = ::rocprim::detail::atomic_load(&values[padding + block_id].words[i]);
        }
        __builtin_memcpy(&value, &v, sizeof(value));
        return value;
#else
        const auto* values = static_cast<const T*>(prefixes_complete_values);
        return values[padding + block_id];
#endif
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    T get_partial_value(const unsigned int block_id)
    {
        const unsigned int padding = ::rocprim::arch::wavefront::size();

#if ROCPRIM_DETAIL_LOOKBACK_SCAN_STATE_WITHOUT_SLOW_FENCES
        T           value;
        const auto* values = static_cast<const value_underlying_type*>(prefixes_partial_values);
        value_underlying_type v;
        for(unsigned int i = 0; i < value_underlying_type::words_no; ++i)
        {
            v.words[i] = ::rocprim::detail::atomic_load(&values[padding + block_id].words[i]);
        }
        __builtin_memcpy(&value, &v, sizeof(value));
        return value;
#else
        const auto* values = static_cast<const T*>(prefixes_partial_values);
        return values[padding + block_id];
#endif
    }

    /// \brief This device function calculates the prefix for the next block, based on this block.
    ///
    /// \tparam F [optional] The type of the scan_op parameter.
    ///
    /// \param [in] scan_op the scan operation used.
    /// \param [in] block_id the index of the prefix to be processed.
    ///
    /// \returns the value of the prefix specified by the block_id.
    template<typename F>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    T get_prefix_forward(F scan_op, unsigned int block_id_)
    {
        unsigned int lookback_block_id = block_id_ - lane_id() - 1;

        int cache_offset = 0;

        lookback_scan_prefix_flag flag = this->get_flag(lookback_block_id);

        while(warp_all(flag != lookback_scan_prefix_flag::complete
                       && flag != lookback_scan_prefix_flag::invalid))
        {
            ++cache_offset;
            lookback_block_id -= arch::wavefront::size();
            flag = this->get_flag(lookback_block_id);
        }

        // If no flags are complete, we have hit either of the following edge cases:
        // - The lookback_block_id is < 0 for all lanes. In this case, we need to go
        //   forward one block and pop one invalid item off the cache.
        // - We have run out of available space in the cache. In this case, wait until
        //   any of the current lookback flags pointed to by lookback_block_id changes
        //   to complete.
        if(warp_all(flag != lookback_scan_prefix_flag::complete))
        {
            if(warp_all(flag == lookback_scan_prefix_flag::invalid))
            {
                // All invalid, so we have to move one block back to
                // get back to known civilization.
                // Don't forget to pop one item off the cache too.
                lookback_block_id += arch::wavefront::size();
                --cache_offset;
            }

            do
            {
                flag = this->get_flag(lookback_block_id);
            }
            while(warp_all(flag != lookback_scan_prefix_flag::complete));
        }

        T block_prefix;
        this->get(lookback_block_id, flag, block_prefix);

        // Now just sum all these values to get the prefix
        // Note that the values are striped across the threads.
        // In the first iteration, the current prefix is at the value cache for the current
        // offset at the lowest warp number that has lookback_scan_prefix_flag::complete set.
        const auto bits   = ballot(flag == lookback_scan_prefix_flag::complete);
        const auto lowest = ctz(bits);

        // Now sum all the values from block_prefix that are lower than the current prefix.
        T prefix = lookback_reduce_forward_init(scan_op, block_prefix, lowest);

        // Now sum all from the prior cache.
        // These are all guaranteed to be PARTIAL
        while(cache_offset > 0)
        {
            lookback_block_id += arch::wavefront::size();
            --cache_offset;
            block_prefix = this->get_partial_value(lookback_block_id);
            prefix       = lookback_reduce_forward(scan_op, prefix, block_prefix);
        }

        return warp_readfirstlane(prefix);
    }

private:
    ROCPRIM_DEVICE ROCPRIM_INLINE
    lookback_scan_prefix_flag get_flag(const unsigned int block_id)
    {
        const unsigned int padding = ::rocprim::arch::wavefront::size();

        const unsigned int SLEEP_MAX     = 32;
        unsigned int       times_through = 1;

        lookback_scan_prefix_flag flag = static_cast<lookback_scan_prefix_flag>(
            ::rocprim::detail::atomic_load(&prefixes_flags[padding + block_id]));
        while(flag == lookback_scan_prefix_flag::empty)
        {
            if(UseSleep)
            {
                for(unsigned int j = 0; j < times_through; j++)
                    __builtin_amdgcn_s_sleep(1);
                if(times_through < SLEEP_MAX)
                    times_through++;
            }

            flag = static_cast<lookback_scan_prefix_flag>(
                ::rocprim::detail::atomic_load(&prefixes_flags[padding + block_id]));
        }
        return flag;
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    void set(const unsigned int block_id, const lookback_scan_prefix_flag flag, const T value)
    {
        const unsigned int padding = ::rocprim::arch::wavefront::size();

#if ROCPRIM_DETAIL_LOOKBACK_SCAN_STATE_WITHOUT_SLOW_FENCES
        auto* values = static_cast<value_underlying_type*>(
            flag == lookback_scan_prefix_flag::partial ? prefixes_partial_values
                                                       : prefixes_complete_values);
        value_underlying_type v;
        __builtin_memcpy(&v, &value, sizeof(value));
        for(unsigned int i = 0; i < value_underlying_type::words_no; ++i)
        {
            ::rocprim::detail::atomic_store(&values[padding + block_id].words[i], v.words[i]);
        }
        // Wait for all atomic stores of prefixes_*_values before signaling complete / partial state
        rocprim::detail::atomic_fence_release_vmem_order_only();
#else
        auto* values               = static_cast<T*>(flag == lookback_scan_prefix_flag::partial
                                                         ? prefixes_partial_values
                                                         : prefixes_complete_values);
        values[padding + block_id] = value;
        ::rocprim::detail::memory_fence_device();
#endif

        ::rocprim::detail::atomic_store(&prefixes_flags[padding + block_id],
                                        static_cast<flag_underlying_type>(flag));
    }

    struct value_underlying_type
    {
        static constexpr int words_no = ceiling_div(sizeof(T), sizeof(unsigned int));

        unsigned int words[words_no];
    };

    // We need to separate arrays for partial and final prefixes, because
    // value can be overwritten before flag is changed (flag and value are
    // not stored in single instruction).
    void*                 prefixes_partial_values;
    void*                 prefixes_complete_values;
    flag_underlying_type* prefixes_flags;
};

template<class T,
         class BinaryFunction,
         class LookbackScanState,
         lookback_scan_determinism Determinism = lookback_scan_determinism::default_determinism,
         ::rocprim::arch::wavefront::target TargetWaveSize
         = ::rocprim::arch::wavefront::get_target(),
         typename Enabled = void>
class lookback_scan_prefix_op
{
    static_assert(std::is_same<T, typename LookbackScanState::value_type>::value,
                  "T must be LookbackScanState::value_type");

public:
    ROCPRIM_DEVICE ROCPRIM_INLINE lookback_scan_prefix_op(unsigned int       block_id,
                                                         BinaryFunction     scan_op,
                                                         LookbackScanState& scan_state)
        : block_id_(block_id), scan_op_(scan_op), scan_state_(scan_state)
    {}

    ROCPRIM_DEVICE ROCPRIM_INLINE ~lookback_scan_prefix_op() = default;

private:
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void reduce_partial_prefixes(unsigned int               block_id,
                                 lookback_scan_prefix_flag& flag,
                                 T&                         partial_prefix)
    {
        // Order of reduction must be reversed, because 0th thread has
        // prefix from the (block_id_ - 1) block, 1st thread has prefix
        // from (block_id_ - 2) block etc.
        using headflag_scan_op_type = reverse_binary_op_wrapper<BinaryFunction, T, T>;
        using warp_reduce_prefix_type
            = warp_reduce_crosslane<T,
                                    ::rocprim::arch::wavefront::size_from_target<TargetWaveSize>(),
                                    false>;

        T block_prefix;
        scan_state_.get(block_id, flag, block_prefix);

        auto headflag_scan_op = headflag_scan_op_type(scan_op_);
        warp_reduce_prefix_type().tail_segmented_reduce(
            block_prefix,
            partial_prefix,
            (flag == lookback_scan_prefix_flag::complete),
            headflag_scan_op);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    T get_prefix()
    {
        if constexpr(Determinism == lookback_scan_determinism::nondeterministic)
        {
            lookback_scan_prefix_flag flag;
            T                         partial_prefix;
            unsigned int              previous_block_id = block_id_ - ::rocprim::lane_id() - 1;

            // reduce last warp_size() number of prefixes to
            // get the complete prefix for this block.
            reduce_partial_prefixes(previous_block_id, flag, partial_prefix);
            T prefix = partial_prefix;

            // while we don't load a complete prefix, reduce partial prefixes
            while(::rocprim::detail::warp_all(flag != lookback_scan_prefix_flag::complete))
            {
                previous_block_id -= ::rocprim::arch::wavefront::size_from_target<TargetWaveSize>();
                reduce_partial_prefixes(previous_block_id, flag, partial_prefix);
                prefix = scan_op_(partial_prefix, prefix);
            }
            return prefix;
        }
        else /* Determinism == lookback_scan_state::deterministic */
        {
            return scan_state_.get_prefix_forward(scan_op_, block_id_);
        }
    }

public:
    ROCPRIM_DEVICE ROCPRIM_INLINE
    T operator()(T reduction)
    {
        // Set partial prefix for next block
        if(::rocprim::lane_id() == 0)
        {
            scan_state_.set_partial(block_id_, reduction);
        }

        // Get prefix
        auto prefix = get_prefix();

        // Set complete prefix for next block
        if(::rocprim::lane_id() == 0)
        {
            scan_state_.set_complete(block_id_, scan_op_(prefix, reduction));
        }
        return prefix;
    }

protected:
    unsigned int       block_id_;
    BinaryFunction     scan_op_;
    LookbackScanState& scan_state_;
};

template<class T,
         class BinaryFunction,
         class LookbackScanState,
         lookback_scan_determinism Determinism>
class lookback_scan_prefix_op<T,
                              BinaryFunction,
                              LookbackScanState,
                              Determinism,
                              ::rocprim::arch::wavefront::target::dynamic>
{
public:
    ROCPRIM_DEVICE ROCPRIM_INLINE lookback_scan_prefix_op(unsigned int       block_id,
                                                         BinaryFunction     scan_op,
                                                         LookbackScanState& scan_state)
        : wave32_op(block_id, scan_op, scan_state), wave64_op(block_id, scan_op, scan_state)
    {}

    ROCPRIM_DEVICE ROCPRIM_INLINE ~lookback_scan_prefix_op() = default;

private:
    using lookback_scan_prefix_op_wave32
        = lookback_scan_prefix_op<T,
                                  BinaryFunction,
                                  LookbackScanState,
                                  Determinism,
                                  ::rocprim::arch::wavefront::target::size32>;
    lookback_scan_prefix_op_wave32 wave32_op;
    using lookback_scan_prefix_op_wave64
        = lookback_scan_prefix_op<T,
                                  BinaryFunction,
                                  LookbackScanState,
                                  Determinism,
                                  ::rocprim::arch::wavefront::target::size64>;
    lookback_scan_prefix_op_wave64 wave64_op;

public:
    template<typename... Args>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto operator()(Args&&... args)
    {
        if(::rocprim::arch::wavefront::size() == ROCPRIM_WARP_SIZE_32)
        {
            return wave32_op(args...);
        }
        else
        {
            return wave64_op(args...);
        }
    }
};

// This is a HOST only API
// It is known that early revisions of MI100 (gfx908) hang in the wait loop of
// lookback_scan_state::get() without sleeping (s_sleep).
// is_sleep_scan_state_used() checks the architecture/revision of the device on host in runtime,
// to select the corresponding kernel (with or without sleep). However, since the check is runtime,
// both versions of the kernel must be compiled for all architectures.
// is_lookback_kernel_runnable() can be used in device code to prevent compilation of the version
// with sleep on all device architectures except gfx908.

ROCPRIM_HOST ROCPRIM_INLINE
hipError_t is_sleep_scan_state_used(const hipStream_t stream, bool& use_sleep)
{
    hipDeviceProp_t prop;
    int             device_id;
    if(const hipError_t error = get_device_from_stream(stream, device_id))
    {
        return error;
    }
    else if(const hipError_t error = hipGetDeviceProperties(&prop, device_id))
    {
        return error;
    }
#if HIP_VERSION >= 307
    const int asicRevision = prop.asicRevision;
#else
    const int asicRevision = 0;
#endif
    use_sleep = std::string(prop.gcnArchName).find("908") != std::string::npos && asicRevision < 2;
    return hipSuccess;
}

template<typename LookbackScanState>
constexpr bool is_lookback_kernel_runnable()
{
#if !defined(ROCPRIM_TARGET_SPIRV) || ROCPRIM_TARGET_SPIRV == 0
    if(device_target_arch() == target_arch::gfx908)
    {
        // For gfx908 kernels with both version of lookback_scan_state can run: with and without
        // sleep
        return true;
    }
    // For other GPUs only a kernel without sleep can run
    return !LookbackScanState::use_sleep;
#else
    return true;
#endif
}

template<typename T>
class offset_lookback_scan_factory
{
private:
    struct storage_type_
    {
        T block_reduction;
        T prefix;
    };

public:
    ROCPRIM_DETAIL_SUPPRESS_DEPRECATION_WITH_PUSH
    using storage_type = detail::raw_storage<storage_type_>;
    ROCPRIM_DETAIL_SUPPRESS_DEPRECATION_POP

    template<typename PrefixOp>
    static ROCPRIM_DEVICE
    auto create(PrefixOp& prefix_op, storage_type& storage)
    {
        return [&](T reduction) mutable
        {
            auto prefix = prefix_op(reduction);
            if(::rocprim::lane_id() == 0)
            {
                storage.get().block_reduction = std::move(reduction);
                storage.get().prefix          = prefix;
            }
            return prefix;
        };
    }

    static ROCPRIM_DEVICE
    T get_reduction(const storage_type& storage)
    {
        return storage.get().block_reduction;
    }

    static ROCPRIM_DEVICE
    T get_prefix(const storage_type& storage)
    {
        return storage.get().prefix;
    }
};

template<class T,
         class LookbackScanState,
         class BinaryOp                        = ::rocprim::plus<T>,
         lookback_scan_determinism Determinism = lookback_scan_determinism::default_determinism>
class offset_lookback_scan_prefix_op
    : public lookback_scan_prefix_op<T, BinaryOp, LookbackScanState, Determinism>
{
private:
    using base_type = lookback_scan_prefix_op<T, BinaryOp, LookbackScanState, Determinism>;
    using factory   = detail::offset_lookback_scan_factory<T>;

    ROCPRIM_DEVICE ROCPRIM_INLINE
    base_type& base()
    {
        return *this;
    }

public:
    using storage_type = typename factory::storage_type;

    ROCPRIM_DEVICE ROCPRIM_INLINE offset_lookback_scan_prefix_op(unsigned int       block_id,
                                                                LookbackScanState& state,
                                                                storage_type&      storage,
                                                                BinaryOp binary_op = BinaryOp())
        : base_type(block_id, BinaryOp(std::move(binary_op)), state), storage(storage)
    {}

    ROCPRIM_DEVICE ROCPRIM_INLINE
    T operator()(T reduction)
    {
        return factory::create(base(), storage)(reduction);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    T get_reduction() const
    {
        return factory::get_reduction(storage);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    T get_prefix() const
    {
        return factory::get_prefix(storage);
    }

    // rocThrust uses this implementation detail of rocPRIM, required for backwards compatibility
    ROCPRIM_DEVICE ROCPRIM_INLINE
    T get_exclusive_prefix() const
    {
        return get_prefix();
    }

private:
    storage_type& storage;
};

} // namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DETAIL_LOOKBACK_SCAN_STATE_HPP_
