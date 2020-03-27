// Copyright (c) 2017-2020 Advanced Micro Devices, Inc. All rights reserved.
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

#include <type_traits>

#include "../../intrinsics.hpp"
#include "../../types.hpp"
#include "../../type_traits.hpp"

#include "../../warp/detail/warp_reduce_crosslane.hpp"
#include "../../warp/detail/warp_scan_crosslane.hpp"

#include "../../detail/various.hpp"
#include "../../detail/binary_op_wrappers.hpp"

extern "C"
{
    void __builtin_amdgcn_s_sleep(int);
}
BEGIN_ROCPRIM_NAMESPACE

// Single pass prefix scan was implemented based on:
// Merrill, D. and Garland, M. Single-pass Parallel Prefix Scan with Decoupled Look-back.
// Technical Report NVR2016-001, NVIDIA Research. Mar. 2016.

namespace detail
{

enum prefix_flag
{
    // flag for padding, values should be discarded
    PREFIX_INVALID = -1,
    // initialized, not result in value
    PREFIX_EMPTY = 0,
    // partial prefix value (from single block)
    PREFIX_PARTIAL = 1,
    // final prefix value
    PREFIX_COMPLETE = 2
};

// lookback_scan_state object keeps track of prefixes status for
// a look-back prefix scan. Initially every prefix can be either
// invalid (padding values) or empty. One thread in a block should
// later set it to partial, and later to complete.
template<class T, bool UseSleep = false, bool IsSmall = (sizeof(T) <= 4)>
struct lookback_scan_state;

// Packed flag and prefix value are loaded/stored in one atomic operation.
template<class T, bool UseSleep>
struct lookback_scan_state<T, UseSleep, true>
{
private:
    using flag_type_ = char;

    // Type which is used in store/load operations of block prefix (flag and value).
    // It is 32-bit or 64-bit int and can be loaded/stored using single atomic instruction.
    using prefix_underlying_type =
        typename std::conditional<
            (sizeof(T) > 2),
            unsigned long long,
            unsigned int
        >::type;

    static constexpr unsigned int padding = ::rocprim::warp_size();

    // Helper struct
    struct prefix_type
    {
        flag_type_ flag;
        T value;
    } __attribute__((aligned(sizeof(prefix_underlying_type))));

    static_assert(sizeof(prefix_underlying_type) == sizeof(prefix_type), "");

public:
    // Type used for flag/flag of block prefix
    using flag_type = flag_type_;
    using value_type = T;

    // temp_storage must point to allocation of get_storage_size(number_of_blocks) bytes
    ROCPRIM_HOST static inline
    lookback_scan_state create(void* temp_storage, const unsigned int number_of_blocks)
    {
        (void) number_of_blocks;
        lookback_scan_state state;
        state.prefixes = reinterpret_cast<prefix_underlying_type*>(temp_storage);
        return state;
    }

    ROCPRIM_HOST static inline
    size_t get_storage_size(const unsigned int number_of_blocks)
    {
        return sizeof(prefix_underlying_type) * (padding + number_of_blocks);
    }

    ROCPRIM_DEVICE inline
    void initialize_prefix(const unsigned int block_id,
                           const unsigned int number_of_blocks)
    {
        if(block_id < number_of_blocks)
        {
            prefix_type prefix;
            prefix.flag = PREFIX_EMPTY;
            prefix_underlying_type p;
            __builtin_memcpy(&p, &prefix, sizeof(prefix_type));
            prefixes[padding + block_id] = p;
        }
        if(block_id < padding)
        {
            prefix_type prefix;
            prefix.flag = PREFIX_INVALID;
            prefix_underlying_type p;
            __builtin_memcpy(&p, &prefix, sizeof(prefix_type));
            prefixes[block_id] = p;
        }
    }

    ROCPRIM_DEVICE inline
    void set_partial(const unsigned int block_id, const T value)
    {
        this->set(block_id, PREFIX_PARTIAL, value);
    }

    ROCPRIM_DEVICE inline
    void set_complete(const unsigned int block_id, const T value)
    {
        this->set(block_id, PREFIX_COMPLETE, value);
    }

    // block_id must be > 0
    ROCPRIM_DEVICE inline
    void get(const unsigned int block_id, flag_type& flag, T& value)
    {
        prefix_type prefix;

        const uint SLEEP_MAX = 32;
        uint times_through = 1;

        prefix_underlying_type p = ::rocprim::detail::atomic_add(&prefixes[padding + block_id], 0);
        __builtin_memcpy(&prefix, &p, sizeof(prefix_type));
        while(prefix.flag == PREFIX_EMPTY)
        {
            if (UseSleep)
            {
                for (uint j = 0; j < times_through; j++)
                    __builtin_amdgcn_s_sleep(1);
                if (times_through < SLEEP_MAX)
                    times_through++;
            }
            // atomic_add(..., 0) is used to load values atomically
            prefix_underlying_type p = ::rocprim::detail::atomic_add(&prefixes[padding + block_id], 0);
            __builtin_memcpy(&prefix, &p, sizeof(prefix_type));
        }

        // return
        flag = prefix.flag;
        value = prefix.value;
    }

private:
    ROCPRIM_DEVICE inline
    void set(const unsigned int block_id, const flag_type flag, const T value)
    {
        prefix_type prefix = { flag, value };
        prefix_underlying_type p;
        __builtin_memcpy(&p, &prefix, sizeof(prefix_type));
        ::rocprim::detail::atomic_exch(&prefixes[padding + block_id], p);
    }

    prefix_underlying_type * prefixes;
};

// Flag, partial and final prefixes are stored in separate arrays.
// Consistency ensured by memory fences between flag and prefixes load/store operations.
template<class T, bool UseSleep>
struct lookback_scan_state<T, UseSleep, false>
{
private:
    static constexpr unsigned int padding = ::rocprim::warp_size();

public:
    using flag_type = char;
    using value_type = T;

    // temp_storage must point to allocation of get_storage_size(number_of_blocks) bytes
    ROCPRIM_HOST static inline
    lookback_scan_state create(void* temp_storage, const unsigned int number_of_blocks)
    {
        const auto n = padding + number_of_blocks;
        lookback_scan_state state;

        auto ptr = reinterpret_cast<char*>(temp_storage);

        state.prefixes_flags = reinterpret_cast<flag_type*>(ptr);
        ptr += ::rocprim::detail::align_size(n * sizeof(flag_type));

        state.prefixes_partial_values = reinterpret_cast<T*>(ptr);
        ptr += ::rocprim::detail::align_size(n * sizeof(T));

        state.prefixes_complete_values = reinterpret_cast<T*>(ptr);
        return state;
    }

    ROCPRIM_HOST static inline
    size_t get_storage_size(const unsigned int number_of_blocks)
    {
        const auto n = padding + number_of_blocks;
        size_t size = ::rocprim::detail::align_size(n * sizeof(flag_type));
        size += 2 * ::rocprim::detail::align_size(n * sizeof(T));
        return size;
    }

    ROCPRIM_DEVICE inline
    void initialize_prefix(const unsigned int block_id,
                           const unsigned int number_of_blocks)
    {
        if(block_id < number_of_blocks)
        {
            prefixes_flags[padding + block_id] = PREFIX_EMPTY;
        }
        if(block_id < padding)
        {
            prefixes_flags[block_id] = PREFIX_INVALID;
        }
    }

    ROCPRIM_DEVICE inline
    void set_partial(const unsigned int block_id, const T value)
    {
        store_volatile(&prefixes_partial_values[padding + block_id], value);
        ::rocprim::detail::memory_fence_device();
        store_volatile<flag_type>(&prefixes_flags[padding + block_id], PREFIX_PARTIAL);
    }

    ROCPRIM_DEVICE inline
    void set_complete(const unsigned int block_id, const T value)
    {
        store_volatile(&prefixes_complete_values[padding + block_id], value);
        ::rocprim::detail::memory_fence_device();
        store_volatile<flag_type>(&prefixes_flags[padding + block_id], PREFIX_COMPLETE);
    }

    // block_id must be > 0
    ROCPRIM_DEVICE inline
    void get(const unsigned int block_id, flag_type& flag, T& value)
    {
        const uint SLEEP_MAX = 32;
        uint times_through = 1;

        flag = load_volatile(&prefixes_flags[padding + block_id]);
        ::rocprim::detail::memory_fence_device();
        while(flag == PREFIX_EMPTY)
        {
            if (UseSleep)
            {
                for (uint j = 0; j < times_through; j++)
                    __builtin_amdgcn_s_sleep(1);
                if (times_through < SLEEP_MAX)
                    times_through++;
            }

            flag = load_volatile(&prefixes_flags[padding + block_id]);
            ::rocprim::detail::memory_fence_device();
        }

        if(flag == PREFIX_PARTIAL)
            value = load_volatile(&prefixes_partial_values[padding + block_id]);
        else
            value = load_volatile(&prefixes_complete_values[padding + block_id]);
    }

private:
    flag_type * prefixes_flags;
    // We need to separate arrays for partial and final prefixes, because
    // value can be overwritten before flag is changed (flag and value are
    // not stored in single instruction).
    T * prefixes_partial_values;
    T * prefixes_complete_values;
};

template<class T, class BinaryFunction, class LookbackScanState>
class lookback_scan_prefix_op
{
    using flag_type = typename LookbackScanState::flag_type;
    static_assert(
        std::is_same<T, typename LookbackScanState::value_type>::value,
        "T must be LookbackScanState::value_type"
    );

public:
    ROCPRIM_DEVICE inline
    lookback_scan_prefix_op(unsigned int block_id,
                            BinaryFunction scan_op,
                            LookbackScanState &scan_state)
        : block_id_(block_id),
          scan_op_(scan_op),
          scan_state_(scan_state)
    {
    }

    ROCPRIM_DEVICE inline
    ~lookback_scan_prefix_op() = default;

    ROCPRIM_DEVICE inline
    void reduce_partial_prefixes(unsigned int block_id,
                                 flag_type& flag,
                                 T& partial_prefix)
    {
        // Order of reduction must be reversed, because 0th thread has
        // prefix from the (block_id_ - 1) block, 1st thread has prefix
        // from (block_id_ - 2) block etc.
        using headflag_scan_op_type = reverse_binary_op_wrapper<
            BinaryFunction, T, T
        >;
        using warp_reduce_prefix_type = warp_reduce_crosslane<
            T, ::rocprim::warp_size(), false
        >;

        T block_prefix;
        scan_state_.get(block_id, flag, block_prefix);

        auto headflag_scan_op = headflag_scan_op_type(scan_op_);
        warp_reduce_prefix_type()
            .tail_segmented_reduce(
                block_prefix,
                partial_prefix,
                (flag == PREFIX_COMPLETE),
                headflag_scan_op
            );
    }

    ROCPRIM_DEVICE inline
    T get_prefix()
    {
        flag_type flag;
        unsigned int previous_block_id = block_id_ - ::rocprim::lane_id() - 1;
        bool is_prefix_initialized = false;
        T prefix;

        do
        {
            // reduce last warp_size() number of prefixes to
            // get the complete prefix for this block.
            T partial_prefix;
            reduce_partial_prefixes(previous_block_id, flag, partial_prefix);
            if(!is_prefix_initialized)
            {
                prefix = partial_prefix;
                is_prefix_initialized = true;
            }
            else
            {
                prefix = scan_op_(partial_prefix, prefix);
            }
            previous_block_id -= ::rocprim::warp_size();
            // while we don't load a complete prefix, reduce partial prefixes
        } while(::rocprim::detail::warp_all(flag != PREFIX_COMPLETE));
        return prefix;
    }

    ROCPRIM_DEVICE inline
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

} // end of detail namespace

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DETAIL_LOOKBACK_SCAN_STATE_HPP_
