// Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../../detail/various.hpp"
#include "../../intrinsics.hpp"
#include "../../types.hpp"

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
//
// is_arithmetic - arithmetic types up to 8 bytes have separate faster
// and simpler implementation. See below.
// TODO: consider other types that can be loaded in single op.
template<class T, bool is_arithmetic = std::is_arithmetic<T>::value>
struct lookback_scan_state;

// Flag and prefix value are load/store in one operation. Volatile
// loads/stores are not used as there is no ordering of load/store
// operation within one prefix (prefix_type).
template<class T>
struct lookback_scan_state<T, true>
{
private:
    using flag_type_ =
        typename std::conditional<
            sizeof(T) == 8,
            long long,
            typename std::conditional<
                sizeof(T) == 4,
                int,
                typename std::conditional<
                    sizeof(T) == 2,
                    short,
                    char
                >::type
            >::type
        >::type;

    // Type which is used in store/load operations of block prefix (flag and value).
    // It is essential that this type is load/store using single instruction.
    using prefix_underlying_type = typename make_vector_type<flag_type_, 2>::type;
    static constexpr unsigned int padding = 0;//::rocprim::warp_size();

    // Helper struct
    struct prefix_type
    {
        flag_type_ flag;
        T value;
    } __attribute__((aligned(sizeof(prefix_underlying_type))));

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
        prefix_underlying_type prefix;
        if(block_id < number_of_blocks)
        {
            reinterpret_cast<prefix_type*>(&prefix)->flag = PREFIX_EMPTY;
            prefixes[padding + block_id] = prefix;
        }
        if(block_id < padding)
        {
            reinterpret_cast<prefix_type*>(&prefix)->flag = PREFIX_INVALID;
            prefixes[block_id] = prefix;
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
        do
        {
            ::rocprim::detail::memory_fence_system();
            auto p = prefixes[padding + block_id];
            prefix = *reinterpret_cast<prefix_type*>(&p);
        } while(prefix.flag == PREFIX_EMPTY);

        // return
        flag = prefix.flag;
        value = prefix.value;
    }

private:
    ROCPRIM_DEVICE inline
    void set(const unsigned int block_id, const flag_type flag, const T value)
    {
        prefix_type prefix = { flag, value };
        prefix_underlying_type p = *reinterpret_cast<prefix_underlying_type*>(&prefix);
        prefixes[padding + block_id] = p;
    }

    prefix_underlying_type * prefixes;
};


#define ROCPRIM_DETAIL_LOOKBACK_SCAN_STATE_USE_VOLATILE 1
// This does not work for unknown reasons. Lookback-based scan should
// be only enabled for arithmetic types for now.
template<class T>
struct lookback_scan_state<T, false>
{
private:
    static constexpr unsigned int padding = 0;//::rocprim::warp_size();

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
        #ifdef ROCPRIM_DETAIL_LOOKBACK_SCAN_STATE_USE_VOLATILE
            store_volatile(&prefixes_partial_values[padding + block_id], value);
            ::rocprim::detail::memory_fence_device();
            store_volatile<flag_type>(&prefixes_flags[padding + block_id], PREFIX_PARTIAL);
        #else
            prefixes_partial_values[padding + block_id] = value;
            // ::rocprim::detail::memory_fence_device() (aka __threadfence()) should be
            // enough, but does not work when T is 32 bytes or bigger.
            ::rocprim::detail::memory_fence_system();
            prefixes_flags[padding + block_id] = PREFIX_PARTIAL;
        #endif
    }

    ROCPRIM_DEVICE inline
    void set_complete(const unsigned int block_id, const T value)
    {
        #ifdef ROCPRIM_DETAIL_LOOKBACK_SCAN_STATE_USE_VOLATILE
            store_volatile(&prefixes_complete_values[padding + block_id], value);
            ::rocprim::detail::memory_fence_device();
            store_volatile<flag_type>(&prefixes_flags[padding + block_id], PREFIX_COMPLETE);
        #else
            prefixes_complete_values[padding + block_id] = value;
            // ::rocprim::detail::memory_fence_device() (aka __threadfence()) should be
            // enough, but does not work when T is 32 bytes or bigger.
            ::rocprim::detail::memory_fence_system();
            prefixes_flags[padding + block_id] = PREFIX_COMPLETE;
        #endif
    }

    // block_id must be > 0
    ROCPRIM_DEVICE inline
    void get(const unsigned int block_id, flag_type& flag, T& value)
    {
        #ifdef ROCPRIM_DETAIL_LOOKBACK_SCAN_STATE_USE_VOLATILE
            do
            {
                ::rocprim::detail::memory_fence_system();
                flag = load_volatile(&prefixes_flags[padding + block_id]);
            } while(flag == PREFIX_EMPTY);

            if(flag == PREFIX_PARTIAL)
                value = load_volatile(&prefixes_partial_values[padding + block_id]);
            else
                value = load_volatile(&prefixes_complete_values[padding + block_id]);
        #else
            do
            {
                ::rocprim::detail::memory_fence_system();
                flag = prefixes_flags[padding + block_id];
            } while(flag == PREFIX_EMPTY);

            if(flag == PREFIX_PARTIAL)
                value = prefixes_partial_values[padding + block_id];
            else
                value = prefixes_complete_values[padding + block_id];
        #endif
    }

private:
    flag_type * prefixes_flags;
    // We need to seprate arrays for partial and final prefixes, because
    // value can be overwritten before flag is changed (flag and value are
    // not stored in single instruction).
    T * prefixes_partial_values;
    T * prefixes_complete_values;
};

} // end of detail namespace

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DETAIL_LOOKBACK_SCAN_STATE_HPP_
