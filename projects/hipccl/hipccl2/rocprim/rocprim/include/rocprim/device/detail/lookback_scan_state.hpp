// Copyright (c) 2017-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../../functional.hpp"
#include "../../intrinsics.hpp"
#include "../../type_traits.hpp"
#include "../../types.hpp"

#include "../../warp/detail/warp_reduce_crosslane.hpp"
#include "../../warp/detail/warp_scan_crosslane.hpp"

#include "../../detail/binary_op_wrappers.hpp"
#include "../../detail/temp_storage.hpp"
#include "../../detail/various.hpp"

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

    // Helper struct
    struct alignas(sizeof(prefix_underlying_type)) prefix_type
    {
        flag_type_ flag;
        T value;
    };

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
        return sizeof(prefix_underlying_type) * (::rocprim::host_warp_size() + number_of_blocks);
    }

    ROCPRIM_HOST static inline detail::temp_storage::layout
        get_temp_storage_layout(const unsigned int number_of_blocks)
    {
        return detail::temp_storage::layout{get_storage_size(number_of_blocks),
                                            alignof(prefix_underlying_type)};
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    void initialize_prefix(const unsigned int block_id,
                           const unsigned int number_of_blocks)
    {
        constexpr unsigned int padding = ::rocprim::device_warp_size();

        if(block_id < number_of_blocks)
        {
            prefix_type prefix;
            prefix.flag = PREFIX_EMPTY;
            prefix_underlying_type p;
#ifndef __HIP_CPU_RT__
            __builtin_memcpy(&p, &prefix, sizeof(prefix_type));
#else
            std::memcpy(&p, &prefix, sizeof(prefix_type));
#endif
            prefixes[padding + block_id] = p;
        }
        if(block_id < padding)
        {
            prefix_type prefix;
            prefix.flag = PREFIX_INVALID;
            prefix_underlying_type p;
#ifndef __HIP_CPU_RT__
            __builtin_memcpy(&p, &prefix, sizeof(prefix_type));
#else
            std::memcpy(&p, &prefix, sizeof(prefix_type));
#endif
            prefixes[block_id] = p;
        }
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    void set_partial(const unsigned int block_id, const T value)
    {
        this->set(block_id, PREFIX_PARTIAL, value);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    void set_complete(const unsigned int block_id, const T value)
    {
        this->set(block_id, PREFIX_COMPLETE, value);
    }

    // block_id must be > 0
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void get(const unsigned int block_id, flag_type& flag, T& value)
    {
        constexpr unsigned int padding = ::rocprim::device_warp_size();

        prefix_type prefix;

        const unsigned int SLEEP_MAX = 32;
        unsigned int times_through = 1;

        prefix_underlying_type p = ::rocprim::detail::atomic_add(&prefixes[padding + block_id], 0);
#ifndef __HIP_CPU_RT__
        __builtin_memcpy(&prefix, &p, sizeof(prefix_type));
#else
        std::memcpy(&prefix, &p, sizeof(prefix_type));
#endif
        while(prefix.flag == PREFIX_EMPTY)
        {
            if (UseSleep)
            {
                for (unsigned int j = 0; j < times_through; j++)
#ifndef __HIP_CPU_RT__
                    __builtin_amdgcn_s_sleep(1);
#else
                    std::this_thread::sleep_for(std::chrono::microseconds{1});
#endif
                if (times_through < SLEEP_MAX)
                    times_through++;
            }
            // atomic_add(..., 0) is used to load values atomically
            prefix_underlying_type p = ::rocprim::detail::atomic_add(&prefixes[padding + block_id], 0);
#ifndef __HIP_CPU_RT__
            __builtin_memcpy(&prefix, &p, sizeof(prefix_type));
#else
            std::memcpy(&prefix, &p, sizeof(prefix_type));
#endif
        }

        // return
        flag = prefix.flag;
        value = prefix.value;
    }

private:
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void set(const unsigned int block_id, const flag_type flag, const T value)
    {
        constexpr unsigned int padding = ::rocprim::device_warp_size();

        prefix_type prefix = { flag, value };
        prefix_underlying_type p;
#ifndef __HIP_CPU_RT__
        __builtin_memcpy(&p, &prefix, sizeof(prefix_type));
#else
        std::memcpy(&p, &prefix, sizeof(prefix_type));
#endif
        ::rocprim::detail::atomic_exch(&prefixes[padding + block_id], p);
    }

    prefix_underlying_type * prefixes;
};

// Flag, partial and final prefixes are stored in separate arrays.
// Consistency ensured by memory fences between flag and prefixes load/store operations.
template<class T, bool UseSleep>
struct lookback_scan_state<T, UseSleep, false>
{

public:
    using flag_type  = unsigned int;
    using value_type = T;

    // temp_storage must point to allocation of get_storage_size(number_of_blocks) bytes
    ROCPRIM_HOST static inline
    lookback_scan_state create(void* temp_storage, const unsigned int number_of_blocks)
    {
        const auto n = ::rocprim::host_warp_size() + number_of_blocks;
        lookback_scan_state state;

        auto ptr = static_cast<char*>(temp_storage);

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
        const auto n = ::rocprim::host_warp_size() + number_of_blocks;
        size_t size = ::rocprim::detail::align_size(n * sizeof(flag_type));
        size += 2 * ::rocprim::detail::align_size(n * sizeof(T));
        return size;
    }

    ROCPRIM_HOST static inline detail::temp_storage::layout
        get_temp_storage_layout(const unsigned int number_of_blocks)
    {
        size_t alignment = std::max(alignof(flag_type), alignof(T));
        return detail::temp_storage::layout{get_storage_size(number_of_blocks), alignment};
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    void initialize_prefix(const unsigned int block_id,
                           const unsigned int number_of_blocks)
    {
        constexpr unsigned int padding = ::rocprim::device_warp_size();
        if(block_id < number_of_blocks)
        {
            prefixes_flags[padding + block_id] = PREFIX_EMPTY;
        }
        if(block_id < padding)
        {
            prefixes_flags[block_id] = PREFIX_INVALID;
        }
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    void set_partial(const unsigned int block_id, const T value)
    {
        constexpr unsigned int padding = ::rocprim::device_warp_size();

        prefixes_partial_values[padding + block_id] = value;
        ::rocprim::detail::memory_fence_device();
        ::rocprim::detail::atomic_exch(&prefixes_flags[padding + block_id], PREFIX_PARTIAL);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    void set_complete(const unsigned int block_id, const T value)
    {
        constexpr unsigned int padding = ::rocprim::device_warp_size();

        prefixes_complete_values[padding + block_id] = value;
        ::rocprim::detail::memory_fence_device();
        ::rocprim::detail::atomic_exch(&prefixes_flags[padding + block_id], PREFIX_COMPLETE);
    }

    // block_id must be > 0
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void get(const unsigned int block_id, flag_type& flag, T& value)
    {
        constexpr unsigned int padding = ::rocprim::device_warp_size();

        const unsigned int SLEEP_MAX = 32;
        unsigned int times_through = 1;

        // atomic_add(..., 0) is used to load values atomically
        flag = ::rocprim::detail::atomic_add(&prefixes_flags[padding + block_id], 0);
        ::rocprim::detail::memory_fence_device();
        while(flag == PREFIX_EMPTY)
        {
            if (UseSleep)
            {
                for (unsigned int j = 0; j < times_through; j++)
#ifndef __HIP_CPU_RT__
                    __builtin_amdgcn_s_sleep(1);
#else
                    std::this_thread::sleep_for(std::chrono::microseconds{1});
#endif
                if (times_through < SLEEP_MAX)
                    times_through++;
            }

            flag = ::rocprim::detail::atomic_add(&prefixes_flags[padding + block_id], 0);
            ::rocprim::detail::memory_fence_device();
        }

        if(flag == PREFIX_PARTIAL)
            value = prefixes_partial_values[padding + block_id];
        else
            value = prefixes_complete_values[padding + block_id];
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
    ROCPRIM_DEVICE ROCPRIM_INLINE
    lookback_scan_prefix_op(unsigned int block_id,
                            BinaryFunction scan_op,
                            LookbackScanState &scan_state)
        : block_id_(block_id),
          scan_op_(scan_op),
          scan_state_(scan_state)
    {
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    ~lookback_scan_prefix_op() = default;

    ROCPRIM_DEVICE ROCPRIM_INLINE
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
            T, ::rocprim::device_warp_size(), false
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

    ROCPRIM_DEVICE ROCPRIM_INLINE
    T get_prefix()
    {
        flag_type flag;
        T partial_prefix;
        unsigned int previous_block_id = block_id_ - ::rocprim::lane_id() - 1;

        // reduce last warp_size() number of prefixes to
        // get the complete prefix for this block.
        reduce_partial_prefixes(previous_block_id, flag, partial_prefix);
        T prefix = partial_prefix;

        // while we don't load a complete prefix, reduce partial prefixes
        while(::rocprim::detail::warp_all(flag != PREFIX_COMPLETE))
        {
            previous_block_id -= ::rocprim::device_warp_size();
            reduce_partial_prefixes(previous_block_id, flag, partial_prefix);
            prefix = scan_op_(partial_prefix, prefix);
        }
        return prefix;
    }

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

inline hipError_t is_sleep_scan_state_used(bool& use_sleep)
{
    hipDeviceProp_t prop;
    int             deviceId;
    if(const hipError_t error = hipGetDevice(&deviceId))
    {
        return error;
    }
    else if(const hipError_t error = hipGetDeviceProperties(&prop, deviceId))
    {
        return error;
    }
#if HIP_VERSION >= 307
    const int asicRevision = prop.asicRevision;
#else
    const int asicRevision = 0;
#endif
    use_sleep = prop.gcnArch == 908 && asicRevision < 2;
    return hipSuccess;
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
    using storage_type = detail::raw_storage<storage_type_>;

    template<typename PrefixOp>
    static ROCPRIM_DEVICE auto create(PrefixOp& prefix_op, storage_type& storage)
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

    static ROCPRIM_DEVICE T get_reduction(const storage_type& storage)
    {
        return storage.get().block_reduction;
    }

    static ROCPRIM_DEVICE T get_prefix(const storage_type& storage)
    {
        return storage.get().prefix;
    }
};

template<class T, class LookbackScanState, class BinaryOp = ::rocprim::plus<T>>
class offset_lookback_scan_prefix_op
    : public lookback_scan_prefix_op<T, BinaryOp, LookbackScanState>
{
private:
    using base_type = lookback_scan_prefix_op<T, BinaryOp, LookbackScanState>;
    using factory   = detail::offset_lookback_scan_factory<T>;

    ROCPRIM_DEVICE ROCPRIM_INLINE base_type& base()
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

    ROCPRIM_DEVICE ROCPRIM_INLINE T operator()(T reduction)
    {
        return factory::create(base(), storage)(reduction);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE T get_reduction() const
    {
        return factory::get_reduction(storage);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE T get_prefix() const
    {
        return factory::get_prefix(storage);
    }

    // rocThrust uses this implementation detail of rocPRIM, required for backwards compatibility
    ROCPRIM_DEVICE ROCPRIM_INLINE T get_exclusive_prefix() const
    {
        return get_prefix();
    }

private:
    storage_type& storage;
};

} // end of detail namespace

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DETAIL_LOOKBACK_SCAN_STATE_HPP_
