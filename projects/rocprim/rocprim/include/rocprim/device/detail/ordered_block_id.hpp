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

#ifndef ROCPRIM_DEVICE_DETAIL_ORDERED_BLOCK_ID_HPP_
#define ROCPRIM_DEVICE_DETAIL_ORDERED_BLOCK_ID_HPP_

#include "../../config.hpp"
#include "../../detail/temp_storage.hpp"
#include "../../intrinsics/atomic.hpp"
#include "../../intrinsics/thread.hpp"
#include "../config_types.hpp"

#include <atomic>
#include <exception>
#include <type_traits>

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

/// Helper struct for generating ordered unique ids for blocks in a grid.
template<class T /* id type */ = unsigned int>
struct ordered_block_id
{
    // It's a bit confusing on how this API *should* be used. So here is how it should be initialized:
    //   using ordered_bid_type = ordered_block_id<>;
    //   ordered_bid_type::id_type* ordered_bid_storage;
    //
    //   detail::temp_storage::make_linear_partition(
    //     detail::temp_storage::make_partition(&ordered_bid_storage, ordered_bid_type::get_temp_storage_layout())
    //   );
    //
    //   auto ordered_bid = ordered_bid_type::create(ordered_bid_storage);
    //
    //   my_kernel<<<x, y>>>(ordered_bid);
    //
    // On the kernel side it should be:
    //   __globbal__ my_kernel(ordered_block_id<> ordered_bid)
    //   {
    //     __shared__ ordered_block_id<>::storage ordered_bid_storage;
    //
    //     auto tid      = threadIdx.x;
    //     auto block_id = ordered_bid.get(tid, ordered_bid_storage);
    //   }
    static_assert(std::is_integral<T>::value, "T must be integer");
    using id_type = T;

    /// Pointer to global memory that contains the atomic counter for block id.
    id_type* id;

    /// Shared memory for sharing the received block id to other threads in the block.
    struct storage_type
    {
        id_type id;
    };

    ROCPRIM_HOST
    static inline ordered_block_id create(id_type* ordered_bid_storage)
    {
        ordered_block_id ordered_id;
        ordered_id.id = ordered_bid_storage;
        return ordered_id;
    }

    ROCPRIM_HOST
    static inline size_t get_storage_size()
    {
        return sizeof(id_type);
    }

    ROCPRIM_HOST
    static inline detail::temp_storage::layout get_temp_storage_layout()
    {
        return detail::temp_storage::layout{get_storage_size(), alignof(id_type)};
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    void reset()
    {
        *id = static_cast<id_type>(0);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    id_type get(unsigned int tid, storage_type& storage)
    {
        if(tid == 0)
        {
            storage.id = ::rocprim::detail::atomic_add(this->id, 1);
        }
        ::rocprim::syncthreads();
        return storage.id;
    }

    /// Resets the ordered block id from host. Don't use this if we have an init kernel!
    /// Call `ordered_block_id::reset()` from that kernel instead.
    ROCPRIM_HOST ROCPRIM_INLINE
    hipError_t reset_from_host(const hipStream_t = hipStreamDefault)
    {
        return hipMemset(id, 0, sizeof(id_type));
    }
};

template<class T = unsigned int, bool UsingOrderedBlockId = false>
struct block_id_wrapper;

template<class T>
struct block_id_wrapper<T, false>
{
    using id_type = T;

    // shared memory temporary storage type
    struct storage_type
    {};

    ROCPRIM_HOST
    static inline block_id_wrapper create(void* /*id*/)
    {
        block_id_wrapper ordered_id;
        return ordered_id;
    }

    ROCPRIM_HOST
    static inline size_t get_storage_size()
    {
        return 0;
    }

    ROCPRIM_HOST
    static inline detail::temp_storage::layout get_temp_storage_layout()
    {
        return detail::temp_storage::layout{get_storage_size(), 0};
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    void reset()
    {}

    ROCPRIM_HOST ROCPRIM_INLINE
    hipError_t reset_from_host(const hipStream_t /*stream*/)
    {
        return hipSuccess;
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    id_type get(unsigned int /*tid*/, storage_type& /*storage*/)
    {
        return ::rocprim::detail::block_id<0>();
    }
};

template<class T>
struct block_id_wrapper<T, true>
{
    using id_type = T;

    using storage_type = typename ::rocprim::detail::ordered_block_id<id_type>::storage_type;

    ROCPRIM_HOST
    static inline block_id_wrapper create(void* id)
    {
        block_id_wrapper id_wrapper;
        id_wrapper.ordered_id
            = detail::ordered_block_id<id_type>::create(static_cast<id_type*>(id));
        return id_wrapper;
    }

    ROCPRIM_HOST
    static inline size_t get_storage_size()
    {
        return ::rocprim::detail::ordered_block_id<id_type>::get_storage_size();
    }

    ROCPRIM_HOST
    static inline detail::temp_storage::layout get_temp_storage_layout()
    {
        return ::rocprim::detail::ordered_block_id<id_type>::get_temp_storage_layout();
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    void reset()
    {
        ordered_id.reset();
    }

    ROCPRIM_HOST ROCPRIM_INLINE
    hipError_t reset_from_host(const hipStream_t stream)
    {
        return hipMemsetAsync(ordered_id.id, 0, sizeof(id_type), stream);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    id_type get(unsigned int tid, storage_type& storage)
    {
        auto id = ordered_id.get(tid, storage);
        ::rocprim::syncthreads();
        return id;
    }

    ::rocprim::detail::ordered_block_id<id_type> ordered_id;
};

ROCPRIM_INLINE ROCPRIM_HOST
hipError_t check_if_using_atomic_block_id(hipStream_t stream, bool& enable)
{
    // Define possible options
    enum class use_atomic_block_id : int
    {
        never  = 0,
        hotfix = 1,
        always = 2,

        default_option = use_atomic_block_id::hotfix,
    };

    struct data_t
    {
        bool                valid  = false;
        use_atomic_block_id option = use_atomic_block_id::hotfix;
    };

    static_assert(std::is_trivially_copyable_v<data_t>);

    // Store our data in a static atomic.
    static std::atomic<data_t> cache{data_t{}};

    // First load the atomic, if it's invalid, we need to check the env vars.
    data_t data = cache.load(std::memory_order_acquire);
    if(!data.valid)
    {
        // Try to parse the env var, if it fails fall back to the default option.
        auto* env = std::getenv("ROCPRIM_USE_ATOMIC_BLOCK_ID");

        data.option = use_atomic_block_id::default_option;
        try
        {
            if(env != nullptr)
            {
                data.option = use_atomic_block_id{std::stoi(env)};
            }
        }
        catch(std::exception)
        {
            data.option = use_atomic_block_id::default_option;
        }

        data.valid = true;
        // Now finally update our static atomic.
        cache.store(data, std::memory_order_release);
    }

    // Now we have our data, we need to check what the behaviour is.
    bool needs_hotfix = false;
    if(data.option == use_atomic_block_id::hotfix)
    {
        // First get the device ID.
        int device_id;
        ROCPRIM_RETURN_ON_ERROR(::rocprim::detail::get_device_from_stream(stream, device_id));

        // Then we get the arch. This property is cached.
        target_arch arch;
        ROCPRIM_RETURN_ON_ERROR(rocprim::detail::get_device_arch(device_id, arch));

        needs_hotfix = (arch == target_arch::gfx942) || (arch == target_arch::gfx950);
    }

    switch(data.option)
    {
            // clang-format off
        case use_atomic_block_id::never:
            enable = false;
            break;
        case use_atomic_block_id::always:
            enable = true;
            break;
        case use_atomic_block_id::hotfix:
        default:
            enable = needs_hotfix;
        //clang-format on
    }

    return hipSuccess;
}

} // end of detail namespace

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DETAIL_ORDERED_BLOCK_ID_HPP_
