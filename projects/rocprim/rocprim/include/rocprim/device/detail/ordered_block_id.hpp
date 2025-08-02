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

#include <type_traits>

#include "../../detail/temp_storage.hpp"
#include "../../intrinsics/atomic.hpp"
#include "../../intrinsics/thread.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

// Helper struct for generating ordered unique ids for blocks in a grid.
template<class T /* id type */ = unsigned int>
struct ordered_block_id
{
    static_assert(std::is_integral<T>::value, "T must be integer");
    using id_type = T;

    // shared memory temporary storage type
    struct storage_type
    {
        id_type id;
    };

    ROCPRIM_HOST static inline
    ordered_block_id create(id_type * id)
    {
        ordered_block_id ordered_id;
        ordered_id.id = id;
        return ordered_id;
    }

    ROCPRIM_HOST static inline
    size_t get_storage_size()
    {
        return sizeof(id_type);
    }

    ROCPRIM_HOST static inline detail::temp_storage::layout get_temp_storage_layout()
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

    id_type* id;
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
    static inline block_id_wrapper create(id_type* /*id*/)
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
    static inline block_id_wrapper create(id_type* id)
    {
        block_id_wrapper id_wrapper;
        id_wrapper.ordered_id = detail::ordered_block_id<id_type>::create(id);
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

} // end of detail namespace

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DETAIL_ORDERED_BLOCK_ID_HPP_
