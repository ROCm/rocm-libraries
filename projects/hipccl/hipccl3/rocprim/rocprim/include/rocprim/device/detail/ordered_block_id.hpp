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

#ifndef ROCPRIM_DEVICE_DETAIL_ORDERED_BLOCK_ID_HPP_
#define ROCPRIM_DEVICE_DETAIL_ORDERED_BLOCK_ID_HPP_

#include <type_traits>
#include <limits>

#include "../../detail/various.hpp"
#include "../../intrinsics.hpp"
#include "../../types.hpp"

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

    ROCPRIM_DEVICE inline
    void reset()
    {
        *id = static_cast<id_type>(0);
    }

    ROCPRIM_DEVICE inline
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

} // end of detail namespace

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DETAIL_ORDERED_BLOCK_ID_HPP_
