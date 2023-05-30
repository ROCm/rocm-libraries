/*
 *  Copyright 2018-2020 NVIDIA Corporation
 *  Modifications Copyright© 2019-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*! \file thrust/system/hip/memory_resource.h
 *  \brief Managing memory associated with Thrust's hip system.
 */

#pragma once

#include <thrust/mr/memory_resource.h>
#include <thrust/system/hip/detail/guarded_hip_runtime_api.h>
#include <thrust/system/hip/pointer.h>

#include <thrust/system/detail/bad_alloc.h>
#include <thrust/system/hip/error.h>
#include <thrust/system/hip/detail/util.h>

#include <thrust/mr/host_memory_resource.h>

THRUST_NAMESPACE_BEGIN

namespace system
{
namespace hip
{
namespace detail
{

    typedef hipError_t (*allocation_fn)(void **, std::size_t);
    typedef hipError_t (*deallocation_fn)(void *);

    template<allocation_fn Alloc, deallocation_fn Dealloc, typename Pointer>
    class hip_memory_resource final : public mr::memory_resource<Pointer>
    {
    public:
        Pointer do_allocate(std::size_t bytes, std::size_t alignment = THRUST_MR_DEFAULT_ALIGNMENT) override
        {
            (void)alignment;

            void * ret;
            hipError_t status = Alloc(&ret, bytes);

            if (status != hipSuccess)
            {
                // Clear the HIP global error state.
                hipError_t clear_error_status = hipGetLastError();
                THRUST_UNUSED_VAR(clear_error_status);
                throw thrust::system::detail::bad_alloc(thrust::hip_category().message(status).c_str());
            }

            return Pointer(ret);
        }

        void do_deallocate(Pointer p, std::size_t bytes, std::size_t alignment) override
        {
            (void)bytes;
            (void)alignment;

            hipError_t status = Dealloc(thrust::detail::pointer_traits<Pointer>::get(p));

            if (status != hipSuccess)
            {
                thrust::hip_rocprim::throw_on_error(status, "HIP free failed");
            }
        }
    };

    inline hipError_t hipMallocManaged(void ** ptr, std::size_t bytes)
    {
        return ::hipMallocManaged(ptr, bytes, hipMemAttachGlobal);
    }

    inline hipError_t hipHostMalloc(void ** ptr, std::size_t bytes)
    {
        return ::hipHostMalloc(ptr, bytes, hipHostMallocMapped);
    }

    typedef detail::hip_memory_resource<hipMalloc, hipFree,
        thrust::hip_rocprim::pointer<void> >
        device_memory_resource;
    typedef detail::hip_memory_resource<detail::hipMallocManaged, hipFree,
        thrust::hip::universal_pointer<void> >
        managed_memory_resource;
    typedef detail::hip_memory_resource<hipHostMalloc, hipHostFree,
        thrust::hip::universal_pointer<void> >
        pinned_memory_resource;

} // end detail
//! \endcond

/*! The memory resource for the HIP system. Uses <tt>hipMalloc</tt> and wraps
 *  the result with \p hip::pointer.
 */
typedef detail::device_memory_resource memory_resource;
/*! The universal memory resource for the HIP system. Uses
 *  <tt>hipMallocManaged</tt> and wraps the result with
 *  \p hip::universal_pointer.
 */
typedef detail::managed_memory_resource universal_memory_resource;
/*! The host pinned memory resource for the HIP system. Uses
 *  <tt>hipMallocHost</tt> and wraps the result with \p
 *  hip::universal_pointer.
 */
typedef detail::pinned_memory_resource universal_host_pinned_memory_resource;

} // end hip
} // end system

namespace hip
{
using thrust::system::hip::memory_resource;
using thrust::system::hip::universal_memory_resource;
using thrust::system::hip::universal_host_pinned_memory_resource;
}
THRUST_NAMESPACE_END
