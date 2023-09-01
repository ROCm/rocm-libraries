/*
 *  Copyright 2018 NVIDIA Corporation
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

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/type_traits/pointer_traits.h>

#include <thrust/mr/memory_resource.h>
#include <thrust/mr/validator.h>

THRUST_NAMESPACE_BEGIN
namespace mr
{

/*! A \p memory_resource derived class that wraps a pointer type.
 *
 * Wrapping the pointer type allows this class to use either the global (static) memory
 * resource (see \p mr::get_global_resource ), or a provided "upstream" memory resource
 * instance. When allocations are performed, the returned pointer is cast to the \p Pointer
 * template parameter type, for convenience.
 *
 * \tparam Upstream - a \p memory_resource derrived type. Will be wrapped by this class.
 * \tparam Pointer - when allocations are performed, the return value will be cast to this type.
 */
template<typename Upstream, typename Pointer>
class fancy_pointer_resource final : public memory_resource<Pointer>, private validator<Upstream>
{
public:
    /*! Constructs a pointer to the global (static) memory resource.
     * See \p mr::get_global_resource.
     */
    fancy_pointer_resource() : m_upstream(get_global_resource<Upstream>())
    {
    }

    /*! Constructs a pointer to the provided memory resource.
     * \param upstream - pointer to the memory_resource to wrap.
     */
    fancy_pointer_resource(Upstream * upstream) : m_upstream(upstream)
    {
    }

    /*! Performs a memory allocation. Returns a pointer of type \p Pointer.
     * \param bytes - the requested size of the allocation, in bytes
     * \param alignment - specifies the alignment for the allocation.
     *  Defaults to THRUST_MR_DEFAULT_ALIGNMENT.
     */
    THRUST_NODISCARD
    virtual Pointer do_allocate(std::size_t bytes, std::size_t alignment = THRUST_MR_DEFAULT_ALIGNMENT) override
    {
        return static_cast<Pointer>(m_upstream->do_allocate(bytes, alignment));
    }

    /*! Deallocates memory that was previously allocated with this allocator.
     * \param p - pointer to the memory that was previously allocated by \p do_allocate
     * \param bytes - the size of the allocation that was requested, in bytes
     * \param alignment - specifies the alignment that was used for the allocation
     */
    virtual void do_deallocate(Pointer p, std::size_t bytes, std::size_t alignment) override
    {
        return m_upstream->do_deallocate(
            static_cast<typename Upstream::pointer>(
                thrust::detail::pointer_traits<Pointer>::get(p)),
            bytes, alignment);
    }

private:
    Upstream * m_upstream;
};

} // end mr
THRUST_NAMESPACE_END

