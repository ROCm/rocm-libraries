/*
 *  Copyright 2018-2019 NVIDIA Corporation
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

#include <thrust/mr/memory_resource.h>

THRUST_NAMESPACE_BEGIN
namespace mr
{

/*! This memory_resource-derived type provides the ability to implement custom allocation behaviour.
 *
 * Since this class inherits from \p memory_resource, an instance of this class can be passed
 * to an allocator constructor (as a memory resource).
 * This class also wraps a \p memory_resource. When interface member functions are called
 * (eg. do_allocate), the underlying (wrapped) \p memory_resource's corresponding member functions
 * are invoked.
 * Together, these two properties enable \p polymorphic_adaptor_resource to be used in the
 * implementation of a polymorphic allocator: an allocator whose behaviour is specified at runtime.
 *
 * \tparam Pointer - the pointer type that will be used to create the memory resource.
 * Defaults to void*.
 */
template<typename Pointer = void *>
class polymorphic_adaptor_resource final : public memory_resource<Pointer>
{
public:
    /*! Constructs a new \p polymorphic_adaptor_resource
     * \param t - A pointer to a memory_resource that this instance will wrap.
     */
    polymorphic_adaptor_resource(memory_resource<Pointer> * t) : upstream_resource(t)
    {
    }

    /*! Performs a memory allocation.
     * \param bytes - the requested size of the allocation, in bytes
     * \param alignment - specifies the alignment for the allocation.
     *  Defaults to THRUST_MR_DEFAULT_ALIGNMENT.
     */
    virtual Pointer do_allocate(std::size_t bytes, std::size_t alignment = THRUST_MR_DEFAULT_ALIGNMENT) override
    {
        return upstream_resource->allocate(bytes, alignment);
    }

    /*! Deallocates memory that was previously allocated with this allocator.
     * \param p - pointer to the memory that was previously allocated by \p do_allocate
     * \param bytes - the size of the allocation that was requested, in bytes
     * \param alignment - specifies the alignment that was used for the allocation
     */
    virtual void do_deallocate(Pointer p, std::size_t bytes, std::size_t alignment) override
    {
        return upstream_resource->deallocate(p, bytes, alignment);
    }

    /*! Compares this \p polymorphic_adaptor_resource with another \p memory_resource
     * to see if they are equal.
     */
    __host__ __device__
    virtual bool do_is_equal(const memory_resource<Pointer> & other) const noexcept override
    {
        return upstream_resource->is_equal(other);
    }

private:
    memory_resource<Pointer> * upstream_resource;
};

} // end mr
THRUST_NAMESPACE_END

