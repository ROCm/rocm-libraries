/*
 *  Copyright 2008-2018 NVIDIA Corporation
 *  Modifications Copyright© 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <thrust/device_malloc_allocator.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/system/cpp/vector.h>

#include "test_header.hpp"

#include <cstddef>
#include <memory>

struct my_allocator_with_custom_construct1 : thrust::device_malloc_allocator<int>
{
    __host__ __device__ my_allocator_with_custom_construct1() {}

    template <typename T>
    __host__ __device__ void construct(T* p)
    {
        *p = 13;
    }
};

TEST(AllocatorTests, TestAllocatorCustomDefaultConstruct)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    thrust::device_vector<int>                                      ref(10, 13);
    thrust::device_vector<int, my_allocator_with_custom_construct1> vec(10);

    ASSERT_EQ(ref, vec);
}

struct my_allocator_with_custom_construct2 : thrust::device_malloc_allocator<int>
{
    __host__ __device__ my_allocator_with_custom_construct2() {}

    template <typename T, typename Arg>
    __host__ __device__ void construct(T* p, const Arg&)
    {
        *p = 13;
    }
};

TEST(AllocatorTests, TestAllocatorCustomCopyConstruct)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    thrust::device_vector<int>                                      ref(10, 13);
    thrust::device_vector<int>                                      copy_from(10, 7);
    thrust::device_vector<int, my_allocator_with_custom_construct2> vec(copy_from.begin(),
                                                                        copy_from.end());

    ASSERT_EQ(ref, vec);
}

static int g_state;

struct my_allocator_with_custom_destroy
{
    using system_type = thrust::cpp::tag;

    using value_type      = int;
    using reference       = int&;
    using const_reference = const int&;

    __host__ __device__ my_allocator_with_custom_destroy() {}

    __host__ __device__
             my_allocator_with_custom_destroy(const my_allocator_with_custom_destroy& other)
        : use_me_to_alloc(other.use_me_to_alloc)
    {
    }

    __host__ __device__ ~my_allocator_with_custom_destroy() {}

    template <typename T>
    __host__ __device__ void destroy(T*)
    {
#if !defined(THRUST_HIP_DEVICE_CODE)
        g_state = 13;
#endif
    }

    value_type* allocate(std::ptrdiff_t n)
    {
        return use_me_to_alloc.allocate(n);
    }

    void deallocate(value_type* ptr, std::ptrdiff_t n)
    {
        use_me_to_alloc.deallocate(ptr, n);
    }

    bool operator==(const my_allocator_with_custom_destroy &) const
    {
        return true;
    }

    bool operator!=(const my_allocator_with_custom_destroy &other) const
    {
        return !(*this == other);
    }

    using is_always_equal = thrust::detail::true_type;


    // use composition rather than inheritance
    // to avoid inheriting std::allocator's member
    // function construct
    std::allocator<int> use_me_to_alloc;
};

TEST(AllocatorTests, TestAllocatorCustomDestroy)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    thrust::cpp::vector<int, my_allocator_with_custom_destroy> vec(10);

    // destroy everything
    vec.shrink_to_fit();

    ASSERT_EQ(13, g_state);
}

struct my_minimal_allocator
{
    using value_type = int;

    // XXX ideally, we shouldn't require
    //     these two typedefs
    using reference = int&;
    using const_reference = const int&;

    __host__ __device__ my_minimal_allocator() {}

    __host__ __device__ my_minimal_allocator(const my_minimal_allocator& other)
        : use_me_to_alloc(other.use_me_to_alloc)
    {
    }

    __host__ __device__ ~my_minimal_allocator() {}

    value_type* allocate(std::ptrdiff_t n)
    {
        return use_me_to_alloc.allocate(n);
    }

    void deallocate(value_type* ptr, std::ptrdiff_t n)
    {
        use_me_to_alloc.deallocate(ptr, n);
    }

    std::allocator<int> use_me_to_alloc;
};

TEST(AllocatorTests, TestAllocatorMinimal)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
    
    thrust::cpp::vector<int, my_minimal_allocator> vec(10, 13);

    // XXX copy to h_vec because ASSERT_EQUAL doesn't know about cpp::vector
    thrust::host_vector<int> h_vec(vec.begin(), vec.end());
    thrust::host_vector<int> ref(10, 13);

    ASSERT_EQ(ref, h_vec);
}
