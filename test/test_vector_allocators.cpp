/*
 *  Copyright 2008-2013 NVIDIA Corporation
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

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "test_header.hpp"

template<typename BaseAlloc, bool PropagateOnSwap>
class stateful_allocator : public BaseAlloc
{
public:
    stateful_allocator(int i) : state(i)
    {
    }

    ~stateful_allocator() {}

    stateful_allocator(const stateful_allocator &other)
        : BaseAlloc(other), state(other.state)
    {
    }

    stateful_allocator & operator=(const stateful_allocator & other)
    {
        state = other.state;
        return *this;
    }

    stateful_allocator(stateful_allocator && other)
        : BaseAlloc(std::move(other)), state(other.state)
    {
        other.state = 0;
    }

    stateful_allocator & operator=(stateful_allocator && other)
    {
        state = other.state;
        other.state = 0;
        return *this;
    }

    static int last_allocated;
    static int last_deallocated;

    using pointer =
        typename thrust::detail::allocator_traits<BaseAlloc>::pointer;

    pointer allocate(std::size_t size)
    {
        last_allocated = state;
        return BaseAlloc::allocate(size);
    }

    void deallocate(pointer ptr, std::size_t size)
    {
        last_deallocated = state;
        return BaseAlloc::deallocate(ptr, size);
    }

    bool operator==(const stateful_allocator &rhs) const
    {
        return state == rhs.state;
    }

    bool operator!=(const stateful_allocator &rhs) const
    {
        return state != rhs.state;
    }

    friend std::ostream & operator<<(std::ostream &os,
        const stateful_allocator & alloc)
    {
        os << "stateful_alloc(" << alloc.state << ")";
        return os;
    }

    using is_always_equal = thrust::detail::false_type;
    using propagate_on_container_copy_assignment = thrust::detail::true_type;
    using propagate_on_container_move_assignment = thrust::detail::true_type;
    using propagate_on_container_swap = thrust::detail::integral_constant<bool, PropagateOnSwap>;

private:
    int state;
};

template<typename BaseAlloc, bool PropagateOnSwap>
int stateful_allocator<BaseAlloc, PropagateOnSwap>::last_allocated = 0;

template<typename BaseAlloc, bool PropagateOnSwap>
int stateful_allocator<BaseAlloc, PropagateOnSwap>::last_deallocated = 0;

using host_alloc = stateful_allocator<std::allocator<int>, true>;
using device_alloc = stateful_allocator<thrust::device_allocator<int>, true>;

using host_vector = thrust::host_vector<int, host_alloc>;
using device_vector = thrust::device_vector<int, device_alloc>;

using host_alloc_nsp = stateful_allocator<std::allocator<int>, false>;
using device_alloc_nsp = stateful_allocator<thrust::device_allocator<int>, false>;

using host_vector_nsp = thrust::host_vector<int, host_alloc_nsp>;
using device_vector_nsp = thrust::device_vector<int, device_alloc_nsp>;

template<typename Vector>
void TestVectorAllocatorConstructors()
{
    using Alloc = typename Vector::allocator_type;
    Alloc alloc1(1);
    Alloc alloc2(2);

    Vector v1(alloc1);
    ASSERT_EQ(v1.get_allocator(), alloc1);

    Vector v2(10, alloc1);
    ASSERT_EQ(v2.size(), 10u);
    ASSERT_EQ(v2.get_allocator(), alloc1);
    ASSERT_EQ(Alloc::last_allocated, 1);
    Alloc::last_allocated = 0;

    Vector v3(10, 17, alloc1);
    ASSERT_EQ((v3 == std::vector<int>(10, 17)), true);
    ASSERT_EQ(v3.get_allocator(), alloc1);
    ASSERT_EQ(Alloc::last_allocated, 1);
    Alloc::last_allocated = 0;

    Vector v4(v3, alloc2);
    ASSERT_EQ((v3 == v4), true);
    ASSERT_EQ(v4.get_allocator(), alloc2);
    ASSERT_EQ(Alloc::last_allocated, 2);
    Alloc::last_allocated = 0;

    // FIXME: uncomment this after the vector_base(vector_base&&, const Alloc&)
    // is fixed and implemented
    // Vector v5(std::move(v3), alloc2);
    // ASSERT_EQ((v4 == v5), true);
    // ASSERT_EQ(v5.get_allocator(), alloc2);
    // ASSERT_EQ(Alloc::last_allocated, 1);
    // Alloc::last_allocated = 0;

    Vector v6(v4.begin(), v4.end(), alloc2);
    ASSERT_EQ((v4 == v6), true);
    ASSERT_EQ(v6.get_allocator(), alloc2);
    ASSERT_EQ(Alloc::last_allocated, 2);
}

TEST(VectorAllocatorTests, TestVectorAllocatorConstructorsHost)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    TestVectorAllocatorConstructors<host_vector>();
}

TEST(VectorAllocatorTests, TestVectorAllocatorConstructorsDevice)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    TestVectorAllocatorConstructors<device_vector>();
}

template<typename Vector>
void TestVectorAllocatorPropagateOnCopyAssignment()
{
    ASSERT_EQ(thrust::detail::allocator_traits<typename Vector::allocator_type>::propagate_on_container_copy_assignment::value, true);

    using Alloc = typename Vector::allocator_type;
    Alloc alloc1(1);
    Alloc alloc2(2);

    Vector v1(10, alloc1);
    Vector v2(15, alloc2);

    v2 = v1;
    ASSERT_EQ((v1 == v2), true);
    ASSERT_EQ(v2.get_allocator(), alloc1);
    ASSERT_EQ(Alloc::last_allocated, 1);
    ASSERT_EQ(Alloc::last_deallocated, 2);
}

TEST(VectorAllocatorTests, TestVectorAllocatorPropagateOnCopyAssignmentHost)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    TestVectorAllocatorPropagateOnCopyAssignment<host_vector>();
}

TEST(VectorAllocatorTests, TestVectorAllocatorPropagateOnCopyAssignmentDevice)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    TestVectorAllocatorPropagateOnCopyAssignment<device_vector>();
}

template<typename Vector>
void TestVectorAllocatorPropagateOnMoveAssignment()
{
    using Alloc = typename Vector::allocator_type;
    ASSERT_EQ(thrust::detail::allocator_traits<typename Vector::allocator_type>::propagate_on_container_copy_assignment::value, true);

    using Alloc = typename Vector::allocator_type;
    Alloc alloc1(1);
    Alloc alloc2(2);

    {
    Vector v1(10, alloc1);
    Vector v2(15, alloc2);

    v2 = std::move(v1);
    ASSERT_EQ(v2.get_allocator(), alloc1);
    ASSERT_EQ(Alloc::last_allocated, 2);
    ASSERT_EQ(Alloc::last_deallocated, 2);
    }

    ASSERT_EQ(Alloc::last_deallocated, 1);
}

TEST(VectorAllocatorTests, TestVectorAllocatorPropagateOnMoveAssignmentHost)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    TestVectorAllocatorPropagateOnMoveAssignment<host_vector>();
}

TEST(VectorAllocatorTests, TestVectorAllocatorPropagateOnMoveAssignmentDevice)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    TestVectorAllocatorPropagateOnMoveAssignment<device_vector>();
}

template<typename Vector>
void TestVectorAllocatorPropagateOnSwap()
{
    using Alloc = typename Vector::allocator_type;
    Alloc alloc1(1);
    Alloc alloc2(2);

    Vector v1(10, alloc1);
    Vector v2(17, alloc1);
    thrust::swap(v1, v2);

    ASSERT_EQ(v1.size(), 17u);
    ASSERT_EQ(v2.size(), 10u);

    Vector v3(15, alloc1);
    Vector v4(31, alloc2);
    ASSERT_THROW(thrust::swap(v3, v4), thrust::detail::allocator_mismatch_on_swap);
}

TEST(VectorAllocatorTests, TestVectorAllocatorPropagateOnSwapHost)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
    
    TestVectorAllocatorPropagateOnSwap<host_vector_nsp>();
}

TEST(VectorAllocatorTests, TestVectorAllocatorPropagateOnSwapDevice)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
    
    TestVectorAllocatorPropagateOnSwap<device_vector_nsp>();
}
