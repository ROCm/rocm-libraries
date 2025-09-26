// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <hipdnn_sdk/test_utilities/TestUtilities.hpp>
#include <hipdnn_sdk/utilities/Allocators.hpp>
#include <list>
#include <memory>
#include <vector>

using namespace hipdnn_sdk::utilities;

// Test that allocators work with STL containers
TEST(TestAllocators, HostAllocatorWithVector)
{
    std::vector<int, HostAllocator<int>> vec;
    vec.push_back(1);
    vec.push_back(2);
    vec.push_back(3);

    EXPECT_EQ(vec.size(), 3);
    EXPECT_EQ(vec[0], 1);
    EXPECT_EQ(vec[1], 2);
    EXPECT_EQ(vec[2], 3);
}

TEST(TestAllocators, HostAllocatorWithList)
{
    std::list<double, HostAllocator<double>> lst;
    lst.push_back(1.5);
    lst.push_back(2.5);
    lst.push_back(3.5);

    auto it = lst.begin();
    EXPECT_EQ(*it++, 1.5);
    EXPECT_EQ(*it++, 2.5);
    EXPECT_EQ(*it++, 3.5);
}

TEST(TestAllocators, HostAllocatorBasicOperations)
{
    HostAllocator<int> alloc;

    // Allocate memory for 10 integers
    auto ptr = alloc.allocate(10);
    ASSERT_NE(ptr, nullptr);

    // Construct objects
    for(int i = 0; i < 10; ++i)
    {
        alloc.construct(&ptr[i], i * 2);
    }

    // Verify values
    for(int i = 0; i < 10; ++i)
    {
        EXPECT_EQ(ptr[i], i * 2);
    }

    // Destroy objects
    for(int i = 0; i < 10; ++i)
    {
        alloc.destroy(&ptr[i]);
    }

    // Deallocate memory
    alloc.deallocate(ptr, 10);
}

TEST(TestAllocators, HostAllocatorRebind)
{
    HostAllocator<int> intAlloc;
    typename HostAllocator<int>::template rebind<double>::other doubleAlloc;

    auto intPtr = intAlloc.allocate(5);
    auto doublePtr = doubleAlloc.allocate(5);

    ASSERT_NE(intPtr, nullptr);
    ASSERT_NE(doublePtr, nullptr);

    intAlloc.deallocate(intPtr, 5);
    doubleAlloc.deallocate(doublePtr, 5);
}

TEST(TestAllocators, HostAllocatorComparison)
{
    HostAllocator<int> alloc1;
    HostAllocator<int> alloc2;
    HostAllocator<double> alloc3;

    EXPECT_TRUE(alloc1 == alloc2);
    EXPECT_FALSE(alloc1 != alloc2);
    EXPECT_TRUE(alloc1 == alloc3);
}

TEST(TestGpuAllocators, PinnedHostAllocatorWithVector)
{
    SKIP_IF_NO_DEVICES();

    std::vector<float, PinnedHostAllocator<float>> vec;
    vec.resize(100);

    for(size_t i = 0; i < vec.size(); ++i)
    {
        vec[i] = static_cast<float>(i) * 0.5f;
    }

    for(size_t i = 0; i < vec.size(); ++i)
    {
        EXPECT_EQ(vec[i], static_cast<float>(i) * 0.5f);
    }
}

TEST(TestGpuAllocators, PinnedHostAllocatorBasicOperations)
{
    SKIP_IF_NO_DEVICES();

    PinnedHostAllocator<int> alloc;

    // Allocate pinned memory for 10 integers
    auto ptr = alloc.allocate(10);
    ASSERT_NE(ptr, nullptr);

    // Construct objects
    for(int i = 0; i < 10; ++i)
    {
        alloc.construct(&ptr[i], i * 3);
    }

    // Verify values
    for(int i = 0; i < 10; ++i)
    {
        EXPECT_EQ(ptr[i], i * 3);
    }

    // Destroy objects
    for(int i = 0; i < 10; ++i)
    {
        alloc.destroy(&ptr[i]);
    }

    // Deallocate memory
    alloc.deallocate(ptr, 10);
}

TEST(TestGpuAllocators, PinnedHostAllocatorRebind)
{
    SKIP_IF_NO_DEVICES();

    PinnedHostAllocator<int> intAlloc;
    typename PinnedHostAllocator<int>::template rebind<double>::other doubleAlloc;

    auto intPtr = intAlloc.allocate(5);
    auto doublePtr = doubleAlloc.allocate(5);

    ASSERT_NE(intPtr, nullptr);
    ASSERT_NE(doublePtr, nullptr);

    intAlloc.deallocate(intPtr, 5);
    doubleAlloc.deallocate(doublePtr, 5);
}

TEST(TestGpuAllocators, DeviceAllocatorBasicOperations)
{
    SKIP_IF_NO_DEVICES();

    DeviceAllocator<float> alloc;

    // Allocate device memory for 100 floats
    auto ptr = alloc.allocate(100);
    ASSERT_NE(ptr, nullptr);

    // Verify we can use the pointer with HIP operations
    std::vector<float> hostData(100);
    for(size_t i = 0; i < hostData.size(); ++i)
    {
        hostData[i] = static_cast<float>(i);
    }

    hipError_t err = hipMemcpy(ptr, hostData.data(), 100 * sizeof(float), hipMemcpyHostToDevice);
    EXPECT_EQ(err, hipSuccess);

    std::vector<float> result(100);
    err = hipMemcpy(result.data(), ptr, 100 * sizeof(float), hipMemcpyDeviceToHost);
    EXPECT_EQ(err, hipSuccess);

    for(size_t i = 0; i < result.size(); ++i)
    {
        EXPECT_EQ(result[i], static_cast<float>(i));
    }

    // Deallocate memory
    alloc.deallocate(ptr, 100);
}

TEST(TestGpuAllocators, DeviceAllocatorRebind)
{
    SKIP_IF_NO_DEVICES();

    DeviceAllocator<int> intAlloc;
    typename DeviceAllocator<int>::template rebind<double>::other doubleAlloc;

    auto intPtr = intAlloc.allocate(5);
    auto doublePtr = doubleAlloc.allocate(5);

    ASSERT_NE(intPtr, nullptr);
    ASSERT_NE(doublePtr, nullptr);

    intAlloc.deallocate(intPtr, 5);
    doubleAlloc.deallocate(doublePtr, 5);
}

TEST(TestGpuAllocators, DeviceAllocatorComparison)
{
    SKIP_IF_NO_DEVICES();

    DeviceAllocator<int> alloc1;
    DeviceAllocator<int> alloc2;
    DeviceAllocator<double> alloc3;

    EXPECT_TRUE(alloc1 == alloc2);
    EXPECT_FALSE(alloc1 != alloc2);
    EXPECT_TRUE(alloc1 == alloc3);
}

// Test allocator traits compatibility
TEST(TestAllocators, AllocatorTraitsCompatibility)
{
    using HostTraits = std::allocator_traits<HostAllocator<int>>;
    using PinnedTraits = std::allocator_traits<PinnedHostAllocator<int>>;
    using DeviceTraits = std::allocator_traits<DeviceAllocator<int>>;

    // Check that all required types are defined
    static_assert(std::is_same_v<HostTraits::value_type, int>);
    static_assert(std::is_same_v<HostTraits::pointer, int*>);
    static_assert(std::is_same_v<HostTraits::const_pointer, const int*>);
    static_assert(std::is_same_v<HostTraits::size_type, std::size_t>);
    static_assert(std::is_same_v<HostTraits::difference_type, std::ptrdiff_t>);

    static_assert(std::is_same_v<PinnedTraits::value_type, int>);
    static_assert(std::is_same_v<PinnedTraits::pointer, int*>);
    static_assert(std::is_same_v<PinnedTraits::const_pointer, const int*>);
    static_assert(std::is_same_v<PinnedTraits::size_type, std::size_t>);
    static_assert(std::is_same_v<PinnedTraits::difference_type, std::ptrdiff_t>);

    static_assert(std::is_same_v<DeviceTraits::value_type, int>);
    static_assert(std::is_same_v<DeviceTraits::pointer, int*>);
    static_assert(std::is_same_v<DeviceTraits::const_pointer, const int*>);
    static_assert(std::is_same_v<DeviceTraits::size_type, std::size_t>);
    static_assert(std::is_same_v<DeviceTraits::difference_type, std::ptrdiff_t>);
}

// Test exception handling
TEST(TestAllocators, AllocationFailure)
{
    HostAllocator<int> alloc;

    // Try to allocate an impossibly large amount of memory
    EXPECT_THROW(std::ignore = alloc.allocate(std::numeric_limits<std::size_t>::max()),
                 std::bad_alloc);
}

TEST(TestGpuAllocators, PinnedAllocationFailure)
{
    SKIP_IF_NO_DEVICES();

    PinnedHostAllocator<int> alloc;

    // Try to allocate an impossibly large amount of memory
    EXPECT_THROW(std::ignore = alloc.allocate(std::numeric_limits<std::size_t>::max()),
                 std::bad_alloc);
}

TEST(TestGpuAllocators, DeviceAllocationFailure)
{
    SKIP_IF_NO_DEVICES();

    DeviceAllocator<int> alloc;

    // Try to allocate an impossibly large amount of memory
    EXPECT_THROW(std::ignore = alloc.allocate(std::numeric_limits<std::size_t>::max()),
                 std::bad_alloc);
}
