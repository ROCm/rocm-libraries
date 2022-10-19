// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "common_test_header.hpp"

#include <rocprim/detail/temp_storage.hpp>
#include <rocprim/rocprim.hpp>

#include "test_utils_types.hpp"

namespace rpts = rocprim::detail::temp_storage;

static bool is_aligned_to(void* const ptr, const size_t alignment)
{
    return reinterpret_cast<intptr_t>(ptr) % alignment == 0;
}

static void* displace_ptr(void* const ptr, const size_t offset)
{
    return reinterpret_cast<char*>(ptr) + offset;
}

TEST(RocprimTemporaryStoragePartitioningTests, Basic)
{
    constexpr size_t size      = 123;
    constexpr size_t alignment = 1;

    hipError_t result;

    size_t*    test_allocation;
    const auto partition = [&](void* const temporary_storage, size_t& storage_size)
    {
        return rpts::partition(temporary_storage,
                               storage_size,
                               rpts::make_partition(&test_allocation, size, alignment));
    };

    size_t storage_size;
    result = partition(nullptr, storage_size);
    ASSERT_EQ(result, hipSuccess);
    ASSERT_EQ(storage_size, size);

    void* temporary_storage;
    HIP_CHECK(test_common_utils::hipMallocHelper(&temporary_storage, storage_size));

    result = partition(temporary_storage, storage_size);
    ASSERT_EQ(result, hipSuccess);
    ASSERT_EQ(storage_size, size);
    ASSERT_EQ(test_allocation, temporary_storage);

    hipFree(temporary_storage);
}

TEST(RocprimTemporaryStoragePartitioningTests, ZeroSizePartition)
{
    constexpr size_t size      = 0;
    constexpr size_t alignment = 1;

    hipError_t result;

    size_t*    test_allocation;
    const auto partition = [&](void* const temporary_storage, size_t& storage_size)
    {
        return rpts::partition(temporary_storage,
                               storage_size,
                               rpts::make_partition(&test_allocation, size, alignment));
    };

    size_t storage_size;
    result = partition(nullptr, storage_size);
    ASSERT_EQ(result, hipSuccess);
    ASSERT_NE(storage_size, 0);

    void* temporary_storage;
    HIP_CHECK(test_common_utils::hipMallocHelper(&temporary_storage, storage_size));

    result = partition(temporary_storage, storage_size);
    ASSERT_EQ(result, hipSuccess);
    ASSERT_EQ(test_allocation, nullptr);

    HIP_CHECK(hipFree(temporary_storage));
}

TEST(RocprimTemporaryStoragePartitioningTests, ZeroSizePartitionInsufficientAllocation)
{
    hipError_t result;

    size_t*    test_allocation;
    const auto partition = [&](void* const temporary_storage, size_t& storage_size)
    {
        return rpts::partition(temporary_storage,
                               storage_size,
                               rpts::make_partition(&test_allocation, 0, 1));
    };

    size_t storage_size;
    result = partition(nullptr, storage_size);
    ASSERT_EQ(result, hipSuccess);
    ASSERT_NE(storage_size, 0);

    void* temporary_storage;
    HIP_CHECK(test_common_utils::hipMallocHelper(&temporary_storage, storage_size));

    storage_size = 0;
    result       = partition(temporary_storage, storage_size);
    ASSERT_EQ(result, hipErrorInvalidValue);

    HIP_CHECK(hipFree(temporary_storage));
}

TEST(RocprimTemporaryStoragePartitioningTests, Sequence)
{
    constexpr rpts::layout layout_a   = {123, 8};
    constexpr rpts::layout layout_b   = {10, 8};
    constexpr rpts::layout layout_c   = {100, 32};
    constexpr size_t       elements_e = 17;
    using type_e                      = test_utils::custom_test_type<double>;

    hipError_t result;

    size_t* test_allocation_a;
    size_t* test_allocation_b;
    size_t* test_allocation_c;
    size_t* test_allocation_d;
    type_e* test_allocation_e;

    const auto partition = [&](void* const temporary_storage, size_t& storage_size)
    {
        return rpts::partition(
            temporary_storage,
            storage_size,
            rpts::make_linear_partition(rpts::make_partition(&test_allocation_a, layout_a),
                                        rpts::make_partition(&test_allocation_b, layout_b),
                                        rpts::make_partition(&test_allocation_c, layout_c),
                                        rpts::ptr_aligned_array(&test_allocation_d, 0),
                                        rpts::ptr_aligned_array(&test_allocation_e, elements_e)));
    };

    size_t storage_size;
    result = partition(nullptr, storage_size);
    ASSERT_EQ(result, hipSuccess);

    const size_t expected_offset_b = rocprim::detail::align_size(layout_a.size, layout_b.alignment);
    const size_t expected_offset_c
        = rocprim::detail::align_size(expected_offset_b + layout_b.size, layout_c.alignment);
    const size_t expected_offset_d = expected_offset_c + layout_c.size;
    const size_t expected_offset_e
        = rocprim::detail::align_size(expected_offset_d, alignof(type_e));
    const size_t expected_size = expected_offset_e + elements_e * sizeof(type_e);
    ASSERT_EQ(storage_size, expected_size);

    void* temporary_storage;
    HIP_CHECK(test_common_utils::hipMallocHelper(&temporary_storage, storage_size));

    result = partition(temporary_storage, storage_size);
    ASSERT_EQ(result, hipSuccess);
    ASSERT_EQ(test_allocation_a, temporary_storage);
    ASSERT_EQ(test_allocation_b, displace_ptr(temporary_storage, expected_offset_b));
    ASSERT_EQ(test_allocation_c, displace_ptr(temporary_storage, expected_offset_c));
    ASSERT_EQ(test_allocation_d, nullptr);
    ASSERT_EQ(test_allocation_e, displace_ptr(temporary_storage, expected_offset_e));

    ASSERT_TRUE(
        is_aligned_to(displace_ptr(temporary_storage, expected_offset_b), layout_b.alignment));
    ASSERT_TRUE(
        is_aligned_to(displace_ptr(temporary_storage, expected_offset_c), layout_c.alignment));
    ASSERT_TRUE(is_aligned_to(displace_ptr(temporary_storage, expected_offset_e), alignof(type_e)));

    HIP_CHECK(hipFree(temporary_storage));
}

TEST(RocprimTemporaryStoragePartitioningTests, MutuallyExclusive)
{
    constexpr size_t elements_a = 1;
    using type_a                = char;

    constexpr rpts::layout layout_b   = {50, 1};
    constexpr size_t       elements_c = 17;
    using type_c                      = test_utils::custom_test_type<double>;

    hipError_t result;

    type_a* test_allocation_a;
    size_t* test_allocation_b;
    type_c* test_allocation_c;
    size_t* test_allocation_d;

    const auto partition = [&](void* const temporary_storage, size_t& storage_size)
    {
        return rpts::partition(
            temporary_storage,
            storage_size,
            rpts::make_linear_partition(
                rpts::ptr_aligned_array(&test_allocation_a, elements_a),
                rpts::make_union_partition(rpts::make_partition(&test_allocation_b, layout_b),
                                           rpts::ptr_aligned_array(&test_allocation_c, elements_c),
                                           rpts::ptr_aligned_array(&test_allocation_d, 0))));
    };

    size_t storage_size;
    result = partition(nullptr, storage_size);
    ASSERT_EQ(result, hipSuccess);

    const size_t expected_align_bc = std::max(alignof(type_c), layout_b.alignment);
    const size_t expected_offset_bc
        = rocprim::detail::align_size(elements_a * sizeof(type_a), expected_align_bc);
    const size_t expected_size
        = expected_offset_bc + std::max(elements_c * sizeof(type_c), layout_b.size);
    ASSERT_EQ(storage_size, expected_size);

    void* temporary_storage;
    HIP_CHECK(test_common_utils::hipMallocHelper(&temporary_storage, storage_size));

    result = partition(temporary_storage, storage_size);
    ASSERT_EQ(result, hipSuccess);
    ASSERT_EQ(test_allocation_a, temporary_storage);
    void* const bc = displace_ptr(temporary_storage, expected_offset_bc);
    ASSERT_EQ(test_allocation_b, bc);
    ASSERT_EQ(test_allocation_c, bc);
    ASSERT_EQ(test_allocation_d, nullptr);

    ASSERT_TRUE(is_aligned_to(bc, layout_b.alignment));
    ASSERT_TRUE(is_aligned_to(bc, alignof(type_c)));

    HIP_CHECK(hipFree(temporary_storage));
}

TEST(RocprimTemporaryStoragePartitioningTests, InsufficientAllocation)
{
    constexpr size_t size      = 100;
    constexpr size_t alignment = 4;

    hipError_t result;

    size_t*    test_allocation_a;
    size_t*    test_allocation_b;
    const auto partition = [&](void* const temporary_storage, size_t& storage_size)
    {
        return rpts::partition(
            temporary_storage,
            storage_size,
            rpts::make_linear_partition(rpts::make_partition(&test_allocation_a, size, alignment),
                                        rpts::make_partition(&test_allocation_b, size, alignment)));
    };

    size_t storage_size;
    result = partition(nullptr, storage_size);
    ASSERT_EQ(result, hipSuccess);
    ASSERT_EQ(storage_size, size * 2);

    void* temporary_storage;
    storage_size -= 1;
    HIP_CHECK(test_common_utils::hipMallocHelper(&temporary_storage, storage_size));

    result = partition(temporary_storage, storage_size);
    ASSERT_EQ(result, hipErrorInvalidValue);

    HIP_CHECK(hipFree(temporary_storage));
}
