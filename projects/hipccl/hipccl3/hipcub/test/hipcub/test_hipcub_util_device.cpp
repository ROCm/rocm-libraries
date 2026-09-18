// MIT License
//
// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

// hipcub API
#include <hipcub/util_device.hpp>

template<class T>
__global__
void alias_temporaries_kernel(T* data, size_t* temp_storage_bytes)
{
    T*     allocations[10];
    size_t allocation_sizes[10] = {1, 2, 3, 5, 8, 13, 21, 34, 55, 89};
    (void)
        hipcub::detail::AliasTemporaries(data, *temp_storage_bytes, allocations, allocation_sizes);
}

TEST(HipcubUtilDevice, AliasTemporariesDevice)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    void*   data                    = nullptr;
    size_t  temp_storage_bytes_host = 0; // Temporary storage on the host
    size_t* device_temp_storage_bytes;
    HIP_CHECK(test_common_utils::hipMallocHelper(&device_temp_storage_bytes, sizeof(size_t)));

    // First kernel call to determine required temp storage size
    alias_temporaries_kernel<void><<<1, 1, 0, 0>>>(data, device_temp_storage_bytes);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipGetLastError());

    // Copy the device storage size to host
    HIP_CHECK(hipMemcpy(&temp_storage_bytes_host,
                        device_temp_storage_bytes,
                        sizeof(size_t),
                        hipMemcpyDeviceToHost));
    ASSERT_GT(temp_storage_bytes_host, 0U);

    // Allocate the actual data buffer on the device
    HIP_CHECK(test_common_utils::hipMallocHelper(&data, temp_storage_bytes_host));

    // Second kernel call with allocated buffer
    alias_temporaries_kernel<void><<<1, 1, 0, 0>>>(data, device_temp_storage_bytes);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipGetLastError());

    // Free device memory
    HIP_CHECK(hipFree(device_temp_storage_bytes));
    HIP_CHECK(hipFree(data));
}

TEST(HipcubUtilDevice, AliasTemporariesHost)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    void*  data               = nullptr;
    size_t temp_storage_bytes = 0;
    void*  allocations[10];
    size_t allocation_sizes[10] = {1, 789, 3, 5, 8, 13, 21, 257, 256, 890};

    size_t min_size = 0;
    for(unsigned int i = 0; i < 10; i++)
    {
        min_size += allocation_sizes[i];
    }

    // Determine storage size
    HIP_CHECK(
        hipcub::detail::AliasTemporaries(data, temp_storage_bytes, allocations, allocation_sizes));

    // Should be larger or equal to the sum of all sizes.
    ASSERT_GT(temp_storage_bytes, min_size - 1);

    // Allocate the actual data buffer on the device
    HIP_CHECK(test_common_utils::hipMallocHelper(&data, temp_storage_bytes));

    size_t zero_size = 0;
    // Check for error if it does not fit
    hipError_t error
        = hipcub::detail::AliasTemporaries(data, zero_size, allocations, allocation_sizes);
    test_utils::assert_eq(error, hipErrorInvalidValue);

    HIP_CHECK(
        hipcub::detail::AliasTemporaries(data, temp_storage_bytes, allocations, allocation_sizes));

    test_utils::assert_eq(data, allocations[0]);

    for(unsigned int i = 1; i < 10; i++)
    {
        // The allocations should be in increasing order.
        ASSERT_GT(allocations[i], allocations[i - 1]);
        size_t current_pointer = (size_t)allocations[i];
        size_t before_pointer  = (size_t)allocations[i - 1];
        size_t distance        = current_pointer - before_pointer;

        // Check if all pointer have enough space
        ASSERT_GT(distance + 1, allocation_sizes[i - 1]);
    }

    size_t last_pointer  = (size_t)allocations[9];
    size_t start_pointer = (size_t)data;
    size_t max_size      = start_pointer + temp_storage_bytes;
    size_t last_size     = max_size - last_pointer;

    // Last size should be equal or larger then the last value in allocation_sizes
    ASSERT_GT(last_size + 1, allocation_sizes[9]);

    HIP_CHECK(hipFree(data));
}
