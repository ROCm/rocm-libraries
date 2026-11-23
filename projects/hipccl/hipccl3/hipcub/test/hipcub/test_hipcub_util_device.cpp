// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <hipcub/util_device.hpp>

#include "common_test_header.hpp"

#include _HIPCUB_LIBCXX_INCLUDE(__device/attributes.h)
#include _HIPCUB_LIBCXX_INCLUDE(__device/device_ref.h)

__global__
void kernel(volatile char* ptr)
{
    __shared__
    char static_smem[4096];
    static_smem[threadIdx.x] = static_cast<char>(threadIdx.x) + ptr[threadIdx.x];
    __syncthreads();
    ptr[blockDim.x - 1 - threadIdx.x] = static_smem[threadIdx.x];
}

TEST(HipcubUtilDeviceTests, MaxPotentialDynamicSmemBytes)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    _HIPCUB_LIBCXX::device_ref device{device_id};

    // Calculate the expected max potential dynamic shared memory size.
    hipFuncAttributes kernel_attrs{};
    HIP_CHECK(hipFuncGetAttributes(&kernel_attrs, reinterpret_cast<void*>(&kernel)));

    // Note: workaround for AMD devices because libhipcxx's _HIPCUB_LIBCXX::device_attributes::reserved_shared_memory_per_block
    // is not returning the correct value
    int reserved_smem_per_block_attribute{};
#ifdef __HIP_PLATFORM_AMD__
    HIP_CHECK(hipDeviceGetAttribute(&reserved_smem_per_block_attribute,
                                    hipDeviceAttributeReservedSharedMemPerBlock,
                                    device_id));
#endif

    const auto max_smem_per_block_optin
        = device.attribute(_HIPCUB_LIBCXX::device_attributes::max_shared_memory_per_block_optin);
    const auto reserved_smem_per_block =
#ifdef __HIP_PLATFORM_AMD__
        reserved_smem_per_block_attribute
#else
        device.attribute(_HIPCUB_LIBCXX::device_attributes::reserved_shared_memory_per_block)
#endif
        ;
    const auto expected
        = static_cast<int>(max_smem_per_block_optin - reserved_smem_per_block - 4096);

    int dyn_smem_size{};
    ASSERT_EQ(hipcub::MaxPotentialDynamicSmemBytes(dyn_smem_size, kernel), hipSuccess);
    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(dyn_smem_size, expected));
}

TEST(HipcubUtilDeviceTests, MaxPotentialDynamicSmemBytesError)

{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    _HIPCUB_LIBCXX::device_ref device{device_id};

    int dyn_smem_size{};
    ASSERT_NE(hipcub::MaxPotentialDynamicSmemBytes(dyn_smem_size, nullptr), hipSuccess);
    ASSERT_NO_FATAL_FAILURE(test_utils::assert_eq(dyn_smem_size, -1));
}
