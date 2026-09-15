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

#include "../common_test_header.hpp"

#include <rocprim/device/config_types.hpp>

#include <hip/hip_runtime.h>

using rocprim::detail::target_arch;

// all the dispatch related functions from config_types.hpp are tested in test_config_dispatch.hpp

TEST(RocprimConfigTypesTests, GetTargetGPUFromName)
{
    using rocprim::detail::get_target_gpu_from_name;
    using rocprim::detail::gpu;

    static_assert(get_target_gpu_from_name("RX 9070") == gpu::rx9070);
    static_assert(get_target_gpu_from_name("foo RX 6900 bar") == gpu::rx6900);
    static_assert(get_target_gpu_from_name("doesn't exist") == gpu::generic);
}

TEST(RocprimConfigTypesTests, GetDeviceArch)
{
    using rocprim::detail::get_device_arch;
    using rocprim::detail::target_arch;

    int device_id = test_common_utils::obtain_device_from_ctest();
    HIP_CHECK(hipSetDevice(device_id));

    rocprim::detail::target_arch arch;
    HIP_CHECK(get_device_arch(device_id, arch));

    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, device_id));

    rocprim::detail::target_arch expected_arch = rocprim::detail::parse_gcn_arch(props.gcnArchName);
    ASSERT_EQ(arch, expected_arch);

    // out of bounds check
    ASSERT_NE(get_device_arch(-1, arch), hipError_t::hipSuccess);
}

TEST(RocprimConfigTypesTests, HostTargetGPU)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    HIP_CHECK(hipSetDevice(device_id));

    hipStream_t stream = hipStreamDefault;
    HIP_CHECK(hipStreamCreate(&stream));

    rocprim::detail::gpu gpu;
    HIP_CHECK(rocprim::detail::host_target_gpu(stream, gpu));

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));

    rocprim::detail::gpu expected_gpu = rocprim::detail::get_target_gpu_from_name(prop.name);
    ASSERT_EQ(gpu, expected_gpu);

    HIP_CHECK(hipStreamDestroy(stream));
}

TEST(RocprimConfigTypesTests, GetHostWarpSizeFromStream)
{
    using namespace rocprim;

    int device_id = test_common_utils::obtain_device_from_ctest();
    HIP_CHECK(hipSetDevice(device_id));

    hipStream_t stream = hipStreamDefault;
    HIP_CHECK(hipStreamCreate(&stream));

    unsigned int warp_size;
    HIP_CHECK(host_warp_size(stream, warp_size));

    unsigned int expected_warp_size;
    HIP_CHECK(host_warp_size(device_id, expected_warp_size));

    ASSERT_EQ(warp_size, expected_warp_size);

    HIP_CHECK(hipStreamDestroy(stream));
}
