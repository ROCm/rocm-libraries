// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <thrust/system/hip/detail/util.h>

#include "test_param_fixtures.hpp"
#include "test_utils.hpp"

// These tests verify that trivial_copy_from_device and trivial_copy_to_device
// use hipMemcpyDefault instead of hipMemcpyDeviceToHost/hipMemcpyHostToDevice.
//
// With explicit copy directions, the copy may fail when the actual memory
// location doesn't match the expected direction (e.g., hipMemcpyDeviceToHost
// with a host source pointer).
//
// With hipMemcpyDefault, HIP determines the correct direction at runtime,
// which is necessary when device-accessible pointers may point to host memory.

TEST(TrivialCopyMemcpyDefaultTests, TestTrivialCopyFromDevice_HostSource)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  int src[]  = {0, 10, 20, 30, 40};
  int dst[5] = {};

  hipError_t status = thrust::hip_rocprim::trivial_copy_from_device(dst, src, 5, hipStreamDefault);

  ASSERT_EQ(status, hipSuccess);
  for (int i = 0; i < 5; i++)
  {
    ASSERT_EQ(dst[i], i * 10);
  }
}

TEST(TrivialCopyMemcpyDefaultTests, TestTrivialCopyToDevice_HostDest)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  int src[]  = {0, 100, 200, 300, 400};
  int dst[5] = {};

  hipError_t status = thrust::hip_rocprim::trivial_copy_to_device(dst, src, 5, hipStreamDefault);

  ASSERT_EQ(status, hipSuccess);
  for (int i = 0; i < 5; i++)
  {
    ASSERT_EQ(dst[i], i * 100);
  }
}
