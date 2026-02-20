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

// Regression test for NVBug 2720132.
//
// Summary of 2720132:
//
// 1. The large allocation fails due to running out of memory.
// 2. A `thrust::system::system_error` exception is thrown.
// 3. Local objects are destroyed as the stack is unwound, leading to the destruction of `x`.
// 4. `x` runs a parallel algorithm in its destructor to call the destructors of all of its elements.
// 5. Launching that parallel algorithm fails because of the prior CUDA out of memory error.
// 6. A `thrust::system::system_error` exception is thrown.
// 7. Because we've already got an active exception, `terminate` is called.

#include <thrust/device_vector.h>

#include <cstdint>

#include <unittest/unittest.h>

struct non_trivial
{
  THRUST_HOST_DEVICE non_trivial() {}
  THRUST_HOST_DEVICE ~non_trivial() {}
};

void test_out_of_memory_recovery()
{
  try
  {
    thrust::device_vector<non_trivial> x(1);

    thrust::device_vector<std::uint32_t> y(0x00ffffffffffffff);
  }
  catch (...)
  {}
}
DECLARE_UNITTEST(test_out_of_memory_recovery);
