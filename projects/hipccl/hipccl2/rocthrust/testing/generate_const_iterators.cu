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

#include <unittest/runtime_static_assert.h>
#include <unittest/unittest.h>

// The runtime_static_assert header needs to come first as we are overwriting thrusts internal static assert
#include <thrust/generate.h>

struct generator
{
  THRUST_HOST_DEVICE int operator()() const
  {
    return 1;
  }
};

void TestGenerateConstIteratorCompilationError()
{
  thrust::host_vector<int> test1(10);

  ASSERT_STATIC_ASSERT(thrust::generate(test1.cbegin(), test1.cend(), generator()));
  ASSERT_STATIC_ASSERT(thrust::generate_n(test1.cbegin(), 10, generator()));
}
DECLARE_UNITTEST(TestGenerateConstIteratorCompilationError);

void TestFillConstIteratorCompilationError()
{
  thrust::host_vector<int> test1(10);
  ASSERT_STATIC_ASSERT(thrust::fill(test1.cbegin(), test1.cend(), 1));
}
DECLARE_UNITTEST(TestFillConstIteratorCompilationError);
