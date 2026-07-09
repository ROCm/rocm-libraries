/*
 *  Copyright 2025 NVIDIA Corporation
 *  Modifications Copyright© 2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include <thrust/iterator/transform_iterator.h>
#include <thrust/logical.h>

#include <unittest/unittest.h>

// see also: https://github.com/NVIDIA/cccl/issues/3541
void TestTransformWithLambda()
{
  auto l = [] __host__ __device__(int v) { return v < 4; };
  thrust::host_vector<int> A{1, 2, 3, 4, 5, 6, 7};
  ASSERT_EQUAL(thrust::any_of(A.begin(), A.end(), l), true);

  thrust::device_vector<int> B{1, 2, 3, 4, 5, 6, 7};
  ASSERT_EQUAL(thrust::any_of(B.begin(), B.end(), l), true);
}

DECLARE_UNITTEST(TestTransformWithLambda);
