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

#include <unittest/unittest.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/device_new.h>
#include <thrust/device_delete.h>

#include <thrust/detail/nv_target.h>

struct Foo
{
  THRUST_HOST_DEVICE
  Foo(void)
    : set_me_upon_destruction{nullptr}
  {}

  THRUST_HOST_DEVICE
  ~Foo(void)
  {
    NV_IF_TARGET(NV_IS_DEVICE, (
      if (set_me_upon_destruction != nullptr)
      {
        *set_me_upon_destruction = true;
      }));
  }

  bool *set_me_upon_destruction;
};

#if !defined(__QNX__)
void TestDeviceDeleteDestructorInvocation(void)
{
  KNOWN_FAILURE;
//
//  thrust::device_vector<bool> destructor_flag(1, false);
//
//  thrust::device_ptr<Foo> foo_ptr  = thrust::device_new<Foo>();
//
//  Foo exemplar;
//  exemplar.set_me_upon_destruction = thrust::raw_pointer_cast(&destructor_flag[0]);
//  *foo_ptr = exemplar;
//
//  ASSERT_EQUAL(false, destructor_flag[0]);
//
//  thrust::device_delete(foo_ptr);
//
//  ASSERT_EQUAL(true, destructor_flag[0]);
}
DECLARE_UNITTEST(TestDeviceDeleteDestructorInvocation);
#endif
