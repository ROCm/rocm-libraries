// MIT License
//
// Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.
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

// Thrust
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/device_new.h>
#include <thrust/device_delete.h>

#include "test_header.hpp"

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

struct Foo
{
  __host__ __device__
  Foo(void)
    :set_me_upon_destruction(0)
  {}

  bool *set_me_upon_destruction;
};

TEST(DeviceDelete, TestDeviceDeleteDestructorInvocation)
{
  thrust::device_vector<bool> destructor_flag(1, false);

  thrust::device_ptr<Foo> foo_ptr  = thrust::device_new<Foo>();

  Foo exemplar;
  exemplar.set_me_upon_destruction = thrust::raw_pointer_cast(&destructor_flag[0]);
  *foo_ptr = exemplar;

  ASSERT_EQ(false, destructor_flag[0]);

  // TODO: Known failure
  //thrust::device_delete(foo_ptr);
  //ASSERT_EQ(true, destructor_flag[0]);
}

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
