// Copyright (c) 2020-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

#include <iostream>

// This example demonstrates how to build a minimal custom
// Thrust backend by intercepting for_each's dispatch.

// We begin by defining a "system", which distinguishes our novel
// backend from other Thrust backends.
// We'll derive my_system from thrust::device_execution_policy to inherit
// the functionality of the default device backend.
// Note that we pass the name of our system as a template parameter
// to thrust::device_execution_policy.
struct my_system : thrust::device_execution_policy<my_system>
{};

// Next, we'll create a novel version of for_each which only
// applies to algorithm invocations executed with my_system.
// Our version of for_each will print a message and then call
// the regular device version of for_each.

// The first parameter to our version for_each is my_system. This allows
// Thrust to locate it when dispatching thrust::for_each.
// The following parameters are as normal.
template <typename Iterator, typename Function>
Iterator for_each(my_system, Iterator first, Iterator last, Function f)
{
  // output a message
  std::cout << "Hello, world from for_each(my_system)!" << std::endl;

  // to call the normal device version of for_each, pass thrust::device as the first parameter.
  return thrust::for_each(thrust::device, first, last, f);
}

int main()
{
  thrust::device_vector<int> vec(1);

  // create an instance of our system
  my_system sys;

  // To invoke our version of for_each, pass sys as the first parameter
  thrust::for_each(sys, vec.begin(), vec.end(), thrust::identity<int>());

  // Other algorithms that Thrust implements with thrust::for_each will also
  // cause our version of for_each to be invoked when we pass an instance of my_system as the first parameter.
  // Even though we did not define a special version of transform, Thrust dispatches the version it knows
  // for thrust::device_execution_policy, which my_system inherits.
  thrust::transform(sys, vec.begin(), vec.end(), vec.begin(), thrust::identity<int>());

  // Invocations without my_system are handled normally.
  thrust::for_each(vec.begin(), vec.end(), thrust::identity<int>());

  return 0;
}
