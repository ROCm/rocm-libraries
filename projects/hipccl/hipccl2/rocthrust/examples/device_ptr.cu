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

#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>

#include <cassert>
#include <iostream>

int main(void)
{
  // allocate memory buffer to store 10 integers on the device
  thrust::device_ptr<int> d_ptr = thrust::device_malloc<int>(10);

  // device_ptr supports pointer arithmetic
  thrust::device_ptr<int> first = d_ptr;
  thrust::device_ptr<int> last  = d_ptr + 10;
  std::cout << "device array contains " << (last - first) << " values\n";

  // algorithms work as expected
  thrust::sequence(first, last);
  std::cout << "sum of values is " << thrust::reduce(first, last) << "\n";

  // device memory can be read and written transparently
  d_ptr[0] = 10;
  d_ptr[1] = 11;
  d_ptr[2] = d_ptr[0] + d_ptr[1];

  // device_ptr can be converted to a "raw" pointer for use in other APIs and kernels, etc.
  int* raw_ptr = thrust::raw_pointer_cast(d_ptr);

  // note: raw_ptr cannot necessarily be accessed by the host!

  // conversely, raw pointers can be wrapped
  thrust::device_ptr<int> wrapped_ptr = thrust::device_pointer_cast(raw_ptr);

  // back to where we started
  assert(wrapped_ptr == d_ptr);
  (void) wrapped_ptr; // for when NDEBUG is defined

  // Avoid warning
  THRUST_UNUSED_VAR(wrapped_ptr);

  // deallocate device memory
  thrust::device_free(d_ptr);

  return 0;
}
