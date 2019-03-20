// MIT License
//
// Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.
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
#include <thrust/sort.h>
#include <thrust/functional.h>

#include "test_header.hpp"

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

template <typename T, unsigned int N>
void _TestStableSortWithLargeKeys(void)
{
    size_t n = (128 * 1024) / sizeof(FixedVector<T,N>);

    thrust::host_vector< FixedVector<T,N> > h_keys(n);

    for(size_t i = 0; i < n; i++)
        h_keys[i] = FixedVector<T,N>(rand());

    thrust::device_vector< FixedVector<T,N> > d_keys = h_keys;

    thrust::stable_sort(h_keys.begin(), h_keys.end());
    thrust::stable_sort(d_keys.begin(), d_keys.end());

    ASSERT_EQ_QUIET(h_keys, d_keys);
}

TEST(StableSortLargeTests, TestStableSortWithLargeKeys)
{
  _TestStableSortWithLargeKeys<int,    1>();
  _TestStableSortWithLargeKeys<int,    2>();
  _TestStableSortWithLargeKeys<int,    4>();
  _TestStableSortWithLargeKeys<int,    8>();
// STREAM HPC investigate and fix `error: local memory limit exceeded`
// (make block size smaller for large keys and values in rocPRIM)
#if THRUST_DEVICE_SYSTEM != THRUST_DEVICE_SYSTEM_HIP
  _TestStableSortWithLargeKeys<int,   16>();
  _TestStableSortWithLargeKeys<int,   32>();
  _TestStableSortWithLargeKeys<int,   64>();
  _TestStableSortWithLargeKeys<int,  128>();
  _TestStableSortWithLargeKeys<int,  256>();
  _TestStableSortWithLargeKeys<int,  512>();
  _TestStableSortWithLargeKeys<int, 1024>();

// XXX these take too long to compile
//    _TestStableSortWithLargeKeys<int, 2048>();
//    _TestStableSortWithLargeKeys<int, 4096>();
//    _TestStableSortWithLargeKeys<int, 8192>();
#endif
}

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
