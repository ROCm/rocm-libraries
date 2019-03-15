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


#include <thrust/sort.h>
#include <thrust/functional.h>

#include "test_utils.hpp"
#include "test_assertions.hpp"

template <typename T>
struct less_div_10
{
  __host__ __device__ bool operator()(const T &lhs, const T &rhs) const {return ((int) lhs) / 10 < ((int) rhs) / 10;}
};

template <typename T>
struct greater_div_10
{
  __host__ __device__ bool operator()(const T &lhs, const T &rhs) const {return ((int) lhs) / 10 > ((int) rhs) / 10;}
};


template <typename T, unsigned int N>
void _TestStableSortByKeyWithLargeKeys(void)
{
    size_t n = (128 * 1024) / sizeof(FixedVector<T,N>);

    thrust::host_vector< FixedVector<T,N> > h_keys(n);
    thrust::host_vector<   unsigned int   > h_vals(n);

    for(size_t i = 0; i < n; i++)
    {
        h_keys[i] = FixedVector<T,N>(rand());
        h_vals[i] = i;
    }

    thrust::device_vector< FixedVector<T,N> > d_keys = h_keys;
    thrust::device_vector<   unsigned int   > d_vals = h_vals;

    thrust::stable_sort_by_key(h_keys.begin(), h_keys.end(), h_vals.begin());
    thrust::stable_sort_by_key(d_keys.begin(), d_keys.end(), d_vals.begin());

    ASSERT_EQ_QUIET(h_keys, d_keys);
    ASSERT_EQ_QUIET(h_vals, d_vals);
}


TEST(StableSortByKeyLargeTests, TestStableSortByKeyWithLargeKeys)
{
  _TestStableSortByKeyWithLargeKeys<int,    4>();
  _TestStableSortByKeyWithLargeKeys<int,    8>();
// XXX these take too long to compile
// also anything larger than 8 causes the compiler to not unroll the loops.
//    _TestStableSortByKeyWithLargeKeys<int,   16>();
//    _TestStableSortByKeyWithLargeKeys<int,   32>();
//    _TestStableSortByKeyWithLargeKeys<int,   64>();
//    _TestStableSortByKeyWithLargeKeys<int,  128>();
//    _TestStableSortByKeyWithLargeKeys<int,  256>();
//    _TestStableSortByKeyWithLargeKeys<int,  512>();
//    _TestStableSortByKeyWithLargeKeys<int, 1024>();
//    _TestStableSortByKeyWithLargeKeys<int, 2048>();
//    _TestStableSortByKeyWithLargeKeys<int, 4096>();
//    _TestStableSortByKeyWithLargeKeys<int, 8192>();
}

template <typename T, unsigned int N>
void _TestStableSortByKeyWithLargeValues(void)
{
    size_t n = (128 * 1024) / sizeof(FixedVector<T,N>);

    thrust::host_vector<   unsigned int   > h_keys(n);
    thrust::host_vector< FixedVector<T,N> > h_vals(n);

    for(size_t i = 0; i < n; i++)
    {
        h_keys[i] = rand();
        h_vals[i] = FixedVector<T,N>(i);
    }

    thrust::device_vector<   unsigned int   > d_keys = h_keys;
    thrust::device_vector< FixedVector<T,N> > d_vals = h_vals;

    thrust::stable_sort_by_key(h_keys.begin(), h_keys.end(), h_vals.begin());
    thrust::stable_sort_by_key(d_keys.begin(), d_keys.end(), d_vals.begin());

    ASSERT_EQ_QUIET(h_keys, d_keys);
    ASSERT_EQ_QUIET(h_vals, d_vals);

    // so cuda::stable_merge_sort_by_key() is called
    thrust::stable_sort_by_key(h_keys.begin(), h_keys.end(), h_vals.begin(), greater_div_10<unsigned int>());
    thrust::stable_sort_by_key(d_keys.begin(), d_keys.end(), d_vals.begin(), greater_div_10<unsigned int>());

    ASSERT_EQ_QUIET(h_keys, d_keys);
    ASSERT_EQ_QUIET(h_vals, d_vals);
}

TEST(StableSortByKeyLargeTests, TestStableSortByKeyWithLargeValues)
{
  _TestStableSortByKeyWithLargeValues<int,    4>();
  _TestStableSortByKeyWithLargeValues<int,    8>();
// XXX these take too long to compile
// also anything larger than 8 causes the compiler to not unroll the loops.
//    _TestStableSortByKeyWithLargeValues<int,   16>();
//    _TestStableSortByKeyWithLargeValues<int,   32>();
//    _TestStableSortByKeyWithLargeValues<int,   64>();
//    _TestStableSortByKeyWithLargeValues<int,  128>();
//    _TestStableSortByKeyWithLargeValues<int,  256>();
//    _TestStableSortByKeyWithLargeValues<int,  512>();
//    _TestStableSortByKeyWithLargeValues<int, 1024>();
//    _TestStableSortByKeyWithLargeValues<int, 2048>();
//    _TestStableSortByKeyWithLargeValues<int, 4096>();
//    _TestStableSortByKeyWithLargeValues<int, 8192>();
}


template <typename T, unsigned int N>
void _TestStableSortByKeyWithLargeKeysAndValues(void)
{
    size_t n = (128 * 1024) / sizeof(FixedVector<T,N>);

    thrust::host_vector< FixedVector<T,N> > h_keys(n);
    thrust::host_vector< FixedVector<T,N> > h_vals(n);

    for(size_t i = 0; i < n; i++)
    {
        h_keys[i] = FixedVector<T,N>(rand());
        h_vals[i] = FixedVector<T,N>(i);
    }

    thrust::device_vector< FixedVector<T,N> > d_keys = h_keys;
    thrust::device_vector< FixedVector<T,N> > d_vals = h_vals;

    thrust::stable_sort_by_key(h_keys.begin(), h_keys.end(), h_vals.begin());
    thrust::stable_sort_by_key(d_keys.begin(), d_keys.end(), d_vals.begin());

    ASSERT_EQ_QUIET(h_keys, d_keys);
    ASSERT_EQ_QUIET(h_vals, d_vals);
}


TEST(StableSortByKeyLargeTests, TestStableSortByKeyWithLargeKeysAndValues)
{
  _TestStableSortByKeyWithLargeKeysAndValues<int,    4>();
  _TestStableSortByKeyWithLargeKeysAndValues<int,    8>();
// XXX these take too long to compile,
// also anything larger than 8 causes the compiler to not unroll the loops.
//    _TestStableSortByKeyWithLargeKeysAndValues<int,   16>();
//    _TestStableSortByKeyWithLargeKeysAndValues<int,   32>();
//    _TestStableSortByKeyWithLargeKeysAndValues<int,   64>();
//    _TestStableSortByKeyWithLargeKeysAndValues<int,  128>();
//    _TestStableSortByKeyWithLargeKeysAndValues<int,  256>();
//    _TestStableSortByKeyWithLargeKeysAndValues<int,  512>();
//    _TestStableSortByKeyWithLargeKeysAndValues<int, 1024>();
//    _TestStableSortByKeyWithLargeKeysAndValues<int, 2048>();
//    _TestStableSortByKeyWithLargeKeysAndValues<int, 4096>();
//    _TestStableSortByKeyWithLargeKeysAndValues<int, 8192>();
}
