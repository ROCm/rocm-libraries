/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Modifications Copyright© 2019-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include <thrust/sort.h>
#include <thrust/functional.h>


template <typename T, unsigned int N>
void _TestStableSortWithLargeKeys(void)
{
    size_t n = (128 * 1024) / sizeof(FixedVector<T,N>);

    thrust::host_vector< FixedVector<T,N> > h_keys(n);

    for(size_t i = 0; i < n; i++)
        // XXX Use proper random number generation facility.
        h_keys[i] = FixedVector<T,N>(rand());

    thrust::device_vector< FixedVector<T,N> > d_keys = h_keys;
    
    thrust::stable_sort(h_keys.begin(), h_keys.end());
    thrust::stable_sort(d_keys.begin(), d_keys.end());

    ASSERT_EQUAL_QUIET(h_keys, d_keys);
}

void TestStableSortWithLargeKeys(void)
{
    _TestStableSortWithLargeKeys<int,    2>();
    _TestStableSortWithLargeKeys<int,   17>();
    _TestStableSortWithLargeKeys<int,  128>();
}
DECLARE_UNITTEST(TestStableSortWithLargeKeys);

