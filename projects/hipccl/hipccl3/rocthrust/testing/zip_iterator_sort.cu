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
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>

template <typename T>
  struct TestZipIteratorStableSort
{
  void operator()(const size_t n)
  {
    using namespace thrust;

    host_vector<T>   h1 = unittest::random_integers<T>(n);
    host_vector<T>   h2 = unittest::random_integers<T>(n);
    
    device_vector<T> d1 = h1;
    device_vector<T> d2 = h2;
    
    // sort on host
    stable_sort( make_zip_iterator(make_tuple(h1.begin(), h2.begin())),
                 make_zip_iterator(make_tuple(h1.end(),   h2.end())) );

    // sort on device
    stable_sort( make_zip_iterator(make_tuple(d1.begin(), d2.begin())),
                 make_zip_iterator(make_tuple(d1.end(),   d2.end())) );
  
    ASSERT_EQUAL_QUIET(h1, d1);
    ASSERT_EQUAL_QUIET(h2, d2);
  }
};
VariableUnitTest<TestZipIteratorStableSort, unittest::type_list<unittest::int8_t,unittest::int16_t,unittest::int32_t> > TestZipIteratorStableSortInstance;

