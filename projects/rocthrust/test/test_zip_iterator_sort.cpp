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

#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>

#include "test_header.hpp"

TESTS_DEFINE(ZipIteratorStableSortTests, UnsignedIntegerTestsParams);

TYPED_TEST(ZipIteratorStableSortTests, TestZipIteratorStableSort)
{
  using namespace thrust;
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    for (auto seed : get_seeds())
    {
      SCOPED_TRACE(testing::Message() << "with seed= " << seed);

      thrust::host_vector<T> h1 =
        get_random_data<T>(size, get_default_limits<T>::min(), get_default_limits<T>::max(), seed);
      thrust::host_vector<T> h2 = get_random_data<T>(
        size, get_default_limits<T>::min(), get_default_limits<T>::max(), seed + seed_value_addition);

      device_vector<T> d1 = h1;
      device_vector<T> d2 = h2;

      // sort on host
      stable_sort(make_zip_iterator(make_tuple(h1.begin(), h2.begin())),
                  make_zip_iterator(make_tuple(h1.end(), h2.end())));

      // sort on device
      stable_sort(make_zip_iterator(make_tuple(d1.begin(), d2.begin())),
                  make_zip_iterator(make_tuple(d1.end(), d2.end())));

      ASSERT_EQ_QUIET(h1, d1);
      ASSERT_EQ_QUIET(h2, d2);
    }
  }
}
