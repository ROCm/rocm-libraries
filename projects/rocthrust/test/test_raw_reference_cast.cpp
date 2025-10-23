/*
 *  Copyright 2025 NVIDIA Corporation
 *  Modifications Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <thrust/detail/raw_reference_cast.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>

#include "test_param_fixtures.hpp"
#include "test_utils.hpp"

#if !_THRUST_HAS_DEVICE_SYSTEM_STD
#  include <type_traits>
#endif

TESTS_DEFINE(RawReferenceCastTests, FullTestsParams);

TEST(RawReferenceCastTests, TestRawReferenceCast)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  using _THRUST_STD::is_same_v;

  {
    [[maybe_unused]] int i        = 42;
    [[maybe_unused]] const int ci = 42;
    static_assert(is_same_v<decltype(thrust::raw_reference_cast(i)), int&>);
    static_assert(is_same_v<decltype(thrust::raw_reference_cast(ci)), const int&>);
  }
  {
    [[maybe_unused]] thrust::host_vector<int> vec(1);
    static_assert(is_same_v<decltype(thrust::raw_reference_cast(*vec.begin())), int&>);
    static_assert(is_same_v<decltype(thrust::raw_reference_cast(*vec.cbegin())), const int&>);

    [[maybe_unused]] auto zip = thrust::make_zip_iterator(vec.begin(), vec.begin());
    static_assert(
      is_same_v<decltype(thrust::raw_reference_cast(*zip)), thrust::detail::tuple_of_iterator_references<int&, int&>>);

    [[maybe_unused]] auto zip2 = thrust::make_zip_iterator(zip, zip);
    static_assert(
      is_same_v<decltype(thrust::raw_reference_cast(*zip2)),
                thrust::detail::tuple_of_iterator_references<thrust::detail::tuple_of_iterator_references<int&, int&>,
                                                             thrust::detail::tuple_of_iterator_references<int&, int&>>>);
  }
  {
    [[maybe_unused]] thrust::device_vector<int> vec(1);
    static_assert(is_same_v<decltype(thrust::raw_reference_cast(*vec.begin())), int&>);
    static_assert(is_same_v<decltype(thrust::raw_reference_cast(*vec.cbegin())), const int&>);

    [[maybe_unused]] auto zip = thrust::make_zip_iterator(vec.begin(), vec.begin());
    static_assert(
      is_same_v<decltype(thrust::raw_reference_cast(*zip)), thrust::detail::tuple_of_iterator_references<int&, int&>>);

    [[maybe_unused]] auto zip2 = thrust::make_zip_iterator(zip, zip);
    static_assert(
      is_same_v<decltype(thrust::raw_reference_cast(*zip2)),
                thrust::detail::tuple_of_iterator_references<thrust::detail::tuple_of_iterator_references<int&, int&>,
                                                             thrust::detail::tuple_of_iterator_references<int&, int&>>>);
  }
}
