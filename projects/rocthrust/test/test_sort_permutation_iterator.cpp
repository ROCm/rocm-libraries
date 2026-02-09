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

#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/sort.h>

#include "test_param_fixtures.hpp"
#include "test_real_assertions.hpp"
#include "test_utils.hpp"

TESTS_DEFINE(SortPermutationIteratorsTests, FullTestsParams);

template <typename Iterator>
class strided_range
{
public:
  using difference_type = typename thrust::iterator_difference<Iterator>::type;

  struct stride_functor
  {
    difference_type stride;

    stride_functor(difference_type stride)
        : stride(stride)
    {}

    THRUST_HOST_DEVICE difference_type operator()(const difference_type& i) const
    {
      return stride * i;
    }
  };

  using CountingIterator    = typename thrust::counting_iterator<difference_type>;
  using TransformIterator   = typename thrust::transform_iterator<stride_functor, CountingIterator>;
  using PermutationIterator = typename thrust::permutation_iterator<Iterator, TransformIterator>;

  // type of the strided_range iterator
  using iterator = PermutationIterator;

  // construct strided_range for the range [first,last)
  strided_range(Iterator first, Iterator last, difference_type stride)
      : first(first)
      , last(last)
      , stride(stride)
  {}

  iterator begin() const
  {
    return PermutationIterator(first, TransformIterator(CountingIterator(0), stride_functor(stride)));
  }

  iterator end() const
  {
    return begin() + ((last - first) + (stride - 1)) / stride;
  }

protected:
  Iterator first;
  Iterator last;
  difference_type stride;
};

TYPED_TEST(SortPermutationIteratorsTests, TestSortPermutationIterator)
{
  using Vector   = typename TestFixture::input_type;
  using Iterator = typename Vector::iterator;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector A{2, 9, 0, 1, 5, 3, 8, 6, 7, 4};

  strided_range<Iterator> S(A.begin(), A.end(), 2);

  thrust::sort(S.begin(), S.end());

  Vector ref{0, 9, 2, 1, 5, 3, 7, 6, 8, 4};
  ASSERT_EQ(A, ref);
}

TYPED_TEST(SortPermutationIteratorsTests, TestStableSortPermutationIterator)
{
  using Vector   = typename TestFixture::input_type;
  using Iterator = typename Vector::iterator;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector A{2, 9, 0, 1, 5, 3, 8, 6, 7, 4};

  strided_range<Iterator> S(A.begin(), A.end(), 2);

  thrust::stable_sort(S.begin(), S.end());

  Vector ref{0, 9, 2, 1, 5, 3, 7, 6, 8, 4};
  ASSERT_EQ(A, ref);
}

TYPED_TEST(SortPermutationIteratorsTests, TestSortByKeyPermutationIterator)
{
  using Vector   = typename TestFixture::input_type;
  using Iterator = typename Vector::iterator;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector A{2, 9, 0, 1, 5, 3, 8, 6, 7, 4};
  Vector B{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  strided_range<Iterator> S(A.begin(), A.end(), 2);
  strided_range<Iterator> T(B.begin(), B.end(), 2);

  thrust::sort_by_key(S.begin(), S.end(), T.begin());

  Vector ref_A{0, 9, 2, 1, 5, 3, 7, 6, 8, 4};
  ASSERT_EQ(A, ref_A);

  Vector ref_B{2, 1, 0, 3, 4, 5, 8, 7, 6, 9};
  ASSERT_EQ(B, ref_B);
}

TYPED_TEST(SortPermutationIteratorsTests, TestStableSortByKeyPermutationIterator)
{
  using Vector   = typename TestFixture::input_type;
  using Iterator = typename Vector::iterator;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector A{2, 9, 0, 1, 5, 3, 8, 6, 7, 4};
  Vector B{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  strided_range<Iterator> S(A.begin(), A.end(), 2);
  strided_range<Iterator> T(B.begin(), B.end(), 2);

  thrust::stable_sort_by_key(S.begin(), S.end(), T.begin());

  Vector ref_A{0, 9, 2, 1, 5, 3, 7, 6, 8, 4};
  ASSERT_EQ(A, ref_A);

  Vector ref_B{2, 1, 0, 3, 4, 5, 8, 7, 6, 9};
  ASSERT_EQ(B, ref_B);
}
