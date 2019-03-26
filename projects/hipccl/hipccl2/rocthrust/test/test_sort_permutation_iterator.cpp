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
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/functional.h>

#include "test_header.hpp"

TESTS_DEFINE(SortPermutationIteratorsTests, FullTestsParams);

template <typename Iterator>
class strided_range
{
    public:

    using difference_type = typename thrust::iterator_difference<Iterator>::type;

    struct stride_functor : public thrust::unary_function<difference_type,difference_type>
    {
        difference_type stride;

        stride_functor(difference_type stride)
            : stride(stride) {}

        __host__ __device__
        difference_type operator()(const difference_type& i) const
        {
            return stride * i;
        }
    };

    using CountingIterator = typename thrust::counting_iterator<difference_type>;
    using TransformIterator = typename thrust::transform_iterator<stride_functor, CountingIterator>;
    using PermutationIterator = typename thrust::permutation_iterator<Iterator,TransformIterator>;

    // type of the strided_range iterator
    using iterator = PermutationIterator;

    // construct strided_range for the range [first,last)
    strided_range(Iterator first, Iterator last, difference_type stride)
        : first(first), last(last), stride(stride) {}

    iterator begin(void) const
    {
        return PermutationIterator(first, TransformIterator(CountingIterator(0), stride_functor(stride)));
    }

    iterator end(void) const
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
  using Vector = typename TestFixture::input_type;
  using T = typename Vector::value_type;
  using Iterator = typename Vector::iterator;

  Vector A(10);
  A[0] = T(2);
  A[1] = T(9);
  A[2] = T(0);
  A[3] = T(1);
  A[4] = T(5);
  A[5] = T(3);
  A[6] = T(8);
  A[7] = T(6);
  A[8] = T(7);
  A[9] = T(4);

  strided_range<Iterator> S(A.begin(), A.end(), 2);

  thrust::sort(S.begin(), S.end());

  ASSERT_EQ(A[0], T(0));
  ASSERT_EQ(A[1], T(9));
  ASSERT_EQ(A[2], T(2));
  ASSERT_EQ(A[3], T(1));
  ASSERT_EQ(A[4], T(5));
  ASSERT_EQ(A[5], T(3));
  ASSERT_EQ(A[6], T(7));
  ASSERT_EQ(A[7], T(6));
  ASSERT_EQ(A[8], T(8));
  ASSERT_EQ(A[9], T(4));
}

TYPED_TEST(SortPermutationIteratorsTests, TestStableSortPermutationIterator)
{
  using Vector = typename TestFixture::input_type;
  using T = typename Vector::value_type;
  using Iterator = typename Vector::iterator;

  Vector A(10);
  A[0] = T(2);
  A[1] = T(9);
  A[2] = T(0);
  A[3] = T(1);
  A[4] = T(5);
  A[5] = T(3);
  A[6] = T(8);
  A[7] = T(6);
  A[8] = T(7);
  A[9] = T(4);

  strided_range<Iterator> S(A.begin(), A.end(), 2);

  thrust::stable_sort(S.begin(), S.end());

  ASSERT_EQ(A[0], T(0));
  ASSERT_EQ(A[1], T(9));
  ASSERT_EQ(A[2], T(2));
  ASSERT_EQ(A[3], T(1));
  ASSERT_EQ(A[4], T(5));
  ASSERT_EQ(A[5], T(3));
  ASSERT_EQ(A[6], T(7));
  ASSERT_EQ(A[7], T(6));
  ASSERT_EQ(A[8], T(8));
  ASSERT_EQ(A[9], T(4));
}

TYPED_TEST(SortPermutationIteratorsTests, TestSortByKeyPermutationIterator)
{
    using Vector = typename TestFixture::input_type;
    using ValueType = typename Vector::value_type;
    using Iterator = typename Vector::iterator;

    Vector A(10), B(10);
    A[0] = ValueType(2); B[0] = ValueType(0);
    A[1] = ValueType(9); B[1] = ValueType(1);
    A[2] = ValueType(0); B[2] = ValueType(2);
    A[3] = ValueType(1); B[3] = ValueType(3);
    A[4] = ValueType(5); B[4] = ValueType(4);
    A[5] = ValueType(3); B[5] = ValueType(5);
    A[6] = ValueType(8); B[6] = ValueType(6);
    A[7] = ValueType(6); B[7] = ValueType(7);
    A[8] = ValueType(7); B[8] = ValueType(8);
    A[9] = ValueType(4); B[9] = ValueType(9);

    strided_range<Iterator> S(A.begin(), A.end(), 2);
    strided_range<Iterator> T(B.begin(), B.end(), 2);

    thrust::sort_by_key(S.begin(), S.end(), T.begin());

    ASSERT_EQ(A[0], ValueType(0));
    ASSERT_EQ(A[1], ValueType(9));
    ASSERT_EQ(A[2], ValueType(2));
    ASSERT_EQ(A[3], ValueType(1));
    ASSERT_EQ(A[4], ValueType(5));
    ASSERT_EQ(A[5], ValueType(3));
    ASSERT_EQ(A[6], ValueType(7));
    ASSERT_EQ(A[7], ValueType(6));
    ASSERT_EQ(A[8], ValueType(8));
    ASSERT_EQ(A[9], ValueType(4));

    ASSERT_EQ(B[0], ValueType(2));
    ASSERT_EQ(B[1], ValueType(1));
    ASSERT_EQ(B[2], ValueType(0));
    ASSERT_EQ(B[3], ValueType(3));
    ASSERT_EQ(B[4], ValueType(4));
    ASSERT_EQ(B[5], ValueType(5));
    ASSERT_EQ(B[6], ValueType(8));
    ASSERT_EQ(B[7], ValueType(7));
    ASSERT_EQ(B[8], ValueType(6));
    ASSERT_EQ(B[9], ValueType(9));
}

TYPED_TEST(SortPermutationIteratorsTests, TestStableSortByKeyPermutationIterator)
{
    using Vector = typename TestFixture::input_type;
    using ValueType = typename Vector::value_type;
    using Iterator = typename Vector::iterator;

    Vector A(10), B(10);
    A[0] = ValueType(2); B[0] = ValueType(0);
    A[1] = ValueType(9); B[1] = ValueType(1);
    A[2] = ValueType(0); B[2] = ValueType(2);
    A[3] = ValueType(1); B[3] = ValueType(3);
    A[4] = ValueType(5); B[4] = ValueType(4);
    A[5] = ValueType(3); B[5] = ValueType(5);
    A[6] = ValueType(8); B[6] = ValueType(6);
    A[7] = ValueType(6); B[7] = ValueType(7);
    A[8] = ValueType(7); B[8] = ValueType(8);
    A[9] = ValueType(4); B[9] = ValueType(9);

    strided_range<Iterator> S(A.begin(), A.end(), 2);
    strided_range<Iterator> T(B.begin(), B.end(), 2);

    thrust::stable_sort_by_key(S.begin(), S.end(), T.begin());

    ASSERT_EQ(A[0], ValueType(0));
    ASSERT_EQ(A[1], ValueType(9));
    ASSERT_EQ(A[2], ValueType(2));
    ASSERT_EQ(A[3], ValueType(1));
    ASSERT_EQ(A[4], ValueType(5));
    ASSERT_EQ(A[5], ValueType(3));
    ASSERT_EQ(A[6], ValueType(7));
    ASSERT_EQ(A[7], ValueType(6));
    ASSERT_EQ(A[8], ValueType(8));
    ASSERT_EQ(A[9], ValueType(4));

    ASSERT_EQ(B[0], ValueType(2));
    ASSERT_EQ(B[1], ValueType(1));
    ASSERT_EQ(B[2], ValueType(0));
    ASSERT_EQ(B[3], ValueType(3));
    ASSERT_EQ(B[4], ValueType(4));
    ASSERT_EQ(B[5], ValueType(5));
    ASSERT_EQ(B[6], ValueType(8));
    ASSERT_EQ(B[7], ValueType(7));
    ASSERT_EQ(B[8], ValueType(6));
    ASSERT_EQ(B[9], ValueType(9));
}
