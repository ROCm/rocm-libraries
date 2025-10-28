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

#include <thrust/copy.h>
#include <thrust/detail/libcxx_wrapper/std/__functional/identity.h>
#include <thrust/detail/libcxx_wrapper/std/__iterator/iterator_traits.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/logical.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>

#include <memory>
#include <vector>

#include <unittest/unittest.h>

#if !_THRUST_HAS_DEVICE_SYSTEM_STD
#  include <functional>
#  include <iterator>
#  include <type_traits>
#endif

// ensure that we properly support thrust::reverse_iterator from _THRUST_STD
void TestTransformIteratorTraits()
{
  using func    = thrust::negate<int>;
  using base_it = thrust::host_vector<int>::iterator;

  using it     = thrust::transform_iterator<func, base_it>;
  using traits = _THRUST_STD::iterator_traits<it>;

  static_assert(_THRUST_STD::is_same_v<traits::difference_type, ptrdiff_t>);
  static_assert(_THRUST_STD::is_same_v<traits::value_type, int>);
  static_assert(_THRUST_STD::is_same_v<traits::pointer, void>);
  static_assert(_THRUST_STD::is_same_v<traits::reference, int>);
  static_assert(_THRUST_STD::is_same_v<traits::iterator_category, _THRUST_STD::random_access_iterator_tag>);

  static_assert(_THRUST_STD::is_same_v<thrust::iterator_traversal_t<it>, thrust::random_access_traversal_tag>);

  static_assert(::internal::is_cpp17_random_access_iterator<it>::value);

#if _THRUST_HAS_DEVICE_SYSTEM_STD || THRUST_STD_VER >= 2020
  static_assert(!_THRUST_STD::output_iterator<it, int>);
  static_assert(_THRUST_STD::input_iterator<it>);
  static_assert(_THRUST_STD::forward_iterator<it>);
  static_assert(_THRUST_STD::bidirectional_iterator<it>);
  static_assert(_THRUST_STD::random_access_iterator<it>);
  static_assert(!_THRUST_STD::contiguous_iterator<it>);
#endif
}
DECLARE_UNITTEST(TestTransformIteratorTraits);

template <class Vector>
void TestTransformIterator()
{
  using T = typename Vector::value_type;

  using UnaryFunction = thrust::negate<T>;
  using Iterator      = typename Vector::iterator;

  Vector input(4);
  Vector output(4);

  // initialize input
  thrust::sequence(input.begin(), input.end(), 1);

  // construct transform_iterator
  thrust::transform_iterator<UnaryFunction, Iterator> iter(input.begin(), UnaryFunction());

  thrust::copy(iter, iter + 4, output.begin());

  Vector ref{-1, -2, -3, -4};
  ASSERT_EQUAL(output, ref);
}
DECLARE_VECTOR_UNITTEST(TestTransformIterator);

template <class Vector>
void TestMakeTransformIterator()
{
  using T = typename Vector::value_type;

  using UnaryFunction = thrust::negate<T>;
  using Iterator      = typename Vector::iterator;

  Vector input(4);
  Vector output(4);

  // initialize input
  thrust::sequence(input.begin(), input.end(), 1);

  // construct transform_iterator
  thrust::transform_iterator<UnaryFunction, Iterator> iter(input.begin(), UnaryFunction());

  thrust::copy(thrust::make_transform_iterator(input.begin(), UnaryFunction()),
               thrust::make_transform_iterator(input.end(), UnaryFunction()),
               output.begin());

  Vector ref{-1, -2, -3, -4};
  ASSERT_EQUAL(output, ref);
}
DECLARE_VECTOR_UNITTEST(TestMakeTransformIterator);

template <typename T>
struct TestTransformIteratorReduce
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T> h_data   = unittest::random_samples<T>(n);
    thrust::device_vector<T> d_data = h_data;

    // run on host
    T h_result = thrust::reduce(thrust::make_transform_iterator(h_data.begin(), thrust::negate<T>()),
                                thrust::make_transform_iterator(h_data.end(), thrust::negate<T>()));

    // run on device
    T d_result = thrust::reduce(thrust::make_transform_iterator(d_data.begin(), thrust::negate<T>()),
                                thrust::make_transform_iterator(d_data.end(), thrust::negate<T>()));

    ASSERT_EQUAL(h_result, d_result);
  }
};
VariableUnitTest<TestTransformIteratorReduce, IntegralTypes> TestTransformIteratorReduceInstance;

struct ExtractValue
{
  int operator()(std::unique_ptr<int> const& n)
  {
    return *n;
  }
};

void TestTransformIteratorNonCopyable()
{
  thrust::host_vector<std::unique_ptr<int>> hv(4);
  hv[0].reset(new int{1});
  hv[1].reset(new int{2});
  hv[2].reset(new int{3});
  hv[3].reset(new int{4});

  auto transformed = thrust::make_transform_iterator(hv.begin(), ExtractValue{});
  ASSERT_EQUAL(transformed[0], 1);
  ASSERT_EQUAL(transformed[1], 2);
  ASSERT_EQUAL(transformed[2], 3);
  ASSERT_EQUAL(transformed[3], 4);
}

DECLARE_UNITTEST(TestTransformIteratorNonCopyable);

struct flip_value
{
  THRUST_HOST_DEVICE bool operator()(bool b) const
  {
    return !b;
  }
};

struct pass_ref
{
  THRUST_HOST_DEVICE const bool& operator()(const bool& b) const
  {
    return b;
  }
};

// a user provided functor that forwards its argument
struct forward
{
  template <class _Tp>
  constexpr _Tp&& operator()(_Tp&& __t) const noexcept
  {
    return _THRUST_STD::forward<_Tp>(__t);
  }
};

void TestTransformIteratorReferenceAndValueType()
{
  using _THRUST_STD::is_same;
  using _THRUST_STD::negate;
  {
    thrust::host_vector<bool> v;

    auto it = v.begin();
    static_assert(is_same<decltype(it)::reference, bool&>::value, ""); // ordinary reference
    static_assert(is_same<decltype(it)::value_type, bool>::value, "");

    auto it_tr_val = thrust::make_transform_iterator(it, flip_value{});
    static_assert(is_same<decltype(it_tr_val)::reference, bool>::value, "");
    static_assert(is_same<decltype(it_tr_val)::value_type, bool>::value, "");
    (void) it_tr_val;

    auto it_tr_ref = thrust::make_transform_iterator(it, pass_ref{});
    static_assert(is_same<decltype(it_tr_ref)::reference, const bool&>::value, "");
    static_assert(is_same<decltype(it_tr_ref)::value_type, bool>::value, "");
    (void) it_tr_ref;

    auto it_tr_fwd = thrust::make_transform_iterator(it, forward{});
    static_assert(is_same<decltype(it_tr_fwd)::reference, bool&&>::value, "");
    static_assert(is_same<decltype(it_tr_fwd)::value_type, bool>::value, "");
    (void) it_tr_fwd;

    auto it_tr_cid = thrust::make_transform_iterator(it, ::internal::identity{});
    static_assert(is_same<decltype(it_tr_cid)::reference, bool>::value, ""); // special handling by
                                                                             // transform_iterator_reference
    static_assert(is_same<decltype(it_tr_cid)::value_type, bool>::value, "");
    (void) it_tr_cid;
  }

  {
    thrust::device_vector<bool> v;

    auto it = v.begin();
    static_assert(is_same<decltype(it)::reference, thrust::device_reference<bool>>::value, ""); // proxy reference
    static_assert(is_same<decltype(it)::value_type, bool>::value, "");

    auto it_tr_val = thrust::make_transform_iterator(it, flip_value{});
    static_assert(is_same<decltype(it_tr_val)::reference, bool>::value, "");
    static_assert(is_same<decltype(it_tr_val)::value_type, bool>::value, "");
    (void) it_tr_val;

    auto it_tr_ref = thrust::make_transform_iterator(it, pass_ref{});
    static_assert(is_same<decltype(it_tr_ref)::reference, const bool&>::value, "");
    static_assert(is_same<decltype(it_tr_ref)::value_type, bool>::value, "");
    (void) it_tr_ref;

    auto it_tr_fwd = thrust::make_transform_iterator(it, forward{});
    static_assert(is_same<decltype(it_tr_fwd)::reference, bool&&>::value, ""); // wrapped reference is decayed
    static_assert(is_same<decltype(it_tr_fwd)::value_type, bool>::value, "");
    (void) it_tr_fwd;

    auto it_tr_cid = thrust::make_transform_iterator(it, ::internal::identity{});
    static_assert(is_same<decltype(it_tr_cid)::reference, bool>::value, ""); // special handling by
                                                                             // transform_iterator_reference
    static_assert(is_same<decltype(it_tr_cid)::value_type, bool>::value, "");
    (void) it_tr_cid;
  }

  {
    std::vector<bool> v;

    auto it = v.begin();
    static_assert(is_same<decltype(it)::reference, std::vector<bool>::reference>::value, ""); // proxy reference
    static_assert(is_same<decltype(it)::value_type, bool>::value, "");

    auto it_tr_val = thrust::make_transform_iterator(it, flip_value{});
    static_assert(is_same<decltype(it_tr_val)::reference, bool>::value, "");
    static_assert(is_same<decltype(it_tr_val)::value_type, bool>::value, "");
    (void) it_tr_val;

    auto it_tr_ref = thrust::make_transform_iterator(it, pass_ref{});
    static_assert(is_same<decltype(it_tr_ref)::reference, const bool&>::value, "");
    static_assert(is_same<decltype(it_tr_ref)::value_type, bool>::value, "");
    (void) it_tr_ref;

    auto it_tr_fwd = thrust::make_transform_iterator(it, forward{});
    static_assert(is_same<decltype(it_tr_fwd)::reference, bool&&>::value, ""); // proxy reference is decayed
    static_assert(is_same<decltype(it_tr_fwd)::value_type, bool>::value, "");
    (void) it_tr_fwd;

    auto it_tr_cid = thrust::make_transform_iterator(it, ::internal::identity{});
    static_assert(is_same<decltype(it_tr_cid)::reference, bool>::value, ""); // special handling by
                                                                             // transform_iterator_reference
    static_assert(is_same<decltype(it_tr_cid)::value_type, bool>::value, "");
    (void) it_tr_cid;
  }
}
DECLARE_UNITTEST(TestTransformIteratorReferenceAndValueType);

void TestTransformIteratorIdentity()
{
  thrust::device_vector<int> v(3, 42);

  ASSERT_EQUAL(*thrust::make_transform_iterator(v.begin(), ::internal::identity{}), 42);
  using namespace thrust::placeholders;
  ASSERT_EQUAL(*thrust::make_transform_iterator(v.begin(), _1), 42);
}

DECLARE_UNITTEST(TestTransformIteratorIdentity);
