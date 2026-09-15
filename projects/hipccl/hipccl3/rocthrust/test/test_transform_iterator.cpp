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

#include <thrust/detail/config.h>

#if THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_NVHPC
// suppress warnings on thrust::identity
THRUST_SUPPRESS_DEPRECATED_PUSH
#endif // THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_NVHPC

#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/universal_vector.h>

#include <memory>
#include <vector>

#include "test_param_fixtures.hpp"
#include "test_real_assertions.hpp"
#include "test_utils.hpp"

#if !_THRUST_HAS_DEVICE_SYSTEM_STD
#  include <functional>
#  include <type_traits>
#  include <utility>
#endif

#if THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_NVHPC
THRUST_SUPPRESS_DEPRECATED_POP
#endif // THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_NVHPC

using VectorTestsParams = ::testing::Types<
  Params<thrust::host_vector<signed char>>,
  Params<thrust::host_vector<short>>,
  Params<thrust::host_vector<int>>,
  Params<thrust::host_vector<float>>,
  Params<thrust::host_vector<int, thrust::mr::stateless_resource_allocator<int, thrust::host_memory_resource>>>,
  Params<thrust::device_vector<signed char>>,
  Params<thrust::device_vector<short>>,
  Params<thrust::device_vector<int>>,
  Params<thrust::device_vector<float>>,
  Params<thrust::device_vector<int, thrust::mr::stateless_resource_allocator<int, thrust::device_memory_resource>>>,
  Params<thrust::universal_vector<int>>,
  Params<thrust::universal_host_pinned_vector<int>>>;

TESTS_DEFINE(TransformIteratorTests, FullTestsParams);
TESTS_DEFINE(PrimitiveTransformIteratorTests, NumericalTestsParams);
TESTS_DEFINE(VectorTransformIteratorTests, VectorTestsParams);

TEST(TransformIteratorTests, UsingHip)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  ASSERT_EQ(THRUST_DEVICE_SYSTEM, THRUST_DEVICE_SYSTEM_HIP);
}

TYPED_TEST(VectorTransformIteratorTests, TestTransformIterator)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

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
  ASSERT_EQ(output, ref);
}

TYPED_TEST(VectorTransformIteratorTests, TestMakeTransformIterator)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

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
  ASSERT_EQ(output, ref);
}

TYPED_TEST(PrimitiveTransformIteratorTests, TestTransformIteratorReduce)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    thrust::host_vector<T> h_data   = random_samples<T>(size);
    thrust::device_vector<T> d_data = h_data;

    // run on host
    T h_result = thrust::reduce(thrust::make_transform_iterator(h_data.begin(), thrust::negate<T>()),
                                thrust::make_transform_iterator(h_data.end(), thrust::negate<T>()));

    // run on device
    T d_result = thrust::reduce(thrust::make_transform_iterator(d_data.begin(), thrust::negate<T>()),
                                thrust::make_transform_iterator(d_data.end(), thrust::negate<T>()));

    ASSERT_EQ(h_result, d_result);
  }
}

struct ExtractValue
{
  int operator()(std::unique_ptr<int> const& n)
  {
    return *n;
  }
};

TEST(TransformIteratorTests, TestTransformIteratorNonCopyable)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::host_vector<std::unique_ptr<int>> hv(4);
  hv[0].reset(new int{1});
  hv[1].reset(new int{2});
  hv[2].reset(new int{3});
  hv[3].reset(new int{4});

  auto transformed = thrust::make_transform_iterator(hv.begin(), ExtractValue{});
  ASSERT_EQ(transformed[0], 1);
  ASSERT_EQ(transformed[1], 2);
  ASSERT_EQ(transformed[2], 3);
  ASSERT_EQ(transformed[3], 4);
}

struct DeviceOp
{
  __device__ int operator()(int)
  {
    return int{};
  }
};

/// Tests compilation of device-only operators.
TEST(TransformIteratorTests, TestDeviceOperator)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> dv(1);
  auto iter = thrust::make_transform_iterator(dv.begin(), DeviceOp{});
  THRUST_UNUSED_VAR(iter);
}

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

TEST(TransformIteratorTests, TestTransformIteratorReferenceAndValueType)
{
  THRUST_SUPPRESS_DEPRECATED_PUSH

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

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

    auto it_tr_tid = thrust::make_transform_iterator(it, thrust::identity<bool>{});
    static_assert(is_same<decltype(it_tr_tid)::reference, bool>::value, ""); // identity<bool>::value_type
    static_assert(is_same<decltype(it_tr_tid)::value_type, bool>::value, "");
    (void) it_tr_tid;

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

    auto it_tr_tid = thrust::make_transform_iterator(it, thrust::identity<bool>{});
    static_assert(is_same<decltype(it_tr_tid)::reference, bool>::value, ""); // identity<bool>::value_type
    static_assert(is_same<decltype(it_tr_tid)::value_type, bool>::value, "");
    (void) it_tr_tid;

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

    auto it_tr_ide = thrust::make_transform_iterator(it, thrust::identity<bool>{});
    static_assert(is_same<decltype(it_tr_ide)::reference, bool>::value, ""); // identity<bool>::value_type
    static_assert(is_same<decltype(it_tr_ide)::value_type, bool>::value, "");
    (void) it_tr_ide;

    auto it_tr_tid = thrust::make_transform_iterator(it, thrust::identity<bool>{});
    static_assert(is_same<decltype(it_tr_tid)::reference, bool>::value, ""); // identity<bool>::value_type
    static_assert(is_same<decltype(it_tr_tid)::value_type, bool>::value, "");
    (void) it_tr_tid;

    auto it_tr_cid = thrust::make_transform_iterator(it, ::internal::identity{});
    static_assert(is_same<decltype(it_tr_cid)::reference, bool>::value, ""); // special handling by
                                                                             // transform_iterator_reference
    static_assert(is_same<decltype(it_tr_cid)::value_type, bool>::value, "");
    (void) it_tr_cid;
  }
  THRUST_SUPPRESS_DEPRECATED_POP
}

TEST(TransformIteratorTests, TestTransformIteratorIdentity)
{
  THRUST_SUPPRESS_DEPRECATED_PUSH

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> v(3, 42);

  ASSERT_EQ(*thrust::make_transform_iterator(v.begin(), thrust::identity<int>{}), 42);
  ASSERT_EQ(*thrust::make_transform_iterator(v.begin(), thrust::identity<>{}), 42);
  ASSERT_EQ(*thrust::make_transform_iterator(v.begin(), ::internal::identity{}), 42);
  using namespace thrust::placeholders;
  ASSERT_EQ(*thrust::make_transform_iterator(v.begin(), _1), 42);
  THRUST_SUPPRESS_DEPRECATED_POP
}
