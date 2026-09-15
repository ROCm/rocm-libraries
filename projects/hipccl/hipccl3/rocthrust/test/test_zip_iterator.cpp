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
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/universal_vector.h>

#include _THRUST_STD_INCLUDE(type_traits)

#include "test_param_fixtures.hpp"
#include "test_real_assertions.hpp"
#include "test_utils.hpp"

using ZipIteratorTests32BitParams = ::testing::Types<Params<int>, Params<unsigned int>, Params<float>>;

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

TESTS_DEFINE(ZipIterator32BitTests, ZipIteratorTests32BitParams);

TESTS_DEFINE(ZipIteratorVectorTests, VectorTestsParams);

TESTS_DEFINE(ZipIteratorNumericTests, NumericalTestsParams);

TEST(ZipIterator32BitTests, UsingHip)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  ASSERT_EQ(THRUST_DEVICE_SYSTEM, THRUST_DEVICE_SYSTEM_HIP);
}

template <typename Vector>
void TestZipIteratorManipulation_test()
{
  using namespace thrust;

  Vector v0(4);
  Vector v1(4);
  Vector v2(4);

  // initialize input
  sequence(v0.begin(), v0.end());
  sequence(v1.begin(), v1.end());
  sequence(v2.begin(), v2.end());

  using IteratorTuple = tuple<typename Vector::iterator, typename Vector::iterator>;

  IteratorTuple t = make_tuple(v0.begin(), v1.begin());

  using ZipIterator = zip_iterator<IteratorTuple>;

  // test construction
  ZipIterator iter0 = make_zip_iterator(t);
  ASSERT_EQ(true, iter0 == ZipIterator{t});

  ASSERT_EQ_QUIET(v0.begin(), get<0>(iter0.get_iterator_tuple()));
  ASSERT_EQ_QUIET(v1.begin(), get<1>(iter0.get_iterator_tuple()));

  // test dereference
  ASSERT_EQ(*v0.begin(), get<0>(*iter0));
  ASSERT_EQ(*v1.begin(), get<1>(*iter0));

  // test equality
  ZipIterator iter1 = iter0;
  ZipIterator iter2 = make_zip_iterator(v0.begin(), v2.begin());
  ZipIterator iter3 = make_zip_iterator(v1.begin(), v2.begin());
  ASSERT_EQ(true, iter0 == iter1);
  ASSERT_EQ(true, iter0 == iter2);
  ASSERT_EQ(false, iter0 == iter3);

  // test inequality
  ASSERT_EQ(false, iter0 != iter1);
  ASSERT_EQ(false, iter0 != iter2);
  ASSERT_EQ(true, iter0 != iter3);

  // test advance
  ZipIterator iter4 = iter0 + 1;
  ASSERT_EQ_QUIET(v0.begin() + 1, get<0>(iter4.get_iterator_tuple()));
  ASSERT_EQ_QUIET(v1.begin() + 1, get<1>(iter4.get_iterator_tuple()));

  // test pre-increment
  ++iter4;
  ASSERT_EQ_QUIET(v0.begin() + 2, get<0>(iter4.get_iterator_tuple()));
  ASSERT_EQ_QUIET(v1.begin() + 2, get<1>(iter4.get_iterator_tuple()));

  // test post-increment
  iter4++;
  ASSERT_EQ_QUIET(v0.begin() + 3, get<0>(iter4.get_iterator_tuple()));
  ASSERT_EQ_QUIET(v1.begin() + 3, get<1>(iter4.get_iterator_tuple()));

  // test pre-decrement
  --iter4;
  ASSERT_EQ_QUIET(v0.begin() + 2, get<0>(iter4.get_iterator_tuple()));
  ASSERT_EQ_QUIET(v1.begin() + 2, get<1>(iter4.get_iterator_tuple()));

  // test post-decrement
  iter4--;
  ASSERT_EQ_QUIET(v0.begin() + 1, get<0>(iter4.get_iterator_tuple()));
  ASSERT_EQ_QUIET(v1.begin() + 1, get<1>(iter4.get_iterator_tuple()));

  // test difference
  ASSERT_EQ(1, iter4 - iter0);
  ASSERT_EQ(-1, iter0 - iter4);
}

TYPED_TEST(ZipIteratorNumericTests, TestZipIteratorManipulation)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  TestZipIteratorManipulation_test<thrust::host_vector<T>>();
  TestZipIteratorManipulation_test<thrust::device_vector<T>>();
}
static_assert(_THRUST_STD::is_trivially_copy_constructible<thrust::zip_iterator<thrust::tuple<int*, int*>>>::value, "");

TYPED_TEST(ZipIteratorNumericTests, TestZipIteratorReference)
{
  using T = typename TestFixture::input_type;
  using namespace thrust;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  // test host types
  using Iterator1      = typename host_vector<T>::iterator;
  using Iterator2      = typename host_vector<T>::const_iterator;
  using IteratorTuple1 = tuple<Iterator1, Iterator2>;
  using ZipIterator1   = zip_iterator<IteratorTuple1>;

  using zip_iterator_reference_type1 = typename iterator_reference<ZipIterator1>::type;

  host_vector<T> h_variable(1);

  using reference_type1 = tuple<T&, const T&>;

  reference_type1 ref1(*h_variable.begin(), *h_variable.cbegin());
  zip_iterator_reference_type1 test1(*h_variable.begin(), *h_variable.cbegin());

  ASSERT_EQ_QUIET(ref1, test1);
  ASSERT_EQ(get<0>(ref1), get<0>(test1));
  ASSERT_EQ(get<1>(ref1), get<1>(test1));

  // test device types
  using Iterator3      = typename device_vector<T>::iterator;
  using Iterator4      = typename device_vector<T>::const_iterator;
  using IteratorTuple2 = tuple<Iterator3, Iterator4>;
  using ZipIterator2   = zip_iterator<IteratorTuple2>;

  using zip_iterator_reference_type2 = typename iterator_reference<ZipIterator2>::type;

  device_vector<T> d_variable(1);

  using reference_type2 = tuple<device_reference<T>, device_reference<const T>>;

  reference_type2 ref2(*d_variable.begin(), *d_variable.cbegin());
  zip_iterator_reference_type2 test2(*d_variable.begin(), *d_variable.cbegin());

  ASSERT_EQ_QUIET(ref2, test2);
  ASSERT_EQ(get<0>(ref2), get<0>(test2));
  ASSERT_EQ(get<1>(ref2), get<1>(test2));
}

TYPED_TEST(ZipIteratorNumericTests, TestZipIteratorTraversal)
{
  // using T = typename TestFixture::input_type;
  using namespace thrust;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

#if 0
  // test host types
  using Iterator1 = typename host_vector<T>::iterator         ;
  using Iterator2 = typename host_vector<T>::const_iterator   ;
  using IteratorTuple1 = tuple<Iterator1,Iterator2>                ;
  using ZipIterator1 = zip_iterator<IteratorTuple1>;

  using zip_iterator_traversal_type1 = typename iterator_traversal<ZipIterator1>::type;
#endif

  // ASSERT_EQ(true, (_THRUST_STD::is_convertible<zip_iterator_traversal_type1,
  // random_access_traversal_tag>::value) );

#if 0
  // test device types
  using Iterator3 = typename device_vector<T>::iterator       ;
  using Iterator4 = typename device_vector<T>::const_iterator ;
  using IteratorTuple2 = tuple<Iterator3,Iterator4>                ;
  using ZipIterator2 = zip_iterator<IteratorTuple2>;

  using zip_iterator_traversal_type2 = typename iterator_traversal<ZipIterator2>::type;
#endif

  // ASSERT_EQ(true, (_THRUST_STD::is_convertible<zip_iterator_traversal_type2,
  // thrust::random_access_traversal_tag>::value) );
}

TYPED_TEST(ZipIteratorNumericTests, TestZipIteratorSystem)
{
  // using T = typename TestFixture::input_type;
  using namespace thrust;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  // XXX these assertions complain about undefined references to integral_constant<...>::value

#if 0
  // test host types
  using Iterator1 = typename host_vector<T>::iterator         ;
  using Iterator2 = typename host_vector<T>::const_iterator   ;
  using IteratorTuple1 = tuple<Iterator1,Iterator2>                ;
  using ZipIterator1 = zip_iterator<IteratorTuple1>;

  using zip_iterator_system_type1 = typename iterator_system<ZipIterator1>::type;
#endif

  // ASSERT_EQ(true, (_THRUST_STD::is_same<zip_iterator_system_type1, experimental::space::host>::value) );

#if 0
  // test device types
  using Iterator3 = typename device_vector<T>::iterator       ;
  using Iterator4 = typename device_vector<T>::const_iterator ;
  using IteratorTuple2 = tuple<Iterator3,Iterator4>                ;
  using ZipIterator2 = zip_iterator<IteratorTuple1>;

  using zip_iterator_system_type2 = typename iterator_system<ZipIterator2>::type;
#endif

  // ASSERT_EQ(true, (_THRUST_STD::is_convertible<zip_iterator_system_type2, experimental::space::device>::value)
  // );

#if 0
  // test any
  using Iterator5 = counting_iterator<T>        ;
  using Iterator6 = counting_iterator<const T>  ;
  using IteratorTuple3 = tuple<Iterator5, Iterator6>               ;
  using ZipIterator3 = zip_iterator<IteratorTuple3>;

  using zip_iterator_system_type3 = typename iterator_system<ZipIterator3>::type;
#endif

  // ASSERT_EQ(true, (_THRUST_STD::is_convertible<zip_iterator_system_type3,
  // thrust::experimental::space::any>::value)
  // );

#if 0
  // test host/any
  using IteratorTuple4 = tuple<Iterator1, Iterator5>               ;
  using ZipIterator4 = zip_iterator<IteratorTuple4>;

  using zip_iterator_system_type4 = typename iterator_system<ZipIterator4>::type;
#endif

  // ASSERT_EQ(true, (_THRUST_STD::is_convertible<zip_iterator_system_type4, thrust::host_system_tag>::value) );

#if 0
  // test any/host
  using IteratorTuple5 = tuple<Iterator5, Iterator1>               ;
  using ZipIterator5 = zip_iterator<IteratorTuple5>;

  using zip_iterator_system_type5 = typename iterator_system<ZipIterator5>::type;
#endif

  // ASSERT_EQ(true, (_THRUST_STD::is_convertible<zip_iterator_system_type5, thrust::host_system_tag>::value) );

#if 0
  // test device/any
  using IteratorTuple6 = tuple<Iterator3, Iterator5>               ;
  using ZipIterator6 = zip_iterator<IteratorTuple6>;

  using zip_iterator_system_type6 = typename iterator_system<ZipIterator6>::type;
#endif

  // ASSERT_EQ(true, (_THRUST_STD::is_convertible<zip_iterator_system_type6, thrust::device_system_tag>::value) );

#if 0
  // test any/device
  using IteratorTuple7 = tuple<Iterator5, Iterator3>               ;
  using ZipIterator7 = zip_iterator<IteratorTuple7>;

  using zip_iterator_system_type7 = typename iterator_system<ZipIterator7>::type;
#endif

  // ASSERT_EQ(true, (_THRUST_STD::is_convertible<zip_iterator_system_type7, thrust::device_system_tag>::value) );
}

TYPED_TEST(ZipIteratorVectorTests, TestZipIteratorCopy)
{
  using Vector = typename TestFixture::input_type;
  using namespace thrust;
  using T = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector input0(4), input1(4);
  Vector output0(4), output1(4);

  // initialize input
  sequence(input0.begin(), input0.end(), T{0});
  sequence(input1.begin(), input1.end(), T{13});

  thrust::copy(make_zip_iterator(input0.begin(), input1.begin()),
               make_zip_iterator(input0.end(), input1.end()),
               make_zip_iterator(output0.begin(), output1.begin()));

  ASSERT_EQ(input0, output0);
  ASSERT_EQ(input1, output1);
}

struct SumTwoTuple
{
  template <typename Tuple>
  THRUST_HOST_DEVICE typename _THRUST_STD::remove_reference<typename thrust::tuple_element<0, Tuple>::type>::type
  operator()(Tuple x) const
  {
    return thrust::get<0>(x) + thrust::get<1>(x);
  }
}; // end SumTwoTuple

struct SumThreeTuple
{
  template <typename Tuple>
  THRUST_HOST_DEVICE typename _THRUST_STD::remove_reference<typename thrust::tuple_element<0, Tuple>::type>::type
  operator()(Tuple x) const
  {
    return thrust::get<0>(x) + thrust::get<1>(x) + thrust::get<2>(x);
  }
}; // end SumThreeTuple

TYPED_TEST(ZipIterator32BitTests, TestZipIteratorTransform)
{
  using T = typename TestFixture::input_type;
  using namespace thrust;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    host_vector<T> h_data0 = random_samples<T>(size);
    host_vector<T> h_data1 = random_samples<T>(size);
    host_vector<T> h_data2 = random_samples<T>(size);

    device_vector<T> d_data0 = h_data0;
    device_vector<T> d_data1 = h_data1;
    device_vector<T> d_data2 = h_data2;

    host_vector<T> h_result(size);
    device_vector<T> d_result(size);

    // Tuples with 2 elements
    transform(make_zip_iterator(h_data0.begin(), h_data1.begin()),
              make_zip_iterator(h_data0.end(), h_data1.end()),
              h_result.begin(),
              SumTwoTuple());
    transform(make_zip_iterator(d_data0.begin(), d_data1.begin()),
              make_zip_iterator(d_data0.end(), d_data1.end()),
              d_result.begin(),
              SumTwoTuple());
    ASSERT_EQ(h_result, d_result);

    // Tuples with 3 elements
    transform(make_zip_iterator(h_data0.begin(), h_data1.begin(), h_data2.begin()),
              make_zip_iterator(h_data0.end(), h_data1.end(), h_data2.end()),
              h_result.begin(),
              SumThreeTuple());
    transform(make_zip_iterator(d_data0.begin(), d_data1.begin(), d_data2.begin()),
              make_zip_iterator(d_data0.end(), d_data1.end(), d_data2.end()),
              d_result.begin(),
              SumThreeTuple());
    ASSERT_EQ(h_result, d_result);
  }
}

TEST(ZipIterator32BitTests, TestZipIteratorCopyAoSToSoA)
{
  using namespace thrust;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  const size_t n = 1;

  using structure                  = tuple<int, int>;
  using host_array_of_structures   = host_vector<structure>;
  using device_array_of_structures = device_vector<structure>;

  using host_structure_of_arrays = zip_iterator<tuple<host_vector<int>::iterator, host_vector<int>::iterator>>;

  using device_structure_of_arrays = zip_iterator<tuple<device_vector<int>::iterator, device_vector<int>::iterator>>;

  host_array_of_structures h_aos(n, make_tuple(7, 13));
  device_array_of_structures d_aos(n, make_tuple(7, 13));

  // host to host
  host_vector<int> h_field0(n), h_field1(n);
  host_structure_of_arrays h_soa = make_zip_iterator(h_field0.begin(), h_field1.begin());

  thrust::copy(h_aos.begin(), h_aos.end(), h_soa);
  ASSERT_EQ_QUIET(make_tuple(7, 13), h_soa[0]);

  // host to device
  device_vector<int> d_field0(n), d_field1(n);
  device_structure_of_arrays d_soa = make_zip_iterator(d_field0.begin(), d_field1.begin());

  thrust::copy(h_aos.begin(), h_aos.end(), d_soa);
  ASSERT_EQ_QUIET(make_tuple(7, 13), d_soa[0]);

  // device to device
  thrust::fill(d_field0.begin(), d_field0.end(), 0);
  thrust::fill(d_field1.begin(), d_field1.end(), 0);

  thrust::copy(d_aos.begin(), d_aos.end(), d_soa);
  ASSERT_EQ_QUIET(make_tuple(7, 13), d_soa[0]);

  // device to host
  thrust::fill(h_field0.begin(), h_field0.end(), 0);
  thrust::fill(h_field1.begin(), h_field1.end(), 0);

  thrust::copy(d_aos.begin(), d_aos.end(), h_soa);
  ASSERT_EQ_QUIET(make_tuple(7, 13), h_soa[0]);
}

TEST(ZipIterator32BitTests, TestZipIteratorCopySoAToAoS)
{
  using namespace thrust;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  const size_t n = 1;

  using structure                  = tuple<int, int>;
  using host_array_of_structures   = host_vector<structure>;
  using device_array_of_structures = device_vector<structure>;

  using host_structure_of_arrays = zip_iterator<tuple<host_vector<int>::iterator, host_vector<int>::iterator>>;

  using device_structure_of_arrays = zip_iterator<tuple<device_vector<int>::iterator, device_vector<int>::iterator>>;

  host_vector<int> h_field0(n, 7), h_field1(n, 13);
  device_vector<int> d_field0(n, 7), d_field1(n, 13);

  host_structure_of_arrays h_soa   = make_zip_iterator(h_field0.begin(), h_field1.begin());
  device_structure_of_arrays d_soa = make_zip_iterator(d_field0.begin(), d_field1.begin());

  host_array_of_structures h_aos(n);
  device_array_of_structures d_aos(n);

  // host to host
  thrust::fill(h_aos.begin(), h_aos.end(), make_tuple(0, 0));

  thrust::copy(h_soa, h_soa + n, h_aos.begin());
  ASSERT_EQ_QUIET(7, get<0>(h_soa[0]));
  ASSERT_EQ_QUIET(13, get<1>(h_soa[0]));

  // host to device
  thrust::fill(d_aos.begin(), d_aos.end(), make_tuple(0, 0));

  thrust::copy(h_soa, h_soa + n, d_aos.begin());
  ASSERT_EQ_QUIET(7, get<0>(d_soa[0]));
  ASSERT_EQ_QUIET(13, get<1>(d_soa[0]));

  // device to device
  thrust::fill(d_aos.begin(), d_aos.end(), make_tuple(0, 0));

  thrust::copy(d_soa, d_soa + n, d_aos.begin());
  ASSERT_EQ_QUIET(7, get<0>(d_soa[0]));
  ASSERT_EQ_QUIET(13, get<1>(d_soa[0]));

  // device to host
  thrust::fill(h_aos.begin(), h_aos.end(), make_tuple(0, 0));

  thrust::copy(d_soa, d_soa + n, h_aos.begin());
  ASSERT_EQ_QUIET(7, get<0>(h_soa[0]));
  ASSERT_EQ_QUIET(13, get<1>(h_soa[0]));
}
