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

// Google Test
#include <gtest/gtest.h>
#include "test_utils.hpp"
#include "test_assertions.hpp"

#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/transform.h>

// HIP API
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>

#define HIP_CHECK(condition) ASSERT_EQ(condition, hipSuccess)
#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC


#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC


template<class InputType> struct Params
{
    using input_type = InputType;
};

template<class Params> class ZipIterator32BitTests : public ::testing::Test
{
public:
    using input_type = typename Params::input_type;
};

template<class Params> class ZipIteratorVectorTests : public ::testing::Test
{
public:
    using input_type = typename Params::input_type;
};

template<class Params> class ZipIteratorNumericTests : public ::testing::Test
{
public:
    using input_type = typename Params::input_type;
};


typedef ::testing::Types<Params<int>, Params<unsigned int>, Params<float> > ZipIteratorTests32BitParams;
typedef ::testing::Types<Params<short>, Params<int>, Params<long long>, Params<unsigned short>, Params<unsigned int>, Params<unsigned long long>, Params<float>, Params<double> > ZipIteratorTestsVectorParams;
typedef ::testing::Types<Params<char>, Params<signed char>, Params<unsigned char>, Params<short>, Params<unsigned short>, Params<int>, Params<unsigned int>, Params<long>, Params<unsigned long>, Params<long long>, Params<unsigned long long>, Params<float> > ZipIteratorTestsNumericParams;

TYPED_TEST_CASE(ZipIterator32BitTests, ZipIteratorTests32BitParams);
TYPED_TEST_CASE(ZipIteratorVectorTests, ZipIteratorTestsVectorParams);
TYPED_TEST_CASE(ZipIteratorNumericTests, ZipIteratorTestsNumericParams);

TEST(ZipIterator32BitTests, UsingHip)
{
    ASSERT_EQ(THRUST_DEVICE_SYSTEM, THRUST_DEVICE_SYSTEM_HIP);
}

TYPED_TEST(ZipIteratorVectorTests, TestZipIteratorManipulation)
{
    using T = typename TestFixture::input_type;
    using namespace thrust;

    thrust::device_vector<T> v0(4);
    thrust::device_vector<T> v1(4);
    thrust::device_vector<T> v2(4);

    // initialize input
    sequence(
            v0.begin(), v0.end());
    sequence(
            v1.begin(), v1.end());
    sequence(
            v2.begin(), v2.end());

    typedef tuple<typename thrust::device_vector<T>::iterator, typename thrust::device_vector<T>::iterator> IteratorTuple;

    IteratorTuple t = make_tuple(
            v0.begin(), v1.begin());

    typedef zip_iterator<IteratorTuple> ZipIterator;

    // test construction
    ZipIterator iter0 = make_zip_iterator(t);

    ASSERT_EQ_QUIET(v0.begin(), get<0>(iter0.get_iterator_tuple()));
    ASSERT_EQ_QUIET(v1.begin(), get<1>(iter0.get_iterator_tuple()));

    // test dereference
    ASSERT_EQ(*v0.begin(), get<0>(*iter0));
    ASSERT_EQ(*v1.begin(), get<1>(*iter0));

    // test equality
    ZipIterator iter1 = iter0;
    ZipIterator iter2 = make_zip_iterator(
            make_tuple(
                    v0.begin(), v2.begin()));
    ZipIterator iter3 = make_zip_iterator(
            make_tuple(
                    v1.begin(), v2.begin()));
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

TYPED_TEST(ZipIteratorVectorTests, TestZipIteratorReference)
{
    using T = typename TestFixture::input_type;
    using namespace thrust;

    // test host types
    typedef typename host_vector<T>::iterator Iterator1;
    typedef typename host_vector<T>::const_iterator Iterator2;
    typedef tuple<Iterator1, Iterator2> IteratorTuple1;
    typedef zip_iterator<IteratorTuple1> ZipIterator1;

    typedef typename iterator_reference<ZipIterator1>::type zip_iterator_reference_type1;

    host_vector<T> h_variable(1);

    typedef tuple<T&, const T&> reference_type1;

    reference_type1 ref1(
            *h_variable.begin(), *h_variable.cbegin());
    zip_iterator_reference_type1 test1(
            *h_variable.begin(), *h_variable.cbegin());

    ASSERT_EQ_QUIET(ref1, test1);
    ASSERT_EQ(get<0>(ref1), get<0>(test1));
    ASSERT_EQ(get<1>(ref1), get<1>(test1));


    // test device types
    typedef typename device_vector<T>::iterator Iterator3;
    typedef typename device_vector<T>::const_iterator Iterator4;
    typedef tuple<Iterator3, Iterator4> IteratorTuple2;
    typedef zip_iterator<IteratorTuple2> ZipIterator2;

    typedef typename iterator_reference<ZipIterator2>::type zip_iterator_reference_type2;

    device_vector<T> d_variable(1);

    typedef tuple<device_reference<T>, device_reference<const T> > reference_type2;

    reference_type2 ref2(
            *d_variable.begin(), *d_variable.cbegin());
    zip_iterator_reference_type2 test2(
            *d_variable.begin(), *d_variable.cbegin());

    ASSERT_EQ_QUIET(ref2, test2);
    ASSERT_EQ(get<0>(ref2), get<0>(test2));
    ASSERT_EQ(get<1>(ref2), get<1>(test2));
}

// undefined reference to `thrust::detail::integral_constant<bool, true>::value' for asserts
TYPED_TEST(ZipIteratorNumericTests, TestZipIteratorTraversal)
{
    //    using T = typename TestFixture::input_type;
    using namespace thrust;

#if 0
    // test host types
    typedef typename host_vector<T>::iterator Iterator1;
    typedef typename host_vector<T>::const_iterator Iterator2;
    typedef tuple<Iterator1, Iterator2> IteratorTuple1;
    typedef zip_iterator<IteratorTuple1> ZipIterator1;

    typedef typename iterator_traversal<ZipIterator1>::type zip_iterator_traversal_type1;

    ASSERT_EQ(true,
              (detail::is_convertible<zip_iterator_traversal_type1, random_access_traversal_tag>::value));
#endif

#if 0
    // test device types
    typedef typename device_vector<T>::iterator        Iterator3;
    typedef typename device_vector<T>::const_iterator  Iterator4;
    typedef tuple<Iterator3,Iterator4>                 IteratorTuple2;
    typedef zip_iterator<IteratorTuple2> ZipIterator2;

    typedef typename iterator_traversal<ZipIterator2>::type zip_iterator_traversal_type2;

    ASSERT_EQ(true,
                  (detail::is_convertible<zip_iterator_traversal_type2, thrust::random_access_traversal_tag>::value));
#endif
}

// undefined reference to `thrust::detail::integral_constant<bool, true>::value' for asserts
// also use of experimental::space::XXXXXXXXXX
TYPED_TEST(ZipIteratorNumericTests, TestZipIteratorSystem)
{
    //    using T = typename TestFixture::input_type;
    using namespace thrust;

#if 0
    // test host types
    typedef typename host_vector<T>::iterator          Iterator1;
    typedef typename host_vector<T>::const_iterator    Iterator2;
    typedef tuple<Iterator1,Iterator2>                 IteratorTuple1;
    typedef zip_iterator<IteratorTuple1> ZipIterator1;

    typedef typename iterator_system<ZipIterator1>::type zip_iterator_system_type1;
#endif

    //    ASSERT_EQ(true, (detail::is_same<zip_iterator_system_type1, experimental::space::host>::value) );


#if 0
    // test device types
    typedef typename device_vector<T>::iterator        Iterator3;
    typedef typename device_vector<T>::const_iterator  Iterator4;
    typedef tuple<Iterator3,Iterator4>                 IteratorTuple2;
    typedef zip_iterator<IteratorTuple1> ZipIterator2;

    typedef typename iterator_system<ZipIterator2>::type zip_iterator_system_type2;
#endif

    //ASSERT_EQUAL(true, (detail::is_convertible<zip_iterator_system_type2, experimental::space::device>::value) );


#if 0
    // test any
    typedef counting_iterator<T>         Iterator5;
    typedef counting_iterator<const T>   Iterator6;
    typedef tuple<Iterator5, Iterator6>                IteratorTuple3;
    typedef zip_iterator<IteratorTuple3> ZipIterator3;

    typedef typename iterator_system<ZipIterator3>::type zip_iterator_system_type3;
#endif

    //ASSERT_EQ(true, (detail::is_convertible<zip_iterator_system_type3, thrust::experimental::space::any>::value) );


#if 0
    // test host/any
    typedef tuple<Iterator1, Iterator5>                IteratorTuple4;
    typedef zip_iterator<IteratorTuple4> ZipIterator4;

    typedef typename iterator_system<ZipIterator4>::type zip_iterator_system_type4;
#endif

    //    ASSERT_EQ(true, (detail::is_convertible<zip_iterator_system_type4, thrust::host_system_tag>::value) );


#if 0
    // test any/host
typedef tuple<Iterator5, Iterator1>                IteratorTuple5;
typedef zip_iterator<IteratorTuple5> ZipIterator5;

typedef typename iterator_system<ZipIterator5>::type zip_iterator_system_type5;
#endif

    //ASSERT_EQ(true, (detail::is_convertible<zip_iterator_system_type5, thrust::host_system_tag>::value) );


#if 0
    // test device/any
typedef tuple<Iterator3, Iterator5>                IteratorTuple6;
typedef zip_iterator<IteratorTuple6> ZipIterator6;

typedef typename iterator_system<ZipIterator6>::type zip_iterator_system_type6;
#endif

    //ASSERT_EQ(true, (detail::is_convertible<zip_iterator_system_type6, thrust::device_system_tag>::value) );


#if 0
    // test any/device
typedef tuple<Iterator5, Iterator3>                IteratorTuple7;
typedef zip_iterator<IteratorTuple7> ZipIterator7;

typedef typename iterator_system<ZipIterator7>::type zip_iterator_system_type7;
#endif

    //        ASSERT_EQ(true, (detail::is_convertible<zip_iterator_system_type7, thrust::device_system_tag>::value) );
}

TYPED_TEST(ZipIteratorVectorTests, TestZipIteratorCopy)
{
    using T = typename TestFixture::input_type;
    using namespace thrust;

    thrust::device_vector<T> input0(4), input1(4);
    thrust::device_vector<T> output0(4), output1(4);

    // initialize input
    sequence(
            input0.begin(), input0.end(), 0);
    sequence(
            input1.begin(), input1.end(), 13);

    copy(
            make_zip_iterator(
                    make_tuple(
                            input0.begin(), input1.begin())), make_zip_iterator(
                    make_tuple(
                            input0.end(), input1.end())), make_zip_iterator(
                    make_tuple(
                            output0.begin(), output1.begin())));

    ASSERT_EQ(input0, output0);
    ASSERT_EQ(input1, output1);
}

struct SumTwoTuple
{
    template<typename Tuple> __host__ __device__
    typename thrust::detail::remove_reference<typename thrust::tuple_element<0, Tuple>::type>::type
    operator()(Tuple x) const
    {
        return thrust::get<0>(x) + thrust::get<1>(x);
    }
}; // end SumTwoTuple

struct SumThreeTuple
{
    template<typename Tuple> __host__ __device__
    typename thrust::detail::remove_reference<typename thrust::tuple_element<0, Tuple>::type>::type
    operator()(Tuple x) const
    {
        return thrust::get<0>(x) + thrust::get<1>(x) + thrust::get<2>(x);
    }
}; // end SumThreeTuple


TYPED_TEST(ZipIterator32BitTests, TestZipIteratorTransform)
{
    using T = typename TestFixture::input_type;

    const std::vector<size_t> sizes = get_sizes();
    for (auto size : sizes)
    {
        using namespace thrust;

        host_vector<T> h_data0 = get_random_data<T>(
                size, 0, 10);
        host_vector<T> h_data1 = get_random_data<T>(
                size, 0, 10);
        host_vector<T> h_data2 = get_random_data<T>(
                size, 0, 10);

        device_vector<T> d_data0 = h_data0;
        device_vector<T> d_data1 = h_data1;
        device_vector<T> d_data2 = h_data2;

        host_vector<T> h_result(size);
        device_vector<T> d_result(size);

        // Tuples with 2 elements
        transform(
                make_zip_iterator(
                        make_tuple(
                                h_data0.begin(), h_data1.begin())), make_zip_iterator(
                        make_tuple(
                                h_data0.end(), h_data1.end())), h_result.begin(),
                SumTwoTuple());
        transform(
                make_zip_iterator(
                        make_tuple(
                                d_data0.begin(), d_data1.begin())), make_zip_iterator(
                        make_tuple(
                                d_data0.end(), d_data1.end())), d_result.begin(),
                SumTwoTuple());
        ASSERT_EQ_QUIET(h_result, d_result);


        // Tuples with 3 elements
        transform(
                make_zip_iterator(
                        make_tuple(
                                h_data0.begin(), h_data1.begin(), h_data2.begin())),
                make_zip_iterator(
                        make_tuple(
                                h_data0.end(), h_data1.end(), h_data2.end())),
                h_result.begin(), SumThreeTuple());
        transform(
                make_zip_iterator(
                        make_tuple(
                                d_data0.begin(), d_data1.begin(), d_data2.begin())),
                make_zip_iterator(
                        make_tuple(
                                d_data0.end(), d_data1.end(), d_data2.end())),
                d_result.begin(), SumThreeTuple());
        ASSERT_EQ_QUIET(h_result, d_result);
    }
}

TEST(ZipIterator32BitTests, TestZipIteratorCopyAoSToSoA)
{
    using namespace thrust;

    const size_t n = 1;

    typedef tuple<int, int> structure;
    typedef host_vector<structure> host_array_of_structures;
    typedef device_vector<structure> device_array_of_structures;

    typedef zip_iterator<tuple<host_vector<int>::iterator, host_vector<int>::iterator> > host_structure_of_arrays;

    typedef zip_iterator<tuple<device_vector<int>::iterator, device_vector<int>::iterator> > device_structure_of_arrays;

    host_array_of_structures h_aos(
            n, make_tuple(
                    7, 13));
    device_array_of_structures d_aos(
            n, make_tuple(
                    7, 13));

    // host to host
    host_vector<int> h_field0(n), h_field1(n);
    host_structure_of_arrays h_soa = make_zip_iterator(
            make_tuple(
                    h_field0.begin(), h_field1.begin()));

    thrust::copy(
            h_aos.begin(), h_aos.end(), h_soa);
    ASSERT_EQ_QUIET(make_tuple(
            7, 13), h_soa[0]);

    // host to device
    device_vector<int> d_field0(n), d_field1(n);
    device_structure_of_arrays d_soa = make_zip_iterator(
            make_tuple(
                    d_field0.begin(), d_field1.begin()));

    thrust::copy(
            h_aos.begin(), h_aos.end(), d_soa);
    ASSERT_EQ_QUIET(make_tuple(
            7, 13), d_soa[0]);

    // device to device
    thrust::fill(
            d_field0.begin(), d_field0.end(), 0);
    thrust::fill(
            d_field1.begin(), d_field1.end(), 0);

    thrust::copy(
            d_aos.begin(), d_aos.end(), d_soa);
    ASSERT_EQ_QUIET(make_tuple(
            7, 13), d_soa[0]);

    // device to host
    thrust::fill(
            h_field0.begin(), h_field0.end(), 0);
    thrust::fill(
            h_field1.begin(), h_field1.end(), 0);

    thrust::copy(
            d_aos.begin(), d_aos.end(), h_soa);
    ASSERT_EQ_QUIET(make_tuple(
            7, 13), h_soa[0]);
}

TEST(ZipIterator32BitTests, TestZipIteratorCopySoAToAoS)
{
    using namespace thrust;

    const size_t n = 1;

    typedef tuple<int, int> structure;
    typedef host_vector<structure> host_array_of_structures;
    typedef device_vector<structure> device_array_of_structures;

    typedef zip_iterator<tuple<host_vector<int>::iterator, host_vector<int>::iterator> > host_structure_of_arrays;
    typedef zip_iterator<tuple<device_vector<int>::iterator, device_vector<int>::iterator> > device_structure_of_arrays;

    host_vector<int> h_field0(
            n, 7), h_field1(
            n, 13);
    device_vector<int> d_field0(
            n, 7), d_field1(
            n, 13);

    host_structure_of_arrays h_soa = make_zip_iterator(
            make_tuple(
                    h_field0.begin(), h_field1.begin()));
    device_structure_of_arrays d_soa = make_zip_iterator(
            make_tuple(
                    d_field0.begin(), d_field1.begin()));

    host_array_of_structures h_aos(n);
    device_array_of_structures d_aos(n);

    // host to host
    thrust::fill(
            h_aos.begin(), h_aos.end(), make_tuple(
                    0, 0));

    thrust::copy(
            h_soa, h_soa + n, h_aos.begin());
    ASSERT_EQ_QUIET(7, get<0>(h_soa[0]));
    ASSERT_EQ_QUIET(13, get<1>(h_soa[0]));

    // host to device
    thrust::fill(
            d_aos.begin(), d_aos.end(), make_tuple(
                    0, 0));

    thrust::copy(
            h_soa, h_soa + n, d_aos.begin());
    ASSERT_EQ_QUIET(7, get<0>(d_soa[0]));
    ASSERT_EQ_QUIET(13, get<1>(d_soa[0]));

    // device to device
    thrust::fill(
            d_aos.begin(), d_aos.end(), make_tuple(
                    0, 0));

    thrust::copy(
            d_soa, d_soa + n, d_aos.begin());
    ASSERT_EQ_QUIET(7, get<0>(d_soa[0]));
    ASSERT_EQ_QUIET(13, get<1>(d_soa[0]));

    // device to host
    thrust::fill(
            h_aos.begin(), h_aos.end(), make_tuple(
                    0, 0));

    thrust::copy(
            d_soa, d_soa + n, h_aos.begin());
    ASSERT_EQ_QUIET(7, get<0>(h_soa[0]));
    ASSERT_EQ_QUIET(13, get<1>(h_soa[0]));
}

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
