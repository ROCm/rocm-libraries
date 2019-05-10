/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Modifications Copyright© 2019 Advanced Micro Devices, Inc. All rights reserved.
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

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/memory.h>
#include <thrust/sequence.h>

#include "test_header.hpp"

#include <list>
#include <iterator>

TESTS_DEFINE(CopyTests, FullTestsParams)
TESTS_DEFINE(CopyIntegerTests, IntegerTestsParams)

TEST(HipThrustCopy, HostToDevice)
{
    const size_t size = 256;
    thrust::device_system_tag dev_tag;
    thrust::host_system_tag host_tag;

    // Malloc on host
    auto h_ptr = thrust::malloc<int>(host_tag, sizeof(int) * size);
    // Malloc on device
    auto d_ptr = thrust::malloc<int>(dev_tag, sizeof(int) * size);

    for(size_t i = 0; i < size; i++)
    {
        *h_ptr = i;
    }

    // Compiles thanks to a temporary fix in
    // thrust/system/detail/generic/for_each.h
    thrust::copy(h_ptr, h_ptr + 256, d_ptr);

    // Free
    thrust::free(host_tag, h_ptr);
    thrust::free(dev_tag, d_ptr);
}

TEST(HipThrustCopy, DeviceToDevice)
{
    const size_t size = 256;
    thrust::device_system_tag dev_tag;

    // Malloc on device
    auto d_ptr1 = thrust::malloc<int>(dev_tag, sizeof(int) * size);
    auto d_ptr2 = thrust::malloc<int>(dev_tag, sizeof(int) * size);

    // Zero d_ptr1 memory
    HIP_CHECK(hipMemset(thrust::raw_pointer_cast(d_ptr1), 0, sizeof(int) * size));
    HIP_CHECK(hipMemset(thrust::raw_pointer_cast(d_ptr2), 0xdead, sizeof(int) * size));

    // Copy device->device
    thrust::copy(d_ptr1, d_ptr1 + 256, d_ptr2);

    std::vector<int> output(size);
    HIP_CHECK(
        hipMemcpy(
            output.data(), thrust::raw_pointer_cast(d_ptr2),
            size * sizeof(int),
            hipMemcpyDeviceToHost
        )
    );

    for(size_t i = 0; i < size; i++)
    {
        ASSERT_EQ(output[i], int(0)) << "where index = " << i;
    }

    // Free
    thrust::free(dev_tag, d_ptr1);
    thrust::free(dev_tag, d_ptr2);
}

TEST(CopyTests, TestCopyFromConstIterator)
{
    using T = int;

    std::vector<T> v(5);
    v[0] = T(0); v[1] = T(1); v[2] = T(2); v[3] = T(3); v[4] = T(4);

    std::vector<int>::const_iterator begin = v.begin();
    std::vector<int>::const_iterator end = v.end();

    // copy to host_vector
    thrust::host_vector<T> h(5, (T) 10);
    thrust::host_vector<T>::iterator h_result = thrust::copy(begin, end, h.begin());
    ASSERT_EQ(h[0], T(0));
    ASSERT_EQ(h[1], T(1));
    ASSERT_EQ(h[2], T(2));
    ASSERT_EQ(h[3], T(3));
    ASSERT_EQ(h[4], T(4));
    ASSERT_EQ_QUIET(h_result, h.end());

    // copy to device_vector
    thrust::device_vector<T> d(5, (T) 10);
    thrust::device_vector<T>::iterator d_result = thrust::copy(begin, end, d.begin());
    ASSERT_EQ(d[0], T(0));
    ASSERT_EQ(d[1], T(1));
    ASSERT_EQ(d[2], T(2));
    ASSERT_EQ(d[3], T(3));
    ASSERT_EQ(d[4], T(4));
    ASSERT_EQ_QUIET(d_result, d.end());
}

TEST(CopyTests, TestCopyToDiscardIterator)
{
    using T = int;

    thrust::host_vector<T> h_input(5,1);
    thrust::device_vector<T> d_input = h_input;

    thrust::discard_iterator<> reference(5);

    // copy from host_vector
    thrust::discard_iterator<> h_result =
        thrust::copy(h_input.begin(), h_input.end(), thrust::make_discard_iterator());

    // copy from device_vector
    thrust::discard_iterator<> d_result =
        thrust::copy(d_input.begin(), d_input.end(), thrust::make_discard_iterator());

    ASSERT_EQ_QUIET(reference, h_result);
    ASSERT_EQ_QUIET(reference, d_result);
}

TEST(CopyTests, TestCopyToDiscardIteratorZipped)
{
    using T = int;

    thrust::host_vector<T> h_input(5,1);
    thrust::device_vector<T> d_input = h_input;

    thrust::host_vector<T>     h_output(5);
    thrust::device_vector<T>   d_output(5);
    thrust::discard_iterator<> reference(5);

    using  Tuple1 = thrust::tuple<thrust::discard_iterator<>,thrust::host_vector<T>::iterator>;
    using  Tuple2 = thrust::tuple<thrust::discard_iterator<>,thrust::device_vector<T>::iterator>;

    using ZipIterator1 = thrust::zip_iterator<Tuple1>;
    using ZipIterator2 = thrust::zip_iterator<Tuple2>;

    // copy from host_vector
    ZipIterator1 h_result =
        thrust::copy(thrust::make_zip_iterator(thrust::make_tuple(h_input.begin(),                 h_input.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(h_input.end(),                   h_input.end())),
                     thrust::make_zip_iterator(thrust::make_tuple(thrust::make_discard_iterator(), h_output.begin())));

    // copy from device_vector
    ZipIterator2 d_result =
        thrust::copy(thrust::make_zip_iterator(thrust::make_tuple(d_input.begin(),                 d_input.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(d_input.end(),                   d_input.end())),
                     thrust::make_zip_iterator(thrust::make_tuple(thrust::make_discard_iterator(), d_output.begin())));

    ASSERT_EQ(h_output, h_input);
    ASSERT_EQ(d_output, d_input);
    ASSERT_EQ_QUIET(reference, thrust::get<0>(h_result.get_iterator_tuple()));
    ASSERT_EQ_QUIET(reference, thrust::get<0>(d_result.get_iterator_tuple()));
}

TYPED_TEST(CopyTests, TestCopyMatchingTypes)
{
    using Vector = typename TestFixture::input_type;
    using T = typename Vector::value_type;

    Vector v(5);
    v[0] = T(0); v[1] = T(1); v[2] = T(2); v[3] = T(3); v[4] = T(4);

    // copy to host_vector
    thrust::host_vector<T> h(5, (T) 10);
    typename thrust::host_vector<T>::iterator h_result = thrust::copy(v.begin(), v.end(), h.begin());
    ASSERT_EQ(h[0], T(0));
    ASSERT_EQ(h[1], T(1));
    ASSERT_EQ(h[2], T(2));
    ASSERT_EQ(h[3], T(3));
    ASSERT_EQ(h[4], T(4));
    ASSERT_EQ_QUIET(h_result, h.end());

    // copy to device_vector
    thrust::device_vector<T> d(5, (T) 10);
    typename thrust::device_vector<T>::iterator d_result = thrust::copy(v.begin(), v.end(), d.begin());
    ASSERT_EQ(d[0], T(0));
    ASSERT_EQ(d[1], T(1));
    ASSERT_EQ(d[2], T(2));
    ASSERT_EQ(d[3], T(3));
    ASSERT_EQ(d[4], T(4));
    ASSERT_EQ_QUIET(d_result, d.end());
}

TYPED_TEST(CopyTests, TestCopyMixedTypes)
{
    using Vector = typename TestFixture::input_type;
    using T = typename Vector::value_type;

    Vector v(5);
    v[0] = T(0); v[1] = T(1); v[2] = T(2); v[3] = T(3); v[4] = T(4);

    // copy to host_vector with different type
    thrust::host_vector<float> h(5, (float) 10);
    typename thrust::host_vector<float>::iterator h_result = thrust::copy(v.begin(), v.end(), h.begin());

    ASSERT_EQ(h[0], T(0));
    ASSERT_EQ(h[1], T(1));
    ASSERT_EQ(h[2], T(2));
    ASSERT_EQ(h[3], T(3));
    ASSERT_EQ(h[4], T(4));
    ASSERT_EQ_QUIET(h_result, h.end());

    // copy to device_vector with different type
    thrust::device_vector<float> d(5, (float) 10);
    typename thrust::device_vector<float>::iterator d_result = thrust::copy(v.begin(), v.end(), d.begin());
    ASSERT_EQ(d[0], T(0));
    ASSERT_EQ(d[1], T(1));
    ASSERT_EQ(d[2], T(2));
    ASSERT_EQ(d[3], T(3));
    ASSERT_EQ(d[4], T(4));
    ASSERT_EQ_QUIET(d_result, d.end());
}

TEST(CopyTests, TestCopyVectorBool)
{
    std::vector<bool> v(3);
    v[0] = true; v[1] = false; v[2] = true;

    thrust::host_vector<bool> h(3);
    thrust::device_vector<bool> d(3);

    thrust::copy(v.begin(), v.end(), h.begin());
    thrust::copy(v.begin(), v.end(), d.begin());

    ASSERT_EQ(h[0], true);
    ASSERT_EQ(h[1], false);
    ASSERT_EQ(h[2], true);

    ASSERT_EQ(d[0], true);
    ASSERT_EQ(d[1], false);
    ASSERT_EQ(d[2], true);
}

TYPED_TEST(CopyTests, TestCopyListTo)
{
    using Vector = typename TestFixture::input_type;
    using T = typename Vector::value_type;

    // copy from list to Vector
    std::list<T> l;
    l.push_back(0);
    l.push_back(1);
    l.push_back(2);
    l.push_back(3);
    l.push_back(4);

    Vector v(l.size());

    typename Vector::iterator v_result = thrust::copy(l.begin(), l.end(), v.begin());

    ASSERT_EQ(v[0], 0);
    ASSERT_EQ(v[1], 1);
    ASSERT_EQ(v[2], 2);
    ASSERT_EQ(v[3], 3);
    ASSERT_EQ(v[4], 4);
    ASSERT_EQ_QUIET(v_result, v.end());

    l.clear();

    thrust::copy(v.begin(), v.end(), std::back_insert_iterator< std::list<T> >(l));

    ASSERT_EQ(l.size(), 5);

    typename std::list<T>::const_iterator iter = l.begin();
    ASSERT_EQ(*iter, 0);  iter++;
    ASSERT_EQ(*iter, 1);  iter++;
    ASSERT_EQ(*iter, 2);  iter++;
    ASSERT_EQ(*iter, 3);  iter++;
    ASSERT_EQ(*iter, 4);  iter++;
}

template<typename T>
struct is_even
{
    __host__ __device__
    bool operator()(T x)
    {
        return (static_cast<unsigned int>(x) & 1) == 0;
    }
};

template<typename T>
struct is_true
{
    __host__ __device__
    bool operator()(T x)
    {
        return x ? true : false;
    }
};

template<typename T>
struct mod_3
{
    __host__ __device__
    unsigned int operator()(T x)
    {
        return static_cast<unsigned int>(x) % 3;
    }
};

TYPED_TEST(CopyTests, TestCopyIfSimple)
{
    using Vector = typename TestFixture::input_type;
    using T = typename Vector::value_type;

    Vector v(5);
    v[0] = T(0); v[1] = T(1); v[2] = T(2); v[3] = T(3); v[4] = T(4);

    Vector dest(3);

    typename Vector::iterator dest_end = thrust::copy_if(v.begin(), v.end(), dest.begin(), is_even<T>());

    ASSERT_EQ(T(0), dest[0]);
    ASSERT_EQ(T(2), dest[1]);
    ASSERT_EQ(T(4), dest[2]);
    ASSERT_EQ_QUIET(dest.end(), dest_end);
}

TYPED_TEST(CopyIntegerTests, TestCopyIf)
{
    using T = typename TestFixture::input_type;

    for (auto size : get_sizes())
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        thrust::host_vector<T> h_data =  get_random_data<T>(size,
                                                            std::numeric_limits<T>::min(),
                                                            std::numeric_limits<T>::max());
        thrust::device_vector<T> d_data = h_data;

        typename thrust::host_vector<T>::iterator   h_new_end;
        typename thrust::device_vector<T>::iterator d_new_end;

        // test with Predicate that returns a bool
        {
            thrust::host_vector<T>   h_result(size);
            thrust::device_vector<T> d_result(size);

            h_new_end = thrust::copy_if(h_data.begin(), h_data.end(), h_result.begin(), is_even<T>());
            d_new_end = thrust::copy_if(d_data.begin(), d_data.end(), d_result.begin(), is_even<T>());

            h_result.resize(h_new_end - h_result.begin());
            d_result.resize(d_new_end - d_result.begin());

            ASSERT_EQ(h_result, d_result);
        }

        // test with Predicate that returns a non-bool
        {
            thrust::host_vector<T>   h_result(size);
            thrust::device_vector<T> d_result(size);

            h_new_end = thrust::copy_if(h_data.begin(), h_data.end(), h_result.begin(), mod_3<T>());
            d_new_end = thrust::copy_if(d_data.begin(), d_data.end(), d_result.begin(), mod_3<T>());

            h_result.resize(h_new_end - h_result.begin());
            d_result.resize(d_new_end - d_result.begin());

            ASSERT_EQ(h_result, d_result);
        }
    }
}

TYPED_TEST(CopyIntegerTests, TestCopyIfStencil)
{
    using T = typename TestFixture::input_type;

    for (auto size : get_sizes())
    {
        thrust::host_vector<T>   h_data(size); thrust::sequence(h_data.begin(), h_data.end());
        thrust::device_vector<T> d_data(size); thrust::sequence(d_data.begin(), d_data.end());

        thrust::host_vector<T>   h_stencil = get_random_data<T>(size,
                                                                std::numeric_limits<T>::min(),
                                                                std::numeric_limits<T>::max());
        thrust::device_vector<T> d_stencil = get_random_data<T>(size,
                                                                std::numeric_limits<T>::min(),
                                                                std::numeric_limits<T>::max());

        thrust::host_vector<T>   h_result(size);
        thrust::device_vector<T> d_result(size);

        typename thrust::host_vector<T>::iterator   h_new_end;
        typename thrust::device_vector<T>::iterator d_new_end;

        // test with Predicate that returns a bool
        {
            thrust::host_vector<T>   h_result(size);
            thrust::device_vector<T> d_result(size);

            h_new_end = thrust::copy_if(h_data.begin(), h_data.end(), h_result.begin(), is_even<T>());
            d_new_end = thrust::copy_if(d_data.begin(), d_data.end(), d_result.begin(), is_even<T>());

            h_result.resize(h_new_end - h_result.begin());
            d_result.resize(d_new_end - d_result.begin());

            ASSERT_EQ(h_result, d_result);
        }

        // test with Predicate that returns a non-bool
        {
            thrust::host_vector<T>   h_result(size);
            thrust::device_vector<T> d_result(size);

            h_new_end = thrust::copy_if(h_data.begin(), h_data.end(), h_result.begin(), mod_3<T>());
            d_new_end = thrust::copy_if(d_data.begin(), d_data.end(), d_result.begin(), mod_3<T>());

            h_result.resize(h_new_end - h_result.begin());
            d_result.resize(d_new_end - d_result.begin());

            ASSERT_EQ(h_result, d_result);
        }
    }
}

TYPED_TEST(CopyTests, TestCopyCountingIterator)
{
    using Vector = typename TestFixture::input_type;
    using T = typename Vector::value_type;

    thrust::counting_iterator<T> iter(1);

    Vector vec(4);

    thrust::copy(iter, iter + 4, vec.begin());

    ASSERT_EQ(vec[0], T(1));
    ASSERT_EQ(vec[1], T(2));
    ASSERT_EQ(vec[2], T(3));
    ASSERT_EQ(vec[3], T(4));
}

TYPED_TEST(CopyTests, TestCopyZipIterator)
{
    using Vector = typename TestFixture::input_type;
    using T = typename Vector::value_type;

    Vector v1(3); v1[0] = T(1); v1[1] = T(2); v1[2] = T(3);
    Vector v2(3); v2[0] = T(4); v2[1] = T(5); v2[2] = T(6);
    Vector v3(3, T(0));
    Vector v4(3, T(0));

    thrust::copy(thrust::make_zip_iterator(thrust::make_tuple(v1.begin(),v2.begin())),
                 thrust::make_zip_iterator(thrust::make_tuple(v1.end(),v2.end())),
                 thrust::make_zip_iterator(thrust::make_tuple(v3.begin(),v4.begin())));

    ASSERT_EQ(v1, v3);
    ASSERT_EQ(v2, v4);
}

TYPED_TEST(CopyTests, TestCopyConstantIteratorToZipIterator)
{
    using Vector = typename TestFixture::input_type;
    using T = typename Vector::value_type;

    Vector v1(3,T(0));
    Vector v2(3,T(0));

    thrust::copy(thrust::make_constant_iterator(thrust::tuple<T,T>(4,7)),
                 thrust::make_constant_iterator(thrust::tuple<T,T>(4,7)) + v1.size(),
                 thrust::make_zip_iterator(thrust::make_tuple(v1.begin(),v2.begin())));

    ASSERT_EQ(v1[0], T(4));
    ASSERT_EQ(v1[1], T(4));
    ASSERT_EQ(v1[2], T(4));
    ASSERT_EQ(v2[0], T(7));
    ASSERT_EQ(v2[1], T(7));
    ASSERT_EQ(v2[2], T(7));
}

template<typename InputIterator, typename OutputIterator>
OutputIterator copy(my_system &system, InputIterator, InputIterator, OutputIterator result)
{
    system.validate_dispatch();
    return result;
}

TEST(CopyTests, TestCopyDispatchExplicit)
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::copy(sys,
                 vec.begin(),
                 vec.end(),
                 vec.begin());

    ASSERT_EQ(true, sys.is_valid());
}

template<typename InputIterator, typename OutputIterator>
OutputIterator copy(my_tag, InputIterator, InputIterator, OutputIterator result)
{
    *result = 13;
    return result;
}

TEST(CopyTests, TestCopyDispatchImplicit)
{
    thrust::device_vector<int> vec(1);

    thrust::copy(thrust::retag<my_tag>(vec.begin()),
                 thrust::retag<my_tag>(vec.end()),
                 thrust::retag<my_tag>(vec.begin()));

    ASSERT_EQ(13, vec.front());
}

template<typename InputIterator, typename OutputIterator, typename Predicate>
__host__ __device__
OutputIterator copy_if(my_system &system, InputIterator, InputIterator, OutputIterator result, Predicate)
{
    system.validate_dispatch();
    return result;
}

TEST(CopyTests, TestCopyIfDispatchExplicit)
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::copy_if(sys,
                    vec.begin(),
                    vec.end(),
                    vec.begin(),
                    0);

    ASSERT_EQ(true, sys.is_valid());
}

template<typename InputIterator, typename OutputIterator, typename Predicate>
__host__ __device__
OutputIterator copy_if(my_tag, InputIterator, InputIterator, OutputIterator result, Predicate)
{
    *result = 13;
    return result;
}

TEST(CopyTests, TestCopyIfDispatchImplicit)
{
    thrust::device_vector<int> vec(1);

    thrust::copy_if(thrust::retag<my_tag>(vec.begin()),
                    thrust::retag<my_tag>(vec.end()),
                    thrust::retag<my_tag>(vec.begin()),
                    0);

    ASSERT_EQ(13, vec.front());
}

template<typename InputIterator1, typename InputIterator2, typename OutputIterator, typename Predicate>
__host__ __device__
OutputIterator copy_if(my_system &system, InputIterator1, InputIterator1, InputIterator2, OutputIterator result, Predicate)
{
    system.validate_dispatch();
    return result;
}

TEST(CopyTests, TestCopyIfStencilDispatchExplicit)
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::copy_if(sys,
                    vec.begin(),
                    vec.end(),
                    vec.begin(),
                    vec.begin(),
                    0);

    ASSERT_EQ(true, sys.is_valid());
}

template<typename InputIterator1, typename InputIterator2, typename OutputIterator, typename Predicate>
__host__ __device__
OutputIterator copy_if(my_tag, InputIterator1, InputIterator1, InputIterator2, OutputIterator result, Predicate)
{
    *result = 13;
    return result;
}

TEST(CopyTests, TestCopyIfStencilDispatchImplicit)
{
    thrust::device_vector<int> vec(1);

    thrust::copy_if(thrust::retag<my_tag>(vec.begin()),
                    thrust::retag<my_tag>(vec.end()),
                    thrust::retag<my_tag>(vec.begin()),
                    thrust::retag<my_tag>(vec.begin()),
                    0);

    ASSERT_EQ(13, vec.front());
}
