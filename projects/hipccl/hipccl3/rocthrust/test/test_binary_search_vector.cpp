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

#include <thrust/binary_search.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/retag.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include "test_header.hpp"

TESTS_DEFINE(BinarySearchVectorTestsInKernel, NumericalTestsParams);

struct custom_less
{
    template<class T>
    __device__ inline
    bool operator()(T a, T b)
    {
        return a < b;
    }
};

template<class T>
__global__
void lower_bound_kernel(size_t n,
                        T* input,
                        ptrdiff_t* output)
{
    thrust::counting_iterator<T> values(0);
    thrust::lower_bound(
        thrust::device, input, input + n, values, values + 10, output, custom_less()
    );
}

TYPED_TEST(BinarySearchVectorTestsInKernel, TestLowerBound)
{
    using T = typename TestFixture::input_type;

    thrust::device_vector<T> d_input(5);
    d_input[0] = 0;
    d_input[1] = 2;
    d_input[2] = 5;
    d_input[3] = 7;
    d_input[4] = 8;

    thrust::device_vector<ptrdiff_t> d_output(10);

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(lower_bound_kernel),
        dim3(1), dim3(1), 0, 0,
        size_t(d_input.size()),
        thrust::raw_pointer_cast(d_input.data()),
        thrust::raw_pointer_cast(d_output.data())
    );

    thrust::host_vector<ptrdiff_t> output = d_output;
    ASSERT_EQ(output[0], 0);
    ASSERT_EQ(output[1], 1);
    ASSERT_EQ(output[2], 1);
    ASSERT_EQ(output[3], 2);
    ASSERT_EQ(output[4], 2);
    ASSERT_EQ(output[5], 2);
    ASSERT_EQ(output[6], 3);
    ASSERT_EQ(output[7], 3);
    ASSERT_EQ(output[8], 4);
    ASSERT_EQ(output[9], 5);
}

template<class T>
__global__
void upper_bound_kernel(size_t n,
                        T* input,
                        ptrdiff_t* output)
{
    thrust::counting_iterator<T> values(0);
    thrust::upper_bound(thrust::device, input, input + n, values, values + 10, output);
}

TYPED_TEST(BinarySearchVectorTestsInKernel, TestUpperBound)
{
    using T = typename TestFixture::input_type;

    thrust::device_vector<T> d_input(5);
    d_input[0] = 0;
    d_input[1] = 2;
    d_input[2] = 5;
    d_input[3] = 7;
    d_input[4] = 8;

    thrust::device_vector<ptrdiff_t> d_output(10);

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(upper_bound_kernel),
        dim3(1), dim3(1), 0, 0,
        size_t(d_input.size()),
        thrust::raw_pointer_cast(d_input.data()),
        thrust::raw_pointer_cast(d_output.data())
    );

    thrust::host_vector<ptrdiff_t> output = d_output;
    ASSERT_EQ(output[0], 1);
    ASSERT_EQ(output[1], 1);
    ASSERT_EQ(output[2], 2);
    ASSERT_EQ(output[3], 2);
    ASSERT_EQ(output[4], 2);
    ASSERT_EQ(output[5], 3);
    ASSERT_EQ(output[6], 3);
    ASSERT_EQ(output[7], 4);
    ASSERT_EQ(output[8], 5);
    ASSERT_EQ(output[9], 5);
}

template<class T>
__global__
void binary_search_kernel(size_t n,
                          T* input,
                          bool* output)
{
    thrust::counting_iterator<T> values(0);
    thrust::binary_search(thrust::device, input, input + n, values, values + 10, output);
}

TYPED_TEST(BinarySearchVectorTestsInKernel, TestBinarySearch)
{
    using T = typename TestFixture::input_type;

    thrust::device_vector<T> d_input(5);
    d_input[0] = 0;
    d_input[1] = 2;
    d_input[2] = 5;
    d_input[3] = 7;
    d_input[4] = 8;

    thrust::device_vector<bool> d_output(10);

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(binary_search_kernel),
        dim3(1), dim3(1), 0, 0,
        size_t(d_input.size()),
        thrust::raw_pointer_cast(d_input.data()),
        thrust::raw_pointer_cast(d_output.data())
    );

    thrust::host_vector<bool> output = d_output;
    ASSERT_EQ(output[0], true);
    ASSERT_EQ(output[1], false);
    ASSERT_EQ(output[2], true);
    ASSERT_EQ(output[3], false);
    ASSERT_EQ(output[4], false);
    ASSERT_EQ(output[5], true);
    ASSERT_EQ(output[6], false);
    ASSERT_EQ(output[7], true);
    ASSERT_EQ(output[8], true);
    ASSERT_EQ(output[9], false);
}

TESTS_DEFINE(BinarySearchVectorTests, FullTestsParams);
TESTS_DEFINE(BinarySearchVectorIntegerTests, SignedIntegerTestsParams);

// convert xxx_vector<T1> to xxx_vector<T2>
template <class ExampleVector, typename NewType>
struct vector_like
{
    typedef typename ExampleVector::allocator_type          alloc;
    typedef typename alloc::template rebind<NewType>::other new_alloc;
    typedef thrust::detail::vector_base<NewType, new_alloc> type;
};

TYPED_TEST(BinarySearchVectorTests, TestScalarLowerBoundSimple)
{
    using Vector = typename TestFixture::input_type;
    Vector vec(5);

    vec[0] = 0;
    vec[1] = 2;
    vec[2] = 5;
    vec[3] = 7;
    vec[4] = 8;

    Vector input(10);
    thrust::sequence(input.begin(), input.end());

    typedef typename vector_like<Vector, int>::type IntVector;

    // test with integral output type
    IntVector integral_output(10);
    thrust::lower_bound(
        vec.begin(), vec.end(), input.begin(), input.end(), integral_output.begin());

    typename IntVector::iterator output_end = thrust::lower_bound(
        vec.begin(), vec.end(), input.begin(), input.end(), integral_output.begin());

    ASSERT_EQ((output_end - integral_output.begin()), 10);

    ASSERT_EQ(integral_output[0], 0);
    ASSERT_EQ(integral_output[1], 1);
    ASSERT_EQ(integral_output[2], 1);
    ASSERT_EQ(integral_output[3], 2);
    ASSERT_EQ(integral_output[4], 2);
    ASSERT_EQ(integral_output[5], 2);
    ASSERT_EQ(integral_output[6], 3);
    ASSERT_EQ(integral_output[7], 3);
    ASSERT_EQ(integral_output[8], 4);
    ASSERT_EQ(integral_output[9], 5);

    //// test with iterator output type
    //typedef typename vector_like<Vector, typename Vector::iterator>::type IteratorVector;
    //IteratorVector iterator_output(10);
    //thrust::lower_bound(vec.begin(), vec.end(), input.begin(), input.end(), iterator_output.begin());

    //ASSERT_EQ(iterator_output[0] - vec.begin(), 0);
    //ASSERT_EQ(iterator_output[1] - vec.begin(), 1);
    //ASSERT_EQ(iterator_output[2] - vec.begin(), 1);
    //ASSERT_EQ(iterator_output[3] - vec.begin(), 2);
    //ASSERT_EQ(iterator_output[4] - vec.begin(), 2);
    //ASSERT_EQ(iterator_output[5] - vec.begin(), 2);
    //ASSERT_EQ(iterator_output[6] - vec.begin(), 3);
    //ASSERT_EQ(iterator_output[7] - vec.begin(), 3);
    //ASSERT_EQ(iterator_output[8] - vec.begin(), 4);
    //ASSERT_EQ(iterator_output[9] - vec.begin(), 5);
}

template <typename ForwardIterator, typename InputIterator, typename OutputIterator>
OutputIterator lower_bound(my_system& system,
                           ForwardIterator,
                           ForwardIterator,
                           InputIterator,
                           InputIterator,
                           OutputIterator output)
{
    system.validate_dispatch();
    return output;
}

TEST(BinarySearchVectorTests, TestVectorLowerBoundDispatchExplicit)
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::lower_bound(sys, vec.begin(), vec.end(), vec.begin(), vec.end(), vec.begin());

    ASSERT_EQ(true, sys.is_valid());
}

template <typename ForwardIterator, typename InputIterator, typename OutputIterator>
OutputIterator lower_bound(
    my_tag, ForwardIterator, ForwardIterator, InputIterator, InputIterator, OutputIterator output)
{
    *output = 13;
    return output;
}

TEST(BinarySearchVectorTests, TestVectorLowerBoundDispatchImplicit)
{
    thrust::device_vector<int> vec(1);

    thrust::lower_bound(thrust::retag<my_tag>(vec.begin()),
                        thrust::retag<my_tag>(vec.end()),
                        thrust::retag<my_tag>(vec.begin()),
                        thrust::retag<my_tag>(vec.end()),
                        thrust::retag<my_tag>(vec.begin()));

    ASSERT_EQ(13, vec.front());
}

TYPED_TEST(BinarySearchVectorTests, TestVectorUpperBoundSimple)
{
    using Vector = typename TestFixture::input_type;
    Vector vec(5);

    vec[0] = 0;
    vec[1] = 2;
    vec[2] = 5;
    vec[3] = 7;
    vec[4] = 8;

    Vector input(10);
    thrust::sequence(input.begin(), input.end());

    typedef typename vector_like<Vector, int>::type IntVector;

    // test with integral output type
    IntVector                    integral_output(10);
    typename IntVector::iterator output_end = thrust::upper_bound(
        vec.begin(), vec.end(), input.begin(), input.end(), integral_output.begin());

    ASSERT_EQ((output_end - integral_output.begin()), 10);

    ASSERT_EQ(integral_output[0], 1);
    ASSERT_EQ(integral_output[1], 1);
    ASSERT_EQ(integral_output[2], 2);
    ASSERT_EQ(integral_output[3], 2);
    ASSERT_EQ(integral_output[4], 2);
    ASSERT_EQ(integral_output[5], 3);
    ASSERT_EQ(integral_output[6], 3);
    ASSERT_EQ(integral_output[7], 4);
    ASSERT_EQ(integral_output[8], 5);
    ASSERT_EQ(integral_output[9], 5);

    //// test with iterator output type
    //typedef typename vector_like<Vector, typename Vector::iterator>::type IteratorVector;
    //IteratorVector iterator_output(10);
    //thrust::upper_bound(vec.begin(), vec.end(), input.begin(), input.end(), iterator_output.begin());

    //ASSERT_EQ(iterator_output[0] - vec.begin(), 1);
    //ASSERT_EQ(iterator_output[1] - vec.begin(), 1);
    //ASSERT_EQ(iterator_output[2] - vec.begin(), 2);
    //ASSERT_EQ(iterator_output[3] - vec.begin(), 2);
    //ASSERT_EQ(iterator_output[4] - vec.begin(), 2);
    //ASSERT_EQ(iterator_output[5] - vec.begin(), 3);
    //ASSERT_EQ(iterator_output[6] - vec.begin(), 3);
    //ASSERT_EQ(iterator_output[7] - vec.begin(), 4);
    //ASSERT_EQ(iterator_output[8] - vec.begin(), 5);
    //ASSERT_EQ(iterator_output[9] - vec.begin(), 5);
}

template <typename ForwardIterator, typename InputIterator, typename OutputIterator>
OutputIterator upper_bound(my_system& system,
                           ForwardIterator,
                           ForwardIterator,
                           InputIterator,
                           InputIterator,
                           OutputIterator output)
{
    system.validate_dispatch();
    return output;
}

TEST(BinarySearchVectorTests, TestVectorUpperBoundDispatchExplicit)
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::upper_bound(sys, vec.begin(), vec.end(), vec.begin(), vec.end(), vec.begin());

    ASSERT_EQ(true, sys.is_valid());
}

template <typename ForwardIterator, typename InputIterator, typename OutputIterator>
OutputIterator upper_bound(
    my_tag, ForwardIterator, ForwardIterator, InputIterator, InputIterator, OutputIterator output)
{
    *output = 13;
    return output;
}

TEST(BinarySearchVectorTests, TestVectorUpperBoundDispatchImplicit)
{
    thrust::device_vector<int> vec(1);

    thrust::upper_bound(thrust::retag<my_tag>(vec.begin()),
                        thrust::retag<my_tag>(vec.end()),
                        thrust::retag<my_tag>(vec.begin()),
                        thrust::retag<my_tag>(vec.end()),
                        thrust::retag<my_tag>(vec.begin()));

    ASSERT_EQ(13, vec.front());
}

TYPED_TEST(BinarySearchVectorTests, TestVectorBinarySearchSimple)
{
    using Vector = typename TestFixture::input_type;
    Vector vec(5);

    vec[0] = 0;
    vec[1] = 2;
    vec[2] = 5;
    vec[3] = 7;
    vec[4] = 8;

    Vector input(10);
    thrust::sequence(input.begin(), input.end());

    typedef typename vector_like<Vector, bool>::type BoolVector;
    typedef typename vector_like<Vector, int>::type  IntVector;

    // test with boolean output type
    BoolVector                    bool_output(10);
    typename BoolVector::iterator bool_output_end = thrust::binary_search(
        vec.begin(), vec.end(), input.begin(), input.end(), bool_output.begin());

    ASSERT_EQ((bool_output_end - bool_output.begin()), 10);

    ASSERT_EQ(bool_output[0], true);
    ASSERT_EQ(bool_output[1], false);
    ASSERT_EQ(bool_output[2], true);
    ASSERT_EQ(bool_output[3], false);
    ASSERT_EQ(bool_output[4], false);
    ASSERT_EQ(bool_output[5], true);
    ASSERT_EQ(bool_output[6], false);
    ASSERT_EQ(bool_output[7], true);
    ASSERT_EQ(bool_output[8], true);
    ASSERT_EQ(bool_output[9], false);

    // test with integral output type
    IntVector                    integral_output(10, 2);
    typename IntVector::iterator int_output_end = thrust::binary_search(
        vec.begin(), vec.end(), input.begin(), input.end(), integral_output.begin());

    ASSERT_EQ((int_output_end - integral_output.begin()), 10);

    ASSERT_EQ(integral_output[0], 1);
    ASSERT_EQ(integral_output[1], 0);
    ASSERT_EQ(integral_output[2], 1);
    ASSERT_EQ(integral_output[3], 0);
    ASSERT_EQ(integral_output[4], 0);
    ASSERT_EQ(integral_output[5], 1);
    ASSERT_EQ(integral_output[6], 0);
    ASSERT_EQ(integral_output[7], 1);
    ASSERT_EQ(integral_output[8], 1);
    ASSERT_EQ(integral_output[9], 0);
}

template <typename ForwardIterator, typename InputIterator, typename OutputIterator>
OutputIterator binary_search(my_system& system,
                             ForwardIterator,
                             ForwardIterator,
                             InputIterator,
                             InputIterator,
                             OutputIterator output)
{
    system.validate_dispatch();
    return output;
}

TEST(BinarySearchVectorTests, TestVectorBinarySearchDispatchExplicit)
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::binary_search(sys, vec.begin(), vec.end(), vec.begin(), vec.end(), vec.begin());

    ASSERT_EQ(true, sys.is_valid());
}

template <typename ForwardIterator, typename InputIterator, typename OutputIterator>
OutputIterator binary_search(
    my_tag, ForwardIterator, ForwardIterator, InputIterator, InputIterator, OutputIterator output)
{
    *output = 13;
    return output;
}

TEST(BinarySearchVectorTests, TestVectorBinarySearchDispatchImplicit)
{
    thrust::device_vector<int> vec(1);

    thrust::binary_search(thrust::retag<my_tag>(vec.begin()),
                          thrust::retag<my_tag>(vec.end()),
                          thrust::retag<my_tag>(vec.begin()),
                          thrust::retag<my_tag>(vec.end()),
                          thrust::retag<my_tag>(vec.begin()));

    ASSERT_EQ(13, vec.front());
}

TYPED_TEST(BinarySearchVectorIntegerTests, TestVectorLowerBound)
{
    using T = typename TestFixture::input_type;
    for(auto size : get_sizes())
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        thrust::host_vector<T> h_vec = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
        thrust::sort(h_vec.begin(), h_vec.end());
        thrust::device_vector<T> d_vec = h_vec;

        thrust::host_vector<T> h_input = get_random_data<T>(
            2 * size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
        thrust::device_vector<T> d_input = h_input;

        thrust::host_vector<int>   h_output(2 * size);
        thrust::device_vector<int> d_output(2 * size);

        thrust::lower_bound(
            h_vec.begin(), h_vec.end(), h_input.begin(), h_input.end(), h_output.begin());
        thrust::lower_bound(
            d_vec.begin(), d_vec.end(), d_input.begin(), d_input.end(), d_output.begin());

        ASSERT_EQ(h_output, d_output);
    }
}

TYPED_TEST(BinarySearchVectorIntegerTests, TestVectorUpperBound)
{
    using T = typename TestFixture::input_type;
    for(auto size : get_sizes())
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        thrust::host_vector<T> h_vec = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
        thrust::sort(h_vec.begin(), h_vec.end());
        thrust::device_vector<T> d_vec = h_vec;

        thrust::host_vector<T> h_input = get_random_data<T>(
            2 * size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
        thrust::device_vector<T> d_input = h_input;

        thrust::host_vector<int>   h_output(2 * size);
        thrust::device_vector<int> d_output(2 * size);

        thrust::upper_bound(
            h_vec.begin(), h_vec.end(), h_input.begin(), h_input.end(), h_output.begin());
        thrust::upper_bound(
            d_vec.begin(), d_vec.end(), d_input.begin(), d_input.end(), d_output.begin());

        ASSERT_EQ(h_output, d_output);
    }
}

TYPED_TEST(BinarySearchVectorIntegerTests, TestVectorBinarySearch)
{
    using T = typename TestFixture::input_type;
    for(auto size : get_sizes())
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        thrust::host_vector<T> h_vec = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
        thrust::sort(h_vec.begin(), h_vec.end());
        thrust::device_vector<T> d_vec = h_vec;

        thrust::host_vector<T> h_input = get_random_data<T>(
            2 * size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
        thrust::device_vector<T> d_input = h_input;

        thrust::host_vector<int>   h_output(2 * size);
        thrust::device_vector<int> d_output(2 * size);

        thrust::binary_search(
            h_vec.begin(), h_vec.end(), h_input.begin(), h_input.end(), h_output.begin());
        thrust::binary_search(
            d_vec.begin(), d_vec.end(), d_input.begin(), d_input.end(), d_output.begin());

        ASSERT_EQ(h_output, d_output);
    }
}

TYPED_TEST(BinarySearchVectorIntegerTests, TestVectorLowerBoundDiscardIterator)
{
    using T = typename TestFixture::input_type;
    for(auto size : get_sizes())
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        thrust::host_vector<T> h_vec = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
        thrust::sort(h_vec.begin(), h_vec.end());
        thrust::device_vector<T> d_vec = h_vec;

        thrust::host_vector<T> h_input = get_random_data<T>(
            2 * size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
        thrust::device_vector<T> d_input = h_input;

        thrust::discard_iterator<> h_result = thrust::lower_bound(h_vec.begin(),
                                                                  h_vec.end(),
                                                                  h_input.begin(),
                                                                  h_input.end(),
                                                                  thrust::make_discard_iterator());
        thrust::discard_iterator<> d_result = thrust::lower_bound(d_vec.begin(),
                                                                  d_vec.end(),
                                                                  d_input.begin(),
                                                                  d_input.end(),
                                                                  thrust::make_discard_iterator());

        thrust::discard_iterator<> reference(2 * size);

        ASSERT_EQ_QUIET(reference, h_result);
        ASSERT_EQ_QUIET(reference, d_result);
    }
}

TYPED_TEST(BinarySearchVectorIntegerTests, TestVectorUpperBoundDiscardIterator)
{
    using T = typename TestFixture::input_type;
    for(auto size : get_sizes())
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        thrust::host_vector<T> h_vec = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
        thrust::sort(h_vec.begin(), h_vec.end());
        thrust::device_vector<T> d_vec = h_vec;

        thrust::host_vector<T> h_input = get_random_data<T>(
            2 * size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
        thrust::device_vector<T> d_input = h_input;

        thrust::discard_iterator<> h_result = thrust::upper_bound(h_vec.begin(),
                                                                  h_vec.end(),
                                                                  h_input.begin(),
                                                                  h_input.end(),
                                                                  thrust::make_discard_iterator());
        thrust::discard_iterator<> d_result = thrust::upper_bound(d_vec.begin(),
                                                                  d_vec.end(),
                                                                  d_input.begin(),
                                                                  d_input.end(),
                                                                  thrust::make_discard_iterator());

        thrust::discard_iterator<> reference(2 * size);

        ASSERT_EQ_QUIET(reference, h_result);
        ASSERT_EQ_QUIET(reference, d_result);
    }
}

TYPED_TEST(BinarySearchVectorIntegerTests, TestVectorBinarySearchDiscardIterator)
{
    using T = typename TestFixture::input_type;
    for(auto size : get_sizes())
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        thrust::host_vector<T> h_vec = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
        thrust::sort(h_vec.begin(), h_vec.end());
        thrust::device_vector<T> d_vec = h_vec;

        thrust::host_vector<T> h_input = get_random_data<T>(
            2 * size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
        thrust::device_vector<T> d_input = h_input;

        thrust::discard_iterator<> h_result
            = thrust::binary_search(h_vec.begin(),
                                    h_vec.end(),
                                    h_input.begin(),
                                    h_input.end(),
                                    thrust::make_discard_iterator());
        thrust::discard_iterator<> d_result
            = thrust::binary_search(d_vec.begin(),
                                    d_vec.end(),
                                    d_input.begin(),
                                    d_input.end(),
                                    thrust::make_discard_iterator());

        thrust::discard_iterator<> reference(2 * size);

        ASSERT_EQ_QUIET(reference, h_result);
        ASSERT_EQ_QUIET(reference, d_result);
    }
}
