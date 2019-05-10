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
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/transform_reduce.h>

#include "test_header.hpp"

TESTS_DEFINE(PermutationIteratorTests, FullTestsParams);

TEST(PermutationIteratorTests, UsingHip)
{
    ASSERT_EQ(THRUST_DEVICE_SYSTEM, THRUST_DEVICE_SYSTEM_HIP);
}

TYPED_TEST(PermutationIteratorTests, PermutationIteratorSimple)
{
    using Vector   = typename TestFixture::input_type;
    using T        = typename Vector::value_type;
    using Iterator = typename Vector::iterator;

    Vector source(8);
    Vector indices(4);

    // initialize input
    thrust::sequence(source.begin(), source.end(), 1);

    indices[0] = 3;
    indices[1] = 0;
    indices[2] = 5;
    indices[3] = 7;

    thrust::permutation_iterator<Iterator, Iterator> begin(source.begin(), indices.begin());
    thrust::permutation_iterator<Iterator, Iterator> end(source.begin(), indices.end());

    ASSERT_EQ(end - begin, 4);
    ASSERT_EQ((begin + 4) == end, true);

    ASSERT_EQ((T)*begin, 4);

    begin++;
    end--;

    ASSERT_EQ((T)*begin, 1);
    ASSERT_EQ((T)*end, 8);
    ASSERT_EQ(end - begin, 2);

    end--;

    *begin = 10;
    *end   = 20;

    ASSERT_EQ(source[0], 10);
    ASSERT_EQ(source[1], 2);
    ASSERT_EQ(source[2], 3);
    ASSERT_EQ(source[3], 4);
    ASSERT_EQ(source[4], 5);
    ASSERT_EQ(source[5], 20);
    ASSERT_EQ(source[6], 7);
    ASSERT_EQ(source[7], 8);
}

TYPED_TEST(PermutationIteratorTests, PermutationIteratorGather)
{
    using Vector   = typename TestFixture::input_type;
    using Iterator = typename Vector::iterator;

    Vector source(8);
    Vector indices(4);
    Vector output(4, 10);

    // initialize input
    thrust::sequence(source.begin(), source.end(), 1);

    indices[0] = 3;
    indices[1] = 0;
    indices[2] = 5;
    indices[3] = 7;

    thrust::permutation_iterator<Iterator, Iterator> p_source(source.begin(), indices.begin());

    thrust::copy(p_source, p_source + 4, output.begin());

    ASSERT_EQ(output[0], 4);
    ASSERT_EQ(output[1], 1);
    ASSERT_EQ(output[2], 6);
    ASSERT_EQ(output[3], 8);
}

TYPED_TEST(PermutationIteratorTests, PermutationIteratorScatter)
{
    using Vector   = typename TestFixture::input_type;
    using Iterator = typename Vector::iterator;

    Vector source(4, 10);
    Vector indices(4);
    Vector output(8);

    // initialize output
    thrust::sequence(output.begin(), output.end(), 1);

    indices[0] = 3;
    indices[1] = 0;
    indices[2] = 5;
    indices[3] = 7;

    // construct transform_iterator
    thrust::permutation_iterator<Iterator, Iterator> p_output(output.begin(), indices.begin());

    thrust::copy(source.begin(), source.end(), p_output);

    ASSERT_EQ(output[0], 10);
    ASSERT_EQ(output[1], 2);
    ASSERT_EQ(output[2], 3);
    ASSERT_EQ(output[3], 10);
    ASSERT_EQ(output[4], 5);
    ASSERT_EQ(output[5], 10);
    ASSERT_EQ(output[6], 7);
    ASSERT_EQ(output[7], 10);
}

TYPED_TEST(PermutationIteratorTests, MakePermutationIterator)
{
    using Vector = typename TestFixture::input_type;

    Vector source(8);
    Vector indices(4);
    Vector output(4, 10);

    // initialize input
    thrust::sequence(source.begin(), source.end(), 1);

    indices[0] = 3;
    indices[1] = 0;
    indices[2] = 5;
    indices[3] = 7;

    thrust::copy(thrust::make_permutation_iterator(source.begin(), indices.begin()),
                 thrust::make_permutation_iterator(source.begin(), indices.begin()) + 4,
                 output.begin());

    ASSERT_EQ(output[0], 4);
    ASSERT_EQ(output[1], 1);
    ASSERT_EQ(output[2], 6);
    ASSERT_EQ(output[3], 8);
}

TYPED_TEST(PermutationIteratorTests, PermutationIteratorReduce)
{
    using Vector   = typename TestFixture::input_type;
    using T        = typename Vector::value_type;
    using Iterator = typename Vector::iterator;

    Vector source(8);
    Vector indices(4);
    Vector output(4, 10);

    // initialize input
    thrust::sequence(source.begin(), source.end(), 1);

    indices[0] = 3;
    indices[1] = 0;
    indices[2] = 5;
    indices[3] = 7;

    // construct transform_iterator
    thrust::permutation_iterator<Iterator, Iterator> iter(source.begin(), indices.begin());

    T result1
        = thrust::reduce(thrust::make_permutation_iterator(source.begin(), indices.begin()),
                         thrust::make_permutation_iterator(source.begin(), indices.begin()) + 4);

    ASSERT_EQ(result1, (T)19);

    T result2 = thrust::transform_reduce(
        thrust::make_permutation_iterator(source.begin(), indices.begin()),
        thrust::make_permutation_iterator(source.begin(), indices.begin()) + 4,
        thrust::negate<T>(),
        T(0),
        thrust::plus<T>());
    ASSERT_EQ(result2, (T)-19);
};

TEST(PermutationIteratorTests, PermutationIteratorHostDeviceGather)
{
    using T              = int;
    using HostVector     = typename thrust::host_vector<T>;
    using DeviceVector   = typename thrust::host_vector<T>;
    using HostIterator   = typename HostVector::iterator;
    using DeviceIterator = typename DeviceVector::iterator;

    HostVector h_source(8);
    HostVector h_indices(4);
    HostVector h_output(4, 10);

    DeviceVector d_source(8);
    DeviceVector d_indices(4);
    DeviceVector d_output(4, 10);

    // initialize source
    thrust::sequence(h_source.begin(), h_source.end(), 1);
    thrust::sequence(d_source.begin(), d_source.end(), 1);

    h_indices[0] = d_indices[0] = 3;
    h_indices[1] = d_indices[1] = 0;
    h_indices[2] = d_indices[2] = 5;
    h_indices[3] = d_indices[3] = 7;

    thrust::permutation_iterator<HostIterator, HostIterator>     p_h_source(h_source.begin(),
                                                                        h_indices.begin());
    thrust::permutation_iterator<DeviceIterator, DeviceIterator> p_d_source(d_source.begin(),
                                                                            d_indices.begin());

    // gather host->device
    thrust::copy(p_h_source, p_h_source + 4, d_output.begin());

    ASSERT_EQ(d_output[0], 4);
    ASSERT_EQ(d_output[1], 1);
    ASSERT_EQ(d_output[2], 6);
    ASSERT_EQ(d_output[3], 8);

    // gather device->host
    thrust::copy(p_d_source, p_d_source + 4, h_output.begin());

    ASSERT_EQ(h_output[0], 4);
    ASSERT_EQ(h_output[1], 1);
    ASSERT_EQ(h_output[2], 6);
    ASSERT_EQ(h_output[3], 8);
}

TEST(PermutationIteratorTests, PermutationIteratorHostDeviceScatter)
{
    using T              = int;
    using HostVector     = typename thrust::host_vector<T>;
    using DeviceVector   = typename thrust::host_vector<T>;
    using HostIterator   = typename HostVector::iterator;
    using DeviceIterator = typename DeviceVector::iterator;

    HostVector h_source(4, 10);
    HostVector h_indices(4);
    HostVector h_output(8);

    DeviceVector d_source(4, 10);
    DeviceVector d_indices(4);
    DeviceVector d_output(8);

    // initialize source
    thrust::sequence(h_output.begin(), h_output.end(), 1);
    thrust::sequence(d_output.begin(), d_output.end(), 1);

    h_indices[0] = d_indices[0] = 3;
    h_indices[1] = d_indices[1] = 0;
    h_indices[2] = d_indices[2] = 5;
    h_indices[3] = d_indices[3] = 7;

    thrust::permutation_iterator<HostIterator, HostIterator>     p_h_output(h_output.begin(),
                                                                        h_indices.begin());
    thrust::permutation_iterator<DeviceIterator, DeviceIterator> p_d_output(d_output.begin(),
                                                                            d_indices.begin());

    // scatter host->device
    thrust::copy(h_source.begin(), h_source.end(), p_d_output);

    ASSERT_EQ(d_output[0], 10);
    ASSERT_EQ(d_output[1], 2);
    ASSERT_EQ(d_output[2], 3);
    ASSERT_EQ(d_output[3], 10);
    ASSERT_EQ(d_output[4], 5);
    ASSERT_EQ(d_output[5], 10);
    ASSERT_EQ(d_output[6], 7);
    ASSERT_EQ(d_output[7], 10);

    // scatter device->host
    thrust::copy(d_source.begin(), d_source.end(), p_h_output);

    ASSERT_EQ(h_output[0], 10);
    ASSERT_EQ(h_output[1], 2);
    ASSERT_EQ(h_output[2], 3);
    ASSERT_EQ(h_output[3], 10);
    ASSERT_EQ(h_output[4], 5);
    ASSERT_EQ(h_output[5], 10);
    ASSERT_EQ(h_output[6], 7);
    ASSERT_EQ(h_output[7], 10);
}

TYPED_TEST(PermutationIteratorTests, PermutationIteratorWithCountingIterator)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;

    typename thrust::counting_iterator<T> input(0), index(0);

    // test copy()
    {
        Vector output(4, 0);

        thrust::copy(thrust::make_permutation_iterator(input, index),
                     thrust::make_permutation_iterator(input, index + output.size()),
                     output.begin());

        ASSERT_EQ(output[0], 0);
        ASSERT_EQ(output[1], 1);
        ASSERT_EQ(output[2], 2);
        ASSERT_EQ(output[3], 3);
    }

    // test copy()
    {
        Vector output(4, 0);

        thrust::transform(thrust::make_permutation_iterator(input, index),
                          thrust::make_permutation_iterator(input, index + 4),
                          output.begin(),
                          thrust::identity<T>());

        ASSERT_EQ(output[0], 0);
        ASSERT_EQ(output[1], 1);
        ASSERT_EQ(output[2], 2);
        ASSERT_EQ(output[3], 3);
    }
}
