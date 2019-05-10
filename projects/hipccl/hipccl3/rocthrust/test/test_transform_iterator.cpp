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

#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>

#include "test_header.hpp"

TESTS_DEFINE(TransformIteratorTests, FullTestsParams);
TESTS_DEFINE(PrimitiveTransformIteratorTests, NumericalTestsParams);

TEST(TransformIteratorTests, UsingHip)
{
    ASSERT_EQ(THRUST_DEVICE_SYSTEM, THRUST_DEVICE_SYSTEM_HIP);
}

TYPED_TEST(TransformIteratorTests, TransformIterator)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;

    using UnaryFunction = thrust::negate<T>;
    using Iterator      = typename Vector::iterator;

    Vector input(4);
    Vector output(4);

    // initialize input
    thrust::sequence(input.begin(), input.end(), 1);

    // construct transform_iterator
    thrust::transform_iterator<UnaryFunction, Iterator> iter(input.begin(), UnaryFunction());

    thrust::copy(iter, iter + 4, output.begin());

    ASSERT_EQ(output[0], (T)-1);
    ASSERT_EQ(output[1], (T)-2);
    ASSERT_EQ(output[2], (T)-3);
    ASSERT_EQ(output[3], (T)-4);
}

TYPED_TEST(TransformIteratorTests, MakeTransformIterator)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;

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

    ASSERT_EQ(output[0], (T)-1);
    ASSERT_EQ(output[1], (T)-2);
    ASSERT_EQ(output[2], (T)-3);
    ASSERT_EQ(output[3], (T)-4);
}

TYPED_TEST(PrimitiveTransformIteratorTests, TransformIteratorReduce)
{
    using T = typename TestFixture::input_type;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        T                        error_margin = (T)0.01 * size;
        thrust::host_vector<T>   h_data       = get_random_data<T>(size, 0, 10);
        thrust::device_vector<T> d_data       = h_data;

        // run on host
        T h_result
            = thrust::reduce(thrust::make_transform_iterator(h_data.begin(), thrust::negate<T>()),
                             thrust::make_transform_iterator(h_data.end(), thrust::negate<T>()));

        // run on device
        T d_result
            = thrust::reduce(thrust::make_transform_iterator(d_data.begin(), thrust::negate<T>()),
                             thrust::make_transform_iterator(d_data.end(), thrust::negate<T>()));

        ASSERT_NEAR(h_result, d_result, error_margin);
    }
}
