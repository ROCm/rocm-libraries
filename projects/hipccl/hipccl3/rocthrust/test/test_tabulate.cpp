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

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>
#include <thrust/tabulate.h>

#include "test_header.hpp"

TESTS_DEFINE(TabulateTests, FullTestsParams);
TESTS_DEFINE(TabulatePrimitiveTests, NumericalTestsParams);

template <typename ForwardIterator, typename UnaryOperation>
void tabulate(my_system& system, ForwardIterator, ForwardIterator, UnaryOperation)
{
    system.validate_dispatch();
}

TEST(TabulateTests, TestTabulateDispatchExplicit)
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::tabulate(sys, vec.begin(), vec.end(), thrust::identity<int>());

    ASSERT_EQ(true, sys.is_valid());
}

template <typename ForwardIterator, typename UnaryOperation>
void tabulate(my_tag, ForwardIterator first, ForwardIterator, UnaryOperation)
{
    *first = 13;
}

TEST(TabulateTests, TestTabulateDispatchImplicit)
{
    thrust::device_vector<int> vec(1);

    thrust::tabulate(thrust::retag<my_tag>(vec.begin()),
                     thrust::retag<my_tag>(vec.end()),
                     thrust::identity<int>());

    ASSERT_EQ(13, vec.front());
}

TYPED_TEST(TabulateTests, TestTabulateSimple)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;
    using namespace thrust::placeholders;

    Vector v(5);

    thrust::tabulate(v.begin(), v.end(), thrust::identity<T>());

    ASSERT_EQ(v[0], T(0));
    ASSERT_EQ(v[1], T(1));
    ASSERT_EQ(v[2], T(2));
    ASSERT_EQ(v[3], T(3));
    ASSERT_EQ(v[4], T(4));

    thrust::tabulate(v.begin(), v.end(), -_1);

    ASSERT_EQ(v[0], T(0));
    ASSERT_EQ(v[1], T(-1));
    ASSERT_EQ(v[2], T(-2));
    ASSERT_EQ(v[3], T(-3));
    ASSERT_EQ(v[4], T(-4));

    thrust::tabulate(v.begin(), v.end(), _1 * _1 * _1);

    ASSERT_EQ(v[0], T(0));
    ASSERT_EQ(v[1], T(1));
    ASSERT_EQ(v[2], T(8));
    ASSERT_EQ(v[3], T(27));
    ASSERT_EQ(v[4], T(64));
}

template<class OutputType>
struct nonconst_op
{
    THRUST_HIP_FUNCTION
    OutputType operator()(size_t idx)
    {
        return (OutputType)(idx >= 3);
    }
};

TYPED_TEST(TabulateTests, TestTabulateSimpleNonConstOP)
{
    using Vector = typename TestFixture::input_type;
    using T = typename Vector::value_type;

    Vector v(5);

    thrust::tabulate(v.begin(), v.end(), nonconst_op<T>());

    ASSERT_EQ(v[0], T(0));
    ASSERT_EQ(v[1], T(0));
    ASSERT_EQ(v[2], T(0));
    ASSERT_EQ(v[3], T(1));
    ASSERT_EQ(v[4], T(1));
}

TYPED_TEST(TabulatePrimitiveTests, TestTabulate)
{
    using T = typename TestFixture::input_type;
    using namespace thrust::placeholders;

    for(auto size : get_sizes())
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        thrust::host_vector<T>   h_data(size);
        thrust::device_vector<T> d_data(size);

        thrust::tabulate(h_data.begin(), h_data.end(), _1 * _1 + T(13));
        thrust::tabulate(d_data.begin(), d_data.end(), _1 * _1 + T(13));

        thrust::host_vector<T> h_result = d_data;
        for(size_t i = 0; i < size; i++)
        {
            ASSERT_EQ(h_data[i], h_result[i]) << "where index = " << i;
        }

        thrust::tabulate(h_data.begin(), h_data.end(), (_1 - T(7)) * _1);
        thrust::tabulate(d_data.begin(), d_data.end(), (_1 - T(7)) * _1);

        ASSERT_EQ(h_data, d_data);
    }
}

TEST(TabulateTests, TestTabulateToDiscardIterator)
{
    for(auto size : get_sizes())
    {
        thrust::tabulate(thrust::discard_iterator<thrust::device_system_tag>(),
                         thrust::discard_iterator<thrust::device_system_tag>(size),
                         thrust::identity<int>());
    }
    // nothing to check -- just make sure it compiles
}
