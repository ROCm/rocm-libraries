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

#include <thrust/functional.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>
#include <thrust/unique.h>

#include "test_header.hpp"

TESTS_DEFINE(UniqueByKeyTests, FullTestsParams);

TESTS_DEFINE(UniqueByKeyIntegralTests, IntegerTestsParams);

template <typename ForwardIterator1, typename ForwardIterator2>
thrust::pair<ForwardIterator1, ForwardIterator2> unique_by_key(my_system&       system,
                                                               ForwardIterator1 keys_first,
                                                               ForwardIterator1,
                                                               ForwardIterator2 values_first)
{
    system.validate_dispatch();
    return thrust::make_pair(keys_first, values_first);
}

TEST(UniqueByKeyTests, TestUniqueByKeyDispatchExplicit)
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::unique_by_key(sys, vec.begin(), vec.begin(), vec.begin());

    ASSERT_EQ(true, sys.is_valid());
}

template <typename ForwardIterator1, typename ForwardIterator2>
thrust::pair<ForwardIterator1, ForwardIterator2> unique_by_key(my_tag,
                                                               ForwardIterator1 keys_first,
                                                               ForwardIterator1,
                                                               ForwardIterator2 values_first)
{
    *keys_first = 13;
    return thrust::make_pair(keys_first, values_first);
}

TEST(UniqueByKeyTests, TestUniqueByKeyDispatchImplicit)
{
    thrust::device_vector<int> vec(1);

    thrust::unique_by_key(thrust::retag<my_tag>(vec.begin()),
                          thrust::retag<my_tag>(vec.begin()),
                          thrust::retag<my_tag>(vec.begin()));

    ASSERT_EQ(13, vec.front());
}

template <typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2>
thrust::pair<OutputIterator1, OutputIterator2> unique_by_key_copy(my_system& system,
                                                                  InputIterator1,
                                                                  InputIterator1,
                                                                  InputIterator2,
                                                                  OutputIterator1 keys_output,
                                                                  OutputIterator2 values_output)
{
    system.validate_dispatch();
    return thrust::make_pair(keys_output, values_output);
}

TEST(UniqueByKeyTests, TestUniqueByKeyCopyDispatchExplicit)
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::unique_by_key_copy(
        sys, vec.begin(), vec.begin(), vec.begin(), vec.begin(), vec.begin());

    ASSERT_EQ(true, sys.is_valid());
}

template <typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2>
thrust::pair<OutputIterator1, OutputIterator2> unique_by_key_copy(my_tag,
                                                                  InputIterator1,
                                                                  InputIterator1,
                                                                  InputIterator2,
                                                                  OutputIterator1 keys_output,
                                                                  OutputIterator2 values_output)
{
    *keys_output = 13;
    return thrust::make_pair(keys_output, values_output);
}

TEST(UniqueByKeyTests, TestUniqueByKeyCopyDispatchImplicit)
{
    thrust::device_vector<int> vec(1);

    thrust::unique_by_key_copy(thrust::retag<my_tag>(vec.begin()),
                               thrust::retag<my_tag>(vec.begin()),
                               thrust::retag<my_tag>(vec.begin()),
                               thrust::retag<my_tag>(vec.begin()),
                               thrust::retag<my_tag>(vec.begin()));

    ASSERT_EQ(13, vec.front());
}

template <typename T>
struct is_equal_div_10_unique
{
    __host__ __device__ bool operator()(const T x, const T& y) const
    {
        return ((int)x / 10) == ((int)y / 10);
    }
};

template <typename Vector>
void initialize_keys(Vector& keys)
{
    keys.resize(9);
    keys[0] = 11;
    keys[1] = 11;
    keys[2] = 21;
    keys[3] = 20;
    keys[4] = 21;
    keys[5] = 21;
    keys[6] = 21;
    keys[7] = 37;
    keys[8] = 37;
}

template <typename Vector>
void initialize_values(Vector& values)
{
    values.resize(9);
    values[0] = 0;
    values[1] = 1;
    values[2] = 2;
    values[3] = 3;
    values[4] = 4;
    values[5] = 5;
    values[6] = 6;
    values[7] = 7;
    values[8] = 8;
}

TYPED_TEST(UniqueByKeyTests, TestUniqueByKeySimple)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;

    Vector keys;
    Vector values;

    typename thrust::pair<typename Vector::iterator, typename Vector::iterator> new_last;

    // basic test
    initialize_keys(keys);
    initialize_values(values);

    new_last = thrust::unique_by_key(keys.begin(), keys.end(), values.begin());

    ASSERT_EQ(new_last.first - keys.begin(), 5);
    ASSERT_EQ(new_last.second - values.begin(), 5);
    ASSERT_EQ(keys[0], 11);
    ASSERT_EQ(keys[1], 21);
    ASSERT_EQ(keys[2], 20);
    ASSERT_EQ(keys[3], 21);
    ASSERT_EQ(keys[4], 37);

    ASSERT_EQ(values[0], 0);
    ASSERT_EQ(values[1], 2);
    ASSERT_EQ(values[2], 3);
    ASSERT_EQ(values[3], 4);
    ASSERT_EQ(values[4], 7);

    // test BinaryPredicate
    initialize_keys(keys);
    initialize_values(values);

    new_last = thrust::unique_by_key(
        keys.begin(), keys.end(), values.begin(), is_equal_div_10_unique<T>());

    ASSERT_EQ(new_last.first - keys.begin(), 3);
    ASSERT_EQ(new_last.second - values.begin(), 3);
    ASSERT_EQ(keys[0], 11);
    ASSERT_EQ(keys[1], 21);
    ASSERT_EQ(keys[2], 37);

    ASSERT_EQ(values[0], 0);
    ASSERT_EQ(values[1], 2);
    ASSERT_EQ(values[2], 7);
}

TYPED_TEST(UniqueByKeyTests, TestUniqueCopyByKeySimple)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;

    Vector keys;
    Vector values;

    typename thrust::pair<typename Vector::iterator, typename Vector::iterator> new_last;

    // basic test
    initialize_keys(keys);
    initialize_values(values);

    Vector output_keys(keys.size());
    Vector output_values(values.size());

    new_last = thrust::unique_by_key_copy(
        keys.begin(), keys.end(), values.begin(), output_keys.begin(), output_values.begin());

    ASSERT_EQ(new_last.first - output_keys.begin(), 5);
    ASSERT_EQ(new_last.second - output_values.begin(), 5);
    ASSERT_EQ(output_keys[0], 11);
    ASSERT_EQ(output_keys[1], 21);
    ASSERT_EQ(output_keys[2], 20);
    ASSERT_EQ(output_keys[3], 21);
    ASSERT_EQ(output_keys[4], 37);

    ASSERT_EQ(output_values[0], 0);
    ASSERT_EQ(output_values[1], 2);
    ASSERT_EQ(output_values[2], 3);
    ASSERT_EQ(output_values[3], 4);
    ASSERT_EQ(output_values[4], 7);

    // test BinaryPredicate
    initialize_keys(keys);
    initialize_values(values);

    new_last = thrust::unique_by_key_copy(keys.begin(),
                                          keys.end(),
                                          values.begin(),
                                          output_keys.begin(),
                                          output_values.begin(),
                                          is_equal_div_10_unique<T>());

    ASSERT_EQ(new_last.first - output_keys.begin(), 3);
    ASSERT_EQ(new_last.second - output_values.begin(), 3);
    ASSERT_EQ(output_keys[0], 11);
    ASSERT_EQ(output_keys[1], 21);
    ASSERT_EQ(output_keys[2], 37);

    ASSERT_EQ(output_values[0], 0);
    ASSERT_EQ(output_values[1], 2);
    ASSERT_EQ(output_values[2], 7);
}

TYPED_TEST(UniqueByKeyIntegralTests, TestUniqueByKey)
{
    using K = typename TestFixture::input_type;
    using V = unsigned int; // ValueType

    for(auto size : get_sizes())
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        thrust::host_vector<K> h_keys = get_random_data<K>(
            size, std::numeric_limits<K>::min(), std::numeric_limits<K>::max());

        thrust::host_vector<V> h_vals = get_random_data<V>(
            size, std::numeric_limits<V>::min(), std::numeric_limits<V>::max());
        ;
        thrust::device_vector<K> d_keys = h_keys;
        thrust::device_vector<V> d_vals = h_vals;

        using HostKeyIterator   = typename thrust::host_vector<K>::iterator;
        using HostValIterator   = typename thrust::host_vector<V>::iterator;
        using DeviceKeyIterator = typename thrust::device_vector<K>::iterator;
        using DeviceValIterator = typename thrust::device_vector<V>::iterator;

        using HostIteratorPair   = typename thrust::pair<HostKeyIterator, HostValIterator>;
        using DeviceIteratorPair = typename thrust::pair<DeviceKeyIterator, DeviceValIterator>;

        HostIteratorPair h_last
            = thrust::unique_by_key(h_keys.begin(), h_keys.end(), h_vals.begin());
        DeviceIteratorPair d_last
            = thrust::unique_by_key(d_keys.begin(), d_keys.end(), d_vals.begin());

        ASSERT_EQ(h_last.first - h_keys.begin(), d_last.first - d_keys.begin());
        ASSERT_EQ(h_last.second - h_vals.begin(), d_last.second - d_vals.begin());

        size_t N = h_last.first - h_keys.begin();

        h_keys.resize(N);
        h_vals.resize(N);
        d_keys.resize(N);
        d_vals.resize(N);

        ASSERT_EQ(h_keys, d_keys);
        ASSERT_EQ(h_vals, d_vals);
    }
}

TYPED_TEST(UniqueByKeyIntegralTests, TestUniqueCopyByKey)
{
    using K = typename TestFixture::input_type;
    using V = unsigned int; // ValueType

    for(auto size : get_sizes())
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        thrust::host_vector<K> h_keys = get_random_data<K>(
            size, std::numeric_limits<K>::min(), std::numeric_limits<K>::max());

        thrust::host_vector<V> h_vals = get_random_data<V>(
            size, std::numeric_limits<V>::min(), std::numeric_limits<V>::max());
        ;
        thrust::device_vector<K> d_keys = h_keys;
        thrust::device_vector<V> d_vals = h_vals;

        thrust::host_vector<K>   h_keys_output(size);
        thrust::host_vector<V>   h_vals_output(size);
        thrust::device_vector<K> d_keys_output(size);
        thrust::device_vector<V> d_vals_output(size);

        using HostKeyIterator   = typename thrust::host_vector<K>::iterator;
        using HostValIterator   = typename thrust::host_vector<V>::iterator;
        using DeviceKeyIterator = typename thrust::device_vector<K>::iterator;
        using DeviceValIterator = typename thrust::device_vector<V>::iterator;

        using HostIteratorPair   = typename thrust::pair<HostKeyIterator, HostValIterator>;
        using DeviceIteratorPair = typename thrust::pair<DeviceKeyIterator, DeviceValIterator>;

        HostIteratorPair   h_last = thrust::unique_by_key_copy(h_keys.begin(),
                                                             h_keys.end(),
                                                             h_vals.begin(),
                                                             h_keys_output.begin(),
                                                             h_vals_output.begin());
        DeviceIteratorPair d_last = thrust::unique_by_key_copy(d_keys.begin(),
                                                               d_keys.end(),
                                                               d_vals.begin(),
                                                               d_keys_output.begin(),
                                                               d_vals_output.begin());

        ASSERT_EQ(h_last.first - h_keys_output.begin(), d_last.first - d_keys_output.begin());
        ASSERT_EQ(h_last.second - h_vals_output.begin(), d_last.second - d_vals_output.begin());

        size_t N = h_last.first - h_keys_output.begin();

        h_keys_output.resize(N);
        h_vals_output.resize(N);
        d_keys_output.resize(N);
        d_vals_output.resize(N);

        ASSERT_EQ(h_keys_output, d_keys_output);
        ASSERT_EQ(h_vals_output, d_vals_output);
    }
}

TYPED_TEST(UniqueByKeyIntegralTests, TestUniqueCopyByKeyToDiscardIterator)
{
    using K = typename TestFixture::input_type;
    using V = unsigned int; // ValueType

    for(auto size : get_sizes())
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        thrust::host_vector<K> h_keys = get_random_data<K>(
            size, std::numeric_limits<K>::min(), std::numeric_limits<K>::max());

        thrust::host_vector<V> h_vals = get_random_data<V>(
            size, std::numeric_limits<V>::min(), std::numeric_limits<V>::max());
        ;
        thrust::device_vector<K> d_keys = h_keys;
        thrust::device_vector<V> d_vals = h_vals;

        thrust::host_vector<V>   h_vals_output(size);
        thrust::device_vector<V> d_vals_output(size);

        thrust::host_vector<K>   h_keys_output(size);
        thrust::device_vector<K> d_keys_output(size);

        thrust::host_vector<K> h_unique_keys = h_keys;
        h_unique_keys.erase(thrust::unique(h_unique_keys.begin(), h_unique_keys.end()),
                            h_unique_keys.end());

        size_t num_unique_keys = h_unique_keys.size();

        // mask both outputs
        thrust::pair<thrust::discard_iterator<>, thrust::discard_iterator<>> h_result1
            = thrust::unique_by_key_copy(h_keys.begin(),
                                         h_keys.end(),
                                         h_vals.begin(),
                                         thrust::make_discard_iterator(),
                                         thrust::make_discard_iterator());

        thrust::pair<thrust::discard_iterator<>, thrust::discard_iterator<>> d_result1
            = thrust::unique_by_key_copy(d_keys.begin(),
                                         d_keys.end(),
                                         d_vals.begin(),
                                         thrust::make_discard_iterator(),
                                         thrust::make_discard_iterator());

        thrust::pair<thrust::discard_iterator<>, thrust::discard_iterator<>> reference1
            = thrust::make_pair(thrust::make_discard_iterator(num_unique_keys),
                                thrust::make_discard_iterator(num_unique_keys));

        ASSERT_EQ_QUIET(reference1, h_result1);
        ASSERT_EQ_QUIET(reference1, d_result1);

        // mask values output
        thrust::pair<typename thrust::host_vector<K>::iterator, thrust::discard_iterator<>>
            h_result2 = thrust::unique_by_key_copy(h_keys.begin(),
                                                   h_keys.end(),
                                                   h_vals.begin(),
                                                   h_keys_output.begin(),
                                                   thrust::make_discard_iterator());

        thrust::pair<typename thrust::device_vector<K>::iterator, thrust::discard_iterator<>>
            d_result2 = thrust::unique_by_key_copy(d_keys.begin(),
                                                   d_keys.end(),
                                                   d_vals.begin(),
                                                   d_keys_output.begin(),
                                                   thrust::make_discard_iterator());

        thrust::pair<typename thrust::host_vector<K>::iterator, thrust::discard_iterator<>>
            h_reference2 = thrust::make_pair(h_keys_output.begin() + num_unique_keys,
                                             thrust::make_discard_iterator(num_unique_keys));

        thrust::pair<typename thrust::device_vector<K>::iterator, thrust::discard_iterator<>>
            d_reference2 = thrust::make_pair(d_keys_output.begin() + num_unique_keys,
                                             thrust::make_discard_iterator(num_unique_keys));

        ASSERT_EQ(h_keys_output, d_keys_output);
        ASSERT_EQ_QUIET(h_reference2, h_result2);
        ASSERT_EQ_QUIET(d_reference2, d_result2);

        // mask keys output
        thrust::pair<thrust::discard_iterator<>, typename thrust::host_vector<V>::iterator>
            h_result3 = thrust::unique_by_key_copy(h_keys.begin(),
                                                   h_keys.end(),
                                                   h_vals.begin(),
                                                   thrust::make_discard_iterator(),
                                                   h_vals_output.begin());

        thrust::pair<thrust::discard_iterator<>, typename thrust::device_vector<V>::iterator>
            d_result3 = thrust::unique_by_key_copy(d_keys.begin(),
                                                   d_keys.end(),
                                                   d_vals.begin(),
                                                   thrust::make_discard_iterator(),
                                                   d_vals_output.begin());

        thrust::pair<thrust::discard_iterator<>, typename thrust::host_vector<V>::iterator>
            h_reference3 = thrust::make_pair(thrust::make_discard_iterator(num_unique_keys),
                                             h_vals_output.begin() + num_unique_keys);

        thrust::pair<thrust::discard_iterator<>, typename thrust::device_vector<V>::iterator>
            d_reference3 = thrust::make_pair(thrust::make_discard_iterator(num_unique_keys),
                                             d_vals_output.begin() + num_unique_keys);

        ASSERT_EQ(h_vals_output, d_vals_output);
        ASSERT_EQ_QUIET(h_reference3, h_result3);
        ASSERT_EQ_QUIET(d_reference3, d_result3);
    }
}
