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
 
#include <unittest/unittest.h>
#include <thrust/logical.h>
#include <thrust/functional.h>
#include <thrust/iterator/retag.h>


template <class Vector>
void TestAllOf(void)
{
    typedef typename Vector::value_type T;

    Vector v(3, 1);

    ASSERT_EQUAL(thrust::all_of(v.begin(), v.end(), thrust::identity<T>()), true);

    v[1] = 0;

    ASSERT_EQUAL(thrust::all_of(v.begin(), v.end(), thrust::identity<T>()), false);

    ASSERT_EQUAL(thrust::all_of(v.begin() + 0, v.begin() + 0, thrust::identity<T>()), true);
    ASSERT_EQUAL(thrust::all_of(v.begin() + 0, v.begin() + 1, thrust::identity<T>()), true);
    ASSERT_EQUAL(thrust::all_of(v.begin() + 0, v.begin() + 2, thrust::identity<T>()), false);
    ASSERT_EQUAL(thrust::all_of(v.begin() + 1, v.begin() + 2, thrust::identity<T>()), false);
}
DECLARE_VECTOR_UNITTEST(TestAllOf);


template <class InputIterator, class Predicate>
__host__ __device__
bool all_of(my_system &system, InputIterator, InputIterator, Predicate)
{
    system.validate_dispatch();
    return false;
}

void TestAllOfDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::all_of(sys, vec.begin(), vec.end(), 0);

    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestAllOfDispatchExplicit);


template <class InputIterator, class Predicate>
__host__ __device__
bool all_of(my_tag, InputIterator first, InputIterator, Predicate)
{
    *first = 13;
    return false;
}

void TestAllOfDispatchImplicit()
{
    thrust::device_vector<int> vec(1);

    thrust::all_of(thrust::retag<my_tag>(vec.begin()),
                   thrust::retag<my_tag>(vec.end()),
                   0);

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestAllOfDispatchImplicit);


template <class Vector>
void TestAnyOf(void)
{
    typedef typename Vector::value_type T;

    Vector v(3, 1);

    ASSERT_EQUAL(thrust::any_of(v.begin(), v.end(), thrust::identity<T>()), true);

    v[1] = 0;

    ASSERT_EQUAL(thrust::any_of(v.begin(), v.end(), thrust::identity<T>()), true);

    ASSERT_EQUAL(thrust::any_of(v.begin() + 0, v.begin() + 0, thrust::identity<T>()), false);
    ASSERT_EQUAL(thrust::any_of(v.begin() + 0, v.begin() + 1, thrust::identity<T>()), true);
    ASSERT_EQUAL(thrust::any_of(v.begin() + 0, v.begin() + 2, thrust::identity<T>()), true);
    ASSERT_EQUAL(thrust::any_of(v.begin() + 1, v.begin() + 2, thrust::identity<T>()), false);
}
DECLARE_VECTOR_UNITTEST(TestAnyOf);


template <class InputIterator, class Predicate>
__host__ __device__
bool any_of(my_system &system, InputIterator, InputIterator, Predicate)
{
    system.validate_dispatch();
    return false;
}

void TestAnyOfDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::any_of(sys, vec.begin(), vec.end(), 0);

    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestAnyOfDispatchExplicit);


template <class InputIterator, class Predicate>
__host__ __device__
bool any_of(my_tag, InputIterator first, InputIterator, Predicate)
{
    *first = 13;
    return false;
}

void TestAnyOfDispatchImplicit()
{
    thrust::device_vector<int> vec(1);

    thrust::any_of(thrust::retag<my_tag>(vec.begin()),
                   thrust::retag<my_tag>(vec.end()),
                   0);

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestAnyOfDispatchImplicit);


template <class Vector>
void TestNoneOf(void)
{
    typedef typename Vector::value_type T;

    Vector v(3, 1);

    ASSERT_EQUAL(thrust::none_of(v.begin(), v.end(), thrust::identity<T>()), false);

    v[1] = 0;

    ASSERT_EQUAL(thrust::none_of(v.begin(), v.end(), thrust::identity<T>()), false);

    ASSERT_EQUAL(thrust::none_of(v.begin() + 0, v.begin() + 0, thrust::identity<T>()), true);
    ASSERT_EQUAL(thrust::none_of(v.begin() + 0, v.begin() + 1, thrust::identity<T>()), false);
    ASSERT_EQUAL(thrust::none_of(v.begin() + 0, v.begin() + 2, thrust::identity<T>()), false);
    ASSERT_EQUAL(thrust::none_of(v.begin() + 1, v.begin() + 2, thrust::identity<T>()), true);
}
DECLARE_VECTOR_UNITTEST(TestNoneOf);


template <class InputIterator, class Predicate>
__host__ __device__
bool none_of(my_system &system, InputIterator, InputIterator, Predicate)
{
    system.validate_dispatch();
    return false;
}

void TestNoneOfDispatchExplicit()
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::none_of(sys, vec.begin(), vec.end(), 0);

    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestNoneOfDispatchExplicit);


template <class InputIterator, class Predicate>
__host__ __device__
bool none_of(my_tag, InputIterator first, InputIterator, Predicate)
{
    *first = 13;
    return false;
}

void TestNoneOfDispatchImplicit()
{
    thrust::device_vector<int> vec(1);

    thrust::none_of(thrust::retag<my_tag>(vec.begin()),
                    thrust::retag<my_tag>(vec.end()),
                    0);

    ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestNoneOfDispatchImplicit);
