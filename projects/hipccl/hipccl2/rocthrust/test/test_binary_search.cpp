/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Modifications Copyright© 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/retag.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

#include "test_header.hpp"

TESTS_DEFINE(BinarySearchTestsInKernel, NumericalTestsParams);

template <typename T>
struct init_scalar
{
    T operator()(T t)
    {
        return t;
    }
};

template <typename T>
struct init_tuple
{
    thrust::tuple<T, T> operator()(T t)
    {
        return thrust::make_tuple(t, t);
    }
};

struct custom_less
{
    template <class T>
    __device__ inline bool operator()(T a, T b)
    {
        return a < b;
    }
};

template <class T>
__global__
THRUST_HIP_LAUNCH_BOUNDS_DEFAULT
void lower_bound_kernel(size_t n, T* input, ptrdiff_t* output)
{
    output[0] = thrust::lower_bound(thrust::device, input, input + n, T(0), custom_less()) - input;
    output[1] = thrust::lower_bound(thrust::device, input, input + n, T(1)) - input;
    output[2] = thrust::lower_bound(thrust::device, input, input + n, T(2)) - input;
    output[3] = thrust::lower_bound(thrust::device, input, input + n, T(3)) - input;
    output[4] = thrust::lower_bound(thrust::device, input, input + n, T(4), custom_less()) - input;
    output[5] = thrust::lower_bound(thrust::device, input, input + n, T(5)) - input;
    output[6] = thrust::lower_bound(thrust::device, input, input + n, T(6)) - input;
    output[7] = thrust::lower_bound(thrust::device, input, input + n, T(7)) - input;
    output[8] = thrust::lower_bound(thrust::device, input, input + n, T(8)) - input;
    output[9] = thrust::lower_bound(thrust::device, input, input + n, T(9), custom_less()) - input;
}

TYPED_TEST(BinarySearchTestsInKernel, TestLowerBound)
{
    using T = typename TestFixture::input_type;

    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    thrust::device_vector<T> d_input(5);
    d_input[0] = 0;
    d_input[1] = 2;
    d_input[2] = 5;
    d_input[3] = 7;
    d_input[4] = 8;

    thrust::device_vector<ptrdiff_t> d_output(10);

    hipLaunchKernelGGL(HIP_KERNEL_NAME(lower_bound_kernel),
                       dim3(1),
                       dim3(1),
                       0,
                       0,
                       size_t(d_input.size()),
                       thrust::raw_pointer_cast(d_input.data()),
                       thrust::raw_pointer_cast(d_output.data()));

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

template <class T>
__global__
THRUST_HIP_LAUNCH_BOUNDS_DEFAULT
void upper_bound_kernel(size_t n, T* input, ptrdiff_t* output)
{
    output[0] = thrust::upper_bound(thrust::device, input, input + n, T(0)) - input;
    output[1] = thrust::upper_bound(thrust::device, input, input + n, T(1)) - input;
    output[2] = thrust::upper_bound(thrust::device, input, input + n, T(2)) - input;
    output[3] = thrust::upper_bound(thrust::device, input, input + n, T(3)) - input;
    output[4] = thrust::upper_bound(thrust::device, input, input + n, T(4)) - input;
    output[5] = thrust::upper_bound(thrust::device, input, input + n, T(5)) - input;
    output[6] = thrust::upper_bound(thrust::device, input, input + n, T(6)) - input;
    output[7] = thrust::upper_bound(thrust::device, input, input + n, T(7)) - input;
    output[8] = thrust::upper_bound(thrust::device, input, input + n, T(8)) - input;
    output[9] = thrust::upper_bound(thrust::device, input, input + n, T(9)) - input;
}

TYPED_TEST(BinarySearchTestsInKernel, TestUpperBound)
{
    using T = typename TestFixture::input_type;

    thrust::device_vector<T> d_input(5);
    d_input[0] = 0;
    d_input[1] = 2;
    d_input[2] = 5;
    d_input[3] = 7;
    d_input[4] = 8;

    thrust::device_vector<ptrdiff_t> d_output(10);

    hipLaunchKernelGGL(HIP_KERNEL_NAME(upper_bound_kernel),
                       dim3(1),
                       dim3(1),
                       0,
                       0,
                       size_t(d_input.size()),
                       thrust::raw_pointer_cast(d_input.data()),
                       thrust::raw_pointer_cast(d_output.data()));

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

template <class T>
__global__
THRUST_HIP_LAUNCH_BOUNDS_DEFAULT
void binary_search_kernel(size_t n, T* input, bool* output)
{
    output[0] = thrust::binary_search(thrust::device, input, input + n, T(0));
    output[1] = thrust::binary_search(thrust::device, input, input + n, T(1));
    output[2] = thrust::binary_search(thrust::device, input, input + n, T(2));
    output[3] = thrust::binary_search(thrust::device, input, input + n, T(3));
    output[4] = thrust::binary_search(thrust::device, input, input + n, T(4));
    output[5] = thrust::binary_search(thrust::device, input, input + n, T(5));
    output[6] = thrust::binary_search(thrust::device, input, input + n, T(6));
    output[7] = thrust::binary_search(thrust::device, input, input + n, T(7));
    output[8] = thrust::binary_search(thrust::device, input, input + n, T(8));
    output[9] = thrust::binary_search(thrust::device, input, input + n, T(9));
}

TYPED_TEST(BinarySearchTestsInKernel, TestBinarySearch)
{
    using T = typename TestFixture::input_type;

    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    thrust::device_vector<T> d_input(5);
    d_input[0] = 0;
    d_input[1] = 2;
    d_input[2] = 5;
    d_input[3] = 7;
    d_input[4] = 8;

    thrust::device_vector<bool> d_output(10);

    hipLaunchKernelGGL(HIP_KERNEL_NAME(binary_search_kernel),
                       dim3(1),
                       dim3(1),
                       0,
                       0,
                       size_t(d_input.size()),
                       thrust::raw_pointer_cast(d_input.data()),
                       thrust::raw_pointer_cast(d_output.data()));

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

TESTS_DEFINE(BinarySearchTests, FullTestsParams);

THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_BEGIN

// accepts device_vector and host_vector
template <typename Vector, typename Policy, typename Initializer>
void test_scalar_lower_bound_simple(Initializer init)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    Vector vec(5);

    vec[0] = init(0);
    vec[1] = init(2);
    vec[2] = init(5);
    vec[3] = init(7);
    vec[4] = init(8);

    ASSERT_EQ(thrust::lower_bound(Policy {}, vec.begin(), vec.end(), init(0)) - vec.begin(), 0);
    ASSERT_EQ(thrust::lower_bound(Policy {}, vec.begin(), vec.end(), init(1)) - vec.begin(), 1);
    ASSERT_EQ(thrust::lower_bound(Policy {}, vec.begin(), vec.end(), init(2)) - vec.begin(), 1);
    ASSERT_EQ(thrust::lower_bound(Policy {}, vec.begin(), vec.end(), init(3)) - vec.begin(), 2);
    ASSERT_EQ(thrust::lower_bound(Policy {}, vec.begin(), vec.end(), init(4)) - vec.begin(), 2);
    ASSERT_EQ(thrust::lower_bound(Policy {}, vec.begin(), vec.end(), init(5)) - vec.begin(), 2);
    ASSERT_EQ(thrust::lower_bound(Policy {}, vec.begin(), vec.end(), init(6)) - vec.begin(), 3);
    ASSERT_EQ(thrust::lower_bound(Policy {}, vec.begin(), vec.end(), init(7)) - vec.begin(), 3);
    ASSERT_EQ(thrust::lower_bound(Policy {}, vec.begin(), vec.end(), init(8)) - vec.begin(), 4);
    ASSERT_EQ(thrust::lower_bound(Policy {}, vec.begin(), vec.end(), init(9)) - vec.begin(), 5);
}

TYPED_TEST(BinarySearchTests, TestScalarLowerBoundSimple)
{
    using Vector = typename TestFixture::input_type;
    using Policy = typename TestFixture::execution_policy;
    using T      = typename Vector::value_type;
    test_scalar_lower_bound_simple<Vector, Policy>(init_scalar<T>());
}

TEST(BinarySearchTests, TestTupleLowerBoundSimple)
{
    {
        using Policy = typename std::decay_t<decltype(thrust::hip::par)>;
        using Vector = thrust::device_vector<thrust::tuple<int, int>>;
        test_scalar_lower_bound_simple<Vector, Policy>(init_tuple<int>());
    }
    {
        using Policy = typename thrust::detail::host_t;
        using Vector = thrust::host_vector<thrust::tuple<int, int>>;
        test_scalar_lower_bound_simple<Vector, Policy>(init_tuple<int>());
    }
}

// accepts device_vector
template <typename Vector, typename Initializer>
void test_scalar_lower_bound_haystack(Initializer init)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    Vector haystack(5);

    haystack[0] = init(0);
    haystack[1] = init(2);
    haystack[2] = init(5);
    haystack[3] = init(7);
    haystack[4] = init(8);

    Vector needles(2);

    needles[0] = init(1);
    needles[1] = init(6);

    thrust::device_vector<int> indices(needles.size());

    thrust::lower_bound(
        haystack.begin(), haystack.end(), needles.begin(), needles.end(), indices.begin());

    thrust::device_vector<int> expected(needles.size());
    expected[0] = 1;
    expected[1] = 3;

    ASSERT_EQ(indices, expected);
}

TEST(BinarySearchTests, TestTupleLowerBoundHayStack)
{
    {
        using Vector = thrust::device_vector<int>;
        test_scalar_lower_bound_haystack<Vector>(init_scalar<int>());
    }
    {
        using Vector = thrust::device_vector<thrust::tuple<int, int>>;
        test_scalar_lower_bound_haystack<Vector>(init_tuple<int>());
    }
}

template <typename ForwardIterator, typename LessThanComparable>
ForwardIterator
lower_bound(my_system& system, ForwardIterator first, ForwardIterator, const LessThanComparable&)
{
    system.validate_dispatch();
    return first;
}

TEST(BinarySearchTests, TestScalarLowerBoundDispatchExplicit)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::lower_bound(sys, vec.begin(), vec.end(), 0);

    ASSERT_EQ(true, sys.is_valid());
}

template <typename ForwardIterator, typename LessThanComparable>
ForwardIterator
lower_bound(my_tag, ForwardIterator first, ForwardIterator, const LessThanComparable&)
{
    *first = 13;
    return first;
}

TEST(BinarySearchTests, TestScalarLowerBoundDispatchImplicit)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    thrust::device_vector<int> vec(1);

    thrust::lower_bound(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.end()), 0);

    ASSERT_EQ(13, vec.front());
}

// accepts device_vector and host_vector
template <typename Vector, typename Policy, typename Initializer>
void test_scalar_upper_bound_simple(Initializer init)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    Vector vec(5);

    vec[0] = init(0);
    vec[1] = init(2);
    vec[2] = init(5);
    vec[3] = init(7);
    vec[4] = init(8);

    ASSERT_EQ(thrust::upper_bound(Policy {}, vec.begin(), vec.end(), init(0)) - vec.begin(), 1);
    ASSERT_EQ(thrust::upper_bound(Policy {}, vec.begin(), vec.end(), init(1)) - vec.begin(), 1);
    ASSERT_EQ(thrust::upper_bound(Policy {}, vec.begin(), vec.end(), init(2)) - vec.begin(), 2);
    ASSERT_EQ(thrust::upper_bound(Policy {}, vec.begin(), vec.end(), init(3)) - vec.begin(), 2);
    ASSERT_EQ(thrust::upper_bound(Policy {}, vec.begin(), vec.end(), init(4)) - vec.begin(), 2);
    ASSERT_EQ(thrust::upper_bound(Policy {}, vec.begin(), vec.end(), init(5)) - vec.begin(), 3);
    ASSERT_EQ(thrust::upper_bound(Policy {}, vec.begin(), vec.end(), init(6)) - vec.begin(), 3);
    ASSERT_EQ(thrust::upper_bound(Policy {}, vec.begin(), vec.end(), init(7)) - vec.begin(), 4);
    ASSERT_EQ(thrust::upper_bound(Policy {}, vec.begin(), vec.end(), init(8)) - vec.begin(), 5);
    ASSERT_EQ(thrust::upper_bound(Policy {}, vec.begin(), vec.end(), init(9)) - vec.begin(), 5);
}

TYPED_TEST(BinarySearchTests, TestScalarUpperBoundSimple)
{
    using Vector = typename TestFixture::input_type;
    using Policy = typename TestFixture::execution_policy;
    using T      = typename Vector::value_type;
    test_scalar_upper_bound_simple<Vector, Policy>(init_scalar<T>());
}

TEST(BinarySearchTests, TestTupleUpperBoundSimple)
{
    {
        using Policy = typename std::decay_t<decltype(thrust::hip::par)>;
        using Vector = thrust::device_vector<thrust::tuple<int, int>>;
        test_scalar_upper_bound_simple<Vector, Policy>(init_tuple<int>());
    }
    {
        using Policy = typename thrust::detail::host_t;
        using Vector = thrust::host_vector<thrust::tuple<int, int>>;
        test_scalar_upper_bound_simple<Vector, Policy>(init_tuple<int>());
    }
}

// accepts device_vector
template <typename Vector, typename Initializer>
void test_scalar_upper_bound_haystack(Initializer init)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    Vector haystack(5);

    haystack[0] = init(0);
    haystack[1] = init(2);
    haystack[2] = init(5);
    haystack[3] = init(7);
    haystack[4] = init(8);

    Vector needles(2);

    needles[0] = init(1);
    needles[1] = init(6);

    thrust::device_vector<int> indices(needles.size());

    thrust::upper_bound(
        haystack.begin(), haystack.end(), needles.begin(), needles.end(), indices.begin());

    thrust::device_vector<int> expected(needles.size());
    expected[0] = 1;
    expected[1] = 3;

    ASSERT_EQ(indices, expected);
}

TEST(BinarySearchTests, TestTupleUpperBoundHayStack)
{
    {
        using Vector = thrust::device_vector<int>;
        test_scalar_upper_bound_haystack<Vector>(init_scalar<int>());
    }
    {
        using Vector = thrust::device_vector<thrust::tuple<int, int>>;
        test_scalar_upper_bound_haystack<Vector>(init_tuple<int>());
    }
}

template <typename ForwardIterator, typename LessThanComparable>
ForwardIterator
upper_bound(my_system& system, ForwardIterator first, ForwardIterator, const LessThanComparable&)
{
    system.validate_dispatch();
    return first;
}

TEST(BinarySearchTests, TestScalarUpperBoundDispatchExplicit)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::upper_bound(sys, vec.begin(), vec.end(), 0);

    ASSERT_EQ(true, sys.is_valid());
}

template <typename ForwardIterator, typename LessThanComparable>
ForwardIterator
upper_bound(my_tag, ForwardIterator first, ForwardIterator, const LessThanComparable&)
{
    *first = 13;
    return first;
}

TEST(BinarySearchTests, TestScalarUpperBoundDispatchImplicit)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    thrust::device_vector<int> vec(1);

    thrust::upper_bound(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.end()), 0);

    ASSERT_EQ(13, vec.front());
}

// accepts device_vector and host_vector
template <typename Vector, typename Policy, typename Initializer>
void test_scalar_binary_search_simple(Initializer init)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    Vector vec(5);

    vec[0] = init(0);
    vec[1] = init(2);
    vec[2] = init(5);
    vec[3] = init(7);
    vec[4] = init(8);

    ASSERT_EQ(thrust::binary_search(Policy {}, vec.begin(), vec.end(), init(0)), true);
    ASSERT_EQ(thrust::binary_search(Policy {}, vec.begin(), vec.end(), init(1)), false);
    ASSERT_EQ(thrust::binary_search(Policy {}, vec.begin(), vec.end(), init(2)), true);
    ASSERT_EQ(thrust::binary_search(Policy {}, vec.begin(), vec.end(), init(3)), false);
    ASSERT_EQ(thrust::binary_search(Policy {}, vec.begin(), vec.end(), init(4)), false);
    ASSERT_EQ(thrust::binary_search(Policy {}, vec.begin(), vec.end(), init(5)), true);
    ASSERT_EQ(thrust::binary_search(Policy {}, vec.begin(), vec.end(), init(6)), false);
    ASSERT_EQ(thrust::binary_search(Policy {}, vec.begin(), vec.end(), init(7)), true);
    ASSERT_EQ(thrust::binary_search(Policy {}, vec.begin(), vec.end(), init(8)), true);
    ASSERT_EQ(thrust::binary_search(Policy {}, vec.begin(), vec.end(), init(9)), false);
}

TYPED_TEST(BinarySearchTests, TestScalarBinarySearchSimple)
{
    using Vector = typename TestFixture::input_type;
    using Policy = typename TestFixture::execution_policy;
    using T      = typename Vector::value_type;
    test_scalar_binary_search_simple<Vector, Policy>(init_scalar<T>());
}

TEST(BinarySearchTests, TestTupleBinarySearchSimple)
{
    {
        using Policy = typename std::decay_t<decltype(thrust::hip::par)>;
        using Vector = thrust::device_vector<thrust::tuple<int, int>>;
        test_scalar_binary_search_simple<Vector, Policy>(init_tuple<int>());
    }
    {
        using Policy = typename thrust::detail::host_t;
        using Vector = thrust::host_vector<thrust::tuple<int, int>>;
        test_scalar_binary_search_simple<Vector, Policy>(init_tuple<int>());
    }
}

// accepts device_vector
template <typename Vector, typename Initializer>
void test_scalar_binary_search_haystack(Initializer init)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    Vector haystack(5);

    haystack[0] = init(0);
    haystack[1] = init(2);
    haystack[2] = init(5);
    haystack[3] = init(7);
    haystack[4] = init(8);

    Vector needles(2);

    needles[0] = init(3);
    needles[1] = init(5);

    thrust::device_vector<bool> indices(needles.size());

    thrust::binary_search(
        haystack.begin(), haystack.end(), needles.begin(), needles.end(), indices.begin());

    thrust::device_vector<bool> expected(needles.size());
    expected[0] = false;
    expected[1] = true;

    ASSERT_EQ(indices, expected);
}

TEST(BinarySearchTests, TestTupleBinarySearchHayStack)
{
    {
        using Vector = thrust::device_vector<int>;
        test_scalar_binary_search_haystack<Vector>(init_scalar<int>());
    }
    {
        using Vector = thrust::device_vector<thrust::tuple<int, int>>;
        test_scalar_binary_search_haystack<Vector>(init_tuple<int>());
    }
}

template <typename ForwardIterator, typename LessThanComparable>
bool binary_search(my_system& system, ForwardIterator, ForwardIterator, const LessThanComparable&)
{
    system.validate_dispatch();
    return false;
}

TEST(BinarySearchTests, TestScalarBinarySearchDispatchExplicit)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::binary_search(sys, vec.begin(), vec.end(), 0);

    ASSERT_EQ(true, sys.is_valid());
}

template <typename ForwardIterator, typename LessThanComparable>
bool binary_search(my_tag, ForwardIterator first, ForwardIterator, const LessThanComparable&)
{
    *first = 13;
    return false;
}

TEST(BinarySearchTests, TestScalarBinarySearchDispatchImplicit)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    thrust::device_vector<int> vec(1);

    thrust::binary_search(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.end()), 0);

    ASSERT_EQ(13, vec.front());
}

TYPED_TEST(BinarySearchTests, TestScalarEqualRangeSimple)
{
    using Vector = typename TestFixture::input_type;
    using Policy = typename TestFixture::execution_policy;
    using T      = typename Vector::value_type;

    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    Vector vec(5);

    vec[0] = 0;
    vec[1] = 2;
    vec[2] = 5;
    vec[3] = 7;
    vec[4] = 8;

    ASSERT_EQ(thrust::equal_range(Policy{}, vec.begin(), vec.end(), T(0)).first - vec.begin(), 0);
    ASSERT_EQ(thrust::equal_range(Policy{}, vec.begin(), vec.end(), T(1)).first - vec.begin(), 1);
    ASSERT_EQ(thrust::equal_range(Policy{}, vec.begin(), vec.end(), T(2)).first - vec.begin(), 1);
    ASSERT_EQ(thrust::equal_range(Policy{}, vec.begin(), vec.end(), T(3)).first - vec.begin(), 2);
    ASSERT_EQ(thrust::equal_range(Policy{}, vec.begin(), vec.end(), T(4)).first - vec.begin(), 2);
    ASSERT_EQ(thrust::equal_range(Policy{}, vec.begin(), vec.end(), T(5)).first - vec.begin(), 2);
    ASSERT_EQ(thrust::equal_range(Policy{}, vec.begin(), vec.end(), T(6)).first - vec.begin(), 3);
    ASSERT_EQ(thrust::equal_range(Policy{}, vec.begin(), vec.end(), T(7)).first - vec.begin(), 3);
    ASSERT_EQ(thrust::equal_range(Policy{}, vec.begin(), vec.end(), T(8)).first - vec.begin(), 4);
    ASSERT_EQ(thrust::equal_range(Policy{}, vec.begin(), vec.end(), T(9)).first - vec.begin(), 5);

    ASSERT_EQ(thrust::equal_range(Policy{}, vec.begin(), vec.end(), T(0)).second - vec.begin(), 1);
    ASSERT_EQ(thrust::equal_range(Policy{}, vec.begin(), vec.end(), T(1)).second - vec.begin(), 1);
    ASSERT_EQ(thrust::equal_range(Policy{}, vec.begin(), vec.end(), T(2)).second - vec.begin(), 2);
    ASSERT_EQ(thrust::equal_range(Policy{}, vec.begin(), vec.end(), T(3)).second - vec.begin(), 2);
    ASSERT_EQ(thrust::equal_range(Policy{}, vec.begin(), vec.end(), T(4)).second - vec.begin(), 2);
    ASSERT_EQ(thrust::equal_range(Policy{}, vec.begin(), vec.end(), T(5)).second - vec.begin(), 3);
    ASSERT_EQ(thrust::equal_range(Policy{}, vec.begin(), vec.end(), T(6)).second - vec.begin(), 3);
    ASSERT_EQ(thrust::equal_range(Policy{}, vec.begin(), vec.end(), T(7)).second - vec.begin(), 4);
    ASSERT_EQ(thrust::equal_range(Policy{}, vec.begin(), vec.end(), T(8)).second - vec.begin(), 5);
    ASSERT_EQ(thrust::equal_range(Policy{}, vec.begin(), vec.end(), T(9)).second - vec.begin(), 5);
}

template <typename ForwardIterator, typename LessThanComparable>
thrust::pair<ForwardIterator, ForwardIterator>
equal_range(my_system& system, ForwardIterator first, ForwardIterator, const LessThanComparable&)
{
    system.validate_dispatch();
    return thrust::make_pair(first, first);
}

TEST(BinarySearchTests, TestScalarEqualRangeDispatchExplicit)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::equal_range(sys, vec.begin(), vec.end(), 0);

    ASSERT_EQ(true, sys.is_valid());
}

template <typename ForwardIterator, typename LessThanComparable>
thrust::pair<ForwardIterator, ForwardIterator>
equal_range(my_tag, ForwardIterator first, ForwardIterator, const LessThanComparable&)
{
    *first = 13;
    return thrust::make_pair(first, first);
}

TEST(BinarySearchTests, TestScalarEqualRangeDispatchImplicit)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    thrust::device_vector<int> vec(1);

    thrust::binary_search(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.end()), 0);

    ASSERT_EQ(13, vec.front());
}

TEST(BinarySearchTests, TestEqualRangeExecutionPolicy)
{
    using thrust_exec_policy_t
        = thrust::detail::execute_with_allocator<thrust::device_allocator<char>,
                                                 thrust::hip_rocprim::execute_on_stream_base>;

    constexpr int              data[] = {1, 2, 3, 4, 4, 5, 6, 7, 8, 9};
    constexpr size_t           size   = sizeof(data) / sizeof(data[0]);
    constexpr int              key    = 4;
    thrust::device_vector<int> d_data(data, data + size);

    thrust::pair<thrust::device_vector<int>::iterator, thrust::device_vector<int>::iterator> range
        = thrust::equal_range(
            thrust_exec_policy_t(thrust::hip_rocprim::execute_on_stream_base<thrust_exec_policy_t>(
                                     hipStreamPerThread),
                                 thrust::device_allocator<char>()),
            d_data.begin(),
            d_data.end(),
            key);

    ASSERT_EQ(*range.first, 4);
    ASSERT_EQ(*range.second, 5);
}

__global__
THRUST_HIP_LAUNCH_BOUNDS_DEFAULT
void BinarySearchKernel(int const N, int* in_array, int*result_array, int search_value)
{
  if(threadIdx.x == 0)
  {
      thrust::device_ptr<int> begin(in_array);
      thrust::device_ptr<int> end(in_array + N);
      result_array[search_value]=thrust::binary_search(thrust::hip::par, begin,end,search_value);
  }
}

TEST(BinarySearchTests, TestBinarySearchDevice)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
    for(auto size : get_sizes() )
    {
        SCOPED_TRACE(testing::Message() << "with size= " << size);

        for(auto seed : get_seeds())
        {
            SCOPED_TRACE(testing::Message() << "with seed= " << seed);

            thrust::host_vector<int> h_data = get_random_data<int>(size, 0, size, seed);
            thrust::device_vector<int> d_data = h_data;

            thrust::host_vector<int> h_result(size*2,-1);
            thrust::device_vector<int> d_result(size*2,-1);

            for(int search_value = 0; search_value < (int)size*2; search_value++)
            {
              SCOPED_TRACE(testing::Message() << "searching for " <<search_value);

              h_result[search_value] = thrust::binary_search(h_data.begin(),h_data.end(),search_value);
              hipLaunchKernelGGL(BinarySearchKernel,
                                 dim3(1, 1, 1),
                                 dim3(128, 1, 1),
                                 0,
                                 0,
                                 size,
                                 thrust::raw_pointer_cast(&d_data[0]),
                                 thrust::raw_pointer_cast(&d_result[0]),
                                 search_value);
            }
            ASSERT_EQ(h_result,d_result);
        }
    }
}
THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_END
