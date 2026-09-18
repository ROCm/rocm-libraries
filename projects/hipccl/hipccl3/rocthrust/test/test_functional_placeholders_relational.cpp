/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Modifications Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <thrust/detail/allocator/allocator_traits.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/universal_vector.h>

#include "test_param_fixtures.hpp"
#include "test_utils.hpp"

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

TESTS_DEFINE(FunctionalPlaceholdersRelationalTest, VectorTestsParams);

static const size_t num_samples = 10000;

template <typename Vector, typename U>
struct rebind_vector;

template <typename T, typename U, typename Allocator>
struct rebind_vector<thrust::host_vector<T, Allocator>, U>
{
  using alloc_traits = typename thrust::detail::allocator_traits<Allocator>;
  using new_alloc    = typename alloc_traits::template rebind_alloc<U>;
  using type         = thrust::host_vector<U, new_alloc>;
};

template <typename T, typename U, typename Allocator>
struct rebind_vector<thrust::device_vector<T, Allocator>, U>
{
  using type = thrust::device_vector<U, typename Allocator::template rebind<U>::other>;
};

template <typename T, typename U, typename Allocator>
struct rebind_vector<thrust::universal_vector<T, Allocator>, U>
{
  using type = thrust::universal_vector<U, typename Allocator::template rebind<U>::other>;
};

#define BINARY_FUNCTIONAL_PLACEHOLDERS_TEST(name, reference_operator, functor)                        \
  TYPED_TEST(FunctionalPlaceholdersRelationalTest, TestFunctionalPlaceholdersBinary##name)            \
  {                                                                                                   \
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());          \
                                                                                                      \
    using Vector      = typename TestFixture::input_type;                                             \
    using T           = typename Vector::value_type;                                                  \
    using bool_vector = typename rebind_vector<Vector, bool>::type;                                   \
    Vector lhs        = random_samples<T>(num_samples);                                               \
    Vector rhs        = random_samples<T>(num_samples);                                               \
                                                                                                      \
    bool_vector reference(lhs.size());                                                                \
    thrust::transform(lhs.begin(), lhs.end(), rhs.begin(), reference.begin(), functor<T>());          \
                                                                                                      \
    using namespace thrust::placeholders;                                                             \
    bool_vector result(lhs.size());                                                                   \
    thrust::transform(lhs.begin(), lhs.end(), rhs.begin(), result.begin(), _1 reference_operator _2); \
                                                                                                      \
    ASSERT_EQ(reference, result);                                                                     \
  }

BINARY_FUNCTIONAL_PLACEHOLDERS_TEST(EqualTo, ==, thrust::equal_to);
BINARY_FUNCTIONAL_PLACEHOLDERS_TEST(NotEqualTo, !=, thrust::not_equal_to);
BINARY_FUNCTIONAL_PLACEHOLDERS_TEST(Greater, >, thrust::greater);
BINARY_FUNCTIONAL_PLACEHOLDERS_TEST(Less, <, thrust::less);
BINARY_FUNCTIONAL_PLACEHOLDERS_TEST(GreaterEqual, >=, thrust::greater_equal);
BINARY_FUNCTIONAL_PLACEHOLDERS_TEST(LessEqual, <=, thrust::less_equal);
