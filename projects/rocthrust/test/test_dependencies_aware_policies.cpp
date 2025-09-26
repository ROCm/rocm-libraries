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

#include <thrust/detail/config.h>

// need to suppress deprecation warnings for execute_with_allocator_and_dependencies inside type traits
THRUST_SUPPRESS_DEPRECATED_PUSH

#include <thrust/detail/seq.h>
#include <thrust/system/cpp/detail/par.h>
#include <thrust/system/hip/detail/par.h>
#include <thrust/system/omp/detail/par.h>
#include <thrust/system/tbb/detail/par.h>

#include "test_param_fixtures.hpp"
#include "test_utils.hpp"

#if !_THRUST_HAS_DEVICE_SYSTEM_STD
#  include <type_traits>
#endif

template <typename T>
struct test_allocator_t
{};

test_allocator_t<int> test_allocator = test_allocator_t<int>();

template <int I>
struct test_dependency_t
{};

template <int I>
test_dependency_t<I> test_dependency()
{
  return {};
}

template <typename Policy, template <typename> class CRTPBase>
struct policy_info
{
  using policy = Policy;

  template <template <template <typename> class, typename...> class Template, typename... Arguments>
  using apply_base_first = Template<CRTPBase, Arguments...>;

  template <template <typename, template <typename> class, typename...> class Template,
            typename First,
            typename... Arguments>
  using apply_base_second = Template<First, CRTPBase, Arguments...>;
};

template <typename PolicyInfo>
struct TestDependencyAttachment
{
  template <typename... Expected, typename T>
  static void assert_correct(T)
  {
    ASSERT_EQ(
      (_THRUST_STD::is_same<
        T,
        typename PolicyInfo::template apply_base_first<thrust::detail::execute_with_dependencies, Expected...>>::value),
      true);
  }

  template <typename Allocator, typename... Expected, typename T>
  static void assert_correct_with_allocator(T)
  {
    ASSERT_EQ((_THRUST_STD::is_same<
                T,
                typename PolicyInfo::template apply_base_second<thrust::detail::execute_with_allocator_and_dependencies,
                                                                Allocator,
                                                                Expected...>>::value),
              true);
  }

  void operator()()
  {
    typename PolicyInfo::policy policy;

    assert_correct<test_dependency_t<1>>(policy.after(test_dependency<1>()));

    assert_correct<test_dependency_t<1>, test_dependency_t<2>>(
      policy.after(test_dependency<1>(), test_dependency<2>()));

    assert_correct<test_dependency_t<1>, test_dependency_t<2>, test_dependency_t<3>>(
      policy.after(test_dependency<1>(), test_dependency<2>(), test_dependency<3>()));

    assert_correct_with_allocator<test_allocator_t<int>&, test_dependency_t<1>>(
      policy(test_allocator).after(test_dependency<1>()));

    assert_correct_with_allocator<test_allocator_t<int>&, test_dependency_t<1>, test_dependency_t<2>>(
      policy(test_allocator).after(test_dependency<1>(), test_dependency<2>()));

    assert_correct_with_allocator<test_allocator_t<int>&,
                                  test_dependency_t<1>,
                                  test_dependency_t<2>,
                                  test_dependency_t<3>>(
      policy(test_allocator).after(test_dependency<1>(), test_dependency<2>(), test_dependency<3>()));
  }
};

using sequential_info = policy_info<thrust::detail::seq_t, thrust::system::detail::sequential::execution_policy>;
using cpp_par_info    = policy_info<thrust::system::cpp::detail::par_t, thrust::system::cpp::detail::execution_policy>;
using omp_par_info    = policy_info<thrust::system::omp::detail::par_t, thrust::system::omp::detail::execution_policy>;
using tbb_par_info    = policy_info<thrust::system::tbb::detail::par_t, thrust::system::tbb::detail::execution_policy>;

using hip_par_info = policy_info<thrust::system::hip::detail::par_t, thrust::hip_rocprim::execute_on_stream_base>;

using PolicyTestsParams = ::testing::Types<
  // TODO: uncomment when dependencies are generalized to all backends
  // Params<sequential_info>,
  // Params<cpp_par_info>,
  // Params<omp_par_info>,
  // Params<tbb_par_info>,
  Params<hip_par_info>>;

TESTS_DEFINE(DependenciesAwarePoliciesTests, PolicyTestsParams);

TYPED_TEST(DependenciesAwarePoliciesTests, TestDependencyAttachmentInstance)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  TestDependencyAttachment<T> test;
  test();
}

THRUST_SUPPRESS_DEPRECATED_POP
