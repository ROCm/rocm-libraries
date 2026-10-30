// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_categories.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>

#include _THRUST_LIBCXX_INCLUDE(iterator)
#include _THRUST_STD_INCLUDE(type_traits)
#include _THRUST_STD_INCLUDE(utility)

#include <vector>

#include "test_param_fixtures.hpp"
#include "test_utils.hpp"

template <typename Cat, typename Tag>
inline constexpr bool system_is = _THRUST_STD::is_same_v<thrust::detail::iterator_category_to_system_t<Cat>, Tag>;

TEST(IteratorTraitsTests, IteratorCategoryToSystem)
{
  static_assert(system_is<thrust::input_device_iterator_tag, thrust::device_system_tag>);
  static_assert(system_is<thrust::output_device_iterator_tag, thrust::device_system_tag>);
  static_assert(system_is<thrust::forward_device_iterator_tag, thrust::device_system_tag>);
  static_assert(system_is<thrust::bidirectional_device_iterator_tag, thrust::device_system_tag>);
  static_assert(system_is<thrust::random_access_device_iterator_tag, thrust::device_system_tag>);

  static_assert(system_is<thrust::input_host_iterator_tag, thrust::host_system_tag>);
  static_assert(system_is<thrust::output_host_iterator_tag, thrust::host_system_tag>);
  static_assert(system_is<thrust::forward_host_iterator_tag, thrust::host_system_tag>);
  static_assert(system_is<thrust::bidirectional_host_iterator_tag, thrust::host_system_tag>);
  static_assert(system_is<thrust::random_access_host_iterator_tag, thrust::host_system_tag>);
}

template <typename Cat, typename Traversal>
inline constexpr bool traversal_is =
  _THRUST_STD::is_same_v<thrust::detail::iterator_category_to_traversal_t<Cat>, Traversal>;

TEST(IteratorTraitsTests, IteratorCategoryToTraversal)
{
  static_assert(traversal_is<thrust::input_device_iterator_tag, thrust::single_pass_traversal_tag>);
  static_assert(traversal_is<thrust::output_device_iterator_tag, thrust::incrementable_traversal_tag>);
  static_assert(traversal_is<thrust::forward_device_iterator_tag, thrust::forward_traversal_tag>);
  static_assert(traversal_is<thrust::bidirectional_device_iterator_tag, thrust::bidirectional_traversal_tag>);
  static_assert(traversal_is<thrust::random_access_device_iterator_tag, thrust::random_access_traversal_tag>);

  static_assert(traversal_is<thrust::input_host_iterator_tag, thrust::single_pass_traversal_tag>);
  static_assert(traversal_is<thrust::output_host_iterator_tag, thrust::incrementable_traversal_tag>);
  static_assert(traversal_is<thrust::forward_host_iterator_tag, thrust::forward_traversal_tag>);
  static_assert(traversal_is<thrust::bidirectional_host_iterator_tag, thrust::bidirectional_traversal_tag>);
  static_assert(traversal_is<thrust::random_access_host_iterator_tag, thrust::random_access_traversal_tag>);
}

struct cuda_make_counting_iterator
{
  template <typename... Args>
  auto operator()(Args&&... args) const
  {
    return _THRUST_LIBCXX::make_counting_iterator(_THRUST_STD::forward<Args>(args)...);
  }
};

struct cuda_make_transform_iterator
{
  template <typename... Args>
  auto operator()(Args&&... args) const
  {
    return _THRUST_LIBCXX::make_transform_iterator(_THRUST_STD::forward<Args>(args)...);
  }
};

struct cuda_make_zip_iterator
{
  template <typename... Args>
  auto operator()(Args&&... args) const
  {
    return _THRUST_LIBCXX::make_zip_iterator(_THRUST_STD::forward<Args>(args)...);
  }
};

struct thrust_make_counting_iterator
{
  template <typename... Args>
  auto operator()(Args&&... args) const
  {
    return thrust::make_counting_iterator(_THRUST_STD::forward<Args>(args)...);
  }
};

struct thrust_make_transform_iterator
{
  template <typename... Args>
  auto operator()(Args&&... args) const
  {
    return thrust::make_transform_iterator(_THRUST_STD::forward<Args>(args)...);
  }
};

struct thrust_make_zip_iterator
{
  template <typename... Args>
  auto operator()(Args&&... args) const
  {
    return thrust::make_zip_iterator(_THRUST_STD::forward<Args>(args)...);
  }
};

template <typename Iterator>
inline constexpr bool has_random_access_traversal =
  _THRUST_STD::is_same_v<thrust::iterator_traversal_t<Iterator>, thrust::random_access_traversal_tag>;

using _THRUST_STD::__type_cartesian_product;
using _THRUST_STD::__type_list;

// Invokes Check{}.operator()<TestType>() for every TestType in the list (replaces Catch2's TEMPLATE_LIST_TEST_CASE)
template <typename Check, typename... TestTypes>
void for_each_test_type(__type_list<TestTypes...>)
{
  (Check{}.template operator()<TestTypes>(), ...);
}

using make_counting_t  = __type_list<cuda_make_counting_iterator, thrust_make_counting_iterator>;
using make_transform_t = __type_list<cuda_make_transform_iterator, thrust_make_transform_iterator>;
using make_zip_t       = __type_list<cuda_make_zip_iterator, thrust_make_zip_iterator>;
using make_it_t        = __type_cartesian_product<make_counting_t, make_transform_t, make_zip_t>;

struct check_any_system
{
  template <typename TestType>
  void operator()() const
  {
    using namespace _THRUST_STD;
    __type_at_c<0, TestType> make_counting_iterator;
    __type_at_c<1, TestType> make_transform_iterator;
    __type_at_c<2, TestType> make_zip_iterator;

    auto counting_it = make_counting_iterator(0);
    static_assert(is_same_v<thrust::iterator_system_t<decltype(counting_it)>, thrust::any_system_tag>);
    static_assert(has_random_access_traversal<decltype(counting_it)>);

    auto transform_it = make_transform_iterator(counting_it, thrust::square<>{});
    static_assert(is_same_v<thrust::iterator_system_t<decltype(transform_it)>, thrust::any_system_tag>);
    static_assert(has_random_access_traversal<decltype(transform_it)>);

    [[maybe_unused]] auto zip_it = make_zip_iterator(counting_it, transform_it);
    static_assert(is_same_v<thrust::iterator_system_t<decltype(zip_it)>, thrust::any_system_tag>);
    static_assert(has_random_access_traversal<decltype(zip_it)>);
  }
};

TEST(IteratorTraitsTests, IteratorSystemAndTraversalPropagationAnySystem)
{
  for_each_test_type<check_any_system>(make_it_t{});
}

auto expected_tag(thrust::device_vector<int>) -> thrust::device_system_tag;
auto expected_tag(thrust::host_vector<int>) -> thrust::host_system_tag;
auto expected_tag(std::vector<int>) -> thrust::host_system_tag;

using vectors           = __type_list<thrust::device_vector<int>, thrust::host_vector<int>, std::vector<int>>;
using make_vec_and_it_t = __type_cartesian_product<vectors, make_transform_t, make_zip_t>;

struct check_vectors
{
  template <typename TestType>
  void operator()() const
  {
    using namespace _THRUST_STD;
    __type_at_c<0, TestType> vec{};
    __type_at_c<1, TestType> make_transform_iterator;
    __type_at_c<2, TestType> make_zip_iterator;

    using tag = decltype(expected_tag(vec));

    auto vec_it = vec.begin();
    static_assert(_THRUST_STD::is_same_v<thrust::iterator_system_t<decltype(vec_it)>, tag>);
    static_assert(has_random_access_traversal<decltype(vec_it)>);

    using thrust::placeholders::_1;
    auto transform_it = make_transform_iterator(vec_it, _1 + 1);
    static_assert(is_same_v<thrust::iterator_system_t<decltype(transform_it)>, tag>);
    static_assert(has_random_access_traversal<decltype(transform_it)>);

    [[maybe_unused]] auto zip_it = make_zip_iterator(vec_it, transform_it);
    static_assert(is_same_v<thrust::iterator_system_t<decltype(zip_it)>, tag>);
    static_assert(has_random_access_traversal<decltype(zip_it)>);
  }
};

TEST(IteratorTraitsTests, IteratorSystemAndTraversalPropagationVectors)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for_each_test_type<check_vectors>(make_vec_and_it_t{});
}

using make_vec_count_zip_it_t = __type_cartesian_product<vectors, make_counting_t, make_zip_t>;

struct check_vectors_and_any_system
{
  template <typename TestType>
  void operator()() const
  {
    using namespace _THRUST_STD;
    __type_at_c<0, TestType> vec{};
    __type_at_c<1, TestType> make_counting_iterator;
    __type_at_c<2, TestType> make_zip_iterator;

    using vec_tag = decltype(expected_tag(vec));

    auto vec_it = vec.begin();
    static_assert(_THRUST_STD::is_same_v<thrust::iterator_system_t<decltype(vec_it)>, vec_tag>);
    static_assert(has_random_access_traversal<decltype(vec_it)>);

    auto counting_it = make_counting_iterator(0);
    static_assert(_THRUST_STD::is_same_v<thrust::iterator_system_t<decltype(counting_it)>, thrust::any_system_tag>);
    static_assert(has_random_access_traversal<decltype(counting_it)>);

    [[maybe_unused]] auto zip_it = make_zip_iterator(vec_it, counting_it);
    static_assert(_THRUST_STD::is_same_v<thrust::iterator_system_t<decltype(zip_it)>, vec_tag>);
    static_assert(has_random_access_traversal<decltype(zip_it)>);
  }
};

TEST(IteratorTraitsTests, IteratorSystemAndTraversalPropagationVectorsAndAnySystem)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for_each_test_type<check_vectors_and_any_system>(make_vec_count_zip_it_t{});
}
