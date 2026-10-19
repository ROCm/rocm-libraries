/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Modifications Copyright© 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <thrust/detail/libcxx_wrapper/__cccl_config.h>
#include <thrust/detail/type_traits.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/detail/iterator_category_to_system.h>
#include <thrust/iterator/detail/iterator_category_to_traversal.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/pair.h>
#include <thrust/tuple.h>
#include <thrust/type_traits/is_contiguous_iterator.h>

#include "test_param_fixtures.hpp"
#include "test_utils.hpp"

#if THRUST_COMPILER(GCC, >=, 7)
// This header pulls in an unsuppressable warning on GCC 6
#  include _THRUST_STD_INCLUDE(complex)
#endif // THRUST_COMPILER(GCC, >=, 7)
#include _THRUST_STD_INCLUDE(tuple)
#include _THRUST_STD_INCLUDE(utility)

#include <vector>

#if _THRUST_HAS_DEVICE_SYSTEM_STD
#  include _THRUST_LIBCXX_INCLUDE(iterator)
#endif

#if !_THRUST_HAS_DEVICE_SYSTEM_STD
#  include <type_traits>
#endif

TESTS_DEFINE(TypeTraitsTests, FullTestsParams);

TEST(TypeTraitsTests, TestIsContiguousIterator)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  using HostVector   = thrust::host_vector<int>;
  using DeviceVector = thrust::device_vector<int>;

  ASSERT_EQ(thrust::is_contiguous_iterator_v<int*>, true);
  ASSERT_EQ(thrust::is_contiguous_iterator_v<thrust::device_ptr<int>>, true);

  ASSERT_EQ(thrust::is_contiguous_iterator_v<HostVector::iterator>, true);
  ASSERT_EQ(thrust::is_contiguous_iterator_v<HostVector::const_iterator>, true);

  ASSERT_EQ(thrust::is_contiguous_iterator_v<DeviceVector::iterator>, true);
  ASSERT_EQ(thrust::is_contiguous_iterator_v<DeviceVector::const_iterator>, true);

  ASSERT_EQ(thrust::is_contiguous_iterator_v<thrust::device_ptr<int>>, true);

  using HostIteratorTuple = thrust::tuple<HostVector::iterator, HostVector::iterator>;

  using ConstantIterator  = thrust::constant_iterator<int>;
  using CountingIterator  = thrust::counting_iterator<int>;
  using TransformIterator = thrust::transform_iterator<_THRUST_STD::identity, HostVector::iterator>;
  using ZipIterator       = thrust::zip_iterator<HostIteratorTuple>;

  ASSERT_EQ(thrust::is_contiguous_iterator_v<ConstantIterator>, false);
  ASSERT_EQ(thrust::is_contiguous_iterator_v<CountingIterator>, false);
  ASSERT_EQ(thrust::is_contiguous_iterator_v<TransformIterator>, false);
  ASSERT_EQ(thrust::is_contiguous_iterator_v<ZipIterator>, false);
}

TEST(TypeTraitsTests, TestIsCommutative)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  {
    using T  = int;
    using Op = _THRUST_STD::plus<T>;
    ASSERT_EQ((bool) thrust::detail::is_commutative<Op>::value, true);
  }
  {
    using T  = int;
    using Op = _THRUST_STD::multiplies<T>;
    ASSERT_EQ((bool) thrust::detail::is_commutative<Op>::value, true);
  }
  {
    using T  = int;
    using Op = _THRUST_LIBCXX::minimum<T>;
    ASSERT_EQ((bool) thrust::detail::is_commutative<Op>::value, true);
  }
  {
    using T  = int;
    using Op = _THRUST_LIBCXX::maximum<T>;
    ASSERT_EQ((bool) thrust::detail::is_commutative<Op>::value, true);
  }
  {
    using T  = int;
    using Op = _THRUST_STD::logical_or<T>;
    ASSERT_EQ((bool) thrust::detail::is_commutative<Op>::value, true);
  }
  {
    using T  = int;
    using Op = _THRUST_STD::logical_and<T>;
    ASSERT_EQ((bool) thrust::detail::is_commutative<Op>::value, true);
  }
  {
    using T  = int;
    using Op = _THRUST_STD::bit_or<T>;
    ASSERT_EQ((bool) thrust::detail::is_commutative<Op>::value, true);
  }
  {
    using T  = int;
    using Op = _THRUST_STD::bit_and<T>;
    ASSERT_EQ((bool) thrust::detail::is_commutative<Op>::value, true);
  }
  {
    using T  = int;
    using Op = _THRUST_STD::bit_xor<T>;
    ASSERT_EQ((bool) thrust::detail::is_commutative<Op>::value, true);
  }

  {
    using T  = char;
    using Op = _THRUST_STD::plus<T>;
    ASSERT_EQ((bool) thrust::detail::is_commutative<Op>::value, true);
  }
  {
    using T  = short;
    using Op = _THRUST_STD::plus<T>;
    ASSERT_EQ((bool) thrust::detail::is_commutative<Op>::value, true);
  }
  {
    using T  = long;
    using Op = _THRUST_STD::plus<T>;
    ASSERT_EQ((bool) thrust::detail::is_commutative<Op>::value, true);
  }
  {
    using T  = long long;
    using Op = _THRUST_STD::plus<T>;
    ASSERT_EQ((bool) thrust::detail::is_commutative<Op>::value, true);
  }
  {
    using T  = float;
    using Op = _THRUST_STD::plus<T>;
    ASSERT_EQ((bool) thrust::detail::is_commutative<Op>::value, true);
  }
  {
    using T  = double;
    using Op = _THRUST_STD::plus<T>;
    ASSERT_EQ((bool) thrust::detail::is_commutative<Op>::value, true);
  }

  {
    using T  = int;
    using Op = _THRUST_STD::minus<T>;
    ASSERT_EQ((bool) thrust::detail::is_commutative<Op>::value, false);
  }
  {
    using T  = int;
    using Op = _THRUST_STD::divides<T>;
    ASSERT_EQ((bool) thrust::detail::is_commutative<Op>::value, false);
  }
  {
    using T  = float;
    using Op = _THRUST_STD::divides<T>;
    ASSERT_EQ((bool) thrust::detail::is_commutative<Op>::value, false);
  }
  {
    using T  = float;
    using Op = _THRUST_STD::minus<T>;
    ASSERT_EQ((bool) thrust::detail::is_commutative<Op>::value, false);
  }

  {
    using T  = thrust::tuple<int, int>;
    using Op = _THRUST_STD::plus<T>;
    ASSERT_EQ((bool) thrust::detail::is_commutative<Op>::value, false);
  }
}

struct NonTriviallyCopyable
{
  NonTriviallyCopyable(const NonTriviallyCopyable&) {}
};
THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(NonTriviallyCopyable);

static_assert(!_THRUST_STD::is_trivially_copyable<NonTriviallyCopyable>::value, "");
static_assert(thrust::is_trivially_relocatable<NonTriviallyCopyable>::value, "");

TEST(TypeTraitsTests, TestTriviallyRelocatable)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  static_assert(thrust::is_trivially_relocatable<int>::value, "");
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA || THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_HIP
  static_assert(thrust::is_trivially_relocatable<__half>::value, "");
  static_assert(thrust::is_trivially_relocatable<int1>::value, "");
  static_assert(thrust::is_trivially_relocatable<int2>::value, "");
  static_assert(thrust::is_trivially_relocatable<int3>::value, "");
  static_assert(thrust::is_trivially_relocatable<int4>::value, "");
#  if !defined(THRUST_DISABLE_INT128_SUPPORT) && (defined(__linux__) || defined(__LP64__)) \
    && ((THRUST_COMPILER(NVRTC) && defined(__CUDACC_RTC_INT128__)) || defined(__SIZEOF_INT128__))
  static_assert(thrust::is_trivially_relocatable<__int128>::value, "");
#  endif // !defined(THRUST_DISABLE_INT128_SUPPORT) && (defined(__linux__) || defined(__LP64__))
         // && ((THRUST_COMPILER(NVRTC) && defined(__CUDACC_RTC_INT128__)) || defined(__SIZEOF_INT128__))
#endif // THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA || THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_HIP
#if THRUST_COMPILER(GCC, >=, 7)
  static_assert(thrust::is_trivially_relocatable<thrust::complex<float>>::value, "");
  static_assert(thrust::is_trivially_relocatable<_THRUST_STD::complex<float>>::value, "");
  static_assert(thrust::is_trivially_relocatable<thrust::pair<int, thrust::complex<float>>>::value, "");
  static_assert(thrust::is_trivially_relocatable<_THRUST_STD::pair<int, _THRUST_STD::complex<float>>>::value, "");
  static_assert(thrust::is_trivially_relocatable<thrust::tuple<int, thrust::complex<float>, char>>::value, "");
  static_assert(thrust::is_trivially_relocatable<_THRUST_STD::tuple<int, _THRUST_STD::complex<float>, char>>::value,
                "");
#endif // THRUST_COMPILER(GCC, >=, 7)
#if _THRUST_HAS_DEVICE_SYSTEM_STD
  static_assert(thrust::is_trivially_relocatable<
                  _THRUST_STD::tuple<thrust::pair<int, thrust::tuple<int, _THRUST_STD::tuple<>>>,
                                     thrust::tuple<_THRUST_STD::pair<int, thrust::tuple<>>, int>>>::value,
                "");
#endif

  static_assert(!thrust::is_trivially_relocatable<thrust::pair<int, std::string>>::value, "");
  static_assert(!thrust::is_trivially_relocatable<_THRUST_STD::pair<int, std::string>>::value, "");
  static_assert(!thrust::is_trivially_relocatable<thrust::tuple<int, float, std::string>>::value, "");
  static_assert(!thrust::is_trivially_relocatable<_THRUST_STD::tuple<int, float, std::string>>::value, "");

  // test propagation of relocatability through pair and tuple
  static_assert(thrust::is_trivially_relocatable<NonTriviallyCopyable>::value, "");
#if _THRUST_HAS_DEVICE_SYSTEM_STD
  static_assert(thrust::is_trivially_relocatable<thrust::pair<NonTriviallyCopyable, int>>::value, "");
#endif
  static_assert(thrust::is_trivially_relocatable<_THRUST_STD::pair<NonTriviallyCopyable, int>>::value, "");
#if _THRUST_HAS_DEVICE_SYSTEM_STD
  static_assert(thrust::is_trivially_relocatable<thrust::tuple<NonTriviallyCopyable>>::value, "");
#endif
  static_assert(thrust::is_trivially_relocatable<_THRUST_STD::tuple<NonTriviallyCopyable>>::value, "");
}

template <typename Cat, typename Tag>
inline constexpr bool system_is = _THRUST_STD::is_same_v<thrust::detail::iterator_category_to_system_t<Cat>, Tag>;

template <typename Cat, typename Traversal>
inline constexpr bool traversal_is =
  _THRUST_STD::is_same_v<thrust::detail::iterator_category_to_traversal_t<Cat>, Traversal>;

TEST(TypeTraitsTests, TestIteratorCategoryToSystem)
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

TEST(TypeTraitsTests, TestIteratorCategoryToTraversal)
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

#if _THRUST_HAS_DEVICE_SYSTEM_STD
struct libcxx_make_counting_iterator
{
  template <typename... Args>
  auto operator()(Args&&... args) const
  {
    return _THRUST_LIBCXX::make_counting_iterator(_THRUST_STD::forward<Args>(args)...);
  }
};

struct libcxx_make_transform_iterator
{
  template <typename... Args>
  auto operator()(Args&&... args) const
  {
    return _THRUST_LIBCXX::make_transform_iterator(_THRUST_STD::forward<Args>(args)...);
  }
};

struct libcxx_make_zip_iterator
{
  template <typename... Args>
  auto operator()(Args&&... args) const
  {
    return _THRUST_LIBCXX::make_zip_iterator(_THRUST_STD::forward<Args>(args)...);
  }
};
#endif // _THRUST_HAS_DEVICE_SYSTEM_STD

template <typename Iterator>
inline constexpr bool has_random_access_traversal =
  _THRUST_STD::is_same_v<thrust::iterator_traversal_t<Iterator>, thrust::random_access_traversal_tag>;

auto expected_tag(thrust::device_vector<int>) -> thrust::device_system_tag;
auto expected_tag(thrust::host_vector<int>) -> thrust::host_system_tag;
auto expected_tag(std::vector<int>) -> thrust::host_system_tag;

template <typename MakeCounting, typename MakeTransform, typename MakeZip>
void check_any_system_propagation()
{
  auto counting_it = MakeCounting{}(0);
  static_assert(_THRUST_STD::is_same_v<thrust::iterator_system_t<decltype(counting_it)>, thrust::any_system_tag>);
  static_assert(has_random_access_traversal<decltype(counting_it)>);

  auto transform_it = MakeTransform{}(counting_it, thrust::square<>{});
  static_assert(_THRUST_STD::is_same_v<thrust::iterator_system_t<decltype(transform_it)>, thrust::any_system_tag>);
  static_assert(has_random_access_traversal<decltype(transform_it)>);

  [[maybe_unused]] auto zip_it = MakeZip{}(counting_it, transform_it);
  static_assert(_THRUST_STD::is_same_v<thrust::iterator_system_t<decltype(zip_it)>, thrust::any_system_tag>);
  static_assert(has_random_access_traversal<decltype(zip_it)>);
}

template <typename Vector, typename MakeTransform, typename MakeZip>
void check_vector_propagation()
{
  Vector vec{};
  using tag = decltype(expected_tag(vec));

  auto vec_it = vec.begin();
  static_assert(_THRUST_STD::is_same_v<thrust::iterator_system_t<decltype(vec_it)>, tag>);
  static_assert(has_random_access_traversal<decltype(vec_it)>);

  using thrust::placeholders::_1;
  auto transform_it = MakeTransform{}(vec_it, _1 + 1);
  static_assert(_THRUST_STD::is_same_v<thrust::iterator_system_t<decltype(transform_it)>, tag>);
  static_assert(has_random_access_traversal<decltype(transform_it)>);

  [[maybe_unused]] auto zip_it = MakeZip{}(vec_it, transform_it);
  static_assert(_THRUST_STD::is_same_v<thrust::iterator_system_t<decltype(zip_it)>, tag>);
  static_assert(has_random_access_traversal<decltype(zip_it)>);
}

template <typename Vector, typename MakeCounting, typename MakeZip>
void check_vector_and_any_system_propagation()
{
  Vector vec{};
  using vec_tag = decltype(expected_tag(vec));

  auto vec_it = vec.begin();
  static_assert(_THRUST_STD::is_same_v<thrust::iterator_system_t<decltype(vec_it)>, vec_tag>);
  static_assert(has_random_access_traversal<decltype(vec_it)>);

  auto counting_it = MakeCounting{}(0);
  static_assert(_THRUST_STD::is_same_v<thrust::iterator_system_t<decltype(counting_it)>, thrust::any_system_tag>);
  static_assert(has_random_access_traversal<decltype(counting_it)>);

  [[maybe_unused]] auto zip_it = MakeZip{}(vec_it, counting_it);
  static_assert(_THRUST_STD::is_same_v<thrust::iterator_system_t<decltype(zip_it)>, vec_tag>);
  static_assert(has_random_access_traversal<decltype(zip_it)>);
}

TEST(TypeTraitsTests, TestIteratorSystemAndTraversalPropagationAnySystem)
{
  check_any_system_propagation<thrust_make_counting_iterator, thrust_make_transform_iterator, thrust_make_zip_iterator>();
#if _THRUST_HAS_DEVICE_SYSTEM_STD
  check_any_system_propagation<thrust_make_counting_iterator, thrust_make_transform_iterator, libcxx_make_zip_iterator>();
  check_any_system_propagation<thrust_make_counting_iterator, libcxx_make_transform_iterator, thrust_make_zip_iterator>();
  check_any_system_propagation<thrust_make_counting_iterator, libcxx_make_transform_iterator, libcxx_make_zip_iterator>();
  check_any_system_propagation<libcxx_make_counting_iterator, thrust_make_transform_iterator, thrust_make_zip_iterator>();
  check_any_system_propagation<libcxx_make_counting_iterator, thrust_make_transform_iterator, libcxx_make_zip_iterator>();
  check_any_system_propagation<libcxx_make_counting_iterator, libcxx_make_transform_iterator, thrust_make_zip_iterator>();
  check_any_system_propagation<libcxx_make_counting_iterator, libcxx_make_transform_iterator, libcxx_make_zip_iterator>();
#endif // _THRUST_HAS_DEVICE_SYSTEM_STD
}

TEST(TypeTraitsTests, TestIteratorSystemAndTraversalPropagationVectors)
{
  check_vector_propagation<thrust::device_vector<int>, thrust_make_transform_iterator, thrust_make_zip_iterator>();
  check_vector_propagation<thrust::host_vector<int>, thrust_make_transform_iterator, thrust_make_zip_iterator>();
  check_vector_propagation<std::vector<int>, thrust_make_transform_iterator, thrust_make_zip_iterator>();
#if _THRUST_HAS_DEVICE_SYSTEM_STD
  check_vector_propagation<thrust::device_vector<int>, thrust_make_transform_iterator, libcxx_make_zip_iterator>();
  check_vector_propagation<thrust::device_vector<int>, libcxx_make_transform_iterator, thrust_make_zip_iterator>();
  check_vector_propagation<thrust::device_vector<int>, libcxx_make_transform_iterator, libcxx_make_zip_iterator>();
  check_vector_propagation<thrust::host_vector<int>, thrust_make_transform_iterator, libcxx_make_zip_iterator>();
  check_vector_propagation<thrust::host_vector<int>, libcxx_make_transform_iterator, thrust_make_zip_iterator>();
  check_vector_propagation<thrust::host_vector<int>, libcxx_make_transform_iterator, libcxx_make_zip_iterator>();
  check_vector_propagation<std::vector<int>, thrust_make_transform_iterator, libcxx_make_zip_iterator>();
  check_vector_propagation<std::vector<int>, libcxx_make_transform_iterator, thrust_make_zip_iterator>();
  check_vector_propagation<std::vector<int>, libcxx_make_transform_iterator, libcxx_make_zip_iterator>();
#endif // _THRUST_HAS_DEVICE_SYSTEM_STD
}

TEST(TypeTraitsTests, TestIteratorSystemAndTraversalPropagationVectorsAndAnySystem)
{
  check_vector_and_any_system_propagation<thrust::device_vector<int>, thrust_make_counting_iterator, thrust_make_zip_iterator>();
  check_vector_and_any_system_propagation<thrust::host_vector<int>, thrust_make_counting_iterator, thrust_make_zip_iterator>();
  check_vector_and_any_system_propagation<std::vector<int>, thrust_make_counting_iterator, thrust_make_zip_iterator>();
#if _THRUST_HAS_DEVICE_SYSTEM_STD
  check_vector_and_any_system_propagation<thrust::device_vector<int>, thrust_make_counting_iterator, libcxx_make_zip_iterator>();
  check_vector_and_any_system_propagation<thrust::device_vector<int>, libcxx_make_counting_iterator, thrust_make_zip_iterator>();
  check_vector_and_any_system_propagation<thrust::device_vector<int>, libcxx_make_counting_iterator, libcxx_make_zip_iterator>();
  check_vector_and_any_system_propagation<thrust::host_vector<int>, thrust_make_counting_iterator, libcxx_make_zip_iterator>();
  check_vector_and_any_system_propagation<thrust::host_vector<int>, libcxx_make_counting_iterator, thrust_make_zip_iterator>();
  check_vector_and_any_system_propagation<thrust::host_vector<int>, libcxx_make_counting_iterator, libcxx_make_zip_iterator>();
  check_vector_and_any_system_propagation<std::vector<int>, thrust_make_counting_iterator, libcxx_make_zip_iterator>();
  check_vector_and_any_system_propagation<std::vector<int>, libcxx_make_counting_iterator, thrust_make_zip_iterator>();
  check_vector_and_any_system_propagation<std::vector<int>, libcxx_make_counting_iterator, libcxx_make_zip_iterator>();
#endif // _THRUST_HAS_DEVICE_SYSTEM_STD
}
