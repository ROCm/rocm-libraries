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
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
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
