/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Modifications Copyright© 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <thrust/detail/type_traits.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/pair.h>
#include <thrust/tuple.h>
#include <thrust/type_traits/is_contiguous_iterator.h>

#include <unittest/unittest.h>

#if defined(THRUST_GCC_VERSION) && THRUST_GCC_VERSION >= 70000
// This header pulls in an unsuppressable warning on GCC 6
#  include _THRUST_STD_INCLUDE(complex)
#endif // defined(THRUST_GCC_VERSION) && THRUST_GCC_VERSION >= 70000
#include _THRUST_STD_INCLUDE(tuple)
#include _THRUST_STD_INCLUDE(utility)

#if !_THRUST_HAS_DEVICE_SYSTEM_STD
#  include <type_traits>
#endif

void TestIsContiguousIterator()
{
  using HostVector   = thrust::host_vector<int>;
  using DeviceVector = thrust::device_vector<int>;

  ASSERT_EQUAL((bool) thrust::is_contiguous_iterator<int*>::value, true);
  ASSERT_EQUAL((bool) thrust::is_contiguous_iterator<thrust::device_ptr<int>>::value, true);

  ASSERT_EQUAL((bool) thrust::is_contiguous_iterator<HostVector::iterator>::value, true);
  ASSERT_EQUAL((bool) thrust::is_contiguous_iterator<HostVector::const_iterator>::value, true);

  ASSERT_EQUAL((bool) thrust::is_contiguous_iterator<DeviceVector::iterator>::value, true);
  ASSERT_EQUAL((bool) thrust::is_contiguous_iterator<DeviceVector::const_iterator>::value, true);

  ASSERT_EQUAL((bool) thrust::is_contiguous_iterator<thrust::device_ptr<int>>::value, true);

  using HostIteratorTuple = thrust::tuple<HostVector::iterator, HostVector::iterator>;

  using ConstantIterator = thrust::constant_iterator<int>;
  using CountingIterator = thrust::counting_iterator<int>;
  THRUST_SUPPRESS_DEPRECATED_PUSH
  using TransformIterator1 = thrust::transform_iterator<thrust::identity<int>, HostVector::iterator>;
  THRUST_SUPPRESS_DEPRECATED_POP
  using TransformIterator2 = thrust::transform_iterator<::internal::identity, HostVector::iterator>;
  using ZipIterator        = thrust::zip_iterator<HostIteratorTuple>;

  ASSERT_EQUAL((bool) thrust::is_contiguous_iterator<ConstantIterator>::value, false);
  ASSERT_EQUAL((bool) thrust::is_contiguous_iterator<CountingIterator>::value, false);
#if THRUST_HOST_COMPILER != THRUST_HOST_COMPILER_NVHPC
  // thrust::identity creates a deprecated warning that could not be worked around
  THRUST_SUPPRESS_DEPRECATED_PUSH
  ASSERT_EQUAL((bool) thrust::is_contiguous_iterator<TransformIterator1>::value, false);
  THRUST_SUPPRESS_DEPRECATED_POP
#endif // THRUST_HOST_COMPILER != THRUST_HOST_COMPILER_NVHPC
  ASSERT_EQUAL((bool) thrust::is_contiguous_iterator<TransformIterator2>::value, false);
  ASSERT_EQUAL((bool) thrust::is_contiguous_iterator<ZipIterator>::value, false);
}
DECLARE_UNITTEST(TestIsContiguousIterator);

void TestIsCommutative()
{
  {
    using T  = int;
    using Op = thrust::plus<T>;
    ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true);
  }
  {
    using T  = int;
    using Op = thrust::multiplies<T>;
    ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true);
  }
  {
    using T  = int;
    using Op = thrust::minimum<T>;
    ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true);
  }
  {
    using T  = int;
    using Op = thrust::maximum<T>;
    ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true);
  }
  {
    using T  = int;
    using Op = thrust::logical_or<T>;
    ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true);
  }
  {
    using T  = int;
    using Op = thrust::logical_and<T>;
    ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true);
  }
  {
    using T  = int;
    using Op = thrust::bit_or<T>;
    ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true);
  }
  {
    using T  = int;
    using Op = thrust::bit_and<T>;
    ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true);
  }
  {
    using T  = int;
    using Op = thrust::bit_xor<T>;
    ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true);
  }

  {
    using T  = char;
    using Op = thrust::plus<T>;
    ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true);
  }
  {
    using T  = short;
    using Op = thrust::plus<T>;
    ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true);
  }
  {
    using T  = long;
    using Op = thrust::plus<T>;
    ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true);
  }
  {
    using T  = long long;
    using Op = thrust::plus<T>;
    ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true);
  }
  {
    using T  = float;
    using Op = thrust::plus<T>;
    ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true);
  }
  {
    using T  = double;
    using Op = thrust::plus<T>;
    ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true);
  }

  {
    using T  = int;
    using Op = thrust::minus<T>;
    ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, false);
  }
  {
    using T  = int;
    using Op = thrust::divides<T>;
    ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, false);
  }
  {
    using T  = float;
    using Op = thrust::divides<T>;
    ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, false);
  }
  {
    using T  = float;
    using Op = thrust::minus<T>;
    ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, false);
  }

  {
    using T  = thrust::tuple<int, int>;
    using Op = thrust::plus<T>;
    ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, false);
  }
}
DECLARE_UNITTEST(TestIsCommutative);

struct NonTriviallyCopyable
{
  NonTriviallyCopyable(const NonTriviallyCopyable&) {}
};
THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(NonTriviallyCopyable);

static_assert(!_THRUST_STD::is_trivially_copyable<NonTriviallyCopyable>::value, "");
static_assert(thrust::is_trivially_relocatable<NonTriviallyCopyable>::value, "");

void TestTriviallyRelocatable()
{
  static_assert(thrust::is_trivially_relocatable<int>::value, "");
  static_assert(thrust::is_trivially_relocatable<__half>::value, "");
  static_assert(thrust::is_trivially_relocatable<int1>::value, "");
  static_assert(thrust::is_trivially_relocatable<int2>::value, "");
  static_assert(thrust::is_trivially_relocatable<int3>::value, "");
  static_assert(thrust::is_trivially_relocatable<int4>::value, "");
#if !(THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC                                          \
      || (THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_NVRTC && !defined(__CUDACC_RTC_INT128__)) \
      || (defined(__NVCC__) && __CUDACC_VER_MAJOR__ * 100 + __CUDACC_VER_MINOR__ < 1105)         \
      || !defined(__SIZEOF_INT128__))
  static_assert(thrust::is_trivially_relocatable<__int128>::value, "");
#endif // (THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC || (THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_NVRTC &&
       // !defined(__CUDACC_RTC_INT128__)) || (defined(__NVCC__) && __CUDACC_VER_MAJOR__ * 100 + __CUDACC_VER_MINOR__ <
       // 1105) || !defined(__SIZEOF_INT128__))
#if defined(THRUST_GCC_VERSION) && THRUST_GCC_VERSION >= 70000
  static_assert(thrust::is_trivially_relocatable<thrust::complex<float>>::value, "");
  static_assert(thrust::is_trivially_relocatable<_THRUST_STD::complex<float>>::value, "");
  static_assert(thrust::is_trivially_relocatable<thrust::pair<int, thrust::complex<float>>>::value, "");
  static_assert(thrust::is_trivially_relocatable<_THRUST_STD::pair<int, _THRUST_STD::complex<float>>>::value, "");
  static_assert(thrust::is_trivially_relocatable<thrust::tuple<int, thrust::complex<float>, char>>::value, "");
  static_assert(thrust::is_trivially_relocatable<_THRUST_STD::tuple<int, _THRUST_STD::complex<float>, char>>::value,
                "");
#endif // defined(THRUST_GCC_VERSION) && THRUST_GCC_VERSION >= 70000
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
};
DECLARE_UNITTEST(TestTriviallyRelocatable);
