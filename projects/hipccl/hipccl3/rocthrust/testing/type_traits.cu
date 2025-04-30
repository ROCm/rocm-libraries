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

#include <unittest/unittest.h>

#include <thrust/detail/type_traits.h>
#include <thrust/type_traits/is_contiguous_iterator.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/device_ptr.h>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>

struct non_pod
{
  // non-pods can have constructors
  non_pod(void)
  {}

  int x; int y;
};

void TestIsPlainOldData(void)
{
    // primitive types
    ASSERT_EQUAL((bool)thrust::detail::is_pod<bool>::value, true);

    ASSERT_EQUAL((bool)thrust::detail::is_pod<char>::value, true);
    ASSERT_EQUAL((bool)thrust::detail::is_pod<signed char>::value, true);
    ASSERT_EQUAL((bool)thrust::detail::is_pod<unsigned char>::value, true);
    
    ASSERT_EQUAL((bool)thrust::detail::is_pod<short>::value, true);
    ASSERT_EQUAL((bool)thrust::detail::is_pod<signed short>::value, true);
    ASSERT_EQUAL((bool)thrust::detail::is_pod<unsigned short>::value, true);

    ASSERT_EQUAL((bool)thrust::detail::is_pod<int>::value, true);
    ASSERT_EQUAL((bool)thrust::detail::is_pod<signed int>::value, true);
    ASSERT_EQUAL((bool)thrust::detail::is_pod<unsigned int>::value, true);
    
    ASSERT_EQUAL((bool)thrust::detail::is_pod<long>::value, true);
    ASSERT_EQUAL((bool)thrust::detail::is_pod<signed long>::value, true);
    ASSERT_EQUAL((bool)thrust::detail::is_pod<unsigned long>::value, true);
    
    ASSERT_EQUAL((bool)thrust::detail::is_pod<long long>::value, true);
    ASSERT_EQUAL((bool)thrust::detail::is_pod<signed long long>::value, true);
    ASSERT_EQUAL((bool)thrust::detail::is_pod<unsigned long long>::value, true);
    
    ASSERT_EQUAL((bool)thrust::detail::is_pod<float>::value, true);
    ASSERT_EQUAL((bool)thrust::detail::is_pod<double>::value, true);
    
    // void
    ASSERT_EQUAL((bool)thrust::detail::is_pod<void>::value, true);

    // structs
    ASSERT_EQUAL((bool)thrust::detail::is_pod<non_pod>::value, false);

    // pointers
    ASSERT_EQUAL((bool)thrust::detail::is_pod<char *>::value, true);
    ASSERT_EQUAL((bool)thrust::detail::is_pod<int *>::value, true);
    ASSERT_EQUAL((bool)thrust::detail::is_pod<int **>::value, true);
    ASSERT_EQUAL((bool)thrust::detail::is_pod<non_pod *>::value, true);

    // const types
    ASSERT_EQUAL((bool)thrust::detail::is_pod<const int>::value, true);
    ASSERT_EQUAL((bool)thrust::detail::is_pod<const int *>::value, true);
}
DECLARE_UNITTEST(TestIsPlainOldData);

void TestIsContiguousIterator(void)
{
    using HostVector  = thrust::host_vector<int>;
    using DeviceVector = thrust::device_vector<int>;
    
    ASSERT_EQUAL((bool) thrust::is_contiguous_iterator< int * >::value, true);
    ASSERT_EQUAL((bool) thrust::is_contiguous_iterator< thrust::device_ptr<int> >::value, true);


    ASSERT_EQUAL((bool) thrust::is_contiguous_iterator<HostVector::iterator>::value, true);
    ASSERT_EQUAL((bool) thrust::is_contiguous_iterator<HostVector::const_iterator>::value, true);

    ASSERT_EQUAL((bool) thrust::is_contiguous_iterator<DeviceVector::iterator>::value, true);
    ASSERT_EQUAL((bool) thrust::is_contiguous_iterator<DeviceVector::const_iterator>::value, true);

    ASSERT_EQUAL((bool) thrust::is_contiguous_iterator< thrust::device_ptr<int> >::value, true);

    using HostIteratorTuple = thrust::tuple<HostVector::iterator, HostVector::iterator>;

    using ConstantIterator  = thrust::constant_iterator<int>;
    using CountingIterator  = thrust::counting_iterator<int>;
    using TransformIterator = thrust::transform_iterator<thrust::identity<int>, HostVector::iterator>;
    using ZipIterator       = thrust::zip_iterator<HostIteratorTuple>;

    ASSERT_EQUAL((bool) thrust::is_contiguous_iterator<ConstantIterator>::value,  false);
    ASSERT_EQUAL((bool) thrust::is_contiguous_iterator<CountingIterator>::value,  false);
    ASSERT_EQUAL((bool) thrust::is_contiguous_iterator<TransformIterator>::value, false);
    ASSERT_EQUAL((bool) thrust::is_contiguous_iterator<ZipIterator>::value,       false);

}
DECLARE_UNITTEST(TestIsContiguousIterator);

void TestIsCommutative(void)
{
  { using T = int; using Op = thrust::plus<T>       ; ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true); }
  { using T = int; using Op = thrust::multiplies<T> ; ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true); }
  { using T = int; using Op = thrust::minimum<T>    ; ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true); }
  { using T = int; using Op = thrust::maximum<T>    ; ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true); }
  { using T = int; using Op = thrust::logical_or<T> ; ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true); }
  { using T = int; using Op = thrust::logical_and<T>; ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true); }
  { using T = int; using Op = thrust::bit_or<T>     ; ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true); }
  { using T = int; using Op = thrust::bit_and<T>    ; ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true); }
  { using T = int; using Op = thrust::bit_xor<T>    ; ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true); }
  
  { using T = char     ; using Op = thrust::plus<T> ; ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true); }
  { using T = short    ; using Op = thrust::plus<T> ; ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true); }
  { using T = long     ; using Op = thrust::plus<T> ; ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true); }
  { using T = long long; using Op = thrust::plus<T> ; ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true); }
  { using T = float    ; using Op = thrust::plus<T> ; ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true); }
  { using T = double   ; using Op = thrust::plus<T> ; ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true); }
  
  { using T = int  ; using Op = thrust::minus<T>  ; ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, false); }
  { using T = int  ; using Op = thrust::divides<T>; ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, false); }
  { using T = float; using Op = thrust::divides<T>; ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, false); }
  { using T = float; using Op = thrust::minus<T>  ; ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, false); }
  
  { using T = thrust::tuple<int,int>; using Op = thrust::plus<T> ; ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, false); }
}
DECLARE_UNITTEST(TestIsCommutative);

