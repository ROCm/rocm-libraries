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

#include <thrust/detail/alignment.h>

#include "test_param_fixtures.hpp"
#include "test_utils.hpp"

struct alignof_mock_0
{
  char a;
  char b;
}; // size: 2 * sizeof(char), alignment: sizeof(char)

struct alignof_mock_1
{
  int n;
  char c;
  // sizeof(int) - sizeof(char) bytes of padding
}; // size: 2 * sizeof(int), alignment: sizeof(int)

struct alignof_mock_2
{
  int n;
  char c;
  // sizeof(int) - sizeof(char) bytes of padding
}; // size: 2 * sizeof(int), alignment: sizeof(int)

struct alignof_mock_3
{
  char c;
  // sizeof(int) - sizeof(char) bytes of padding
  int n;
}; // size: 2 * sizeof(int), alignment: sizeof(int)

struct alignof_mock_4
{
  char c0;
  // sizeof(int) - sizeof(char) bytes of padding
  int n;
  char c1;
  // sizeof(int) - sizeof(char) bytes of padding
}; // size: 3 * sizeof(int), alignment: sizeof(int)

struct alignof_mock_5
{
  char c0;
  char c1;
  // sizeof(int) - 2 * sizeof(char) bytes of padding
  int n;
}; // size: 2 * sizeof(int), alignment: sizeof(int)

struct alignof_mock_6
{
  int n;
  char c0;
  char c1;
  // sizeof(int) - 2 * sizeof(char) bytes of padding
}; // size: 2 * sizeof(int), alignment: sizeof(int)

TEST(AlignmentTests, test_alignof_mocks_sizes)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  ASSERT_EQ(sizeof(alignof_mock_0), 2 * sizeof(char));
  ASSERT_EQ(sizeof(alignof_mock_1), 2 * sizeof(int));
  ASSERT_EQ(sizeof(alignof_mock_2), 2 * sizeof(int));
  ASSERT_EQ(sizeof(alignof_mock_3), 2 * sizeof(int));
  ASSERT_EQ(sizeof(alignof_mock_4), 3 * sizeof(int));
  ASSERT_EQ(sizeof(alignof_mock_5), 2 * sizeof(int));
  ASSERT_EQ(sizeof(alignof_mock_6), 2 * sizeof(int));
}

TEST(AlignmentTests, test_alignof)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  ASSERT_EQ(alignof(bool), sizeof(bool));
  ASSERT_EQ(alignof(signed char), sizeof(signed char));
  ASSERT_EQ(alignof(unsigned char), sizeof(unsigned char));
  ASSERT_EQ(alignof(char), sizeof(char));
  ASSERT_EQ(alignof(short int), sizeof(short int));
  ASSERT_EQ(alignof(unsigned short int), sizeof(unsigned short int));
  ASSERT_EQ(alignof(int), sizeof(int));
  ASSERT_EQ(alignof(unsigned int), sizeof(unsigned int));
  ASSERT_EQ(alignof(long int), sizeof(long int));
  ASSERT_EQ(alignof(unsigned long int), sizeof(unsigned long int));
  ASSERT_EQ(alignof(long long int), sizeof(long long int));
  ASSERT_EQ(alignof(unsigned long long int), sizeof(unsigned long long int));
  ASSERT_EQ(alignof(float), sizeof(float));
  ASSERT_EQ(alignof(double), sizeof(double));
  ASSERT_EQ(alignof(long double), sizeof(long double));

  ASSERT_EQ(alignof(alignof_mock_0), sizeof(char));
  ASSERT_EQ(alignof(alignof_mock_1), sizeof(int));
  ASSERT_EQ(alignof(alignof_mock_2), sizeof(int));
  ASSERT_EQ(alignof(alignof_mock_3), sizeof(int));
  ASSERT_EQ(alignof(alignof_mock_4), sizeof(int));
  ASSERT_EQ(alignof(alignof_mock_5), sizeof(int));
  ASSERT_EQ(alignof(alignof_mock_6), sizeof(int));
}

TEST(AlignmentTests, test_alignment_of)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  ASSERT_EQ(thrust::detail::alignment_of<bool>::value, sizeof(bool));
  ASSERT_EQ(thrust::detail::alignment_of<signed char>::value, sizeof(signed char));
  ASSERT_EQ(thrust::detail::alignment_of<unsigned char>::value, sizeof(unsigned char));
  ASSERT_EQ(thrust::detail::alignment_of<char>::value, sizeof(char));
  ASSERT_EQ(thrust::detail::alignment_of<short int>::value, sizeof(short int));
  ASSERT_EQ(thrust::detail::alignment_of<unsigned short int>::value, sizeof(unsigned short int));
  ASSERT_EQ(thrust::detail::alignment_of<int>::value, sizeof(int));
  ASSERT_EQ(thrust::detail::alignment_of<unsigned int>::value, sizeof(unsigned int));
  ASSERT_EQ(thrust::detail::alignment_of<long int>::value, sizeof(long int));
  ASSERT_EQ(thrust::detail::alignment_of<unsigned long int>::value, sizeof(unsigned long int));
  ASSERT_EQ(thrust::detail::alignment_of<long long int>::value, sizeof(long long int));
  ASSERT_EQ(thrust::detail::alignment_of<unsigned long long int>::value, sizeof(unsigned long long int));
  ASSERT_EQ(thrust::detail::alignment_of<float>::value, sizeof(float));
  ASSERT_EQ(thrust::detail::alignment_of<double>::value, sizeof(double));
  ASSERT_EQ(thrust::detail::alignment_of<long double>::value, sizeof(long double));

  ASSERT_EQ(thrust::detail::alignment_of<alignof_mock_0>::value, sizeof(char));
  ASSERT_EQ(thrust::detail::alignment_of<alignof_mock_1>::value, sizeof(int));
  ASSERT_EQ(thrust::detail::alignment_of<alignof_mock_2>::value, sizeof(int));
  ASSERT_EQ(thrust::detail::alignment_of<alignof_mock_3>::value, sizeof(int));
  ASSERT_EQ(thrust::detail::alignment_of<alignof_mock_4>::value, sizeof(int));
  ASSERT_EQ(thrust::detail::alignment_of<alignof_mock_5>::value, sizeof(int));
  ASSERT_EQ(thrust::detail::alignment_of<alignof_mock_6>::value, sizeof(int));
}

template <std::size_t Align>
void test_aligned_type_instantiation()
{
  using type = typename thrust::detail::aligned_type<Align>::type;
  ASSERT_GE(sizeof(type), 1lu);
  ASSERT_EQ(alignof(type), Align);
  ASSERT_EQ(thrust::detail::alignment_of<type>::value, Align);
}

TEST(AlignmentTests, test_aligned_type)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  test_aligned_type_instantiation<1>();
  test_aligned_type_instantiation<2>();
  test_aligned_type_instantiation<4>();
  test_aligned_type_instantiation<8>();
  test_aligned_type_instantiation<16>();
  test_aligned_type_instantiation<32>();
  test_aligned_type_instantiation<64>();
  test_aligned_type_instantiation<128>();
}

TEST(AlignmentTests, test_max_align_t)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  ASSERT_GE(alignof(thrust::detail::max_align_t), alignof(bool));
  ASSERT_GE(alignof(thrust::detail::max_align_t), alignof(signed char));
  ASSERT_GE(alignof(thrust::detail::max_align_t), alignof(unsigned char));
  ASSERT_GE(alignof(thrust::detail::max_align_t), alignof(char));
  ASSERT_GE(alignof(thrust::detail::max_align_t), alignof(short int));
  ASSERT_GE(alignof(thrust::detail::max_align_t), alignof(unsigned short int));
  ASSERT_GE(alignof(thrust::detail::max_align_t), alignof(int));
  ASSERT_GE(alignof(thrust::detail::max_align_t), alignof(unsigned int));
  ASSERT_GE(alignof(thrust::detail::max_align_t), alignof(long int));
  ASSERT_GE(alignof(thrust::detail::max_align_t), alignof(unsigned long int));
  ASSERT_GE(alignof(thrust::detail::max_align_t), alignof(long long int));
  ASSERT_GE(alignof(thrust::detail::max_align_t), alignof(unsigned long long int));
  ASSERT_GE(alignof(thrust::detail::max_align_t), alignof(float));
  ASSERT_GE(alignof(thrust::detail::max_align_t), alignof(double));
  ASSERT_GE(alignof(thrust::detail::max_align_t), alignof(long double));
}

TEST(AlignmentTests, test_aligned_reinterpret_cast)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::detail::aligned_type<1>* a1 = 0;

  thrust::detail::aligned_type<2>* a2 = 0;

  // Cast to type with stricter (larger) alignment requirement.
  a2 = thrust::detail::aligned_reinterpret_cast<thrust::detail::aligned_type<2>*>(a1);

  // Cast to type with less strict (smaller) alignment requirement.
  a1 = thrust::detail::aligned_reinterpret_cast<thrust::detail::aligned_type<1>*>(a2);
}
