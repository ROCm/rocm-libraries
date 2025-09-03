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

#include <thrust/functional.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/random.h>
#include <thrust/scan.h>

#include "test_param_fixtures.hpp"
#include "test_real_assertions.hpp"
#include "test_utils.hpp"

TESTS_DEFINE(ScanByKeyInclusiveTests, FullTestsParams);

TESTS_DEFINE(ScanByKeyInclusiveVariablesTests, NumericalTestsParams);

TESTS_DEFINE(ScanByKeyInclusiveVectorTests, VectorSignedIntegerTestsParams);

TYPED_TEST(ScanByKeyInclusiveVectorTests, TestInclusiveScanByKeySimple)
{
  using Vector   = typename TestFixture::input_type;
  using T        = typename Vector::value_type;
  using Iterator = typename Vector::iterator;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector keys{0, 1, 1, 1, 2, 3, 3};
  Vector vals{1, 2, 3, 4, 5, 6, 7};
  Vector output(7, 0);

  Iterator iter = thrust::inclusive_scan_by_key(keys.begin(), keys.end(), vals.begin(), output.begin());

  ASSERT_EQ_QUIET(iter, output.end());

  Vector ref{1, 2, 5, 9, 5, 6, 13};
  ASSERT_EQ(output, ref);

  thrust::inclusive_scan_by_key(
    keys.begin(), keys.end(), vals.begin(), output.begin(), thrust::equal_to<T>(), thrust::multiplies<T>());

  ref = {1, 2, 6, 24, 5, 6, 42};
  ASSERT_EQ(output, ref);

  thrust::inclusive_scan_by_key(keys.begin(), keys.end(), vals.begin(), output.begin(), thrust::equal_to<T>());

  ref = {1, 2, 5, 9, 5, 6, 13};
  ASSERT_EQ(output, ref);
}

template <typename InputIterator1, typename InputIterator2, typename OutputIterator>
OutputIterator
inclusive_scan_by_key(my_system& system, InputIterator1, InputIterator1, InputIterator2, OutputIterator result)
{
  system.validate_dispatch();
  return result;
}

TEST(ScanByKeyInclusiveTests, TestInclusiveScanByKeyDispatchExplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::inclusive_scan_by_key(sys, vec.begin(), vec.begin(), vec.begin(), vec.begin());

  ASSERT_EQ(true, sys.is_valid());
}

template <typename InputIterator1, typename InputIterator2, typename OutputIterator>
OutputIterator inclusive_scan_by_key(my_tag, InputIterator1, InputIterator1, InputIterator2, OutputIterator result)
{
  *result = 13;
  return result;
}

TEST(ScanByKeyInclusiveTests, TestInclusiveScanByKeyDispatchImplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  thrust::inclusive_scan_by_key(
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.begin()));

  ASSERT_EQ(13, vec.front());
}

struct head_flag_predicate
{
  template <typename T>
  THRUST_HOST_DEVICE bool operator()(const T&, const T& b)
  {
    return b ? false : true;
  }
};

TYPED_TEST(ScanByKeyInclusiveVectorTests, TestScanByKeyHeadFlags)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector keys{0, 1, 0, 0, 1, 1, 0};
  Vector vals{1, 2, 3, 4, 5, 6, 7};

  Vector output(7, 0);

  thrust::inclusive_scan_by_key(
    keys.begin(), keys.end(), vals.begin(), output.begin(), head_flag_predicate(), thrust::plus<T>());

  Vector ref{1, 2, 5, 9, 5, 6, 13};
  ASSERT_EQ(output, ref);
}

TYPED_TEST(ScanByKeyInclusiveVectorTests, TestInclusiveScanByKeyTransformIterator)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector keys{0, 1, 1, 1, 2, 3, 3};
  Vector vals{1, 2, 3, 4, 5, 6, 7};
  Vector output(7, 0);

  thrust::inclusive_scan_by_key(
    keys.begin(), keys.end(), thrust::make_transform_iterator(vals.begin(), thrust::negate<T>()), output.begin());

  Vector ref{-1, -2, -5, -9, -5, -6, -13};
  ASSERT_EQ(output, ref);
}

TYPED_TEST(ScanByKeyInclusiveVectorTests, TestScanByKeyReusedKeys)
{
  using Vector = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector keys{0, 1, 1, 1, 0, 1, 1};
  Vector vals{1, 2, 3, 4, 5, 6, 7};
  Vector output(7, 0);

  thrust::inclusive_scan_by_key(keys.begin(), keys.end(), vals.begin(), output.begin());

  Vector ref{1, 2, 5, 9, 5, 6, 13};
  ASSERT_EQ(output, ref);
}

TYPED_TEST(ScanByKeyInclusiveVariablesTests, TestInclusiveScanByKey)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    thrust::host_vector<int> h_keys(size);
    thrust::default_random_engine rng;
    for (size_t i = 0, k = 0; i < size; i++)
    {
      h_keys[i] = static_cast<int>(k);
      if (rng() % 10 == 0)
      {
        k++;
      }
    }
    thrust::device_vector<int> d_keys = h_keys;

    for (auto seed : get_seeds())
    {
      SCOPED_TRACE(testing::Message() << "with seed= " << seed);

      thrust::host_vector<T> h_vals =
        get_random_data<int>(size, get_default_limits<int>::min(), get_default_limits<int>::max(), seed);
      for (size_t i = 0; i < size; i++)
      {
        h_vals[i] = static_cast<int>(i % 10);
      }
      thrust::device_vector<T> d_vals = h_vals;

      thrust::host_vector<T> h_output(size);
      thrust::device_vector<T> d_output(size);

      thrust::inclusive_scan_by_key(h_keys.begin(), h_keys.end(), h_vals.begin(), h_output.begin());
      thrust::inclusive_scan_by_key(d_keys.begin(), d_keys.end(), d_vals.begin(), d_output.begin());
      ASSERT_EQ(d_output, h_output);
    }
  }
}

TYPED_TEST(ScanByKeyInclusiveVariablesTests, TestInclusiveScanByKeyInPlace)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    thrust::host_vector<int> h_keys(size);
    thrust::default_random_engine rng;
    for (size_t i = 0, k = 0; i < size; i++)
    {
      h_keys[i] = static_cast<int>(k);
      if (rng() % 10 == 0)
      {
        k++;
      }
    }
    thrust::device_vector<int> d_keys = h_keys;

    for (auto seed : get_seeds())
    {
      SCOPED_TRACE(testing::Message() << "with seed= " << seed);

      thrust::host_vector<T> h_vals =
        get_random_data<int>(size, get_default_limits<int>::min(), get_default_limits<int>::max(), seed);
      for (size_t i = 0; i < size; i++)
      {
        h_vals[i] = static_cast<int>(i % 10);
      }
      thrust::device_vector<T> d_vals = h_vals;

      thrust::host_vector<T> h_output(size);
      thrust::device_vector<T> d_output(size);

      // in-place scans: in/out values aliasing
      h_output = h_vals;
      d_output = d_vals;
      thrust::inclusive_scan_by_key(h_keys.begin(), h_keys.end(), h_output.begin(), h_output.begin());
      thrust::inclusive_scan_by_key(d_keys.begin(), d_keys.end(), d_output.begin(), d_output.begin());
      ASSERT_EQ(d_output, h_output);

      // in-place scans: in/out keys aliasing
      thrust::inclusive_scan_by_key(h_keys.begin(), h_keys.end(), h_vals.begin(), h_keys.begin());
      thrust::inclusive_scan_by_key(d_keys.begin(), d_keys.end(), d_vals.begin(), d_keys.begin());
      ASSERT_EQ(d_keys, h_keys);
    }
  }
}

TEST(ScanByKeyInclusiveTests, TestScanByKeyMixedTypes)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  const unsigned int n = 113;

  thrust::host_vector<int> h_keys(n);
  thrust::default_random_engine rng;
  for (size_t i = 0, k = 0; i < n; i++)
  {
    h_keys[i] = static_cast<int>(k);
    if (rng() % 10 == 0)
    {
      k++;
    }
  }
  thrust::device_vector<int> d_keys = h_keys;

  for (auto seed : get_seeds())
  {
    SCOPED_TRACE(testing::Message() << "with seed= " << seed);

    thrust::host_vector<unsigned int> h_vals = get_random_data<unsigned int>(
      n, get_default_limits<unsigned int>::min(), get_default_limits<unsigned int>::max(), seed);
    for (size_t i = 0; i < n; i++)
    {
      h_vals[i] %= 10;
    }
    thrust::device_vector<unsigned int> d_vals = h_vals;

    thrust::host_vector<float> h_float_output(n);
    thrust::device_vector<float> d_float_output(n);
    thrust::host_vector<int> h_int_output(n);
    thrust::device_vector<int> d_int_output(n);

    // mixed vals/output types
    thrust::inclusive_scan_by_key(h_keys.begin(), h_keys.end(), h_vals.begin(), h_float_output.begin());
    thrust::inclusive_scan_by_key(d_keys.begin(), d_keys.end(), d_vals.begin(), d_float_output.begin());
    ASSERT_EQ(d_float_output, h_float_output);
  }
}

TYPED_TEST(ScanByKeyInclusiveVariablesTests, TestScanByKeyDiscardOutput)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    thrust::host_vector<T> h_keys(size);
    thrust::default_random_engine rng;

    for (size_t i = 0, k = 0; i < size; i++)
    {
      h_keys[i] = static_cast<T>(k);
      if (rng() % 10 == 0)
      {
        k++;
      }
    }
    thrust::device_vector<T> d_keys = h_keys;

    thrust::host_vector<T> h_vals(size);
    for (size_t i = 0; i < size; i++)
    {
      h_vals[i] = static_cast<T>(i % 10);
    }
    thrust::device_vector<T> d_vals = h_vals;

    auto out = thrust::make_discard_iterator();

    // These are no-ops, but they should compile.
    thrust::inclusive_scan_by_key(d_keys.cbegin(), d_keys.cend(), d_vals.cbegin(), out);
    thrust::inclusive_scan_by_key(d_keys.cbegin(), d_keys.cend(), d_vals.cbegin(), out, thrust::equal_to<T>{});
    thrust::inclusive_scan_by_key(
      d_keys.cbegin(), d_keys.cend(), d_vals.cbegin(), out, thrust::equal_to<T>{}, thrust::multiplies<T>{});
  }
}

TEST(ScanByKeyInclusiveTests, TestScanByKeyLargeInput)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  const unsigned int N = 1 << 20;

  for (auto seed : get_seeds())
  {
    SCOPED_TRACE(testing::Message() << "with seed= " << seed);

    thrust::host_vector<unsigned int> vals_sizes = get_random_data<unsigned int>(
      10, get_default_limits<unsigned int>::min(), get_default_limits<unsigned int>::max(), seed);

    thrust::host_vector<unsigned int> h_vals = get_random_data<unsigned int>(
      N, get_default_limits<unsigned int>::min(), get_default_limits<unsigned int>::max(), seed + seed_value_addition);
    thrust::device_vector<unsigned int> d_vals = h_vals;

    thrust::host_vector<unsigned int> h_output(N, 0);
    thrust::device_vector<unsigned int> d_output(N, 0);

    for (unsigned int i = 0; i < vals_sizes.size(); i++)
    {
      const unsigned int n = vals_sizes[i] % N;

      // define segments
      thrust::host_vector<unsigned int> h_keys(n);
      thrust::default_random_engine rng;
      for (size_t j = 0, k = 0; j < n; j++)
      {
        h_keys[j] = static_cast<unsigned int>(k);
        if (rng() % 100 == 0)
        {
          k++;
        }
      }
      thrust::device_vector<unsigned int> d_keys = h_keys;

      thrust::inclusive_scan_by_key(h_keys.begin(), h_keys.begin() + n, h_vals.begin(), h_output.begin());
      thrust::inclusive_scan_by_key(d_keys.begin(), d_keys.begin() + n, d_vals.begin(), d_output.begin());
      ASSERT_EQ(d_output, h_output);
    }
  }
}

template <typename T, unsigned int N>
void _TestScanByKeyWithLargeTypes()
{
  size_t n = (64 * 1024) / sizeof(FixedVector<T, N>);

  thrust::host_vector<unsigned int> h_keys(n);
  thrust::host_vector<FixedVector<T, N>> h_vals(n);
  thrust::host_vector<FixedVector<T, N>> h_output(n);

  thrust::default_random_engine rng;
  for (size_t i = 0, k = 0; i < h_vals.size(); i++)
  {
    h_keys[i] = static_cast<unsigned int>(k);
    h_vals[i] = FixedVector<T, N>(static_cast<T>(i));
    if (rng() % 5 == 0)
    {
      k++;
    }
  }

  thrust::device_vector<unsigned int> d_keys      = h_keys;
  thrust::device_vector<FixedVector<T, N>> d_vals = h_vals;
  thrust::device_vector<FixedVector<T, N>> d_output(n);

  thrust::inclusive_scan_by_key(h_keys.begin(), h_keys.end(), h_vals.begin(), h_output.begin());
  thrust::inclusive_scan_by_key(d_keys.begin(), d_keys.end(), d_vals.begin(), d_output.begin());

  ASSERT_EQ_QUIET(h_output, d_output);
}

TEST(ScanByKeyInclusiveTests, TestScanByKeyWithLargeTypes)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  _TestScanByKeyWithLargeTypes<int, 1>();
  _TestScanByKeyWithLargeTypes<int, 2>();
  _TestScanByKeyWithLargeTypes<int, 4>();
  _TestScanByKeyWithLargeTypes<int, 8>();

  // too many resources requested for launch:
  //_TestScanByKeyWithLargeTypes<int,   16>();
  //_TestScanByKeyWithLargeTypes<int,   32>();

  // too large to pass as argument
  //_TestScanByKeyWithLargeTypes<int,   64>();
  //_TestScanByKeyWithLargeTypes<int,  128>();
  //_TestScanByKeyWithLargeTypes<int,  256>();
  //_TestScanByKeyWithLargeTypes<int,  512>();
  //_TestScanByKeyWithLargeTypes<int, 1024>();
}

__global__ THRUST_HIP_LAUNCH_BOUNDS_DEFAULT void
InclusiveScanByKeyKernel(int const N, int* in_array, int* keys_array, int* out_array)
{
  if (threadIdx.x == 0)
  {
    thrust::device_ptr<int> in_begin(in_array);
    thrust::device_ptr<int> in_end(in_array + N);
    thrust::device_ptr<int> keys_begin(keys_array);
    thrust::device_ptr<int> keys_end(keys_array + N);
    thrust::device_ptr<int> out_begin(out_array);

    thrust::inclusive_scan_by_key(thrust::hip::par, keys_begin, keys_end, in_begin, out_begin);
  }
}

TEST(ScanByKeyInclusiveTests, TestInclusiveScanByKeyDevice)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    thrust::host_vector<int> h_keys(size);
    thrust::default_random_engine rng;
    for (size_t i = 0, k = 0; i < size; i++)
    {
      h_keys[i] = k;
      if (rng() % 10 == 0)
      {
        k++;
      }
    }
    thrust::device_vector<int> d_keys = h_keys;

    for (auto seed : get_seeds())
    {
      SCOPED_TRACE(testing::Message() << "with seed= " << seed);

      thrust::host_vector<int> h_vals =
        get_random_data<int>(size, get_default_limits<int>::min(), get_default_limits<int>::max(), seed);
      for (size_t i = 0; i < size; i++)
      {
        h_vals[i] = i % 10;
      }
      thrust::device_vector<int> d_vals = h_vals;

      thrust::host_vector<int> h_output(size);
      thrust::device_vector<int> d_output(size);

      thrust::inclusive_scan_by_key(h_keys.begin(), h_keys.end(), h_vals.begin(), h_output.begin());

      hipLaunchKernelGGL(
        InclusiveScanByKeyKernel,
        dim3(1, 1, 1),
        dim3(128, 1, 1),
        0,
        0,
        size,
        thrust::raw_pointer_cast(&d_vals[0]),
        thrust::raw_pointer_cast(&d_keys[0]),
        thrust::raw_pointer_cast(&d_output[0]));

      ASSERT_EQ(d_output, h_output);
    }
  }
}
