/*
 *  Copyright© 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <thrust/async/scan.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "test_header.hpp"

TESTS_DEFINE(AsyncScanTests, NumericalTestsParams)

enum wait_policy
{
  wait_for_futures,
  do_not_wait_for_futures
};

#define FIRST_ARG(N, ...)  N
#define SECOND_ARG(N, ...) FIRST_ARG(__VA_ARGS__)

#define DEFINE_INCLUSIVE_SCAN_INVOKER(name, ...)                                                                \
  template <typename T>                                                                                         \
  struct name                                                                                                   \
  {                                                                                                             \
    template <typename ForwardIt, typename Sentinel, typename OutputIt>                                         \
    __host__ static void sync(ForwardIt&& first, Sentinel&& last, OutputIt&& out)                               \
    {                                                                                                           \
      ::thrust::inclusive_scan(                                                                                 \
        THRUST_FWD(first),                                                                                      \
        THRUST_FWD(last),                                                                                       \
        THRUST_FWD(out) THRUST_PP_COMMA_IF(THRUST_PP_ARITY(FIRST_ARG(__VA_ARGS__))) FIRST_ARG(__VA_ARGS__));    \
    }                                                                                                           \
                                                                                                                \
    template <typename ForwardIt, typename Sentinel, typename OutputIt>                                         \
    __host__ static auto async(ForwardIt&& first, Sentinel&& last, OutputIt&& out)                              \
      THRUST_DECLTYPE_RETURNS(::thrust::async::inclusive_scan(                                                  \
        SECOND_ARG(__VA_ARGS__) THRUST_PP_COMMA_IF(THRUST_PP_ARITY(SECOND_ARG(__VA_ARGS__))) THRUST_FWD(first), \
        THRUST_FWD(last),                                                                                       \
        THRUST_FWD(out) THRUST_PP_COMMA_IF(THRUST_PP_ARITY(FIRST_ARG(__VA_ARGS__))) FIRST_ARG(__VA_ARGS__)))    \
  } /**/

#define DEFINE_EXCLUSIVE_SCAN_INVOKER(name, init, ...)                                                          \
  template <typename T>                                                                                         \
  struct name                                                                                                   \
  {                                                                                                             \
    template <typename ForwardIt, typename Sentinel, typename OutputIt>                                         \
    __host__ static void sync(ForwardIt&& first, Sentinel&& last, OutputIt&& out)                               \
    {                                                                                                           \
      ::thrust::exclusive_scan(                                                                                 \
        THRUST_FWD(first),                                                                                      \
        THRUST_FWD(last),                                                                                       \
        THRUST_FWD(out),                                                                                        \
        init THRUST_PP_COMMA_IF(THRUST_PP_ARITY(FIRST_ARG(__VA_ARGS__))) FIRST_ARG(__VA_ARGS__));               \
    }                                                                                                           \
                                                                                                                \
    template <typename ForwardIt, typename Sentinel, typename OutputIt>                                         \
    __host__ static auto async(ForwardIt&& first, Sentinel&& last, OutputIt&& out)                              \
      THRUST_DECLTYPE_RETURNS(::thrust::async::exclusive_scan(                                                  \
        SECOND_ARG(__VA_ARGS__) THRUST_PP_COMMA_IF(THRUST_PP_ARITY(SECOND_ARG(__VA_ARGS__))) THRUST_FWD(first), \
        THRUST_FWD(last),                                                                                       \
        THRUST_FWD(out),                                                                                        \
        init THRUST_PP_COMMA_IF(THRUST_PP_ARITY(FIRST_ARG(__VA_ARGS__))) FIRST_ARG(__VA_ARGS__)))               \
  } /**/

DEFINE_INCLUSIVE_SCAN_INVOKER(inclusive_scan_invoker);
DEFINE_INCLUSIVE_SCAN_INVOKER(inclusive_scan_invoker_device, thrust::plus<T>{}, thrust::device);
DEFINE_EXCLUSIVE_SCAN_INVOKER(exclusive_scan_invoker, T(0));
DEFINE_EXCLUSIVE_SCAN_INVOKER(exclusive_scan_invoker_device, T(0), thrust::plus<T>{}, thrust::device);

DEFINE_INCLUSIVE_SCAN_INVOKER(inclusive_scan_invoker_maximum, thrust::maximum<T>{});
DEFINE_INCLUSIVE_SCAN_INVOKER(inclusive_scan_invoker_maximum_device, thrust::maximum<T>{}, thrust::device);
DEFINE_EXCLUSIVE_SCAN_INVOKER(exclusive_scan_invoker_maximum, T(1), thrust::maximum<T>{});
DEFINE_EXCLUSIVE_SCAN_INVOKER(exclusive_scan_invoker_maximum_device, T(1), thrust::maximum<T>{}, thrust::device);

#undef DEFINE_SCAN_INVOKER
#undef DEFINE_SCAN_OP_INVOKER
#undef FIRST_ARG
#undef SECOND_ARD

///////////////////////////////////////////////////////////////////////////////

template <typename T, template <typename> class ScanInvoker, wait_policy WaitPolicy>
void TestAsyncScan()
{
  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size = " << size);
    for (auto seed : get_seeds())
    {
      SCOPED_TRACE(testing::Message() << "with seed= " << seed);

      thrust::host_vector<T> h0_data =
        get_random_data<T>(size, get_default_limits<T>::min(), get_default_limits<T>::max(), seed);
      thrust::device_vector<T> d0_data(h0_data);
      thrust::host_vector<T> h0_output(h0_data);
      thrust::device_vector<T> d0_output(d0_data);

      ScanInvoker<T>::sync(h0_data.begin(), h0_data.end(), h0_output.begin());

      auto f0 = ScanInvoker<T>::async(d0_data.begin(), d0_data.end(), d0_output.begin());

      THRUST_IF_CONSTEXPR (wait_for_futures == WaitPolicy)
      {
        f0.wait();
        test_equality_scan(h0_output, d0_output);
      }
    }
  }
}

TYPED_TEST(AsyncScanTests, AsyncInclusiveScan)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  using T = typename TestFixture::input_type;
  TestAsyncScan<T, inclusive_scan_invoker, wait_for_futures>();
};

TYPED_TEST(AsyncScanTests, AsyncInclusiveScanNoWait)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  using T = typename TestFixture::input_type;
  TestAsyncScan<T, inclusive_scan_invoker, do_not_wait_for_futures>();
};

TYPED_TEST(AsyncScanTests, AsyncInclusiveScanPolicy)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  using T = typename TestFixture::input_type;
  TestAsyncScan<T, inclusive_scan_invoker_device, wait_for_futures>();
};

TYPED_TEST(AsyncScanTests, AsyncInclusiveScanMaximum)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  using T = typename TestFixture::input_type;
  TestAsyncScan<T, inclusive_scan_invoker_maximum, wait_for_futures>();
};

TYPED_TEST(AsyncScanTests, AsyncInclusiveScanMaximumNoWait)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  using T = typename TestFixture::input_type;
  TestAsyncScan<T, inclusive_scan_invoker_maximum, do_not_wait_for_futures>();
};

TYPED_TEST(AsyncScanTests, AsyncInclusiveScanPolicyMaximum)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  using T = typename TestFixture::input_type;
  TestAsyncScan<T, inclusive_scan_invoker_maximum_device, wait_for_futures>();
};

TYPED_TEST(AsyncScanTests, AsyncInclusiveScanPolicyMaximumNoWait)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  using T = typename TestFixture::input_type;
  TestAsyncScan<T, inclusive_scan_invoker_maximum_device, do_not_wait_for_futures>();
};

TYPED_TEST(AsyncScanTests, AsyncExclusiveScan)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  using T = typename TestFixture::input_type;
  TestAsyncScan<T, exclusive_scan_invoker, wait_for_futures>();
};

TYPED_TEST(AsyncScanTests, AsyncExclusiveScanNoWait)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  using T = typename TestFixture::input_type;
  TestAsyncScan<T, exclusive_scan_invoker, do_not_wait_for_futures>();
};

TYPED_TEST(AsyncScanTests, AsyncExclusiveScanPolicy)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  using T = typename TestFixture::input_type;
  TestAsyncScan<T, exclusive_scan_invoker_device, wait_for_futures>();
};

TYPED_TEST(AsyncScanTests, AsyncExclusiveScanMaximum)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  using T = typename TestFixture::input_type;
  TestAsyncScan<T, exclusive_scan_invoker_maximum, wait_for_futures>();
};

TYPED_TEST(AsyncScanTests, AsyncExclusiveScanMaximumNoWait)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  using T = typename TestFixture::input_type;
  TestAsyncScan<T, exclusive_scan_invoker_maximum, do_not_wait_for_futures>();
};

TYPED_TEST(AsyncScanTests, AsyncExclusiveScanPolicyMaximum)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  using T = typename TestFixture::input_type;
  TestAsyncScan<T, exclusive_scan_invoker_maximum_device, wait_for_futures>();
};

TYPED_TEST(AsyncScanTests, AsyncExclusiveScanPolicyMaximumNoWait)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
  using T = typename TestFixture::input_type;
  TestAsyncScan<T, exclusive_scan_invoker_maximum_device, do_not_wait_for_futures>();
};

///////////////////////////////////////////////////////////////////////////////
