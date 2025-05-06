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

#include <thrust/limits.h>
#include <thrust/async/copy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "test_header.hpp"

TESTS_DEFINE(AsyncCopyTests, NumericalTestsParams);

#define DEFINE_ASYNC_COPY_CALLABLE(name, ...)                                 \
  struct THRUST_PP_CAT2(name, _fn)                                            \
  {                                                                           \
    template <typename ForwardIt, typename Sentinel, typename OutputIt>       \
    __host__                                                                  \
    auto operator()(                                                          \
      ForwardIt&& first, Sentinel&& last, OutputIt&& output                   \
    ) const                                                                   \
    THRUST_RETURNS(                                                           \
      ::thrust::async::copy(                                                  \
        __VA_ARGS__                                                           \
        THRUST_PP_COMMA_IF(THRUST_PP_ARITY(__VA_ARGS__))                      \
        THRUST_FWD(first), THRUST_FWD(last), THRUST_FWD(output)               \
      )                                                                       \
    )                                                                         \
  };                                                                          \
  /**/

DEFINE_ASYNC_COPY_CALLABLE(
  invoke_async_copy
);

DEFINE_ASYNC_COPY_CALLABLE(
  invoke_async_copy_host,   thrust::host
);
DEFINE_ASYNC_COPY_CALLABLE(
  invoke_async_copy_device, thrust::device
);

DEFINE_ASYNC_COPY_CALLABLE(
  invoke_async_copy_host_to_device,    thrust::host,   thrust::device
);
DEFINE_ASYNC_COPY_CALLABLE(
  invoke_async_copy_device_to_host,    thrust::device, thrust::host
);
DEFINE_ASYNC_COPY_CALLABLE(
  invoke_async_copy_host_to_host,      thrust::host,   thrust::host
);
DEFINE_ASYNC_COPY_CALLABLE(
  invoke_async_copy_device_to_device,  thrust::device, thrust::device
);

#undef DEFINE_ASYNC_COPY_CALLABLE

template <typename T, typename AsyncCopyCallable>
void AsyncCopyHostToDevice()
{
  for(auto size : get_sizes())
  {
      SCOPED_TRACE(testing::Message() << "with size = " << size);
      for(auto seed : get_seeds())
      {
          SCOPED_TRACE(testing::Message() << "with seed= " << seed);

          thrust::host_vector<T>   h0 = get_random_data<T>(
              size, get_default_limits<T>::min(), get_default_limits<T>::max(), seed);
          thrust::device_vector<T> d0(size);

          auto f0 = AsyncCopyCallable{}(
            h0.begin(), h0.end(), d0.begin()
          );

          f0.wait();

          ASSERT_EQ(h0, d0);
      }
  }
}

TYPED_TEST(AsyncCopyTests, TestAsyncTriviallyRelocatableElementsHostToDevice)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
    using T = typename TestFixture::input_type;
    AsyncCopyHostToDevice<T, invoke_async_copy_fn>();
};

TYPED_TEST(AsyncCopyTests, TestAsyncTriviallyRelocatableElementsHostToDevicePolicies)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
    using T = typename TestFixture::input_type;
    AsyncCopyHostToDevice<T, invoke_async_copy_host_to_device_fn>();
};

template <typename T, typename AsyncCopyCallable>
void AsyncCopyDeviceToHost()
{
  for(auto size : get_sizes())
  {
      SCOPED_TRACE(testing::Message() << "with size = " << size);
      for(auto seed : get_seeds())
      {
          SCOPED_TRACE(testing::Message() << "with seed= " << seed);

          thrust::host_vector<T>   h0 = get_random_data<T>(
              size, get_default_limits<T>::min(), get_default_limits<T>::max(), seed);
          thrust::device_vector<T> h1(size);
          thrust::device_vector<T> d0(size);

          thrust::copy(h0.begin(), h0.end(), d0.begin());

          ASSERT_EQ(h0, d0);

          auto f0 = AsyncCopyCallable{}(
            d0.begin(), d0.end(), h1.begin()
          );

          f0.wait();

          ASSERT_EQ(h0, d0);
          ASSERT_EQ(d0, h1);
      }
  }
}

TYPED_TEST(AsyncCopyTests, TestAsyncCopyTriviallyRelocatableDeviceToHost)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
    using T = typename TestFixture::input_type;
    AsyncCopyDeviceToHost<T, invoke_async_copy_fn>();
};

TYPED_TEST(AsyncCopyTests, TestAsyncCopyTriviallyRelocatableDeviceToHostPolicies)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
    using T = typename TestFixture::input_type;
    AsyncCopyDeviceToHost<T, invoke_async_copy_device_to_host_fn>();
};

/*TYPED_TEST(AsyncCopyTests, TestAsyncCopyDevicetoDevice)
{
    using T = typename TestFixture::input_type;
    for(auto size : get_sizes())
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);
        for(auto seed : get_seeds())
        {
            SCOPED_TRACE(testing::Message() << "with seed= " << seed);

            thrust::host_vector<T>   h0_data = get_random_data<T>(
                size, get_default_limits<T>::min(), get_default_limits<T>::max(), seed);
            thrust::device_vector<T> d0_data(size);
            thrust::device_vector<T> d1_data(size);

            thrust::copy(h0_data.begin(), h0_data.end(), d0_data.begin());

            ASSERT_EQ(h0_data, d0_data);

            auto f0 = AsyncCopyCallable{}(
              d0.begin(), d0.end(), d1.begin()
            );

            f0.wait();

            ASSERT_EQ(h0_data, d0_data);
            ASSERT_EQ(d0_data, d1_data);
        }
    }
}

TYPED_TEST(AsyncCopyTests, TestAsyncCopyDevicetoDeviceWithPolicy)
{
    using T = typename TestFixture::input_type;
    for(auto size : get_sizes())
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);
        for(auto seed : get_seeds())
        {
            SCOPED_TRACE(testing::Message() << "with seed= " << seed);

            thrust::host_vector<T>   h0_data = get_random_data<T>(
                size, get_default_limits<T>::min(), get_default_limits<T>::max(), seed);
            thrust::device_vector<T> d0_data(size);
            thrust::device_vector<T> d1_data(size);

            thrust::copy(h0_data.begin(), h0_data.end(), d0_data.begin());

            ASSERT_EQ(h0_data, d0_data);

            auto f0 = AsyncCopyCallable{}(
              first, last, d1.begin()
            );

            f0.wait();

            ASSERT_EQ(h0_data, d0_data);
            ASSERT_EQ(d0_data, d1_data);
        }
    }
};*/

// TODO: device_to_device implicit.

// TODO: device_to_device NonContiguousIterator input (counting_iterator).

// TODO: device_to_device NonContiguousIterator output (discard_iterator).

// TODO: host_to_device non trivially relocatable.

// TODO: device_to_host non trivially relocatable.

// TODO: host_to_device NonContiguousIterator input (counting_iterator).

// TODO: host_to_device NonContiguousIterator output (discard_iterator).

// TODO: device_to_host NonContiguousIterator input (counting_iterator).

// TODO: device_to_host NonContiguousIterator output (discard_iterator).

// TODO: Mixed types, needs loosening of `is_trivially_relocatable_to` logic.

// TODO: H->D copy, then dependent D->H copy (round trip).
// Can't do this today because we can't do cross-system with explicit policies.
