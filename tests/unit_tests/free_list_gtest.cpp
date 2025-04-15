/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#include "free_list_gtest.hpp"

#include <thrust/sort.h>

#include "../src/util.hpp"

using namespace rocshmem;

/*****************************************************************************
 ******************************* Fixture Tests *******************************
 *****************************************************************************/

namespace rocshmem {

template <typename List, typename Value>
__global__ void pop_all(List* list, Value* values, const std::size_t count) {
  const auto stride = blockDim.x * gridDim.x;
  const auto thread_index = blockIdx.x * blockDim.x + threadIdx.x;
  // One push per block. block size is always WF_SIZE
  for (std::size_t i = thread_index; i < count * WF_SIZE; i += stride) {
    if (is_thread_zero_in_wave()) {
      auto last = list->pop_front();
      if (values != nullptr) {
        values[i / WF_SIZE] = last.value;
      }
    }
  }
}

template <typename List, typename Value>
__global__ void push_all(List* list, const Value* values,
                         const std::size_t count) {
  const auto stride = blockDim.x * gridDim.x;
  const auto thread_index = blockIdx.x * blockDim.x + threadIdx.x;
  // One push per block. block size is always WF_SIZE
  for (std::size_t i = thread_index; i < count * WF_SIZE; i += stride) {
    if (is_thread_zero_in_wave()) {
      list->push_back(values[i / WF_SIZE]);
    }
  }
}

template <typename List>
__global__ void pop_empty(List* list, bool* empty) {
  auto pop_result = list->pop_front();
  *empty = !pop_result.success;
}
}  // namespace rocshmem

TYPED_TEST(FreeListTestFixture, pop_empty_device) {
  using Allocator = typename TestFixture::Allocator;
  using T = typename TestFixture::T;

  auto& h_input = this->h_input;
  auto& d_input = this->d_input;
  auto& free_list = this->free_list;

  FreeListProxy<Allocator, T> empty_list_proxy{};
  FreeList<T, Allocator>* empty_free_list{empty_list_proxy.get()};

  thrust::device_vector<bool> is_empty(1);
  rocshmem::pop_empty<<<1, 1>>>(empty_free_list, is_empty.data().get());
  CHECK_HIP(hipDeviceSynchronize());
  EXPECT_TRUE(is_empty[0]);
}

TYPED_TEST(FreeListTestFixture, push_host_pop_device) {
  using Allocator = typename TestFixture::Allocator;
  using T = typename TestFixture::T;

  auto& h_input = this->h_input;
  auto& d_input = this->d_input;
  auto& free_list = this->free_list;

  thrust::device_vector<T> results(h_input.size());
  const auto block_size = WF_SIZE;
  rocshmem::pop_all<<<1, block_size>>>(free_list, results.data().get(),
                                       results.size());
  CHECK_HIP(hipDeviceSynchronize());

  for (std::size_t i = 0; i < results.size(); i++) {
    EXPECT_EQ(results[i], h_input[i]);
  }

  thrust::device_vector<bool> is_empty(1);
  rocshmem::pop_empty<<<1, 1>>>(free_list, is_empty.data().get());
  CHECK_HIP(hipDeviceSynchronize());

  EXPECT_TRUE(is_empty[0]);
}

TYPED_TEST(FreeListTestFixture, push_host_concurrent_pop_device) {
  using Allocator = typename TestFixture::Allocator;
  using T = typename TestFixture::T;

  auto& h_input = this->h_input;
  auto& d_input = this->d_input;
  auto& free_list = this->free_list;

  thrust::device_vector<T> results(h_input.size());
  const auto num_blocks = h_input.size();
  const auto block_size = WF_SIZE;
  rocshmem::pop_all<<<num_blocks, block_size>>>(free_list, results.data().get(),
                                                results.size());
  CHECK_HIP(hipDeviceSynchronize());

  // sort to guarantee that the ordering is correct
  thrust::sort(results.begin(), results.end());
  thrust::sort(h_input.begin(), h_input.end());

  for (std::size_t i = 0; i < results.size(); i++) {
    EXPECT_EQ(results[i], h_input[i]);
  }

  thrust::device_vector<bool> is_empty(1);
  rocshmem::pop_empty<<<1, 1>>>(free_list, is_empty.data().get());
  CHECK_HIP(hipDeviceSynchronize());

  EXPECT_TRUE(is_empty[0]);
}

TYPED_TEST(FreeListTestFixture, push_host_pop_push_device) {
  using Allocator = typename TestFixture::Allocator;
  using T = typename TestFixture::T;
  using FreeListType = FreeList<T, Allocator>;

  auto& h_input = this->h_input;
  auto& d_input = this->d_input;
  auto& free_list = this->free_list;

  const auto block_size = WF_SIZE;

  rocshmem::pop_all<FreeListType, T><<<1, block_size>>>(free_list, nullptr, 0);
  CHECK_HIP(hipDeviceSynchronize());

  rocshmem::push_all<<<1, 1>>>(free_list, d_input.data().get(), d_input.size());
  CHECK_HIP(hipDeviceSynchronize());

  thrust::device_vector<T> results(d_input.size());
  rocshmem::pop_all<<<1, block_size>>>(free_list, results.data().get(),
                                       results.size());

  for (std::size_t i = 0; i < results.size(); i++) {
    EXPECT_EQ(results[i], h_input[i]);
  }
}

TYPED_TEST(FreeListTestFixture, push_host_pop_concurrent_push_device) {
  using Allocator = typename TestFixture::Allocator;
  using T = typename TestFixture::T;
  using FreeListType = FreeList<T, Allocator>;

  auto& h_input = this->h_input;
  auto& d_input = this->d_input;
  auto& free_list = this->free_list;

  const auto block_size = WF_SIZE;
  rocshmem::pop_all<FreeListType, T><<<1, block_size>>>(free_list, nullptr, 0);
  CHECK_HIP(hipDeviceSynchronize());

  // Concurrently push all values
  const auto num_blocks = h_input.size();
  rocshmem::push_all<<<num_blocks, block_size>>>(
      free_list, d_input.data().get(), d_input.size());
  CHECK_HIP(hipDeviceSynchronize());

  thrust::device_vector<T> results(d_input.size());
  rocshmem::pop_all<<<1, block_size>>>(free_list, results.data().get(),
                                       results.size());

  // Sort to guarantee that the ordering is correct
  thrust::sort(results.begin(), results.end());
  thrust::sort(h_input.begin(), h_input.end());

  for (std::size_t i = 0; i < results.size(); i++) {
    EXPECT_EQ(results[i], h_input[i]);
  }
}

TYPED_TEST(FreeListTestFixture, push_host_concurrent_pop_push_device) {
  using Allocator = typename TestFixture::Allocator;
  using T = typename TestFixture::T;
  using FreeListType = FreeList<T, Allocator>;

  auto& h_input = this->h_input;
  auto& d_input = this->d_input;
  auto& free_list = this->free_list;

  const auto block_size = WF_SIZE;
  rocshmem::pop_all<FreeListType, T><<<1, block_size>>>(free_list, nullptr, 0);
  CHECK_HIP(hipDeviceSynchronize());

  // Concurrently push all values
  const auto num_blocks = h_input.size();
  rocshmem::push_all<<<num_blocks, block_size>>>(
      free_list, d_input.data().get(), d_input.size());
  CHECK_HIP(hipDeviceSynchronize());

  // Concurrently pop all values
  thrust::device_vector<T> results(d_input.size());
  rocshmem::pop_all<<<num_blocks, block_size>>>(free_list, results.data().get(),
                                                results.size());

  // Sort to guarantee that the ordering is correct
  thrust::sort(results.begin(), results.end());
  thrust::sort(h_input.begin(), h_input.end());

  for (std::size_t i = 0; i < results.size(); i++) {
    EXPECT_EQ(results[i], h_input[i]);
  }
}
