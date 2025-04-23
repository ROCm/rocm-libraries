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

#include "single_heap_gtest.hpp"

using namespace rocshmem;

TEST_F(SingleHeapTestFixture, unallocated_size_check) {
  ASSERT_EQ(single_heap_.get_size(), 1 << 30);
}

TEST_F(SingleHeapTestFixture, unallocated_avail_check) {
  ASSERT_EQ(single_heap_.get_avail(), 1 << 30);
}

TEST_F(SingleHeapTestFixture, unallocated_used_check) {
  ASSERT_EQ(single_heap_.get_used(), 0);
}

TEST_F(SingleHeapTestFixture, free_null) {
  void* ptr{nullptr};
  single_heap_.free(ptr);
}

TEST_F(SingleHeapTestFixture, alloc_0) {
  size_t request_size{0};
  void* ptr{nullptr};

  single_heap_.malloc(&ptr, request_size);
  ASSERT_EQ(ptr, nullptr);

  size_t expected_used{request_size};
  ASSERT_EQ(single_heap_.get_used(), expected_used);
  size_t expected_avail{single_heap_.get_size() - expected_used};
  ASSERT_EQ(single_heap_.get_avail(), expected_avail);

  single_heap_.free(ptr);

  expected_used = 0;
  ASSERT_EQ(single_heap_.get_used(), expected_used);
  expected_avail = single_heap_.get_size();
  ASSERT_EQ(single_heap_.get_avail(), expected_avail);
}

TEST_F(SingleHeapTestFixture, alloc_1) {
  size_t request_size{1};
  void* ptr{nullptr};

  single_heap_.malloc(&ptr, request_size);
  ASSERT_NE(ptr, nullptr);

  size_t expected_used{128};
  ASSERT_EQ(single_heap_.get_used(), expected_used);
  size_t expected_avail{single_heap_.get_size() - expected_used};
  ASSERT_EQ(single_heap_.get_avail(), expected_avail);

  single_heap_.free(ptr);

  expected_used = 0;
  ASSERT_EQ(single_heap_.get_used(), expected_used);
  expected_avail = single_heap_.get_size();
  ASSERT_EQ(single_heap_.get_avail(), expected_avail);
}

TEST_F(SingleHeapTestFixture, alloc_256) {
  size_t request_size{256};
  void* ptr{nullptr};

  single_heap_.malloc(&ptr, request_size);
  ASSERT_NE(ptr, nullptr);

  size_t expected_used{request_size};
  ASSERT_EQ(single_heap_.get_used(), expected_used);
  size_t expected_avail{single_heap_.get_size() - expected_used};
  ASSERT_EQ(single_heap_.get_avail(), expected_avail);

  single_heap_.free(ptr);

  expected_used = 0;
  ASSERT_EQ(single_heap_.get_used(), expected_used);
  expected_avail = single_heap_.get_size();
  ASSERT_EQ(single_heap_.get_avail(), expected_avail);
}

TEST_F(SingleHeapTestFixture, alloc_1024) {
  size_t request_size{1024};
  void* ptr{nullptr};

  single_heap_.malloc(&ptr, request_size);
  ASSERT_NE(ptr, nullptr);

  size_t expected_used{request_size};
  ASSERT_EQ(single_heap_.get_used(), expected_used);
  size_t expected_avail{single_heap_.get_size() - expected_used};
  ASSERT_EQ(single_heap_.get_avail(), expected_avail);

  single_heap_.free(ptr);

  expected_used = 0;
  ASSERT_EQ(single_heap_.get_used(), expected_used);
  expected_avail = single_heap_.get_size();
  ASSERT_EQ(single_heap_.get_avail(), expected_avail);
}

TEST_F(SingleHeapTestFixture, alloc_4097) {
  size_t request_size{4097};
  void* ptr{nullptr};

  single_heap_.malloc(&ptr, request_size);
  ASSERT_NE(ptr, nullptr);

  size_t expected_used{8192};
  ASSERT_EQ(single_heap_.get_used(), expected_used);
  size_t expected_avail{single_heap_.get_size() - expected_used};
  ASSERT_EQ(single_heap_.get_avail(), expected_avail);

  single_heap_.free(ptr);

  expected_used = 0;
  ASSERT_EQ(single_heap_.get_used(), expected_used);
  expected_avail = single_heap_.get_size();
  ASSERT_EQ(single_heap_.get_avail(), expected_avail);
}
